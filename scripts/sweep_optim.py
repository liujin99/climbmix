#!/usr/bin/env python3
# ═════════════════════════════════════════════════════════════════════
#  sweep_optim.py — 清扫无人消费的 mid optim_* 分片 (2026-09-22 裁决)
#
#  用法 (本地, 默认 dry-run):
#    python3 scripts/sweep_optim.py /home/ma-user/work/nanochat_model_dir/mid_checkpoints
#    python3 scripts/sweep_optim.py result/prod4_current \
#        /home/ma-user/work/nanochat_model_dir/mid_checkpoints
#    python3 scripts/sweep_optim.py <上述目录> --apply        # 真删
#
#  用法 (OBS, 服务器上, 同样默认 dry-run):
#    python3 scripts/sweep_optim.py --remote-config <remote_config.json> \
#        --obs-prefix obs://<bucket>/<内部前缀>/prod4/exps
#    # 确认清单无误后加 --apply
#
#  背景: mid_train 每 rank 写 optim_<step>_rank<r>.pt (fp32 动量/双矩,
#  ≈1.5× 权重大小), 但全链路零消费者 — eval 只读 model
#  (load_optimizer=False), warm-start 只读 base_checkpoints, mid 训练无
#  续训路径, 实验级重试 fast-path 只 glob model_*.pt。2026-09-22 起写入
#  端已全部过滤 (worker 上传 / executor 下载 / 归档 copytree / 训练侧
#  即时清扫), 本脚本只回收历史存量: 本地 {base}/mid_checkpoints/{tag}/
#  optim_*、归档 {exp}/mid_checkpoint/optim_*、OBS 上的同款对象。
#
#  守卫 (防脚枪):
#    1. 只删路径含 mid_checkpoint / mid_checkpoints 组件下的 optim_* —
#       base_checkpoints 的 optim 是 d28 单节点臂 warm-start 的必需品,
#       结构性永不相碰 (ROOT 指到 nanochat_model_dir 根也不会扫到它,
#       只会对着 base 侧打印告警)
#    2. --min-age-hours (默认 12): 更新的文件跳过 — 保护在跑的实验
#       (smoke / 臂训练窗口)。OBS 无 mtime API, OBS 模式只对已收官轮次
#       的 exps 前缀使用
#    3. 默认 dry-run; --apply 才真删
#
#  stdlib-only (本地模式); OBS 模式懒加载 climbmix 远端栈。
# ═════════════════════════════════════════════════════════════════════
import argparse
import os
import sys
import time


def _is_mid_path(path: str) -> bool:
    """True = 路径组件里含 mid_checkpoint / mid_checkpoints (可删侧)。"""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    return "mid_checkpoint" in parts or "mid_checkpoints" in parts


def _is_base_path(path: str) -> bool:
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    return "base_checkpoints" in parts or "base_checkpoint" in parts


def iter_local_targets(roots, min_age_ts):
    """yield (path, size) — mid 侧、足够老、名为 optim_* 的文件。

    base 侧的同名文件打印告警并跳过 (结构性守卫, 永不删除)。"""
    for root in roots:
        if not os.path.isdir(root):
            print(f"[skip] 不存在: {root}")
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                if not f.startswith("optim_"):
                    continue
                p = os.path.join(dirpath, f)
                if _is_base_path(p):
                    print(f"[guard] base 侧 optim, 不删: {p}")
                    continue
                if not _is_mid_path(p):
                    print(f"[guard] 不在 mid_checkpoint 路径下, 不删: {p}")
                    continue
                try:
                    st = os.stat(p)
                except OSError as e:
                    print(f"[skip] stat 失败: {p} ({e})")
                    continue
                if st.st_mtime > min_age_ts:
                    age_h = (time.time() - st.st_mtime) / 3600
                    print(f"[guard] 太新 ({age_h:.1f}h < 阈值), 跳过: {p}")
                    continue
                yield p, st.st_size


def sweep_local(roots, min_age_ts, apply):
    total = n = 0
    for p, size in iter_local_targets(roots, min_age_ts):
        n += 1
        total += size
        tag = "DEL " if apply else "list"
        print(f"[{tag}] {p}  ({size / 2**30:.2f} GiB)")
        if apply:
            try:
                os.unlink(p)
            except OSError as e:
                print(f"[fail] 删除失败: {p} ({e})")
    print(f"\n[本地] {'已删除' if apply else '可删除'} {n} 个文件, "
          f"共 {total / 2**30:.2f} GiB" + ("" if apply else " (dry-run, 加 --apply 真删)"))
    return n


def sweep_obs(remote_config_path, obs_prefix, apply):
    """OBS 侧: 递归列出 prefix 下对象, 删 /mid_checkpoint/optim_ 键。

    真实后端的 list_objects 是键前缀列举 (递归); dry-run 先看清单再
    --apply。OBS 无 mtime — 只对已收官轮次的 exps 前缀使用。"""
    for _p in ("src", "climbmix-ma"):
        _d = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", _p))
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)
    from climbmix.remote.backends import resolve_backend
    from climbmix.remote.remote_executor import RemoteConfig

    remote_config = RemoteConfig.from_json_file(remote_config_path)
    bundle = resolve_backend(remote_config)
    obs = bundle.make_obs_storage(remote_config)

    prefix = obs_prefix.rstrip("/")
    keys = obs.list_objects(prefix)
    targets = [k for k in sorted(keys)
               if "/mid_checkpoint/optim_" in k]
    skipped = len(keys) - len(targets)
    print(f"[OBS] {prefix} 下共 {len(keys)} 个对象, "
          f"其中 mid optim {len(targets)} 个 (其余 {skipped} 个不动)")
    for k in targets:
        print(f"[{'DEL ' if apply else 'list'}] {k}")
        if apply:
            try:
                obs.delete(k)
            except Exception as e:
                print(f"[fail] 删除失败: {k} ({e})")
    print(f"\n[OBS] {'已删除' if apply else '可删除'} {len(targets)} 个对象"
          + ("" if apply else " (dry-run, 加 --apply 真删)"))
    return len(targets)


def main():
    ap = argparse.ArgumentParser(
        description="清扫无人消费的 mid optim_* 分片 (默认 dry-run)")
    ap.add_argument("roots", nargs="*",
                    help="本地根目录 (可多个): nanochat mid_checkpoints / "
                         "result/<run> 归档目录")
    ap.add_argument("--apply", action="store_true", help="真删 (默认 dry-run)")
    ap.add_argument("--min-age-hours", type=float, default=12.0,
                    help="文件最小年龄 (小时, 默认 12) — 保护在跑的实验")
    ap.add_argument("--remote-config", default=None,
                    help="OBS 模式: remote_config.json 路径")
    ap.add_argument("--obs-prefix", default=None,
                    help="OBS 模式: 已收官轮次的 exps 前缀 "
                         "(obs://bucket/.../<run>/exps)")
    args = ap.parse_args()

    if not args.roots and not (args.remote_config and args.obs_prefix):
        ap.error("需要本地 roots 或 --remote-config + --obs-prefix")

    min_age_ts = time.time() - args.min_age_hours * 3600
    n = 0
    if args.roots:
        n += sweep_local(args.roots, min_age_ts, args.apply)
    if args.remote_config and args.obs_prefix:
        n += sweep_obs(args.remote_config, args.obs_prefix, args.apply)

    if not args.apply and n:
        print("\n[提示] 以上为 dry-run 清单; 确认后加 --apply 执行删除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
