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
#    python3 scripts/sweep_optim.py --remote-config result/prod4_current/remote_config.json
#    # exps 前缀自动推导 = remote_config 的 obs_prefix + /exps (run 目录里
#    # 那份的 obs_prefix 就是该轮前缀, 内部值零手抄); 显式覆盖用
#    # --obs-prefix obs://<bucket>/<前缀>/<run>/exps
#    # 确认清单后加 --apply
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
import re
import sys
import time

_EXP_RE = re.compile(r"exp_(\d{4})")


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
    """OBS 侧: 找出并删除 {exps}/exp_XXXX/mid_checkpoint/optim_* 对象。

    obs_prefix 缺省 = {remote_config.obs_prefix}/exps —— 传 run 目录里
    的 remote_config.json（其 obs_prefix 就是该轮前缀）即可, 内部值零
    手抄。候选 exp id 优先取自 run 目录的 exp_XXXX 清单（注入历史轮次
    的 id 从偏移起步, 本地清单才是全量）; 无本地清单时探测兜底。SDK
    后端一次递归列举直出; mount/mock 单层列举走逐 exp 遍历。dry-run
    先看清单再 --apply。OBS 无 mtime — 只对已收官轮次使用。"""
    for _p in ("src", "climbmix-ma"):
        _d = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", _p))
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)
    from climbmix.remote.backends import resolve_backend
    from climbmix.remote.remote_executor import RemoteConfig

    remote_config = RemoteConfig.from_json_file(remote_config_path)
    prefix = (obs_prefix or remote_config.obs_prefix).rstrip("/")
    if not prefix:
        raise SystemExit(
            "obs 前缀为空: remote_config.json 里没有 obs_prefix, 且未传 "
            "--obs-prefix")
    if not obs_prefix:
        prefix = f"{prefix}/exps"
        print(f"[OBS] 前缀推导自 remote_config: {prefix}")
    bundle = resolve_backend(remote_config)
    obs = bundle.make_obs_storage(remote_config)

    # ── 列举: 兼容两种后端形状 ──
    # SDK 后端 list_objects = 键前缀列举 (递归, 一次拿全对象键); mount/mock
    # 后端 = 单层列举 (只回该层文件, 目录不可见)。递归形状直接匹配; 否则
    # 逐 exp 目录列举 mid_checkpoint。候选 exp id 优先取自本地 run 目录
    # (remote_config.json 所在目录) 的 exp_XXXX 清单 —— 注入历史的轮次 exp
    # id 从 len(history) 起步 (inject_history 语义), 前段 id 只在本地物化、
    # 无 OBS 目录, 从 0000 探起会错过真起点; 而远程 exp 的 CSV/meta 落地
    # 本地同名目录, 本地清单 = 全量。无本地清单时兜底探测 (连续 64 缺或
    # 512 封顶)。
    keys = obs.list_objects(prefix)
    targets = [k for k in sorted(set(keys))
               if "/mid_checkpoint/optim_" in k]
    mode = "递归列举"
    if not targets:
        mode = "exp 目录遍历"
        run_dir = os.path.dirname(os.path.abspath(remote_config_path))
        ids = sorted(
            int(m.group(1))
            for m in (_EXP_RE.fullmatch(n) for n in
                      (os.listdir(run_dir) if os.path.isdir(run_dir) else []))
            if m)
        if ids:
            print(f"[OBS] 候选 exp {len(ids)} 个 (本地 run 目录, "
                  f"id {ids[0]:04d}..{ids[-1]:04d})")
        else:
            i, misses = 0, 0
            while i < 512 and misses < 64:
                if obs.stat(f"{prefix}/exp_{i:04d}"):
                    ids.append(i)
                    misses = 0
                else:
                    misses += 1
                i += 1
            print(f"[OBS] 候选 exp {len(ids)} 个 (探测 0..{i - 1:04d})")
        for eid in ids:
            mid_uri = f"{prefix}/exp_{eid:04d}/mid_checkpoint"
            targets.extend(
                k for k in obs.list_objects(mid_uri)
                if k.rsplit("/", 1)[-1].startswith("optim_"))
        targets = sorted(set(targets))
    print(f"[OBS] mid optim {len(targets)} 个 ({mode}; 其余对象不动)")
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
                    help="OBS 模式: remote_config.json 路径 (run 目录里那份的 "
                         "obs_prefix = 该轮前缀, exps 前缀自动推导)")
    ap.add_argument("--obs-prefix", default=None,
                    help="OBS 模式: 显式 exps 前缀 (obs://bucket/.../<run>/exps), "
                         "缺省用 remote_config.obs_prefix/exps")
    args = ap.parse_args()

    if not args.roots and not args.remote_config:
        ap.error("需要本地 roots 或 --remote-config")
    if args.obs_prefix and not args.remote_config:
        ap.error("--obs-prefix 需要 --remote-config (后端身份从那来)")

    min_age_ts = time.time() - args.min_age_hours * 3600
    n = 0
    if args.roots:
        n += sweep_local(args.roots, min_age_ts, args.apply)
    if args.remote_config:
        n += sweep_obs(args.remote_config, args.obs_prefix, args.apply)

    if not args.apply and n:
        print("\n[提示] 以上为 dry-run 清单; 确认后加 --apply 执行删除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
