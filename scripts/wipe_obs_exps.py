#!/usr/bin/env python3
# ═════════════════════════════════════════════════════════════════════
#  wipe_obs_exps.py — 作废轮次的 OBS exps 前缀清扫 (停发重发流程)
#
#  用法 (服务器上, 默认 dry-run):
#    python3 scripts/wipe_obs_exps.py result/prod5_current --exp-name prod5
#    # 确认清单后加 --apply 真删
#
#  场景: 错形发射 (对账红灯) / 中止重发。停引擎 (pkill -f
#  run_experiment.sh) 后远端在跑作业会各自跑完并上传 — 等 ≥2h 再扫;
#  迟到上传的残留会被重发的同 id 覆盖 (重发 id 空间 ⊇ 作废轮),
#  终态自愈。
#
#  守卫 (防脚枪):
#    1. 三方同名: --exp-name == run 目录 launch_env.json 的 EXP_NAME ==
#       remote_config.obs_prefix 末段 — 指错目录/名字即拒
#    2. 已收官轮次拒扫: run 目录存在 optimal_mixture_weights.json
#       (搜索收官才写, climb_pipeline._save_outputs) → 该轮 exps 是
#       搜索存档, 本工具不碰
#    3. 结构性: 前缀固定 = obs_prefix + /exps (无手抄覆盖口, 内部值
#       零硬编码), 只删 {前缀}/exp_XXXX 整目录 — 其他一切 (assets/
#       顶层杂散文件) 不动
#    4. 默认 dry-run; --apply 真删, 删后复验 stat, 幸存者大声报
#    5. search.log 30 分钟内有写入 → 警告引擎可能还在跑
#
#  候选 exp id = 本地 run 目录 exp_XXXX 清单 ∪ OBS 探测 (0..511,
#  连续 64 缺即止) — 迟到上传 (本地清单之后落 OBS 的) 也被探测兜住。
#  mount 后端 delete(目录) = rmtree 全递归, mock 同形可测; esdk 路径
#  的目录删除是浅层的 (单层键列举) — 本工具按 mount 服务器设计。
# ═════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import re
import sys
import time

_EXP_RE = re.compile(r"exp_(\d{4})")


def _bootstrap_paths():
    for _p in ("src", "climbmix-ma"):
        _d = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", _p))
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)


def main():
    ap = argparse.ArgumentParser(
        description="作废轮次的 OBS exps 前缀清扫 (默认 dry-run)")
    ap.add_argument("run_dir",
                    help="run 目录 (含 launch_env.json + remote_config.json)")
    ap.add_argument("--exp-name", required=True,
                    help="轮次名 — 必须与 launch_env EXP_NAME 和 obs_prefix "
                         "末段三方一致")
    ap.add_argument("--apply", action="store_true",
                    help="真删 (默认 dry-run)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    rc_path = os.path.join(run_dir, "remote_config.json")
    le_path = os.path.join(run_dir, "launch_env.json")
    for p in (rc_path, le_path):
        if not os.path.isfile(p):
            raise SystemExit(f"[拒] 缺 {p} — 需要 run 目录里的发射实录")

    with open(le_path) as f:
        env_exp = str(json.load(f).get("EXP_NAME") or "").strip()
    if env_exp != args.exp_name:
        raise SystemExit(f"[拒] --exp-name {args.exp_name!r} != launch_env "
                         f"EXP_NAME {env_exp!r}")

    if os.path.isfile(os.path.join(run_dir, "optimal_mixture_weights.json")):
        raise SystemExit(
            "[拒] run 目录已有 optimal_mixture_weights.json (搜索收官产物) — "
            "已收官轮次的 exps 是存档, 本工具不碰")

    log = os.path.join(run_dir, "search.log")
    if os.path.isfile(log) and time.time() - os.path.getmtime(log) < 1800:
        print(f"[警] search.log {(time.time() - os.path.getmtime(log)) / 60:.0f} "
              f"分钟前仍有写入 — 引擎可能还在跑, 先 pkill -f "
              f"run_experiment.sh 再扫")

    _bootstrap_paths()
    from climbmix.remote.backends import resolve_backend
    from climbmix.remote.remote_executor import RemoteConfig

    remote_config = RemoteConfig.from_json_file(rc_path)
    prefix = remote_config.obs_prefix.rstrip("/")
    tail = prefix.rsplit("/", 1)[-1]
    if tail != args.exp_name:
        raise SystemExit(f"[拒] obs_prefix 末段 {tail!r} != EXP_NAME "
                         f"{args.exp_name!r} — 前缀与轮次对不上")
    exps_prefix = f"{prefix}/exps"
    print(f"[OBS] exps 前缀 = {exps_prefix}")

    obs = resolve_backend(remote_config).make_obs_storage(remote_config)

    # ── 候选: 本地清单 ∪ 探测 (迟到上传兜底) ──
    local = sorted(int(m.group(1)) for m in
                   (_EXP_RE.fullmatch(n) for n in os.listdir(run_dir)) if m)
    ids = set(local)
    if local:
        print(f"[OBS] 本地 run 目录候选 exp {len(local)} 个 "
              f"(id {local[0]:04d}..{local[-1]:04d})")
    probe_ids, i, misses = [], 0, 0
    while i < 512 and misses < 64:
        if obs.stat(f"{exps_prefix}/exp_{i:04d}"):
            probe_ids.append(i)
            misses = 0
        else:
            misses += 1
        i += 1
    late = sorted(set(probe_ids) - ids)
    ids.update(probe_ids)
    print(f"[OBS] 探测 0..{i - 1:04d}: {len(probe_ids)} 个存在"
          + (f", 其中 {len(late)} 个不在本地清单 (迟到上传): "
             + ",".join(f"{x:04d}" for x in late) if late else ""))

    for k in obs.list_objects(exps_prefix):
        if not _EXP_RE.fullmatch(k.rsplit("/", 1)[-1]):
            print(f"[info] exps/ 顶层非 exp_XXXX 条目, 不动: {k}")

    targets = sorted(x for x in ids if obs.stat(f"{exps_prefix}/exp_{x:04d}"))
    print(f"[OBS] 待清 exp 目录 {len(targets)} 个 (其余对象不动)")
    for x in targets:
        uri = f"{exps_prefix}/exp_{x:04d}"
        print(f"[{'DEL ' if args.apply else 'list'}] {uri}")
        if args.apply:
            try:
                obs.delete(uri)
            except Exception as e:
                print(f"[fail] 删除失败: {uri} ({e})")

    if args.apply:
        survivors = [x for x in targets
                     if obs.stat(f"{exps_prefix}/exp_{x:04d}")]
        for x in survivors:
            print(f"[fail] 仍存在: {exps_prefix}/exp_{x:04d}")
        print(f"\n[OBS] 已删除 {len(targets) - len(survivors)} 个 exp 目录"
              + (f", {len(survivors)} 个幸存 (见上, 重跑本命令续扫)"
                 if survivors else ""))
    else:
        print(f"\n[OBS] 可清 {len(targets)} 个 exp 目录 "
              f"(dry-run, 加 --apply 真删)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
