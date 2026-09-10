#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  preflight_launch.py — 远端发射前自检 (固化 prod2 五轮修复的坑)
#
#  用法 (服务器, 仓库根目录):
#    python3 scripts/diagnostics/preflight_launch.py \
#        --env /tmp/prod2_remote.env \
#        --run-dir result/prod2_k15bal_current \
#        --main-log prod2_k15bal.log
#    # env 与 run-dir 至少给一个; 缺的检查项跳过并说明
#
#  检查项 (对应五坑 + 清洁度):
#    1. flavor 卡数一致性 — 解析将生效的 flavor (REMOTE_FLAVOR || 平台
#       default_flavor) vs REMOTE_NPU_PER_JOB; 4 卡坑
#    2. priority — vendored 后端 _build_job_body 带 schedule_policy +
#       平台配置 job_priority (默认 1); priority-0 坑
#    3. 资产挂载活性 — REMOTE_ASSET_MOUNTS 每个 obs:// uri stat 得到;
#       搜索舰队要求 d20 在列 (asset-死路径坑 + d20-漏挂坑)
#    4. 进程清洁 — run_climbmix.sh / dispatch_target_arm.py /
#       arm_watcher 残留 → 红灯 (先杀后取消!)
#    5. audit 短路 — target_arm_random.json 非 SUCCEEDED 会拒绝重发
#    6. 非终态历史作业 — 先杀 dispatcher 再 API cancel
#    7. remote_config.json (若已存在) 与 env 一致性 — 信息项
#
#  退出码: 0 = 全绿/黄, 1 = 有红灯。发射前必须全绿。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob as globmod
import json
import os
import re
import subprocess
import sys

SUBMIT_RE = re.compile(r"\[([^\]]+)\] submitted job ([0-9a-f-]{8,})")
FLAVOR_CARDS_RE = re.compile(r"(\d+)\s*x", re.IGNORECASE)

RED, YELLOW, GREEN = "RED", "YELLOW", "GREEN"
results = []


def check(level, msg, hint=""):
    results.append((level, msg, hint))
    sym = {"RED": "✗", "YELLOW": "·", "GREEN": "✓"}[level]
    print(f"  [{sym}] {msg}")
    if hint:
        for line in hint.splitlines():
            print(f"        {line}")


def _bootstrap_syspath():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for p in (os.path.join(repo, "src"),
              os.path.join(repo, "climbmix-ma")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return repo


def parse_env_file(path):
    """source 兼容子集: KEY=VALUE, 值可带引号, # 注释, export 前缀。"""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def flavor_cards(flavor):
    m = FLAVOR_CARDS_RE.search(flavor or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description="remote launch preflight")
    ap.add_argument("--env", default="/tmp/prod2_remote.env")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--main-log", default="prod2_k15bal.log")
    ap.add_argument("--skip-d20", action="store_true",
                    help="arms-only 发射 (搜索舰队不需要时不查 d20)")
    ap.add_argument("--skip-jobs", action="store_true",
                    help="跳过历史作业终态检查 (无凭证/离线)")
    args = ap.parse_args()

    repo = _bootstrap_syspath()
    env = parse_env_file(args.env)
    print("═" * 66)
    print(f"  preflight — env={args.env} run_dir={args.run_dir or '(none)'}")
    print("═" * 66)

    rc_file = None
    rc = None
    if args.run_dir:
        rc_file = os.path.join(args.run_dir, "remote_config.json")
        if os.path.isfile(rc_file):
            try:
                from climbmix.remote.remote_executor import RemoteConfig
                rc = RemoteConfig.from_json_file(rc_file)
            except Exception as e:
                check(RED, f"remote_config.json unreadable: {e}")
        else:
            check(YELLOW, f"{rc_file} not found (pre-launch; will be "
                  f"written by the launcher preamble)")
    if not env and rc is None:
        check(RED, "no env file and no remote_config.json — nothing to "
              "check against", "provide --env and/or --run-dir")
        return 1

    # 后端栈 (可缺 — 相关检查降级为 YELLOW)
    ma_cfg = None
    ma_api_mod = None
    try:
        from climbmix.remote.remote_executor import RemoteConfig  # noqa
        if rc is None and env:
            d = {}
            if env.get("REMOTE_OBS_PREFIX"):
                d["obs_prefix"] = env["REMOTE_OBS_PREFIX"]
            if env.get("REMOTE_BACKEND"):
                d["backend"] = env["REMOTE_BACKEND"]
            if env.get("REMOTE_BACKEND_MODULE"):
                d["backend_module"] = env["REMOTE_BACKEND_MODULE"]
            if d.get("backend"):
                rc = RemoteConfig.from_dict(d)
    except ImportError as e:
        check(YELLOW, f"climbmix import failed ({e}) — flavor/priority/"
              f"asset checks degraded")
    try:
        import climbmix_ma.modelarts_job_api as m
        ma_api_mod = m
        from climbmix_ma.modelarts_job_api import load_ma_config
        platform_config = (env.get("REMOTE_PLATFORM_CONFIG")
                           or (rc.platform_config if rc else "") or None)
        ma_cfg = load_ma_config(platform_config or None)
    except Exception as e:
        check(YELLOW, f"platform config unavailable ({type(e).__name__}: "
              f"{e}) — flavor/priority 只能查 env 侧")

    # ── 1. flavor 卡数一致性 ───────────────────────────────────────
    print("── 1. flavor / card count ──")
    env_flavor = env.get("REMOTE_FLAVOR", "")
    rc_flavor = rc.flavor if rc is not None else ""
    ma_default = str((ma_cfg or {}).get("default_flavor") or "")
    resolved = env_flavor or rc_flavor or ma_default
    src = ("env REMOTE_FLAVOR" if env_flavor else
           "remote_config.json" if rc_flavor else "platform default_flavor")
    npu_env = env.get("REMOTE_NPU_PER_JOB", "")
    npu_rc = rc.npu_per_job if rc is not None else None
    cards = flavor_cards(resolved)
    if not resolved:
        check(RED, "no flavor resolves (env/remote_config/platform all "
              "empty) — submit will fail-fast")
    else:
        if src == "platform default_flavor":
            check(YELLOW, f"flavor='{resolved}' from PLATFORM DEFAULT "
                  f"({src})", "explicit REMOTE_FLAVOR recommended — the "
                  "default flipped once already (4-card pitfall)")
        else:
            check(GREEN, f"flavor='{resolved}' (from {src})")
        if cards is None:
            check(YELLOW, f"cannot parse card count from '{resolved}'")
        else:
            ref = int(npu_env or npu_rc or 0)
            if ref and cards != ref:
                check(RED, f"flavor says {cards} cards but "
                      f"REMOTE_NPU_PER_JOB/npu_per_job = {ref}",
                      "flavor 决定实际卡数, npu_per_job 只是记账 — 必须一致")
            else:
                check(GREEN, f"card count consistent: {cards} cards "
                      f"(npu_per_job={ref or '?'})")

    # ── 2. priority ────────────────────────────────────────────────
    print("── 2. job priority ──")
    if ma_api_mod is not None:
        src_file = ma_api_mod.__file__
        body = open(src_file, errors="replace").read()
        if "schedule_policy" in body:
            check(GREEN, "backend carries schedule_policy (b2f0c8a+)")
        else:
            check(RED, f"{src_file} lacks schedule_priority — jobs will "
                  f"launch at gateway default 0",
                  "git pull the vendored backend AND restart any running "
                  "main (running processes keep the old module)")
        prio = (ma_cfg or {}).get("job_priority", 1)
        check(GREEN if int(prio) >= 1 else YELLOW,
              f"platform job_priority = {prio}")
    else:
        check(YELLOW, "backend module not importable — cannot verify "
              "schedule_policy / job_priority")

    # ── 3. 资产挂载活性 ────────────────────────────────────────────
    print("── 3. asset mounts ──")
    mounts = {}
    raw = env.get("REMOTE_ASSET_MOUNTS", "")
    if raw:
        try:
            mounts = json.loads(raw)
        except ValueError as e:
            check(RED, f"REMOTE_ASSET_MOUNTS is not valid JSON: {e}")
    if not mounts and rc is not None:
        am = getattr(rc, "asset_mounts", None)
        if isinstance(am, dict) and am:
            mounts = am
            print(f"        (from remote_config.json)")
    if not mounts:
        check(YELLOW, "no asset mounts found (env/remote_config) — "
              "jobs inherit the platform config's global set")
    if mounts and not args.skip_d20 and "d20" not in mounts:
        check(RED, "d20 NOT in asset mounts — search fleet jobs will "
              "fail at boot (remote_worker fail-fast)",
              "arms-only relaunch? then pass --skip-d20")
    elif mounts and "d20" in mounts:
        check(GREEN, "d20 in asset mounts")
    if mounts:
        obs = None
        if rc is not None:
            try:
                from climbmix.remote.backends import resolve_backend
                obs = resolve_backend(rc).make_obs_storage(rc)
            except Exception as e:
                check(YELLOW, f"obs storage unavailable ({type(e).__name__}: "
                      f"{e}) — existence check skipped")
        if obs is not None:
            for name, uri in sorted(mounts.items()):
                try:
                    ok = bool(obs.stat(uri))
                except Exception as e:
                    ok = False
                    print(f"        (stat error: {e})")
                check(GREEN if ok else RED,
                      f"asset '{name}' -> {uri}"
                      + ("" if ok else "  MISSING on OBS"))

    # ── 4. 进程清洁 ────────────────────────────────────────────────
    print("── 4. leftover processes ──")
    for pat in ("run_climbmix.sh", "dispatch_target_arm.py", "arm_watcher"):
        r = subprocess.run(["pgrep", "-af", pat],
                           capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        if lines:
            check(RED, f"{pat}: {len(lines)} process(es) running",
                  "\n".join("        " + l[:90] for l in lines[:4])
                  + "\n        kill BEFORE cancelling jobs — a live "
                    "dispatcher seeing CANCELLED writes a FAILED audit "
                    "that short-circuits the next random dispatch")
        else:
            check(GREEN, f"{pat}: none")

    # ── 5. audit 短路 ──────────────────────────────────────────────
    print("── 5. arm audit files ──")
    if args.run_dir:
        # Any arm incl. custom ones (docs/reuse_design.md §4.4) —
        # target_arm_<name>.json with a path-safe name.
        import re as _re
        arm_files = sorted(
            f for f in os.listdir(args.run_dir)
            if _re.fullmatch(r"target_arm_[A-Za-z0-9_-]+\.json", f))
        found = 0
        for af in arm_files:
            arm = af[len("target_arm_"):-len(".json")]
            p = os.path.join(args.run_dir, af)
            found += 1
            try:
                d = json.load(open(p))
            except ValueError:
                d = {}
            st = d.get("status", "?")
            if arm == "random" and st != "SUCCEEDED":
                check(RED, f"target_arm_random.json status={st} — next "
                      f"random dispatch will SHORT-CIRCUIT",
                      f"rm {p} before relaunching")
            else:
                check(YELLOW, f"target_arm_{arm}.json status={st} "
                      f"(informational)")
        if not found:
            check(GREEN, "no target_arm_*.json — clean slate")
    else:
        check(YELLOW, "no run-dir — arm audit check skipped")

    # ── 6. 非终态历史作业 ──────────────────────────────────────────
    print("── 6. non-terminal historical jobs ──")
    if args.skip_jobs:
        check(YELLOW, "skipped (--skip-jobs)")
    else:
        sources = [args.main_log]
        if args.run_dir:
            sources.append(os.path.join(args.run_dir, "search.log"))
            sources += sorted(globmod.glob(
                os.path.join(args.run_dir, "dispatch_*.log")))
        ids = set()
        for path in sources:
            if path and os.path.isfile(path):
                with open(path, errors="replace") as f:
                    for line in f:
                        m = SUBMIT_RE.search(line)
                        if m:
                            ids.add(m.group(2))
        if not ids:
            check(YELLOW, f"no 'submitted job' lines found in "
                  f"{args.main_log} (+ dispatch logs) — first launch?")
        else:
            api = None
            try:
                if rc is not None:
                    from climbmix.remote.backends import resolve_backend
                    api = resolve_backend(rc).make_job_api(rc)
            except Exception as e:
                check(YELLOW, f"job api unavailable ({type(e).__name__}: "
                      f"{e}) — status check skipped")
            if api is not None:
                non_terminal = []
                unknown = 0
                errs = 0
                for jid in sorted(ids):
                    try:
                        st = api.status(jid)
                        if st is None or st.value == "UNKNOWN":
                            unknown += 1
                        elif not st.is_terminal:
                            non_terminal.append(jid)
                    except Exception:
                        errs += 1
                if non_terminal:
                    check(RED, f"{len(non_terminal)}/{len(ids)} jobs still "
                          f"non-terminal (PENDING/RUNNING)",
                          "kill dispatchers first (check 4), then cancel "
                          "via API; leaving them running double-books the "
                          "pool")
                else:
                    note = ""
                    if unknown:
                        note += f"; {unknown} UNKNOWN (expired/purged — ok)"
                    if errs:
                        note += f"; {errs} status errors"
                    check(GREEN, f"{len(ids)} historical jobs all terminal"
                          + note)

    # ── 7. env vs remote_config.json ───────────────────────────────
    print("── 7. env vs remote_config.json ──")
    if rc is None or not env:
        check(YELLOW, "one side missing — consistency check skipped")
    else:
        mism = []
        if env.get("REMOTE_FLAVOR") and rc.flavor and \
                env["REMOTE_FLAVOR"] != rc.flavor:
            mism.append(f"flavor: env={env['REMOTE_FLAVOR']} "
                        f"rc={rc.flavor}")
        if env.get("REMOTE_MAX_JOBS") and \
                int(env["REMOTE_MAX_JOBS"]) != rc.max_concurrent_jobs:
            mism.append(f"max_jobs: env={env['REMOTE_MAX_JOBS']} "
                        f"rc={rc.max_concurrent_jobs}")
        if mism:
            check(YELLOW, "; ".join(mism),
                  "remote_config.json is rewritten at every launch — "
                  "this only matters for RESUMED runs reading the old file")
        else:
            check(GREEN, "consistent (flavor / max_jobs)")

    # ── 汇总 ───────────────────────────────────────────────────────
    n_red = sum(1 for r in results if r[0] == RED)
    n_yel = sum(1 for r in results if r[0] == YELLOW)
    print("═" * 66)
    print(f"  PREFLIGHT: {'FAIL' if n_red else 'PASS'} "
          f"({n_red} red, {n_yel} yellow, "
          f"{len(results) - n_red - n_yel} green)"
          + ("" if n_red else "  — safe to launch"))
    print("═" * 66)
    return 1 if n_red else 0


if __name__ == "__main__":
    sys.exit(main())
