#!/usr/bin/env python3
"""dispatch_fleet.py — fire the ENTIRE d28 validation fleet from one command.

⑬t (2026-09-28 用户裁决: "执行 run_experiment.sh 之后,就能自动完成所有
事情" / "所有 arm 在 run_experiment.sh 执行时自动发射"): the arm family —
终选臂 climb + top-k 实测候选 (topk_mixture_candidates.json 自动展开为
climb-cfgN) + 基线 uniform/natural/domainfix + base 锚点 — 此前是搜索收官
后 ~6 条人工 dispatch 命令 (runbook §4.6 "全部臂命令可连发")。本编排器把
那次发射变成一次调用: run_experiment.sh Step 4 调它; 它也能**独立**对任何
run 目录跑 (prod5 桥接: 老代码驱动跑完经典两臂后, 剩余臂族 = 对着已收官
目录一条命令)。

Per arm (全部幂等, marker 门控):
  1. landed 即跳过 (.done_eval_<arm>; base 锚点 = eval_base_remote.csv)
  2. 本地备料 (既有 CLI, .done 门控 — 已备即秒过):
       climb        sampled_dataset.parquet → prepare_shards → mix
       uniform      1/K 簇等权基线 (prepare_random_baseline 默认权重)
       climb-cfgNN  topk 候选权重 → prepare_random_baseline --weights
       natural      池占比权重 (gen_natural_weights) → --weights
       domainfix    四域固定配比 → --label-source domain --weights
       base         无备料 (eval-only 锚点, 恒单节点)
  3. dispatch 为**后台进程** (scripts/dispatch_target_arm.py, 日志 →
     dispatch_<arm>.log, 独立 session: 父进程被杀不牵连落地/报告)。
     并发安全性全在 dispatch 内建: per-arm mutex + .validation_fleet
     节点预算注册表 (REMOTE_MAX_VALIDATION_NODES, 默认 16 = 2 臂 × 8
     节点, 超限自动排队) + 报告刷新 flock。
  4. wait 全部返回后 salvage 扫尾: 训练落地 (.done_mid_train) 但评测缺失
     的臂, 补一次 best-effort 本地评测 (runs/lib/target_arm.sh)
  5. 汇总表 + 全齐写 .done_fleet (run_experiment.sh 的完成标记) +
     final_report --auto (幂等; 落地钩子通常已盖过章, 这里兜底)

Fleet contract = **remote-only, fail-loud** (自动流不做本地 torchrun 兜底
—— 7 臂族全fallback到主节点 8 卡 = ~70h 串行灾难; 重跑本脚本即对缺失臂的
幂等重试, 已落地臂零成本秒过)。本地臂路径保留为手动应急手段
(runs/lib/target_arm.sh)。臂作业排队不设限 (2026-09-28 裁决, 与搜索侧
同哲学); 运行时天花板 = dispatch 缺省 9h 多节点 / 13h 单节点
(--job-timeout-h 0 可显式去掉)。

Exit 0 iff 计划内全部臂 landed。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "climbmix-ma"):
    _d = os.path.join(REPO_ROOT, _p)
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatch_target_arm import (  # noqa: E402
    load_launch_env, mix_subprocess_env, run_logged)
from climbmix.utils.io_utils import stage1_pair  # noqa: E402

FLEET_TOKENS = ("climb", "topk", "uniform", "natural", "domainfix", "base")
DEFAULT_ARMS = "climb,topk,uniform,natural,domainfix,base"
# domainfix 四域固定配比 (P-0 预注册; 原值 = quadmix manual_ratio,
# docs/experiment_prod4.md §4e)。域名/顺序 = config/schema_stem.yaml。
DOMAINFIX_WEIGHTS = {"数学": 0.60, "物理": 0.15,
                     "化学": 0.125, "生物学": 0.125}
TOPK_ARM_RE = re.compile(r"^climb-cfg\d+$")


# ── planning (pure-ish: the testable core) ─────────────────────────────

def load_topk_candidates(output_dir: str):
    """topk_mixture_candidates.json → [{"arm": "climb-cfgN", "weights":
    ...}], 候选序 (分数降序)。文件缺失/为空 = fail-loud (搜索收官必有:
    D19 A3 每条终选路径都导出)。"""
    path = os.path.join(output_dir, "topk_mixture_candidates.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"✗ {path} not found — the search stage writes it at close "
            f"(D19 A3); cannot expand the 'topk' fleet token without it")
    try:
        with open(path) as f:
            topk = json.load(f)
        cands = topk.get("candidates") or []
    except (OSError, ValueError) as e:
        raise SystemExit(f"✗ unreadable topk file {path}: {e}")
    if not cands:
        raise SystemExit(f"✗ {path} carries no candidates — refusing to "
                         f"expand 'topk' into an empty arm list")
    out = []
    for c in cands:
        try:
            cid = int(c["config_id"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"✗ topk candidate missing config_id ({e}): "
                             f"{json.dumps(c)[:200]}")
        if "weights" not in c:
            raise SystemExit(f"✗ topk candidate cfg{cid} carries no weights")
        out.append({"arm": f"climb-cfg{cid}", "weights": c["weights"]})
    return out


def expand_arms(spec: str, output_dir: str):
    """FLEET_ARMS tokens → 具体 臂名列表 (保序去重; topk → climb-cfgN)。
    显式 `climb-cfgN` 也合法 (单臂补发 / 排除某候选时直接点名) — 须存在于
    topk json, 否则 fail-loud。"""
    _cands = None

    def cands():
        nonlocal _cands
        if _cands is None:
            _cands = load_topk_candidates(output_dir)
        return _cands

    arms: list = []
    seen = set()
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok == "topk":
            new = [c["arm"] for c in cands()]
        elif TOPK_ARM_RE.match(tok):
            if tok not in {c["arm"] for c in cands()}:
                raise SystemExit(
                    f"✗ explicit fleet arm '{tok}' is not in "
                    f"topk_mixture_candidates.json — point at a real "
                    f"candidate or use the 'topk' token")
            new = [tok]
        elif tok in FLEET_TOKENS:
            new = ["base_eval_check" if tok == "base" else tok]
        else:
            raise SystemExit(
                f"✗ unknown FLEET_ARMS token '{tok}' "
                f"(valid: {','.join(FLEET_TOKENS)} or an explicit "
                f"climb-cfgN; comma list, empty string = skip the fleet "
                f"entirely)")
        for arm in new:
            if arm not in seen:
                seen.add(arm)
                arms.append(arm)
    return arms


def _weights_match(a, b) -> bool:
    """label 键控 dict 权重逐键 allclose (list 形状 = 保守不比)。"""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    keys = set(a) | set(b)
    if not keys:
        return False
    return all(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) <= 1e-9
               for k in keys)


def dedupe_final_duplicate(output_dir: str, plans):
    """no-claim 结构性去重: no-claim 时终选 = best measured = topk#1
    (构造性同一 config) → 该候选的 climb-cfgN 臂与终选臂 climb 的数据
    **逐字节相同** (同一 select_data_by_mixture + 同 seed 42 + 同预算),
    跑两遍 = 纯重复。climb 在计划内或已落地时, 权重与
    optimal_mixture_weights.json 相同的 topk 臂被剔除 (大声注明)。
    要强制跑复本: 手动 dispatch_target_arm --arm <cfg> 并自管
    expected_arms.txt。"""
    opt_path = os.path.join(output_dir, "optimal_mixture_weights.json")
    if not os.path.isfile(opt_path):
        return plans
    try:
        with open(opt_path) as f:
            opt_w = json.load(f)
    except (OSError, ValueError):
        return plans
    if not isinstance(opt_w, dict) or not opt_w:
        return plans
    climb_active = (arm_landed(output_dir, "climb")
                    or any(p["arm"] == "climb" and not p["landed"]
                           for p in plans))
    if not climb_active:
        return plans
    out, dropped = [], []
    for p in plans:
        if (TOPK_ARM_RE.match(p["arm"]) and not p["landed"]
                and _weights_match(p.get("weights_payload"), opt_w)):
            dropped.append(p["arm"])
            continue
        out.append(p)
    for arm in dropped:
        print(f"  [fleet] {arm}: skipped — weights identical to the "
              f"final-selection climb arm (no-claim 终选 = topk#1; 同 "
              f"selector/seed/预算 → 同数据, 跑两遍是纯重复). 要强制复本 = "
              f"手动 dispatch_target_arm --arm {arm} + 自管 expected_arms.txt",
              flush=True)
    return out


def arm_landed(output_dir: str, arm: str) -> bool:
    if arm == "base_eval_check":
        return os.path.isfile(os.path.join(output_dir,
                                           "eval_base_remote.csv"))
    return os.path.isfile(os.path.join(output_dir, f".done_eval_{arm}"))


def check_balanced_profile(output_dir: str, launch_env: dict) -> None:
    """任何簇键控备料前的 run 级结构门 (原 --arm random 生命周期同款:
    K_final == K_ENHANCED + max_share <= 15% — 基线与赢家必须同簇空间)。"""
    path = os.path.join(output_dir, "balanced_profile.json")
    try:
        with open(path) as f:
            prof = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"✗ unreadable balanced profile {path}: {e}")
    k_final = int(prof.get("K_final", 0))
    k_expected = int(launch_env.get("K_ENHANCED") or 0)
    if k_expected and k_final != k_expected:
        raise SystemExit(f"✗ balanced profile K_final={k_final} != "
                         f"K_ENHANCED={k_expected} — refusing to prep "
                         f"cluster-keyed baselines against a changed space")
    max_share = float(prof.get("max_share", 0.0))
    if max_share > 0.15:
        raise SystemExit(f"✗ balanced profile max_share={max_share:.1%} > "
                         f"15% — cluster structure gate would reject this run")


def build_plan(arms, output_dir: str, launch_env: dict):
    """Per-arm execution plan (纯构建, 零副作用 — dry-run 安全):
    {arm, data_dir, weights_payload, prep: [(cmd, kind), ...], landed}。
    weights_payload = 备料期才落盘的权重 (topk 候选 / domainfix 固定配比);
    natural 的权重文件由 prep 首步的 gen_natural_weights 命令产出。
    备料门 = <arm>_mixed 的 .done (mixed 完成即整段跳过;
    prepare_random_baseline 侧再由 <arm>_shards/.done 双检)。
    climb 的输入 sampled_dataset.parquet 与簇键控臂的 stage1 pair /
    balanced_profile 在此 fail-loud。"""
    climbmix_dir = launch_env.get("CLIMBMIX_DIR") or REPO_ROOT
    schema = os.path.join(climbmix_dir, "config", "schema_stem.yaml")
    num_npu = launch_env.get("NUM_NPU") or "8"
    target_tokens = launch_env.get("TARGET_TOKENS") or "1B"
    stem_ratio = launch_env.get("STEM_RATIO") or "0.7"

    def select_cmd(arm, shards, weights_file="", label_source=False):
        cmd = [sys.executable,
               os.path.join(climbmix_dir, "scripts",
                            "prepare_random_baseline.py"),
               "--data-dir", launch_env["DATA_DIR"],
               "--output-dir", shards,
               "--cluster-cache", stage1_npz,
               "--schema", schema,
               "--target-tokens", target_tokens,
               "--seed", "42", "--num-npu", num_npu]
        if weights_file:
            cmd += ["--weights", weights_file]
        if label_source:
            cmd += ["--label-source", "domain"]
        return cmd

    def mix_cmd(shards, mixed):
        return ([sys.executable,
                 os.path.join(climbmix_dir, "scripts", "mix_general_data.py"),
                  "--stem-dir", shards, "--output-dir", mixed,
                  "--climbmix-dir", launch_env["GENERAL_DATA_DIR"],
                  "--stem-ratio", stem_ratio,
                  "--num-workers", num_npu, "--num-npu", num_npu])

    def weights_path(arm):
        return os.path.join(output_dir, "fleet_weights", f"{arm}.json")

    needs_cluster = [a for a in arms
                     if a not in ("climb", "base_eval_check")
                     and not arm_landed(output_dir, a)]
    stage1_npz = ""
    if needs_cluster:
        stage1_npz, _info = stage1_pair(output_dir)
        if not os.path.isfile(stage1_npz):
            raise SystemExit(
                f"✗ stage-1 cluster cache not found (looked for "
                f"{stage1_npz}) — cluster-keyed fleet arms "
                f"({','.join(needs_cluster)}) cannot prep without it")
        check_balanced_profile(output_dir, launch_env)

    topk = {c["arm"]: c for c in load_topk_candidates(output_dir)} \
        if any(TOPK_ARM_RE.match(a) for a in arms) else {}

    plans = []
    for arm in arms:
        plan = {"arm": arm, "data_dir": None, "weights_payload": None,
                "prep": [], "landed": arm_landed(output_dir, arm)}
        if arm == "base_eval_check":
            plans.append(plan)
            continue
        shards = os.path.join(output_dir, f"{arm}_shards")
        mixed = os.path.join(output_dir, f"{arm}_mixed")
        plan["data_dir"] = mixed
        if plan["landed"] or os.path.isfile(os.path.join(mixed, ".done")):
            plans.append(plan)      # landed or data already prepped
            continue
        if arm == "climb":
            sampled = os.path.join(output_dir, "sampled_dataset.parquet")
            if not os.path.isfile(sampled):
                raise SystemExit(
                    f"✗ {sampled} not found — the climb arm consumes the "
                    f"search's final selection (run the search first)")
            plan["prep"] = [
                ([sys.executable,
                  os.path.join(climbmix_dir, "scripts", "prepare_shards.py"),
                  "--input", sampled, "--output-dir", shards,
                  "--num-npu", num_npu], None),
                (mix_cmd(shards, mixed), "mix"),
            ]
        else:
            steps = []
            if not os.path.isfile(os.path.join(shards, ".done")):
                if TOPK_ARM_RE.match(arm):
                    plan["weights_payload"] = topk[arm]["weights"]
                    wf = weights_path(arm)
                elif arm == "natural":
                    wf = os.path.join(output_dir, "fleet_weights",
                                      "natural_weights.json")
                    steps.append((
                        [sys.executable,
                         os.path.join(climbmix_dir, "scripts",
                                      "gen_natural_weights.py"),
                         "--pool-dir", launch_env["DATA_DIR"],
                         "--cluster-cache", stage1_npz,
                         "--output", wf], None))
                elif arm == "domainfix":
                    plan["weights_payload"] = DOMAINFIX_WEIGHTS
                    wf = weights_path(arm)
                elif arm != "uniform":
                    raise SystemExit(f"✗ unhandled fleet arm '{arm}'")
                else:
                    wf = ""
                steps.append((select_cmd(arm, shards, wf,
                                         label_source=(arm == "domainfix")),
                              None))
            steps.append((mix_cmd(shards, mixed), "mix"))
            plan["prep"] = steps
        plans.append(plan)
    return plans


# ── data prep (subprocess; .done-gated by build_plan) ──────────────────

def write_weights(output_dir: str, arm: str, payload) -> str:
    """备料期落盘权重文件 (plan 只带 payload — dry-run 零副作用)。"""
    d = os.path.join(output_dir, "fleet_weights")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{arm}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def run_prep(plan: dict, output_dir: str, launch_env: dict,
             nanochat_dir: str) -> None:
    if plan["weights_payload"] is not None:
        write_weights(output_dir, plan["arm"], plan["weights_payload"])
    for cmd, kind in plan["prep"]:
        env = None
        if kind == "mix":
            env = mix_subprocess_env(os.environ, launch_env, nanochat_dir)
        run_logged(cmd, env=env, label=plan["arm"])


# ── dispatch (background process per arm) ──────────────────────────────

def dispatch_cmd(arm: str, output_dir: str, climbmix_dir: str,
                 data_dir: str, retry_failed: bool):
    cmd = [sys.executable,
           os.path.join(climbmix_dir, "scripts", "dispatch_target_arm.py"),
           "--arm", arm, "--output-dir", output_dir]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if retry_failed:
        cmd += ["--retry-failed"]
    return cmd


def launch(plan: dict, output_dir: str, climbmix_dir: str,
           retry_failed: bool):
    log_path = os.path.join(output_dir, f"dispatch_{plan['arm']}.log")
    log = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            dispatch_cmd(plan["arm"], output_dir, climbmix_dir,
                         plan["data_dir"], retry_failed),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        log.close()
        raise
    print(f"  [fleet] dispatched {plan['arm']} (pid {proc.pid}, "
          f"log: {os.path.basename(log_path)})", flush=True)
    return proc, log


def wait_fleet(running):
    """running = [(plan, proc, log)] — poll loop with a heartbeat (the
    driver log must stay alive for monitoring: arms run ~10h each)."""
    pending = list(running)
    last_beat = time.time()
    while pending:
        alive = []
        for plan, proc, log in pending:
            if proc.poll() is None:
                alive.append((plan, proc, log))
            else:
                log.close()
        pending = alive
        if not pending:
            break
        now = time.time()
        if now - last_beat >= 600.0:
            names = ", ".join(p["arm"] for p, _p, _l in pending)
            print(f"  [fleet] {len(pending)} dispatch in flight: {names}",
                  flush=True)
            last_beat = now
        time.sleep(30.0)


# ── post-wait: salvage evals + summary + .done_fleet + final report ────

def salvage_local_eval(plan: dict, output_dir: str, launch_env: dict,
                       exp_name: str) -> bool:
    """训练落地但评测缺失的臂 → 一次 best-effort 本地评测 (dispatch 的
    salvage 契约: mid_train_rc=0 时落 ckpt + .done_mid_train, 评测回退本
    地 — 编排器接住它, 否则该臂永远缺 eval)。"""
    arm = plan["arm"]
    if (arm == "base_eval_check" or arm_landed(output_dir, arm)
            or not os.path.isfile(os.path.join(output_dir,
                                               f".done_mid_train_{arm}"))):
        return True
    climbmix_dir = launch_env.get("CLIMBMIX_DIR") or REPO_ROOT
    depth = launch_env.get("TARGET_DEPTH") or "28"
    tag = f"d{depth}_{arm}_{exp_name}"
    script = (f'set -e; source "{climbmix_dir}/runs/lib/target_arm.sh"; '
              f'target_arm_eval "{tag}" "{arm}"')
    env = dict(os.environ)
    for k in ("CLIMBMIX_DIR", "NANOCHAT_DIR", "NANOCHAT_BASE_DIR", "NUM_NPU",
              "EVAL_BENCHMARKS", "EVAL_MAX_PER_TASK", "EVAL_DEVICE_BATCH_SIZE",
              "EVAL_CORE_BATCH_SIZE", "STEM_RATIO"):
        if launch_env.get(k):
            env[k] = launch_env[k]
    env["OUTPUT_DIR"] = output_dir
    env["EXP_NAME"] = exp_name
    print(f"  [fleet] salvage: local eval for {arm} (tag {tag}) — "
          f"training landed but eval did not", flush=True)
    r = subprocess.run(["bash", "-c", script], env=env)
    if r.returncode == 0 and os.path.isfile(
            os.path.join(output_dir, f"eval_{arm}.csv")):
        open(os.path.join(output_dir, f".done_eval_{arm}"), "w").close()
        print(f"  [fleet] salvage: {arm} local eval landed", flush=True)
        return True
    print(f"  [fleet] salvage: {arm} local eval FAILED (rc={r.returncode}) "
          f"— manual: source runs/lib/target_arm.sh + target_arm_eval "
          f"{tag} {arm}", flush=True)
    return False


def final_report_auto(output_dir: str, climbmix_dir: str) -> None:
    """落地钩子通常已盖章; 这里幂等兜底一次 (fail-loud 对 DRAFT 只注记)。"""
    fin = os.path.join(climbmix_dir, "scripts", "diagnostics",
                       "final_report.py")
    if not os.path.isfile(fin):
        return
    r = subprocess.run([sys.executable, fin, output_dir, "--auto"],
                       capture_output=True, text=True, timeout=600)
    lines = [ln for ln in (r.stdout or "").strip().splitlines()
             if ln.strip()]
    print(f"  [fleet] final_report --auto: "
          f"{lines[-1] if lines else f'rc={r.returncode}'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="fire the entire d28 validation fleet (one command)")
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", ""),
                    help="run dir (default: $OUTPUT_DIR)")
    ap.add_argument("--arms", default="",
                    help=f"comma list of fleet tokens (default: "
                         f"$FLEET_ARMS or '{DEFAULT_ARMS}'; climb=终选臂 / "
                         f"topk=top-k 实测候选 / uniform / natural / "
                         f"domainfix / base 锚点; '' = no-op)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="forward --retry-failed to dispatches (re-attempt "
                         "arms whose prior remote attempt failed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (arms, prep + dispatch commands) "
                         "and exit without side effects")
    args = ap.parse_args()

    output_dir = args.output_dir
    if not output_dir:
        raise SystemExit("✗ --output-dir or $OUTPUT_DIR required")
    launch_env = load_launch_env(output_dir)
    exp_name = launch_env.get("EXP_NAME") or "main"
    climbmix_dir = launch_env.get("CLIMBMIX_DIR") or REPO_ROOT
    rc_path = os.path.join(output_dir, "remote_config.json")
    if not os.path.isfile(rc_path):
        raise SystemExit(
            f"✗ {rc_path} not found — fleet arms are remote-only "
            f"(REMOTE_ENABLED=1 run); the local torchrun path is the "
            f"manual emergency route (runs/lib/target_arm.sh)")

    spec = args.arms or os.environ.get("FLEET_ARMS", "") or DEFAULT_ARMS
    arms = expand_arms(spec, output_dir)
    plans = build_plan(arms, output_dir, launch_env)
    plans = dedupe_final_duplicate(output_dir, plans)

    # expected_arms.txt: 终报印章等的是**本次舰队的实际计划**(no-claim
    # 去重后), 不是推导清单 — 派发前落册, 落地钩子的 final_report --auto
    # 读它。已存在 = 操作者手动锁定 (历史命名/条件臂) — 尊重, 内容不同
    # 时只提醒 (同步 = 删除该文件后重跑)。
    expected = []
    if (arm_landed(output_dir, "climb")
            or any(p["arm"] == "climb" for p in plans)):
        expected.append("climb")
    expected += [p["arm"] for p in plans
                 if p["arm"] not in ("base_eval_check", "climb")]
    exp_path = os.path.join(output_dir, "expected_arms.txt")
    if expected and not args.dry_run:
        if os.path.isfile(exp_path):
            cur = [ln.strip() for ln in open(exp_path) if ln.strip()]
            if cur != expected:
                print(f"  [fleet] expected_arms.txt 已存在且与本次计划不同"
                      f" (保留现状):\n    现状: {', '.join(cur)}"
                      f"\n    计划: {', '.join(expected)}", flush=True)
        else:
            with open(exp_path, "w") as f:
                f.write("\n".join(expected) + "\n")
            print(f"  [fleet] expected_arms.txt <- {', '.join(expected)} "
                  f"(终报印章按本次舰队计划)", flush=True)

    print(f"[Fleet] {len(plans)} arm(s) from '{spec}': "
          f"{', '.join(p['arm'] for p in plans)}")
    if args.dry_run:
        for p in plans:
            state = ("landed" if p["landed"]
                     else ("data ready" if not p["prep"] else "needs prep"))
            suffix = (f" (data: {os.path.basename(p['data_dir'])})"
                      if p["data_dir"] else "")
            print(f"  - {p['arm']}: {state}{suffix}")
            for cmd, _kind in p["prep"]:
                print(f"      prep: {' '.join(cmd)}")
            if not p["landed"]:
                dcmd = " ".join(dispatch_cmd(
                    p["arm"], output_dir, climbmix_dir, p["data_dir"],
                    args.retry_failed))
                print(f"      dispatch: {dcmd}")
        return 0

    todo = [p for p in plans if not p["landed"]]
    if not todo:
        print("[Fleet] every planned arm already landed — nothing to do")
    else:
        # base 锚点先行 (1 节点, 免节点预算; P-0), 其余备料完成即发射
        todo.sort(key=lambda p: p["arm"] != "base_eval_check")
        running = []
        for plan in todo:
            if plan["prep"]:
                run_prep(plan, output_dir, launch_env,
                         launch_env["NANOCHAT_DIR"])
            # launch() 返回 (proc, log) 二元组; wait_fleet 的契约是
            # (plan, proc, log) 三元组 — 直接 append 会把二元组塞进去,
            # 首次解包即 ValueError (prod5 首飞实炸, 2026-09-28)
            proc, log = launch(plan, output_dir, climbmix_dir,
                               args.retry_failed)
            running.append((plan, proc, log))
        wait_fleet(running)

    ok = all(salvage_local_eval(p, output_dir, launch_env, exp_name)
             and arm_landed(output_dir, p["arm"]) for p in plans)

    print("\n[Fleet] summary")
    for p in plans:
        state = "OK" if arm_landed(output_dir, p["arm"]) else "MISSING"
        log = os.path.join(output_dir, f"dispatch_{p['arm']}.log")
        hint = "" if state == "OK" else (
            f" — log: {os.path.basename(log)}; idempotent re-run retries "
            f"exactly this arm (--retry-failed to override a prior "
            f"failed attempt)")
        print(f"  {p['arm']:<18} {state}{hint}")

    if ok:
        marker = os.path.join(output_dir, ".done_fleet")
        with open(marker, "w") as f:
            json.dump({"arms": [p["arm"] for p in plans],
                       "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)
        print("[Fleet] all planned arms landed — .done_fleet written")
        final_report_auto(output_dir, climbmix_dir)
        return 0
    print("✗ [Fleet] incomplete — see the MISSING rows above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
