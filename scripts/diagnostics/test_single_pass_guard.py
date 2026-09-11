#!/usr/bin/env python3
"""single-pass (epoch<=1) guard verification — measure_train_tokens /
read_total_batch_size / check_single_pass / derive CLI e2e / fingerprint
neutrality / wiring presence in run_climbmix.sh + dispatch + target_runner
+ run_arm_only.sh (single-knob on the arm-reuse path).

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_single_pass_guard.py
Exit 0 = all checks pass.
"""
import glob
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from climbmix.sampling.single_pass import (  # noqa: E402
    check_single_pass, derive_num_iterations, measure_train_tokens,
    read_total_batch_size)
from climbmix.utils.token_estimate import parse_token_count  # noqa: E402
from climbmix.utils.fingerprint import _stages_for  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def write_shard(path, texts):
    pq.write_table(pa.table({"text": texts}), path, row_group_size=2)


# ── 1. measure_train_tokens: val excluded, .tmp ignored, chars/4 ──────────
with tempfile.TemporaryDirectory() as td:
    write_shard(os.path.join(td, "shard_00000.parquet"),
                ["a" * 400, "b" * 800])            # 1200 chars
    write_shard(os.path.join(td, "shard_00001.parquet"),
                ["c" * 1600])                       # 1600 chars
    write_shard(os.path.join(td, "shard_00002.parquet"),
                ["VAL" * 4000])                     # val (last) — excluded
    write_shard(os.path.join(td, "shard_00003.tmp.parquet"),
                ["x" * 999999])                     # crashed partial — ignored
    got = measure_train_tokens(td)
    check("measure: (400+800+1600)/4, val+tmp excluded", got == 700, f"got {got}")

    # empty text / unicode: utf8_length counts code points like len()
    with tempfile.TemporaryDirectory() as td2:
        write_shard(os.path.join(td2, "shard_00000.parquet"), ["", "数学" * 2])
        write_shard(os.path.join(td2, "shard_00001.parquet"), ["z" * 8])
        write_shard(os.path.join(td2, "shard_00002.parquet"), ["VAL"])  # val
        got2 = measure_train_tokens(td2)
        check("measure: empty + CJK chars, val excluded", got2 == 3,
              f"got {got2}")

    # single-shard dir (only val) -> loud error, not a silent 0
    with tempfile.TemporaryDirectory() as td3:
        write_shard(os.path.join(td3, "shard_00000.parquet"), ["only val"])
        try:
            measure_train_tokens(td3)
            check("measure: single-shard dir raises", False)
        except ValueError:
            check("measure: single-shard dir raises", True)

    # no shards at all
    with tempfile.TemporaryDirectory() as td4:
        try:
            measure_train_tokens(td4)
            check("measure: empty dir raises FileNotFoundError", False)
        except FileNotFoundError:
            check("measure: empty dir raises FileNotFoundError", True)

# ── 2. read_total_batch_size ──────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    with open(os.path.join(td, "meta_step000100.json"), "w") as f:
        json.dump({"total_batch_size": 1048576}, f)
    with open(os.path.join(td, "meta_step000099.json"), "w") as f:
        json.dump({"total_batch_size": 524288}, f)
    check("tbs: newest meta wins (lexicographic, repo convention)",
          read_total_batch_size(td) == 1048576)
    check("tbs: missing dir -> None", read_total_batch_size(None) is None)
    with tempfile.TemporaryDirectory() as td2:
        check("tbs: empty dir -> None", read_total_batch_size(td2) is None)
        with open(os.path.join(td2, "meta_bad.json"), "w") as f:
            f.write("{not json")
        check("tbs: corrupt meta -> None", read_total_batch_size(td2) is None)

# ── 3. check_single_pass: OK / boundary / violation ───────────────────────
info = check_single_pass(2000, 1048576, 2_860_000_000,
                         stem_ratio=0.7, context="unit")
check("check: prod1-shaped config passes (0.73 epoch)",
      info["consume_tokens"] == 2000 * 1048576
      and info["epochs_x100"] == 73, str(info))

check("check: epochs == 1.00 passes (<= max 1.02)",
      check_single_pass(100, 1000, 100_000)["epochs_x100"] == 100)

try:
    check_single_pass(100, 1000, 99_000)  # 1.0101 epochs — within 1.02
    check("check: 1.01 epochs tolerated (measurement-noise margin)", True)
except ValueError:
    check("check: 1.01 epochs tolerated (measurement-noise margin)", False)

try:
    check_single_pass(4000, 1048576, 2_860_000_000,
                      stem_ratio=0.7, context="climb arm")
    check("check: steps-doubled config raises", False)
except ValueError as e:
    msg = str(e)
    ok = ("single-pass violation" in msg and "1.47" in msg
          and "--target-tokens" in msg and "--target-steps" in msg
          and "2,860,000,000" in msg)
    check("check: steps-doubled config raises with both fixes", ok, msg[:160])

try:
    check_single_pass(1000, 1024, 0)
    check("check: zero pool raises", False)
except ValueError:
    check("check: zero pool raises", True)

# ── 4. CLI end-to-end ─────────────────────────────────────────────────────
cli = os.path.join(REPO, "scripts", "check_single_pass.py")
with tempfile.TemporaryDirectory() as td:
    write_shard(os.path.join(td, "shard_00000.parquet"), ["a" * 4_000_000])
    write_shard(os.path.join(td, "shard_00001.parquet"), ["b" * 4_000_000])
    write_shard(os.path.join(td, "shard_00002.parquet"), ["v" * 100])
    # pool = 8,000,000/4 = 2,000,000 tokens; 1000 x 1024 = 1,024,000 -> 0.51
    r = subprocess.run(
        [sys.executable, cli, "--data-dir", td, "--num-iterations", "1000",
         "--total-batch-size", "1024", "--context", "cli ok"],
        capture_output=True, text=True)
    check("cli: under-budget exits 0 with epoch summary",
          r.returncode == 0 and "0.51 epoch" in r.stdout
          and "2,000,000" in r.stdout, r.stdout.strip()[:120] or r.stderr[:120])
    # 3000 x 1024 = 3,072,000 > 2,000,000 -> 1.54 epochs
    r = subprocess.run(
        [sys.executable, cli, "--data-dir", td, "--num-iterations", "3000",
         "--total-batch-size", "1024"],
        capture_output=True, text=True)
    check("cli: over-budget exits 1 with actionable stderr",
          r.returncode == 1 and "single-pass violation" in r.stderr
          and "--target-steps" in r.stderr, r.stderr.strip()[:120])
    # tbs unresolvable -> exit 2 (never guess)
    r = subprocess.run(
        [sys.executable, cli, "--data-dir", td, "--num-iterations", "100"],
        capture_output=True, text=True)
    check("cli: unresolvable tbs exits 2",
          r.returncode == 2 and "total_batch_size" in r.stderr)
    # ckpt-dir resolution path
    with tempfile.TemporaryDirectory() as ck:
        with open(os.path.join(ck, "meta_x.json"), "w") as f:
            json.dump({"total_batch_size": 512}, f)
        r = subprocess.run(
            [sys.executable, cli, "--data-dir", td, "--num-iterations", "1000",
             "--ckpt-dir", ck],
            capture_output=True, text=True)
        check("cli: --ckpt-dir meta resolution (1000x512=512k <= 2M)",
              r.returncode == 0 and "0.26 epoch" in r.stdout, r.stdout[:120])


# ── 5. fingerprint neutrality: the guard must not shift run state ─────────
check("fingerprint: guard files excluded from all stages",
      _stages_for("scripts/check_single_pass.py") == set()
      and _stages_for("src/climbmix/sampling/single_pass.py") == set())
check("fingerprint: target_runner still target-only",
      _stages_for("src/climbmix/pipeline/target_runner.py") == {"target"})
check("fingerprint: classifier self-excluded",
      _stages_for("src/climbmix/utils/fingerprint.py") == set())

# ── 6. wiring: run_climbmix.sh / dispatch / target_runner ─────────────────
sh = os.path.join(REPO, "runs", "run_climbmix.sh")
r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True)
src = open(sh).read()
check("shell: run_climbmix.sh bash -n", r.returncode == 0, r.stderr[:120])
check("shell: run_arm guards before dispatch AND fallback",
      "check_single_pass.py" in src and "single-pass guard failed" in src)

for py in ("scripts/dispatch_target_arm.py",
           "src/climbmix/pipeline/target_runner.py",
           "src/climbmix/sampling/single_pass.py",
           "scripts/check_single_pass.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile",
                        os.path.join(REPO, py)],
                       capture_output=True, text=True)
    check(f"py_compile: {py}", r.returncode == 0, r.stderr[:160])

disp = open(os.path.join(REPO, "scripts/dispatch_target_arm.py")).read()
check("dispatch: guard runs after .done check, before submit",
      "single-pass OK" in disp and "measure_train_tokens(data_dir)" in disp
      and "--total-batch-size" in disp)
check("dispatch: no default-1000 steps fallback (loud on missing)",
      'or "1000"' not in disp and "launch_env_target_steps" in disp)

tr = open(os.path.join(REPO, "src/climbmix/pipeline/target_runner.py")).read()
check("target_runner: guard called after mixture prep",
      "_guard_single_pass(" in tr and "single-pass OK" in tr)

# ── 7. derive_num_iterations: TARGET_TOKENS as the single source of truth ─
check("derive: 2B / 1,048,576 -> 1907 steps",
      derive_num_iterations(2_000_000_000, 1_048_576) == 1907)
check("derive: 1B / 524,288 (proxy-shaped) -> 1907 steps",
      derive_num_iterations(1_000_000_000, 524_288) == 1907)
check("derive: budget < one step -> 1 (guard then rules it out)",
      derive_num_iterations(1_000, 1_048_576) == 1)
for bad in ((0, 1024), (-5, 1024), (1000, 0)):
    try:
        derive_num_iterations(*bad)
        check(f"derive: invalid input {bad} raises", False)
    except ValueError:
        check(f"derive: invalid input {bad} raises", True)

# derived steps are single-pass BY CONSTRUCTION (epoch ~= stem_ratio):
for tokens, tbs in ((1_000_000_000, 1_048_576), (2_000_000_000, 1_048_576),
                    (4_000_000_000, 1_048_576)):
    steps = derive_num_iterations(tokens, tbs)
    pool = tokens * 10 // 7  # ~ tokens / stem_ratio (0.7)
    try:
        info = check_single_pass(steps, tbs, pool, stem_ratio=0.7)
        ok = info["epochs_x100"] <= 71
    except ValueError:
        ok = False
    check(f"derive+guard: T={tokens:,} stays single-pass", ok,
          f"steps={steps}, epochs_x100={info.get('epochs_x100')}")

# e2e composition exactly as the run_climbmix.sh heredoc does it:
with tempfile.TemporaryDirectory() as ck:
    with open(os.path.join(ck, "meta_x.json"), "w") as f:
        json.dump({"total_batch_size": 1024}, f)
    tokens = parse_token_count("4K")
    tbs = read_total_batch_size(ck)
    check("derive e2e: '4K' + fake ckpt meta -> 3 steps",
          tbs == 1024 and derive_num_iterations(tokens, tbs) == 3)
    check("derive e2e: missing meta -> None -> shell exits (covered)",
          read_total_batch_value := read_total_batch_size(
              os.path.join(ck, "nope")) is None)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/derive_target_steps.py"),
         "--target-tokens", "4K", "--ckpt-dir", ck],
        capture_output=True, text=True, timeout=60)
    check("CLI: derive_target_steps.py 4K @ tbs=1024 -> 3",
          r.returncode == 0 and r.stdout.strip() == "3",
          (r.stdout + r.stderr).strip()[:100])
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/derive_target_steps.py"),
         "--target-tokens", "4K", "--ckpt-dir", os.path.join(ck, "nope")],
        capture_output=True, text=True, timeout=60)
    check("CLI: unresolvable tbs -> exit 2 (loud, never guessed)",
          r.returncode == 2 and "total_batch_size" in r.stderr,
          (r.stdout + r.stderr).strip()[:100])

check("shell: TARGET_STEPS derivation block present",
      "derive_target_steps.py" in src and
      "TARGET_STEPS derived from TARGET_TOKENS" in src and
      "no longer a knob" in src and
      "invalid for target arms" in src)
check("shell: TARGET_TOKENS defined before derivation",
      src.index('TARGET_TOKENS="${TARGET_TOKENS:-2B}"')
      < src.index("derive_target_steps.py"))

# real invocations of the two launch-time abort branches (the config block
# sits before any side effect — mkdir/stage gate all come later)
env_base = {k: v for k, v in os.environ.items() if k != "TARGET_STEPS"}
r = subprocess.run(["bash", sh],
                   env={**env_base, "TARGET_STEPS": "2000"},
                   capture_output=True, text=True, timeout=60)
check("shell: external TARGET_STEPS aborts at launch",
      r.returncode != 0 and "no longer a knob" in r.stdout + r.stderr,
      (r.stdout + r.stderr).strip()[:100])
r = subprocess.run(["bash", sh],
                   env={**env_base, "TARGET_TOKENS": "0"},
                   capture_output=True, text=True, timeout=60)
check("shell: TARGET_TOKENS=0 aborts at launch",
      r.returncode != 0 and "invalid for target arms" in r.stdout + r.stderr,
      (r.stdout + r.stderr).strip()[:100])

# ── 8. run_arm_only.sh: single-knob on the arm-reuse path ──────────────
arm_sh = os.path.join(REPO, "runs/run_arm_only.sh")
arm = open(arm_sh).read()
check("arm-reuse: external TARGET_STEPS rejected (same rule)",
      "no longer a knob" in arm)
check("arm-reuse: derivation shares the same CLI (no second formula)",
      "derive_target_steps.py" in arm)
check("arm-reuse: TARGET_STEPS dropped from the forward list",
      "for v in MID_DEVICE_BATCH_SIZE" in arm)
r = subprocess.run(["bash", arm_sh],
                   env={**env_base, "TARGET_STEPS": "2000"},
                   capture_output=True, text=True, timeout=60)
check("arm-reuse: external TARGET_STEPS aborts at launch",
      r.returncode != 0 and "no longer a knob" in r.stdout + r.stderr,
      (r.stdout + r.stderr).strip()[:100])

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL PASS")
