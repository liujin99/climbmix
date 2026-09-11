#!/usr/bin/env python3
"""single-pass (epoch<=1) guard verification — measure_train_tokens /
read_total_batch_size / check_single_pass / derive CLI e2e / fingerprint
neutrality / wiring presence in run_climbmix.sh + dispatch + target_runner
+ run_arm_only.sh (single-knob on the arm-reuse path) / proxy single-knob
(PROXY_NUM_ITERATIONS derived from PROXY_TARGET_TOKENS) / large-scale
sampler smoke (runs/run_large_scale_sample.sh).

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
    # step-prefixed names agree on tbs -> returned regardless of order
    for name, tbs in (("meta_step000100.json", 1048576),
                      ("meta_step000099.json", 1048576)):
        with open(os.path.join(td, name), "w") as f:
            json.dump({"total_batch_size": tbs}, f)
    check("tbs: step-prefixed metas agree -> value",
          read_total_batch_size(td) == 1048576)
    # mtime resolution (disagreement) is covered in §12
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
check("derive: 1B / 524,288 (legacy-assumed proxy tbs) -> 1907 steps",
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
      src.index('TARGET_TOKENS="${TARGET_TOKENS:-1000Mi}"')
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

# ── 9. proxy single-knob: PROXY_NUM_ITERATIONS derived from budget ────
check("Mi parse: 500Mi = 524,288,000 (binary magnitude)",
      parse_token_count("500Mi") == 524_288_000)
check("Mi parse: 1000Mi = exactly 1000 steps @ real tbs 1,048,576 "
      "(d20 AND d28 — prod1/prod2 step-continuous)",
      parse_token_count("1000Mi") == 1_048_576_000)
check("Mi parse: 2000Mi = 2,097,152,000 (2000-step magnitude)",
      parse_token_count("2000Mi") == 2_097_152_000)
check("Mi parse: 2Ki/1.5Gi binary magnitudes",
      parse_token_count("2Ki") == 2048
      and parse_token_count("1.5Gi") == 1_610_612_736)
check("Mi parse: decimal suffixes unchanged (back-compat)",
      parse_token_count("2B") == 2_000_000_000
      and parse_token_count("400M") == 400_000_000)
try:
    parse_token_count("5X")
    check("Mi parse: garbage raises", False)
except ValueError:
    check("Mi parse: garbage raises", True)

check("proxy: knob default removed (value now derived)",
      'PROXY_NUM_ITERATIONS="${PROXY_NUM_ITERATIONS:-1000}"' not in src)
check("proxy: budget default 1000Mi (= 1000 steps @ real tbs, "
      "prod2 step-continuous)",
      'PROXY_TARGET_TOKENS:-1000Mi' in src)
check("proxy: derivation block present",
      "PROXY_NUM_ITERATIONS derived from PROXY_TARGET_TOKENS" in src and
      "DERIVED from PROXY_TARGET_TOKENS" in src)

with tempfile.TemporaryDirectory() as base:
    # REAL server metas (2026-09-11): both d20 and d28 carry tbs 1,048,576 —
    # mid_train inherits tbs from the pretrain meta; 524,288 is only the
    # missing-key fallback (the false assumption behind prod2's 3-4 epoch wrap).
    for d, tbs in (("d28", 1_048_576), ("d20", 1_048_576)):
        dd = os.path.join(base, "base_checkpoints", d)
        os.makedirs(dd)
        with open(os.path.join(dd, "meta_x.json"), "w") as f:
            json.dump({"total_batch_size": tbs}, f)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/derive_target_steps.py"),
         "--target-tokens", "1000Mi",
         "--ckpt-dir", os.path.join(base, "base_checkpoints", "d20")],
        capture_output=True, text=True, timeout=60)
    check("proxy: CLI 1000Mi @ real d20 tbs 1,048,576 -> exactly 1000",
          r.returncode == 0 and r.stdout.strip() == "1000",
          (r.stdout + r.stderr).strip()[:100])
    r = subprocess.run(["bash", sh],
                       env={**env_base, "NANOCHAT_BASE_DIR": base,
                            "PROXY_NUM_ITERATIONS": "999"},
                       capture_output=True, text=True, timeout=60)
    check("proxy: external PROXY_NUM_ITERATIONS aborts at launch",
          r.returncode != 0
          and "PROXY_NUM_ITERATIONS=999 is no longer a knob" in r.stdout + r.stderr,
          (r.stdout + r.stderr).strip()[-120:])

# ── 10. run_large_scale_sample.sh: decoupled production sampler ──────
lss = os.path.join(REPO, "runs/run_large_scale_sample.sh")
lsrc = open(lss).read()
check("lss: data-only by contract (no training/dispatch/eval)",
      "dispatch_target_arm" not in lsrc and "mid_train" not in lsrc)
check("lss: reuses search artifacts (alpha + cluster cache)",
      "optimal_mixture_weights.json" in lsrc and "cluster_cache.npz" in lsrc)
check("lss: manifest records measured truth + warnings",
      "manifest.json" in lsrc and "measure_train_tokens" in lsrc
      and "warnings" in lsrc)
r = subprocess.run(["bash", lss],
                   env={**env_base, "RUN_DIR": "/nonexistent_run_xyz"},
                   capture_output=True, text=True, timeout=60)
check("lss: missing run dir refused",
      r.returncode != 0 and "RUN_DIR" in r.stdout + r.stderr,
      (r.stdout + r.stderr).strip()[:100])

with tempfile.TemporaryDirectory() as fake_run:
    with open(os.path.join(fake_run, "launch_env.json"), "w") as f:
        json.dump({"DATA_DIR": "/tmp/opencode/fake_pool",
                   "GENERAL_DATA_DIR": "/tmp/opencode/fake_general"}, f)
    with open(os.path.join(fake_run, "optimal_mixture_weights.json"), "w") as f:
        json.dump({"0": 0.5, "1": 0.5}, f)
    open(os.path.join(fake_run, "cluster_cache.npz"), "w").close()
    r = subprocess.run(["bash", lss],
                       env={**env_base, "RUN_DIR": fake_run,
                            "TARGET_TOKENS": "0"},
                       capture_output=True, text=True, timeout=60)
    check("lss: TARGET_TOKENS=0 refused",
          r.returncode != 0 and "明确预算" in r.stdout + r.stderr,
          (r.stdout + r.stderr).strip()[:100])
    r = subprocess.run(["bash", lss],
                       env={**env_base, "RUN_DIR": fake_run,
                            "TARGET_TOKENS": "20B", "LAUNCH": "0"},
                       capture_output=True, text=True, timeout=60)
    check("lss: dry-run validates + prints both tool commands",
          r.returncode == 0 and "prepare_random_baseline.py" in r.stdout
          and "mix_general_data.py" in r.stdout and "manifest.json" in r.stdout,
          (r.stdout + r.stderr).strip()[-160:])

# ── 11. mix supply preflight: general repetition is loud by default ───
# mix_general_data draws BOTH sides from cycling generators — a short
# general supply used to repeat docs silently (ratio intact!). The
# preflight must turn that into a loud error, with an explicit opt-in.
mix_src = open(os.path.join(REPO, "scripts/mix_general_data.py")).read()
check("mix: supply preflight present (count vs need + binomial margin)",
      "supply preflight" in mix_src and "binomial" in mix_src)
check("mix: explicit opt-in flag exists",
      "--allow-general-repeat" in mix_src)
check("shell: proxy paper-strength warning at launch",
      "proxy 信号强度" in src and "800M" in src)

import importlib.util
with tempfile.TemporaryDirectory() as td:
    # stub nanochat.dataset (stream_texts_uniform = plain parquet reader)
    stub = os.path.join(td, "nanochat_stub")
    os.makedirs(os.path.join(stub, "nanochat"))
    with open(os.path.join(stub, "nanochat", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(stub, "nanochat", "dataset.py"), "w") as f:
        f.write(
            "import pyarrow.parquet as pq\n"
            "MAX_SHARD = 2000\n"
            "def index_to_filename(i):\n    return f'train_{i:05d}.parquet'\n"
            "def download_single_file(i, d, t):\n    return True\n"
            "def stream_texts_uniform(files):\n"
            "    for fp in files:\n"
            "        for t in pq.read_table(fp)['text'].to_pylist():\n"
            "            yield t\n")
    os.environ["NANOCHAT_REPO"] = stub
    spec = importlib.util.spec_from_file_location(
        "mix_general_data", os.path.join(REPO, "scripts/mix_general_data.py"))
    mix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mix)

    stem_dir = os.path.join(td, "stem")
    os.makedirs(stem_dir)
    write_shard(os.path.join(stem_dir, "shard_00000.parquet"),
                [f"s{i}" for i in range(40)])
    write_shard(os.path.join(stem_dir, "shard_00001.parquet"), ["VAL"])
    climb_dir = os.path.join(td, "climb")
    os.makedirs(climb_dir)
    c0 = os.path.join(climb_dir, "c0.parquet")
    write_shard(c0, [f"g{i}" for i in range(4)])        # 4 docs — short supply

    def out_texts(out):
        texts = []
        for p in sorted(glob.glob(os.path.join(out, "shard_*.parquet")))[:-1]:
            texts += pq.read_table(p)["text"].to_pylist()
        return texts

    try:
        mix.mix_data(stem_dir, [c0], os.path.join(td, "out1"),
                     2, 20, num_npu=1, stem_ratio=0.5)
        check("mix: short general supply raises (no silent repetition)", False)
    except ValueError as e:
        check("mix: short general supply raises (no silent repetition)",
              "general data insufficient" in str(e)
              and "allow-general-repeat" in str(e), str(e)[:90])

    out2 = os.path.join(td, "out2")
    mix.mix_data(stem_dir, [c0], out2, 2, 20, num_npu=1, stem_ratio=0.5,
                 allow_general_repeat=True)
    g = [t for t in out_texts(out2) if t.startswith("g")]
    check("mix: opt-in proceeds and repetition is real (dups observable)",
          len(g) >= 10 and len(set(g)) <= 4,
          f"{len(g)} draws from {len(set(g))} unique docs")
    check("mix: .done records ratio",
          json.load(open(os.path.join(out2, ".done")))["stem_ratio"] == 0.5)

    c1 = os.path.join(climb_dir, "c1.parquet")
    write_shard(c1, [f"h{i}" for i in range(40)])       # ample supply
    out3 = os.path.join(td, "out3")
    mix.mix_data(stem_dir, [c1], out3, 2, 20, num_npu=1, stem_ratio=0.5)
    h = [t for t in out_texts(out3) if t.startswith("h")]
    check("mix: ample supply mixes clean (no duplicates)",
          len(h) == len(set(h)), f"{len(h)} draws / {len(set(h))} unique")

# ── 12. read_total_batch_size: no silent lexicographic pick ──────────
# meta_999 vs meta_1000 sorts the wrong way lexically; a disagreeing meta
# set must resolve by mtime (newest) and say so, never silently.
import time
with tempfile.TemporaryDirectory() as ck:
    for name in ("meta_a.json", "meta_b.json"):
        with open(os.path.join(ck, name), "w") as f:
            json.dump({"total_batch_size": 1024}, f)
    check("tbs: metas agree -> value, no drama",
          read_total_batch_size(ck) == 1024)
    with open(os.path.join(ck, "meta_999.json"), "w") as f:    # lexicographic trap
        json.dump({"total_batch_size": 2048}, f)
    with open(os.path.join(ck, "meta_1000.json"), "w") as f:
        json.dump({"total_batch_size": 1024}, f)
    old, new = time.time() - 3600, time.time()
    os.utime(os.path.join(ck, "meta_999.json"), (old, old))    # old mtime, big value
    os.utime(os.path.join(ck, "meta_1000.json"), (new, new))   # new mtime, right value
    check("tbs: disagreement -> newest mtime wins (not lexicographic)",
          read_total_batch_size(ck) == 1024)
with tempfile.TemporaryDirectory() as ck2:
    with open(os.path.join(ck2, "meta_x.json"), "w") as f:
        json.dump({"other": 1}, f)
    check("tbs: no meta carries the key -> None (caller fails loud)",
          read_total_batch_size(ck2) is None)

prs = open(os.path.join(REPO, "src/climbmix/pipeline/proxy_runner.py")).read()
check("proxy: guard wired after mixture prep, before mid_train",
      "_guard_single_pass(" in prs
      and "proxy_num_iterations, tbs, pool_tokens" in prs
      and prs.index("_prepare_mixture_data(mixture_config")
      < prs.index("self._guard_single_pass(")
      < prs.index("mid_cmd = self._build_mid_train_cmd"),
      "search path now guarded like the target stage")

# ── 13. pool sizing / shard constant / remote guard wiring (2026-09-11) ─
# Three prod2-era data-face bugs, all server-verified (paper_deviations D16):
#   (a) mix total docs = STEM docs (pool ~ budget x 0.98, NOT /ratio) — the
#       loader wrapped 3-4 epochs (mid_train.log epoch lines);
#   (b) CLIMBMIX_DOCS_PER_SHARD 500K vs real ~85K (6x) — general download
#       under-provisioned, the cycling draw repeated docs;
#   (c) the remote executor bypassed the guard entirely (prod search path!).
for name, path in (("proxy", "src/climbmix/pipeline/proxy_runner.py"),
                   ("target", "src/climbmix/pipeline/target_runner.py")):
    rsrc = open(os.path.join(REPO, path)).read()
    check(f"{name}: pool sized at STEM docs / stem_ratio (floor, not len)",
          "stem_docs // int(detected_batch * self.stem_ratio)" in rsrc
          and "num_output_files = len(stem_train_files)" not in rsrc)
check("mix CLI: default output sized at STEM / ratio",
      "stem_docs // int(batch_per_file * STEM_RATIO)" in mix_src
      and "default_output_files" in mix_src)
check("mix: ClimbMix shard constant = 85K (measured 2026-09-11)",
      "CLIMBMIX_DOCS_PER_SHARD = 85000" in mix_src)
check("mix: 85K constant actually drives the shard-count math",
      mix.CLIMBMIX_DOCS_PER_SHARD == 85000
      and mix.calc_climbmix_count(1_000_000, 0.7) == 6          # ceil(428.6K/85K)
      and mix.calc_climbmix_count(10_000_000, 0.7) == 50)       # 51 -> cap binds
re_src = open(os.path.join(
    REPO, "src/climbmix/remote/remote_executor.py")).read()
check("remote: guard wired between mixture prep and upload "
      "(the path prod2/prod3 actually take)",
      re_src.index("self._prepare_mixture_data(")
      < re_src.index("self._guard_single_pass(")
      < re_src.index("self._upload_dir(mixture_data_dir"),
      "fleet experiments were guardless before this")
check("lss: shard cap 150 (20B needs ~127 at 85K docs/shard)",
      'MAX_CLIMBMIX_SHARDS:-150' in lsrc)

# sizing math, floor semantics: draw must stay UNDER the STEM supply
check("sizing: floor keeps the STEM draw under supply (5-sigma safe)",
      all(0.7 * ((n // int(b * 0.7)) * b) <= n
          for n, b in ((503_500, 10_000), (710_000, 10_000),
                       (43_000, 10_000), (99, 10_000), (1_260_000, 10_000))))

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL PASS")
