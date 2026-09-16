#!/usr/bin/env python3
"""prod4 fix verification — OBS mixture content keying (dispatch) /
.done identity guards (prepare_random_baseline + mix_general_data local
twins of the stale-data landmine) / mix subprocess HF_ENDPOINT / arm
disk budget preflight / arm local derived-data cleanup.

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_prod4_fixes.py
Exit 0 = all checks pass.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from climbmix.utils.io_utils import shard_content_key  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mk_shard(d, name, size):
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"x" * size)


# ── 1. shard_content_key: determinism / sensitivity / scoping ──────────
with tempfile.TemporaryDirectory() as td:
    _mk_shard(td, "shard_00000.parquet", 100)
    _mk_shard(td, "shard_00001.parquet", 200)
    with open(os.path.join(td, ".done"), "w") as f:
        json.dump({"stem_ratio": 0.7}, f)
    _mk_shard(td, "shard_00002.parquet.tmp.parquet", 50)

    k1, k2 = shard_content_key(td), shard_content_key(td)
    check("key: deterministic", k1 == k2, k1)
    check("key: 12 hex chars", len(k1) == 12)

    with open(os.path.join(td, ".done"), "w") as f:
        json.dump({"stem_ratio": 0.9}, f)  # non-shard file change
    check("key: ignores .done/.tmp (marker rewrite)",
          shard_content_key(td) == k1)

    _mk_shard(td, "shard_00001.parquet", 201)  # same name, new size
    k_size = shard_content_key(td)
    check("key: size change moves key", k_size != k1, k_size)

    _mk_shard(td, "shard_00002.parquet", 300)  # new member of the set
    check("key: shard-set change moves key",
          shard_content_key(td) not in (k1, k_size))

    try:
        shard_content_key(os.path.join(td, "missing"))
        check("key: missing dir raises", False)
    except ValueError:
        check("key: missing dir raises", True)
    empty = os.path.join(td, "empty")
    os.mkdir(empty)
    try:
        shard_content_key(empty)
        check("key: dir without shards raises", False)
    except ValueError:
        check("key: dir without shards raises", True)

# ── 2. dispatch mix_subprocess_env: HF_ENDPOINT chain ─────────────────
dispatch = load_script("dispatch_target_arm")

env = dispatch.mix_subprocess_env({"HF_ENDPOINT": "https://shell"},
                                  {"HF_ENDPOINT": "https://le"}, "/nc")
check("env: shell value wins (setdefault)", env["HF_ENDPOINT"] == "https://shell")

env = dispatch.mix_subprocess_env({}, {"HF_ENDPOINT": "https://le"}, "/nc")
check("env: launch_env fallback", env["HF_ENDPOINT"] == "https://le")

env = dispatch.mix_subprocess_env({}, {}, "/nc")
check("env: hf-mirror house default",
      env["HF_ENDPOINT"] == "https://hf-mirror.com")

base = {"OTHER": "1"}
dispatch.mix_subprocess_env(base, {}, "/nc")
check("env: NANOCHAT_REPO set + input dict not mutated",
      base == {"OTHER": "1"})

# ── 3. dispatch mixture_obs_uri: content-addressed OBS path ──────────
with tempfile.TemporaryDirectory() as td:
    _mk_shard(td, "shard_00000.parquet", 100)
    uri = dispatch.mixture_obs_uri("obs://p/run/target_arms/climb", td)
    check("uri: empty data_dir keeps legacy unkeyed path",
          dispatch.mixture_obs_uri("obs://p/r/a", "") == "obs://p/r/a/mixture_data")
    check("uri: keyed suffix matches content key",
          uri == f"obs://p/run/target_arms/climb/mixture_data_k{shard_content_key(td)}", uri)
    _mk_shard(td, "shard_00001.parquet", 300)
    check("uri: content change moves uri",
          dispatch.mixture_obs_uri("obs://p/run/target_arms/climb", td) != uri)

# ── 4. prepare_random_baseline: weights identity + .done guard ───────
prepare = load_script("prepare_random_baseline")

check("weights_id: uniform for empty spec",
      prepare._weights_identity("") == "uniform")
wid_str = prepare._weights_identity("0.2,0.3,0.5")
check("weights_id: comma string stable + tagged",
      wid_str.startswith("str:") and prepare._weights_identity("0.2,0.3,0.5") == wid_str)
with tempfile.TemporaryDirectory() as td:
    wf = os.path.join(td, "w.json")
    with open(wf, "w") as f:
        json.dump({"C0": 0.5, "C1": 0.5}, f)
    wid_file = prepare._weights_identity(wf)
    check("weights_id: file path hashed by content",
          wid_file.startswith("file:"))
    with open(wf, "w") as f:
        json.dump({"C0": 0.9, "C1": 0.1}, f)
    check("weights_id: file content change moves id",
          prepare._weights_identity(wf) != wid_file)

done = {"seed": 42, "requested_target_tokens": 3000000000,
        "weights_id": "str:abc"}
check("done_guard: exact match passes",
      prepare._done_matches(done, seed=42, requested_tokens=3000000000,
                            weights_id="str:abc"))
check("done_guard: seed mismatch rejected",
      not prepare._done_matches(done, seed=43, requested_tokens=3000000000,
                                weights_id="str:abc"))
check("done_guard: budget mismatch rejected",
      not prepare._done_matches(done, seed=42, requested_tokens=6000000000,
                                weights_id="str:abc"))
check("done_guard: weights mismatch rejected",
      not prepare._done_matches(done, seed=42, requested_tokens=3000000000,
                                weights_id="str:zzz"))
legacy = {"seed": 42, "target_tokens": 3000000000}  # pre-fix .done
check("done_guard: legacy .done unverifiable -> rejected",
      not prepare._done_matches(legacy, seed=42, requested_tokens=3000000000,
                                weights_id="str:abc"))

# ── 5. mix_general_data: .done staleness over ratio + both inputs ────
mix = load_script("mix_general_data")
fresh = {"stem_ratio": 0.7, "stem_key": "aaa", "general_key": "bbb"}
check("mix_guard: exact match not stale",
      not mix._mix_done_stale(fresh, ratio=0.7, stem_key="aaa",
                              general_key="bbb"))
check("mix_guard: ratio change stale",
      mix._mix_done_stale(fresh, ratio=0.6, stem_key="aaa",
                          general_key="bbb"))
check("mix_guard: stem set change stale (new budget upstream)",
      mix._mix_done_stale(fresh, ratio=0.7, stem_key="ccc",
                          general_key="bbb"))
check("mix_guard: general supply change stale (+1 safety shard)",
      mix._mix_done_stale(fresh, ratio=0.7, stem_key="aaa",
                          general_key="ddd"))
check("mix_guard: legacy .done (no keys) stale",
      mix._mix_done_stale({"stem_ratio": 0.7}, ratio=0.7, stem_key="aaa",
                          general_key="bbb"))

# ── 6. check_disk_budget: estimator + gate ───────────────────────────
cdb = load_script("check_disk_budget")
check("disk: both .done -> zero new usage",
      cdb.estimate_arm_gb(3_000_000_000, shards_done=True,
                          mixed_done=True) == 0.0)
check("disk: fresh 3B ~ 37.95 GB (4.5+6.5)/B x1.15 x3)",
      abs(cdb.estimate_arm_gb(3_000_000_000, shards_done=False,
                              mixed_done=False) - 37.95) < 0.01)
check("disk: shards reused -> mixed-only cost",
      abs(cdb.estimate_arm_gb(3_000_000_000, shards_done=True,
                              mixed_done=False) - 22.425) < 0.01)
with tempfile.TemporaryDirectory() as td:
    r = cdb.check_budget(td, "climb", 3_000_000_000, free_gb=200.0)
    check("disk: 3B with 200G free passes",
          r["ok"] and abs(r["required_gb"] - (37.95 * 2 + 50)) < 0.01)
    r = cdb.check_budget(td, "climb", 3_000_000_000, free_gb=100.0)
    check("disk: 3B with 100G free fails loud",
          not r["ok"])
    for part in ("shards", "mixed"):
        os.makedirs(os.path.join(td, f"climb_{part}"), exist_ok=True)
        open(os.path.join(td, f"climb_{part}", ".done"), "w").close()
    r = cdb.check_budget(td, "climb", 3_000_000_000, free_gb=1.0)
    check("disk: everything .done -> passes regardless of free",
          r["ok"] and r["mixed_done"] and r["shards_done"])

# ── 7. clean_derived_data: guards + dry-run default + apply ──────────
CLI = os.path.join(REPO, "scripts", "clean_derived_data.py")


def cli(run_dir, *extra):
    return subprocess.run(
        [sys.executable, CLI, "--run-dir", run_dir, *extra],
        capture_output=True, text=True)


with tempfile.TemporaryDirectory() as td:
    for arm, succeeded, complete in (("climb", True, True),
                                     ("cfg11", False, True),
                                     ("badmix", True, False)):
        for part in ("shards", "mixed"):
            d = os.path.join(td, f"{arm}_{part}")
            os.makedirs(d, exist_ok=True)
            _mk_shard(d, "shard_00000.parquet", 1024)
            if complete or part == "shards":
                open(os.path.join(d, ".done"), "w").close()
        if succeeded:
            open(os.path.join(td, f".done_mid_train_{arm}"), "w").close()

    out = cli(td).stdout
    check("clean: dry-run default (nothing deleted)",
          os.path.isdir(os.path.join(td, "climb_mixed")) and
          "reclaimable" in out)
    check("clean: unsucceeded arm kept (retry capital)",
          "[keep] cfg11" in out and
          os.path.isdir(os.path.join(td, "cfg11_mixed")))
    check("clean: incomplete mix kept",
          "[keep] badmix" in out and
          os.path.isdir(os.path.join(td, "badmix_mixed")))
    check("clean: auto-discovery finds all three arms",
          all(a in out for a in ("climb", "cfg11", "badmix")))

    out = cli(td, "--apply").stdout
    check("clean: apply deletes the succeeded arm only",
          not os.path.exists(os.path.join(td, "climb_mixed")) and
          not os.path.exists(os.path.join(td, "climb_shards")) and
          os.path.exists(os.path.join(td, ".done_mid_train_climb")) and
          os.path.isdir(os.path.join(td, "cfg11_mixed")) and
          os.path.isdir(os.path.join(td, "badmix_mixed")))

    out = cli(td, "--arms", "cfg11", "--apply", "--force").stdout
    check("clean: --force overrides the guard",
          not os.path.exists(os.path.join(td, "cfg11_mixed")))

# ── 8. parallel writes/uploads: byte-identical determinism ────────────
import glob as _glob  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from climbmix.remote.obs import MockObsStorage  # noqa: E402


def _mk_text_dir(d, n_files, docs_per):
    os.makedirs(d, exist_ok=True)
    for i in range(n_files):
        t = pa.table({"text": ["%s-f%d-d%d" % (os.path.basename(d), i, j)
                               for j in range(docs_per)]})
        pq.write_table(t, os.path.join(d, "shard_%05d.parquet" % i))


def _dir_bytes(d):
    return {f: open(os.path.join(d, f), "rb").read()
            for f in sorted(os.listdir(d))}


with tempfile.TemporaryDirectory() as td:
    stem = os.path.join(td, "stem")
    gen = os.path.join(td, "gen")
    _mk_text_dir(stem, 4, 50)   # 3 train (last = val) = 150 train docs
    _mk_text_dir(gen, 2, 50)    # 100 general docs
    climb_files = sorted(_glob.glob(os.path.join(gen, "shard_*.parquet")))
    outs = {}
    for wk in ("1", "8"):
        os.environ["CLIMB_MIX_WRITE_WORKERS"] = wk
        od = os.path.join(td, "mixout_" + wk)
        mix._mix_data_locked(stem, climb_files, od, num_output_files=5,
                             batch_per_file=20, num_npu=8, stem_ratio=0.7)
        outs[wk] = _dir_bytes(od)
    os.environ.pop("CLIMB_MIX_WRITE_WORKERS", None)
    check("mix_par: 5 train + val + .done produced",
          sorted(outs["1"]) == sorted(
              ["shard_%05d.parquet" % i for i in range(6)] + [".done"]),
          str(sorted(outs["1"])))
    check("mix_par: workers=8 byte-identical to serial workers=1",
          outs["1"] == outs["8"])

with tempfile.TemporaryDirectory() as td:
    train = ["t%d" % i for i in range(250)]
    val = ["v%d" % i for i in range(10)]
    out = os.path.join(td, "out")
    os.makedirs(out)
    prepare._write_shards(train, val, out, n_shards=3, shard_size=100,
                          rg_size=6, workers=8)
    ref = os.path.join(td, "ref")
    os.makedirs(ref)
    for i in range(3):
        s, e = i * 100, min((i + 1) * 100, 250)
        pq.write_table(pa.table({"text": train[s:e]}),
                       os.path.join(ref, "shard_%05d.parquet" % i),
                       row_group_size=6)
    pq.write_table(pa.table({"text": val}),
                   os.path.join(ref, "shard_00003.parquet"), row_group_size=1)
    check("prep_par: parallel _write_shards == serial reference",
          _dir_bytes(out) == _dir_bytes(ref))

with tempfile.TemporaryDirectory() as td:
    local = os.path.join(td, "local")
    os.makedirs(local)
    for i in range(20):
        with open(os.path.join(local, "shard_%05d.parquet" % i), "wb") as f:
            f.write(b"d" * 100)
    obs = MockObsStorage(os.path.join(td, "obs"))
    for i in range(5):  # pre-place 5 -> dedup must skip them
        obs.upload_file(os.path.join(local, "shard_%05d.parquet" % i),
                        "obs://b/mix/shard_%05d.parquet" % i)
    n = dispatch.upload_dir_if_missing(obs, local, "obs://b/mix", "t",
                                       workers=8)
    check("upload_par: returns missing count (15)", n == 15)
    check("upload_par: all 20 present after parallel upload",
          all(obs.stat("obs://b/mix/shard_%05d.parquet" % i)
              for i in range(20)))
    n2 = dispatch.upload_dir_if_missing(obs, local, "obs://b/mix", "t",
                                        workers=8)
    check("upload_par: dedup re-run returns 0", n2 == 0)
    n3 = dispatch.upload_dir_if_missing(obs, local, "obs://b/mix2", "t",
                                        workers=1)
    check("upload_par: serial path (workers=1) works", n3 == 20)

    class _FailObs(MockObsStorage):
        def upload_file(self, p, u):
            if "shard_00007" in u:
                raise RuntimeError("boom")
            super().upload_file(p, u)

    try:
        dispatch.upload_dir_if_missing(_FailObs(os.path.join(td, "obs3")),
                                       local, "obs://b/mix", "t", workers=4)
        check("upload_par: worker failure propagates loud", False)
    except RuntimeError:
        check("upload_par: worker failure propagates loud", True)

# ── summary ───────────────────────────────────────────────────────────
print()
if FAILED:
    print(f"FAILED: {len(FAILED)}: {', '.join(FAILED)}")
    sys.exit(1)
print("All checks passed.")
