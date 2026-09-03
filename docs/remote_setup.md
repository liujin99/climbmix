# Remote fleet setup — architecture, backends, validation, launch

> Two-repo model (2026-08-29 split): THIS repo is platform-neutral — it
> ships the remote-execution contracts, the executor, the worker, and the
> built-in `mock` simulation backend. The platform adapter (gateway REST
> client, auth/token provider, object-store adapter, platform config
> template, hello-world calibration tool) lives in a SEPARATE, access-
> restricted backend repository. Nothing platform-specific (API shapes,
> endpoints, IDs, images, flavors) appears here — on purpose.

Production topology (decided 2026-08-28): local 8x910B4 host = scheduler +
mixer; remote cards from a shared pool (fluctuating — see
`docs/parallel_k_selection.md`) = compute via a job backend; OBS = data
plane. One job per proxy experiment (`npu_per_job` cards, no cross-node
collectives). Results materialize as local `exp_XXXX/` dirs; search resume
and stage fingerprints unchanged.

## 0. How remote execution is wired

```
RemoteExecutor (remote_executor.py)         scheduler — platform-neutral
  ├── job_api:     JobAPI protocol          compute plane (job_api.py)
  └── obs:         ObsStorage protocol      data plane (obs.py)
                 ▲
     backends.py  │ BackendBundle registry  (built-in "mock" | out-of-tree)
                 │
     your backend repo (private): gateway client + auth + storage adapter
```

- `RemoteConfig.backend` selects the backend: `"mock"` (built-in local
  simulation) or a backend name.
- Out-of-tree backends register in ONE of two ways:
  1. `RemoteConfig.backend_module = "pkg.mod:create_backend"` — the
     backend repo cloned and on `PYTHONPATH` (no installation). The
     factory returns a `BackendBundle` (see §Writing a backend).
  2. pip-installed backends expose an entry point in group
     `"climbmix.backends"` named after `RemoteConfig.backend`
     (auto-discovered; `backend_module` unnecessary).
- `RemoteConfig.platform_config` (shell knob `REMOTE_PLATFORM_CONFIG`) is
  an opaque path passed THROUGH to the backend — its schema is
  backend-defined (gateway endpoint, IDs, auth...). Real values live
  outside this public repo (e.g. `~/.config/climbmix/`); credentials are
  per-user and live NOWHERE else.
- Per-launch shape overrides: `REMOTE_IMAGE`/`REMOTE_FLAVOR`/
  `REMOTE_POOL_NAME` are passed via `RemoteConfig` to the backend, which
  lets them beat its config-file defaults.
- Run `python3 scripts/check_repo_secrets.py` before pushing anything
  that touched `remote/`, `scripts/`, or `docs/`.

The launch fail-fast (`runs/run_climbmix.sh`, real backend): resolve the
bundle and run its `validate(remote_config)` at launch — gateway/auth/
image mistakes die with a clear message instead of mid-search.

## 1. Simulation first (mock backend — laptop, zero platform)

The 129-check end-to-end simulation (`tests` workflow in TODO.md M2) runs
the REAL worker subprocess against `MockJobAPI` + a filesystem fake OBS:
`backend="mock"`, `storage_kind="local"`, `storage_root=<dir>`. Every
executor behavior (dynamic submission, resume L1/L2, hybrid fleet,
timeouts) is exercised without any platform access.

## 2. Writing a backend (contract for adapter authors)

A backend = one package exposing `create_backend(remote_config) ->
BackendBundle` (dataclass in `climbmix.remote.backends`):

| Field | Meaning |
|---|---|
| `make_job_api(rc)` | construct your `JobAPI` implementation |
| `make_obs_storage(rc)` | construct your `ObsStorage` implementation |
| `default_worker_path` | container path of `remote_worker.py` under your platform's code-delivery convention; `""` = executor uses the locally staged path |
| `validate(rc)` | optional launch-time fail-fast (load your platform config, resolve image/flavor; never print secrets) |

**JobAPI** (`job_api.py`): `submit(name, command, env, workdir) -> job_id`
— `command` is the WORKER argv; your adapter makes it run in the target
environment (e.g. wrap it into a boot shell that bootstraps assets, then
exec). `status(job_id) -> JobStatus` — map to
PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED; anything unmappable → UNKNOWN
(the executor treats it as non-terminal — the safe default). `logs(job_id,
tail)`, `cancel(job_id)` (tolerate already-gone). Optional
`free_job_slots() -> int|None` — live capacity in JOBS (you normalize
cards→jobs); `None` = no query API (executor falls back to fixed limit +
submit-rejected backoff).

**Error semantics** (the executor's dynamic submission depends on them):
capacity/quota/throttle rejections raise `TransientSubmitError` (backoff +
retry, a config is never burned); hard failures (bad image/auth/argv)
raise `RuntimeError` (siblings fast-fail). Calibrate the mapping from real
responses and keep it in your platform CONFIG (data, not code) so users
never edit the adapter.

**ObsStorage** (`obs.py`): `upload_file/download_file/upload_bytes/
download_bytes/list_objects/stat/delete` over `obs://bucket/key` URIs. The
worker's `--storage` flag selects its own in-container storage mode
(`local` fake, `moxing` SDK) — a backend usually ships both submit-side
and worker-side pieces.

**What runs in the job**: the executor auto-uploads a two-file worker
bundle (`remote_worker.py` + `nanochat_cmds.py`) to `{obs_prefix}/assets`
and points your `code_dir`-equivalent at it; the worker reads `spec.json`
(fully-built torchrun commands — platform job parameters are deliberately
unused), downloads shards from OBS, trains, evals, and uploads
result.json/logs to `{result_uri}` even on failure (the primary debugging
path if your platform has no console/log API). Big one-time assets
(repo tarball, base checkpoints, tokenizer, eval datasets) are uploaded
manually to `{obs_prefix}/assets_big/` — see your backend repo's README
for its boot shell's expected layout; `scripts/dispatch_remote.py
--check-assets` verifies the full set.

## 3. Launch knobs (`runs/run_climbmix.sh`)

| Knob | Meaning |
|---|---|
| `REMOTE_ENABLED` | 1 = search uses RemoteExecutor |
| `REMOTE_BACKEND` | `mock` or your backend name (default `mock`) |
| `REMOTE_BACKEND_MODULE` | `pkg:create_backend` for out-of-tree backends |
| `REMOTE_PLATFORM_CONFIG` | path to the backend's platform config JSON |
| `REMOTE_OBS_PREFIX` | `obs://<bucket>/<series-root>` |
| `REMOTE_IMAGE` / `REMOTE_FLAVOR` / `REMOTE_POOL_NAME` | shape overrides (else backend config defaults) |
| `REMOTE_NPU_PER_JOB` | cards per job (defaults to `NPU_PER_EXP` — k stays fleet-wide fixed for score comparability) |
| `REMOTE_MAX_JOBS` / `REMOTE_MAX_PREP` / `REMOTE_SUBMIT_RETRY_H` / `REMOTE_JOB_TIMEOUT_H` | in-flight cap / local prep semaphore / rejection-retry net / per-job timeout |
| `REMOTE_LOCAL_PARALLEL` | 1 = master-node cards join the fleet via the local parallel path |

## 4. M3 consistency validation (gate to production)

0. Backend hello-world green (its calibration tool — status maps +
   transient-error patterns land in the platform CONFIG) +
   `--check-assets` green.
1. Short smoke: dispatch one 5-step experiment remotely
   (`scripts/dispatch_remote.py ... --proxy-num-iterations 5`).
2. Full check: run exp_0000's weights BOTH locally (ProxyRunner) and
   remotely (dispatch_remote) into separate throwaway output dirs;
   compare `mixture_data/shard_*.parquet` sha256 (must be byte-identical)
   and `stem_metric` (Δ < 0.002 — TODO.md M3 criterion).
3. Concurrency wave: dispatch 2-4 experiments at once (each its own
   `--exp-id` + `--weights` + throwaway `--output-dir`) and watch the
   dynamic scheduler against the REAL pool: job completion frees a slot
   and the next config starts with zero backoff; capacity rejections
   back off and retry; the in-flight peak matches the intended slots.
   The wave also verifies the two-anchor throughput prediction T(4) ≈
   5.3h incidentally and calibrates the pool-full error code.
4. Green → `REMOTE_ENABLED=1 ... bash runs/run_climbmix.sh`.

## 4b. Multi-node pool embedding (TODO E — `scripts/embed_dispatch.py`)

The full-pool embedding cache (~116M docs, ~40h single-node) can be
built as UNITS across the job fleet: each unit = `--unit-shards`
parquet shards (default 16) embedded by one job, uploading
`partial_block.npz` + `manifest.json` back through its result mount.
Unit granularity = progress banked (a crashed job re-embeds only its
own in-flight shards); a unit with a partial already on OBS is skipped
on re-run (`--force` overrides). J jobs → ~40/J hours.

Prerequisites (once, on the submit host):
- The resource package — one OBS directory holding every static shared
  asset, declared in ONE command (see the backend repo's README):

  ```
  climbmix_resource_package/          # upload once, team-shared
  ├── d20/           base checkpoint  # d<depth> naming
  ├── tokenizer/     tokenizer
  ├── stella/        stella_en_400M_v5 model dir (this pipeline's embedder)
  ├── eval_bundle/   eval datasets (bundle)
  └── eval_stem/     eval datasets (stem)

  python3 <backend repo>/scripts/setup.py \
      --resource-package obs://<bucket>/<you>/climbmix_resource_package ...
  ```

  Each recognized dir becomes a DIRECT asset mount (zero duplicate
  storage; the container has no network, so everything the jobs read
  must be reachable this way).
- The EMBED wave's own assets — the pool parquet set (197 GB, DATA,
  not a package asset) and the stella model dir — are PER-LAUNCH
  mounts, not global config: they stage into embed jobs only, via
  `embed_dispatch.py --pool-uri/--model-uri` (the `runs/embed_wave.sh`
  `POOL_URI`/`MODEL_URI` env knobs). Keep them OUT of the global
  backend config on purpose: every job class stages the global set, so
  a 197 GB pool there would be pulled into every proxy-train job too.
  (`--model-uri` may point straight at the resource package's
  `stella/` — the same objects, no second copy.)
- The dispatcher is fresh-prefix self-sufficient (uploads the worker
  bundle to `{prefix}/assets` + the `assets_big` placeholder the
  gateway validates); the obs_prefix itself is the user config knob —
  set it to the production area (the backend config's obs-prefix
  option) before a full production run, so `embed_units/` lands in the
  production area.

Where the finished embeddings live (three tiers):
1. Unit partials (~475 GB total) stay on OBS under
   `{obs_prefix}/embed_units/` — they are the DURABLE tier: a wiped
   submit-host disk re-merges from them in ~1-2h instead of paying the
   40h embed again. Do not delete after merging.
2. The merged pool cache lands at
   `<EMBEDDING_CACHE_DIR>/<content-key>/embedding_cache.npy` — the
   exact path run_climbmix.sh Step 1 hits (written there by
   `scripts/embed_merge.py`, see below). Point `EMBEDDING_CACHE_DIR`
   at the production tree (e.g. `<data-mix-run>/climbmix/cache/
   embeddings`) and the cache investment lives with production.
   The merge is a pure streaming append (peak disk = cache + ONE
   downloaded unit ~7.5 GB, no second copy) and the cache is a raw
   .npy that Step 1 mmaps — a pool-sized cache never materializes in
   RAM. That also means `EMBEDDING_CACHE_DIR` MAY point at a
    FUSE-mounted OBS path (e.g. <obs mount root>/...) when local disk
    can't
   hold ~475 GB: the merge writes through the mount and Step 1 mmaps
   through it (slower than local NVMe, but zero local footprint).
    Legacy single-node `.npz` caches keep working (picked when no .npy
    exists).
    Read-speed note: the OBS-mount cache is fine for the merge, but
    Step 1's clustering does multiple mmap passes over the 475 GB —
    through FUSE that's slower (first full pull ~1-1.5h, then page
    cache). Default plan: OBS-direct and measure the first production
    run; IF Step 1 becomes the bottleneck, the zero-code fix is a
    one-time `cp` of the .npy onto the run host's working disk and
    pointing `EMBEDDING_CACHE_DIR` there (the .npy format is
    mmap-friendly either way — no tooling change).
3. Optional: upload the merged cache to OBS as a redundancy copy
   (overnight, mount-speed).

Re-use semantics — the cache is a one-time investment:
- The cache key hashes only the pool's shard name+size manifest, the
  model name, and the truncate length. Every downstream knob
  (K_enhanced/K_max/merge_distance/prune_threshold/lr/iterations/...)
  is deliberately OUT of the key: ALL ClimbMix experiments on the same
  pool share ONE embedded pool — each run's Step 1 cache-hits and
  continues from clustering onward.
- Pool GREW by appending shards: re-run `runs/embed_wave.sh` (old
  units' partials are resume-skipped — only the new shards' units get
  embedded) + re-run `runs/embed_merge.sh` (fresh cache at the new
  key). Incremental cost = embed(new docs) + one merge. Structural
  changes (insert/delete/reorder shards) shift unit boundaries and
  global offsets → the merge fails loudly; that's a new pool and a
  full re-embed.
- Old-key cache dirs stay on disk after pool growth; delete them once
  the new pool is trusted.

Smoke (one 8-card job, 2 shards, ~10 min end to end — verifies the
mount-read → NPU fp16 math → upload chain; `--compare-local` re-embeds
the same shards through climbmix's own single-card path and demands
byte-identical output). The `runs/embed_wave.sh` wrapper carries the
boilerplate; every knob is an env var (MAX_JOBS, UNIT_SHARDS, ... —
same convention as run_climbmix.sh):

```
SMOKE=2 REMOTE_CONFIG=<backend repo>/config/remote_config.json \
    POOL_URI=obs://<bucket>/<pool dir> \
    MODEL_URI=obs://<bucket>/<pkg>/stella bash runs/embed_wave.sh
# 等价的裸命令 (wrapper 内部执行):
# python3 scripts/embed_dispatch.py \
#     --remote-config <backend repo>/config/remote_config.json \
#     --shard-info <pool>/metadata_shard_info.json \
#     --pool-uri obs://<bucket>/<pool dir> \
#     --model-uri obs://<bucket>/<pkg>/stella \
#     --smoke 2 --flavor <backend 8-card flavor> --npu-per-job 8 \
#     --output-dir /tmp/embed_smoke \
#     --local-model-dir <server-local stella dir> \
#     --compare-local <server-local pool dir>
```

Full pool — the WAVE (63 units at `--unit-shards 16`, `MAX_JOBS=6`
= 48 cards in flight → ~7h wall clock; raise/lower to match what the
shared pool can spare). A FAILED unit never stops its siblings: the
wave drains everything, prints the failed list, and re-running the
SAME command retries only the failures (completed units are
resume-skipped). submit() rejections (pool full) back off and retry
with the RemoteConfig `submit_retry_*` knobs.

```
bash runs/embed_wave.sh                  # MAX_JOBS=6 默认 (POOL_URI/MODEL_URI 同烟雾)
MAX_JOBS=8 bash runs/embed_wave.sh       # 64 卡
SHARD_OFFSET=160 FORCE=1 ...             # 续波 / 强制重发
```

Merge — when the wave drains green, assemble the partials into the
canonical pool cache (verifies exact shard coverage + num_docs +
global offsets against metadata_shard_info.json, validates the whole
pool, `atomic_savez`s at the Step-1 key, resumes from its ledger on a
crash, is idempotent on re-run; `--model`/`--truncate-len` MUST equal
the run config's discovery values — the key depends on them):

```
bash runs/embed_merge.sh
# wrapper 默认: --data-dir = run_climbmix.sh 的 DATA_DIR,
#   --cache-dir = 同名 EMBEDDING_CACHE_DIR 旋钮 (指向生产树即落生产缓存)
# 磁盘需求 (cache 所在文件系统): 池字节 (~475 GB, 流式追加, 无 2x)
#   + ~7.5 GB/在飞单元; 放不下可把 EMBEDDING_CACHE_DIR 指向 OBS 挂载路径
```

After it exits green, the next pipeline run cache-hits Step 1 and
skips the ~40h embed. The OBS partials are intentionally kept (tier 1
above).


## Production launch (M5) — recommended knobs & calculation

### First production run (recommended 2026-08-28)

| Knob | Value | Rationale |
|---|---|---|
| `NPU_PER_EXP` | **4** | λ_max ≈ 5h for the first run (disaster cap ~74 card·h @32-card pool, half-day verification window — docs/parallel_k_selection.md §5.1). `REMOTE_NPU_PER_JOB` defaults to `NPU_PER_EXP`; do NOT set it separately — k stays fleet-wide fixed for score comparability. |
| `REMOTE_MAX_JOBS` | **⌊A_eff/4⌋, typically 6-8** | A_eff = expected pool median~P25 (24-32 cards), NOT the peak. Mid-run growth is absorbed by the capacity monitor (jobs start as idle cards appear); shrink only queues pickups, never kills in-flight jobs. |
| `REMOTE_LOCAL_PARALLEL` | 1 (default) | Master node's 8 cards join the fleet: 2 more k=4 slots. |
| `REMOTE_FLAVOR` | 4-card flavor | From your backend's platform config (`default_flavor`) or the console; override per launch via the knob. |
| `REMOTE_MAX_PREP` | 4 (default) | Prep/upload semaphore on the master node — raise only if I/O is confirmed idle; it never needs to match the job count. |
| `REMOTE_SUBMIT_RETRY_H` | 24 (default) | Backoff net for probe/submit races; also the whole capacity story when the platform has NO quota-query API (fixed limit + rejected-submit retry). |
| `REMOTE_JOB_TIMEOUT_H` | 6 (default) | T(4) ≈ 5.3h + margin. |

Wall-clock estimate for [20,10,5] (35 configs): C = 2 local + 6-8 remote ≈ 8
slots → waves ⌈20/8⌉+⌈10/8⌉+⌈5/8⌉ = 3+2+1 = 6 → **Wall ≈ 5.3h × 6 ≈ 32h**
(search ~1.5 days). Target arms stay local (ws=8, ~6h each + eval).

### Recalculating for the next run (the 30-second method)

Full decision model: `docs/parallel_k_selection.md`. Short version:

1. **Slots** `C(k) = 8/k (local) + ⌊A_eff/k⌋ (remote, node-packed)` —
   A_eff = pool median~P25, k ∈ your platform's flavor granularity.
2. **Wall** `Wall(k) = T(k) × Σᵢ ⌈nᵢ/C(k)⌉` with T(k) from the cost table
   (T(1)=18.6h, T(2)=10.0h, T(4)=5.3h, T(8)=2.8h; full-set eval included).
3. **Constraints on k**: λ_max = T(k) ≤ 5h while the pipeline is
   unverified (→ k ≥ 4); relax to ~10h (→ k ≥ 2) once one run is green —
   k=2 saves scaling tax (7.5% vs 13%) plus node-packing leftovers.
   **k never changes mid-run** (score comparability).
4. Optionally re-shape {nᵢ} to multiples of C so every wave packs 100%
   (zero barrier idle — the dominant cost, 30-80% ≫ the scaling tax).

Second run (after first green): λ_max ≈ 10h → **k=2**, recompute C and
`REMOTE_MAX_JOBS` with the same formulas; A_eff should be re-eyeballed
from the first run's observed pool.

### Backend repo layout

Two equivalent layouts — only `PYTHONPATH` matters:

```bash
# nested (recommended): the backend clone lives INSIDE this checkout
git clone <your-backend-repo-url> climbmix/climbmix-<name>
export PYTHONPATH=$PWD/climbmix/src:$PWD/climbmix/climbmix-<name>

# side-by-side: two sibling checkouts
export PYTHONPATH=$PWD/climbmix/src:$PWD/climbmix-<name>
```

Nested clones are ignored by git (`.gitignore` `/climbmix-*/`), so they
never pollute `git status` or get committed by accident. Do NOT use git
submodules — the submodule URL would leak the private repo's existence
into this public history.

### Launch command

```bash
# once: backend repo cloned + on PYTHONPATH (or pip-installed), platform
# config filled (see the backend repo's README)
REMOTE_ENABLED=1 \
REMOTE_BACKEND=<backend-name> \
REMOTE_BACKEND_MODULE=<backend_pkg>:create_backend \
REMOTE_OBS_PREFIX=obs://<bucket>/climbmix_prod \
NPU_PER_EXP=4 \
REMOTE_MAX_JOBS=6 \
bash runs/run_climbmix.sh
# image/pool default to the platform config; REMOTE_IMAGE/REMOTE_POOL_NAME
# override per launch. REMOTE_NPU_PER_JOB defaults to NPU_PER_EXP (=4,
# fleet-wide k); REMOTE_LOCAL_PARALLEL defaults to 1 (local 8 cards =
# 2 slots); REMOTE_MAX_PREP=4, REMOTE_SUBMIT_RETRY_H=24,
# REMOTE_JOB_TIMEOUT_H=6 (defaults).
```

Decision record (2026-08-28): the launch planner (k-selection §5 three-step
method) is deliberately NOT automated — it runs once per production launch
with human eyeballs on the live pool shape; the table + formulas above are
the entire procedure. S(k) is likewise not separately benchmarked: the
two-anchor model (S(4)≈3.5) gets its first real datapoint from M3's
concurrency wave, retroactively validating T(4).

All `REMOTE_*` knobs are execution-shape only (num_npu precedent) and are
deliberately excluded from the stage fingerprints. NOTE: the remote CODE
itself (`src/climbmix/remote/`, `scripts/remote_worker.py`,
`nanochat_cmds.py`, `proxy_runner.py`) IS fingerprinted (BOTH stages) — land
it before starting the production search; do not edit these files
mid-search. (The 2026-08-29 two-repo split moved the platform adapter OUT
of the fingerprinted tree: backend code now changes WITHOUT invalidating
stage fingerprints — by design, adapters are execution transport, not
experiment semantics.)
