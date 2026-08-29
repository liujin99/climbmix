# Remote fleet setup (ModelArts + OBS) — M1 environment survey checklist

> Status: adapters are IN (2026-08-29): `ModelArtsJobAPI` (gateway REST +
> IAM token auth + boot shell), `ModelArtsObsStorage` (moxing),
> `IamTokenProvider`. What remains on the SERVER is the ✅ checklist below
> (config file + hello-world + assets), then the M3 validation gate
> (`scripts/ma_hello_world.py` → `scripts/dispatch_remote.py`).

Production topology (decided 2026-08-28): local 8x910B4 host = scheduler +
mixer; remote cards (shared pool, fluctuating 10-200, typical ~32 — see
`docs/parallel_k_selection.md`) = compute via ModelArts Job API; OBS = data
plane. One job per proxy experiment (`npu_per_job` cards, no cross-node
collectives). Results materialize as local `exp_XXXX/` dirs; search resume
and stage fingerprints unchanged.

## 0. Platform config file (once, on the server — the repo is PUBLIC)

All internal values — gateway endpoint, project/workspace IDs, pool id,
SWR image + repo id, IAM credentials — live OUTSIDE the repo:

```bash
mkdir -p ~/.config/climbmix
cp config/remote_ma.example.json ~/.config/climbmix/remote_ma.json
# fill in the real values (never commit them; .gitignore +
# scripts/check_repo_secrets.py are the double guard)
```

Resolution order: `RemoteConfig.ma_config` (`REMOTE_MA_CONFIG` shell knob)
→ `$CLIMBMIX_MA_CONFIG` → `~/.config/climbmix/remote_ma.json`.

- Auth: `auth.account + auth.secret` (JWT, preferred) or
  `auth.domain_name + auth.username + auth.password`, or a static
  `auth.x_auth_token`. Tokens are cached in `~/.cache/climbmix/
  iam_tokens.json` and auto-roll ~5 min before their 24h expiry — a 32h
  production run crosses token expiry without interruption.
- Per-launch overrides (shell knobs) beat the config file:
  `REMOTE_IMAGE` > `image_url`, `REMOTE_FLAVOR` > `default_flavor`,
  `REMOTE_POOL_NAME` > `pool_id`. Leaving them empty = config-file values.
- Run `python3 scripts/check_repo_secrets.py` before pushing anything that
  touched `remote/`, `scripts/`, or `docs/`.

## 1. Submit-host environment (on the 8x910B4 server)

- [ ] `python3 -c "import moxing"` works (submit-side storage: spec/shard
      upload, result download — `ModelArtsObsStorage`). If it fails with an
      auth error, fill the optional `obs.ak/sk` section of the config; if
      the module is missing, `pip install moxing` (ModelArts images ship
      it).
- [ ] `python3 -c "import requests"` (gateway client).
- [ ] Token fetch works: `python3 -c "from climbmix.remote.iam_token import
      IamTokenProvider; IamTokenProvider().get_token(<auth dict from your
      config>)"` returns without error and writes the cache file.

## 2. Compute pool & flavor

- Known: dedicated pool via `pool_id` (config file), flavor granularity
  {1, 2, 4, 6, 8} cards — flavor strings are dynamic per pool state.
- [ ] Note the exact 4-card flavor string for the first production run
      (k=4; a 4xlarge-style name is expected — confirm on the console).
- [ ] **Hello-world job** (calibrates auth/pool/image AND the status/error
      tables in one shot):
      `python3 scripts/ma_hello_world.py --remote-config <remote_config.json>`
      It submits `npu-smi info` + a moxing read of the OBS prefix and polls.
      Compare its logged status transitions against the console job page:
      fix `_INT_STATUS`/`_STR_STATUS` in
      `src/climbmix/remote/modelarts_job_api.py` if they disagree.
- [ ] **Submit behavior under quota pressure** (drives
      `remote_executor._submit_with_retry`): when the pool is full, does
      submit FAIL with a quota/capacity error (→ already mapped heuristically
      to `TransientSubmitError`; extend `_TRANSIENT_MESSAGE_PATTERNS` with
      the REAL error code/text once observed) or does the job go PENDING and
      queue server-side? The M3 concurrency wave answers this incidentally.
- [ ] Preemption: can a running job be reclaimed by the pool (→ would
      justify job-level checkpoint chaining later), or do jobs always run
      to terminal state?

## 3. Image — RESOLVED

The production image is the same SWR image the climbmix server runs
(torch_npu + CANN 8.5.1 + pyarrow + moxing all present). `image_url` +
`image_repo_id` (REQUIRED by the gateway even for custom SWR images) go in
the config file. Container workdir convention `/home/ma-user/work` matches
the RemoteConfig `container_*` defaults.

## 4. One-time asset upload (big assets, manual — NOT auto-uploaded)

Upload to `obs://<bucket>/climbmix_prod/assets_big/`:

| Asset | Source on server | Target |
|---|---|---|
| nanochat-npu repo (code, pinned commit) | `tar czf nanochat-npu.tar.gz -C /home/ma-user/work nanochat-npu` | `assets_big/nanochat-npu.tar.gz` |
| d20 base checkpoint | `$NANOCHAT_BASE_DIR/base_checkpoints/d20` (model+meta, NO optim shards needed for proxy: `--load-optimizer 0`) | `assets_big/d20/` |
| tokenizer | `$NANOCHAT_BASE_DIR/tokenizer` | `assets_big/tokenizer/` |
| eval datasets | `$NANOCHAT_BASE_DIR/eval_bundle`, `eval_stem` | `assets_big/eval_bundle/`, `assets_big/eval_stem/` |

Verify with: `python3 scripts/dispatch_remote.py --remote-config <cfg.json>
--check-assets` (also checks the auto-uploaded worker bundle under
`assets/`).

The job boot shell (composed by `ModelArtsJobAPI.submit`) untars the repo
to `/home/ma-user/work/nanochat-npu` and copies the data assets to
`/home/ma-user/work/nanochat_base/{base_checkpoints/d*,tokenizer,
eval_bundle,eval_stem}` — every `d<depth>` dir under assets_big is
auto-discovered, so any proxy_depth works without extra knobs. Marker
files (`.climbmix_asset_ok`) skip the re-download when the pool reuses a
node.

## 5. Network

- [ ] Container egress to `hf-mirror.com` (eval fallback downloads).
      `REMOTE_JOB_ENV HF_ENDPOINT=https://hf-mirror.com` is already wired.
- [ ] OBS reachable from job containers (same-region VPC default; the
      hello-world job's moxing read proves it).

## 6. Adapter status (was "fill the two adapters")

DONE (2026-08-29) — `modelarts_job_api.py` (submit/status/cancel/logs/
free_job_slots + `submit_raw` for calibration jobs), `iam_token.py`,
`ModelArtsObsStorage` (moxing). The Mock implementations in `job_api.py` /
`obs.py` remain the behavioral reference (the 129-check simulation in
`/tmp/opencode/test_remote_executor.py` + 77-check adapter suite in
`/tmp/opencode/test_modelarts_adapter.py` are the contract). Runtime
calibration points (hello-world + M3 wave): the v2 status table, the
transient-error patterns, DELETE-on-cancel.

## 7. M3 consistency validation (gate to production)

0. Hello-world job green (§2) + `--check-assets` green.
1. Short smoke: dispatch one 5-step experiment remotely
   (`scripts/dispatch_remote.py ... --proxy-num-iterations 5`).
2. Full check: run exp_0000's weights BOTH locally (ProxyRunner) and remotely
   (dispatch_remote) into separate throwaway output dirs; compare
   `mixture_data/shard_*.parquet` sha256 (must be byte-identical) and
   `stem_metric` (Δ < 0.002 — TODO.md M3 criterion).
3. Concurrency wave: dispatch 2-4 experiments at once (separate shells /
   sequential submissions while earlier jobs still run — each its own
   `--exp-id` + `--weights` + throwaway `--output-dir`) and watch the
   dynamic scheduler against the REAL pool: job completion frees a slot
   and the next config starts with zero backoff; capacity rejections
   back off and retry; the in-flight peak matches the intended slots.
   This wave also verifies the two-anchor throughput prediction
   T(4) ≈ 5.3h incidentally (dedicated S(k) measurement skipped by
   decision 2026-08-28 — the two-anchor model's S(4)≈3.5 assumption gets
   its first real datapoint here, no extra experiment needed), and it
   calibrates the pool-full error code (§2).
4. Green → set `REMOTE_ENABLED=1 ... bash runs/run_climbmix.sh`.

## Production launch (M5) — recommended knobs & calculation

### First production run (recommended 2026-08-28)

| Knob | Value | Rationale |
|---|---|---|
| `NPU_PER_EXP` | **4** | λ_max ≈ 5h for the first run (disaster cap ~74 card·h @32-card pool, half-day verification window — docs/parallel_k_selection.md §5.1). `REMOTE_NPU_PER_JOB` defaults to `NPU_PER_EXP`; do NOT set it separately — k stays fleet-wide fixed for score comparability. |
| `REMOTE_MAX_JOBS` | **⌊A_eff/4⌋, typically 6-8** | A_eff = expected pool median~P25 (24-32 cards), NOT the peak. Mid-run growth is absorbed by the capacity monitor (jobs start as idle cards appear); shrink only queues pickups, never kills in-flight jobs. |
| `REMOTE_LOCAL_PARALLEL` | 1 (default) | Master node's 8 cards join the fleet: 2 more k=4 slots (configs run via the local parallel path while remote jobs are in flight). |
| `REMOTE_FLAVOR` | 4-card flavor | From the console (§2); or `default_flavor` in the platform config. |
| `REMOTE_MAX_PREP` | 4 (default) | Prep/upload semaphore on the master node — raise only if I/O is confirmed idle; it never needs to match the job count. |
| `REMOTE_SUBMIT_RETRY_H` | 24 (default) | Backoff net for probe/submit races; also the whole capacity story when the region has NO quota-query API (fixed limit + rejected-submit retry). |
| `REMOTE_JOB_TIMEOUT_H` | 6 (default) | T(4) ≈ 5.3h + margin. |

Wall-clock estimate for [20,10,5] (35 configs): C = 2 local + 6-8 remote ≈ 8
slots → waves ⌈20/8⌉+⌈10/8⌉+⌈5/8⌉ = 3+2+1 = 6 → **Wall ≈ 5.3h × 6 ≈ 32h**
(search ~1.5 days). Target arms stay local (ws=8, ~6h each + eval).

### Recalculating for the next run (the 30-second method)

Full decision model: `docs/parallel_k_selection.md`. Short version:

1. **Slots** `C(k) = 8/k (local) + ⌊A_eff/k⌋ (remote, node-packed)` —
   A_eff = pool median~P25, k ∈ {1,2,4,6,8} (flavor granularity).
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
from the first run's observed pool (console monitoring, or the
submission cadence visible in the `[Exp N] submitted job ...` lines).

### Launch command

```bash
# once: ~/.config/climbmix/remote_ma.json filled (gateway/auth/image/pool)
REMOTE_ENABLED=1 \
REMOTE_OBS_PREFIX=obs://<bucket>/climbmix_prod \
REMOTE_FLAVOR=<4-card-flavor> \
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
it before starting the production search (this changeset; do not edit
these files mid-search).
