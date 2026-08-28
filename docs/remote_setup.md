# Remote fleet setup (ModelArts + OBS) — M1 environment survey checklist

> Status: PENDING (all code is in — M2 done; this file collects the facts you
> must gather on the server / ModelArts console before the first real job).
> Owner workflow: fill each ✅ below; when all are green, implement the two
> `NotImplementedError` adapters (`ModelArtsJobAPI`, `ModelArtsObsStorage`)
> and run the M3 validation (`scripts/dispatch_remote.py`).

Production topology (decided 2026-08-28): local 8x910B4 host = scheduler +
mixer; remote cards (shared pool, fluctuating 10-200, typical ~32 — see
`docs/parallel_k_selection.md`) = compute via ModelArts Job API; OBS = data
plane. One job per proxy experiment (`npu_per_job` cards, no cross-node
collectives). Results materialize as local `exp_XXXX/` dirs; search resume
and stage fingerprints unchanged.

## 1. SDK & auth (on the submit host = 8x910B4 server)

- [ ] Which packages exist in the ma-user environment?
      `pip list | grep -Ei 'moxing|modelarts|esdk|obs'`
      Expected: `moxing` (usually preinstalled on ModelArts images). If none:
      decide `pip install esdk-obs-python` vs plain REST + `obsutil` binary.
- [ ] Credentials: AK/SK + project_id + region (console → "My Credentials").
      Env convention used by the stubs: `MA_AK / MA_SK / MA_PROJECT_ID /
      MA_REGION` (jobs) and `OBS_AK / OBS_SK / OBS_SERVER` (storage).
- [ ] Bucket: create/confirm one bucket, note its endpoint (`OBS_SERVER`,
      e.g. `obs.cn-xxx-4.myhuaweicloud.com`), and pick a layout root, e.g.
      `obs://<bucket>/climbmix_prod` (= `REMOTE_OBS_PREFIX`).

## 2. Compute quota & flavor

- [ ] Dedicated pool (recommended — stable 910B4 nodes) vs public pool.
      Note pool name (`REMOTE_POOL_NAME`, empty = public).
- [ ] Ascend flavor for a 1-card job (and the 8-card variant if you later
      want remote target arms): note its exact spec string
      (`REMOTE_FLAVOR`).
- [ ] Max concurrent jobs in the pool / region (drives `REMOTE_MAX_JOBS`;
      the k-selection doc's fleet sizing uses the same number).
- [ ] **Submit behavior under quota pressure** (drives the dynamic-submission
      policy, `remote_executor._submit_with_retry`): when the pool is full,
      does submit FAIL with a quota/capacity error code, or does the job go
      PENDING and queue server-side? Which HTTP/SDK error codes/texts mean
      "retryable" (map them to `TransientSubmitError` in
      `ModelArtsJobAPI.submit`) vs hard (bad image/auth/spec → RuntimeError)?
- [ ] Is there a quota/usage QUERY API (free cards right now)? If yes,
      implement `ModelArtsJobAPI.free_job_slots()` (return
      `free_cards // npu_per_job`, None while unsupported): the executor's
      capacity monitor already re-probes every poll interval and grows/
      shrinks the in-flight limit mid-batch — the 98-exp/16-card scenario
      then drains as idle cards appear. Without it the executor falls back
      to fixed-cap + submit-rejected backoff (correct, just slower to
      react).
- [ ] Preemption: can a running job be reclaimed by the pool (→ would justify
      job-level checkpoint chaining later), or do jobs always run to
      terminal state?
- [ ] Submit ONE hello-world Ascend job from the console (image below,
      `npu-smi info` as the command) to prove quota + network end to end.

## 3. Image

- [ ] Preferred: ModelArts built-in Ascend engine image matching CANN
      8.5.1 / torch_npu of the local host (check `npu-smi info` +
      `pip show torch-npu` versions on the server first).
- [ ] Else: bake a SWR image (base Ascend image + `pip install
      torch-npu==<same version> pyarrow numpy pandas lightgbm` + nothing
      else — the worker is stdlib-only) and note the SWR URI
      (`REMOTE_IMAGE`).
- [ ] Container workdir convention is `/home/ma-user/work` (matches the
      RemoteConfig defaults `container_*`); confirm the image's user can
      write there.

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

The job boot shell (implemented in `ModelArtsJobAPI.submit`, M1) untars the
repo to `/home/ma-user/work/nanochat-npu` and rsyncs the data assets to
`/home/ma-user/work/nanochat_base/{base_checkpoints/d20,tokenizer,...}` once
per node (cache across jobs if the pool reuses nodes).

## 5. Network

- [ ] Container egress to `hf-mirror.com` (eval fallback downloads).
      `REMOTE_JOB_ENV HF_ENDPOINT=https://hf-mirror.com` is already wired.
- [ ] OBS endpoint reachable from job containers (same-region VPC default).

## 6. Fill the two adapters (~1-2 h after facts are known)

- `src/climbmix/remote/job_api.py` → `ModelArtsJobAPI` (submit/status/logs/
  cancel over the chosen SDK; compose the boot shell around the worker argv).
- `src/climbmix/remote/obs.py` → `ModelArtsObsStorage` (moxing or esdk).
  Both interfaces are final; the Mock implementations in the same files are
  the behavioral reference (the 90-check simulation in
  `/tmp/opencode/test_remote_executor.py` is the contract).

## 7. M3 consistency validation (gate to production)

1. Short smoke: dispatch one 5-step experiment remotely
   (`scripts/dispatch_remote.py ... --proxy-num-iterations 5`).
2. Full check: run exp_0000's weights BOTH locally (ProxyRunner) and remotely
   (dispatch_remote) into separate throwaway output dirs; compare
   `mixture_data/shard_*.parquet` sha256 (must be byte-identical) and
   `stem_metric` (Δ < 0.002 — TODO.md M3 criterion).
3. Green → set `REMOTE_ENABLED=1 ... bash runs/run_climbmix.sh`.

## Production launch (M5) — knob summary

```bash
REMOTE_ENABLED=1 \
REMOTE_OBS_PREFIX=obs://<bucket>/climbmix_prod \
REMOTE_IMAGE=<swr-uri-or-built-in> \
REMOTE_FLAVOR=<ascend-flavor> \
REMOTE_NPU_PER_JOB=1 \          # k per docs/parallel_k_selection.md
REMOTE_MAX_JOBS=<pool capacity> \
REMOTE_LOCAL_PARALLEL=1 \       # hybrid: local 8 cards join the fleet
bash runs/run_climbmix.sh
```

All `REMOTE_*` knobs are execution-shape only (num_npu precedent) and are
deliberately excluded from the stage fingerprints. NOTE: the remote CODE
itself (`src/climbmix/remote/`, `scripts/remote_worker.py`,
`nanochat_cmds.py`, `proxy_runner.py`) IS fingerprinted (BOTH stages) — land
it before starting the production search (done: this changeset; do not edit
these files mid-search).
