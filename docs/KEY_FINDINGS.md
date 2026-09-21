# Takeaways — What We Learned Reproducing Nemotron-CLIMB

> The condensed, transferable findings from four production rounds (a full
> validation round costs ≈1,800 NPU-hours: d20 search fleet + 8 target arms
> at ~1.5B scaling / 3B tokens + anchors). Every claim below is evidenced in
> the round records; this page is the index-grade summary. Detail chain:
> [experiment_prod4.md](experiment_prod4.md) (verdicts, §5–§6) ·
> [algorithm_review.md](algorithm_review.md) (predictor/clustering audits) ·
> [paper_deviations.md](paper_deviations.md) (D1–D19, every deliberate
> deviation from the paper).

## Headline results

- **The CLIMB premise holds at target scale.** Search-found mixtures beat
  uniform-cluster / natural (pool-proportional) / hand-tuned domain-ratio
  baselines by **+0.014–0.031 STEM** on a ~1.5B model at a 3B-token
  mid-training budget. The gap is carried by generative math reasoning:
  gsm8k **2.4–2.9× the uniform family**, 11× the base model.
- **The novel finding the paper does not discuss: final selection is the
  weak link.** The predictor's design-space argmin (its "there is a better
  point over there" claim) added nothing over *measured* configs — both
  measured top configs beat the never-measured extrapolated winner. A soft
  winner's curse, and it repeats a pattern: the extrapolated optimum sits
  exactly in the directions where the predictor has the least data.

## Eight transferable design rules

Condensed from experiment_prod4.md §6; each row's full evidence chain is in
the source table.

| # | Rule | One-line evidence |
|---|---|---|
| 1 | **Select best-measured by default; extrapolate only with a claim that clears a noise-calibrated margin** (no-claim guard) | the prod4 argmin winner's claimed gain was **−0.47** in replay; the guard fires at any margin ≥ 0 |
| 2 | **Consume the search output as a region (top-k measured), not a single point** | d20↔d28 rank flips inside the hot zone (proxy online ρ ≈ 0.42–0.50); the d28 winner was the d20 runner-up |
| 3 | **Keep the skeleton: clustering + iterative guided search + predictor pruning** | all four CLIMB points clear the uniform band ceiling; the d28 winner came from a *guided* round, not the initial sweep |
| 4 | **Cluster-level selection is not replaceable by domain-level ratios** | a hand-tuned math-60% domain mix lands inside the uniform band; the win is in cluster-level structure |
| 5 | **Mid-training on a skewed mix is a capability swap, not a free win** | uniform-family mean ≈ base model; MC tasks drift down while gsm8k triples — composite scores hide this |
| 6 | **Read arms by large-N / reproducible columns first** | gpqa (N=198) swings ±.03–.07 on a single seed and dominates composite noise; gsm8k and NLL columns are the reliable invariants |
| 7 | **Pool supply constraints become active constraints at scale — and couple with mixture choice** | winner-cluster 20B oversampling ratio ≈ 2.6 vs cap 2; plan the policy before scaling budgets |
| 8 | **Run a reproducibility machine every round** — seed pairs, remote-anchor reconciliation, protocol freeze windows, data cross-accounting | 4-decimal anchor matches, zero-incident rounds; every "weird number" so far was caught by the machine, not by luck |

## The predictor boundary (what the surrogate model can and cannot be trusted for)

Measured on the 111-point prod4 fleet (`scripts/diagnostics/predictor_audit.py`,
zero-NPU replay):

- **Reliable: ranking regions.** Top-10 predicted vs top-10 measured overlap
  8/10; pooled held-out ρ 0.58; two independently fitted models agree at
  ρ = 0.935. Region-level guidance (which area to sample next) is where the
  value is.
- **Unreliable: single-point argmin.** The extrapolated minimum lands in
  thin-response directions (2–3% feature importance across split/gain/SHAP
  scorings) where the trees are effectively unconstrained — and the
  "winning valley" did not replicate across LightGBM builds
  (implementation-sensitivity as independent evidence of extrapolation
  fragility).
- **Refit on full data after early stopping is free.** Out-of-fold
  ρ 0.606 → 0.633, R² +18% relative — the split-fit model was leaving
  ~20% of the measured points on the table at final-selection time.
- **Late rounds over-exploit.** The pruning band's membership churned ~50%
  across model refits; round-3 narrowing produced no improvement over
  round-1 best. Root-cause fixes (seed ensembles + LCB acquisition) are
  queued as next-round work, not shipped.

## Measurement methodology: noise floor first

- **Training-seed band ±0.016–0.032 STEM** is the largest noise component.
  Report gaps as bands; crown within-band ties explicitly instead of
  re-rolling for a preferred winner.
- **Benchmark sampling noise is task-dependent.** Small-N tasks (gpqa N=198)
  swing ±.03–.07 on one seed; large-N generative tasks and NLL columns are
  bitwise-reproducible invariants — read those first.
- **Eval-protocol changes act non-uniformly across arms.** A generation-cap
  fix moved math +36% on climb-type mixtures and 0% on random-type ones
  (capability floor vs truncation floor). Freeze the protocol within a
  round; when it must change, re-evaluate the whole scoreboard under the
  new protocol before comparing across rounds.

## Engineering lessons that saved rounds

- **Identity guards on every cache.** Reuse keyed on (seed, token budget,
  weights content-hash, label-source) — without it, "same output directory,
  different mixture" fails silently. This class of bug appeared three times
  in three disguises (local `.done`, OBS stale keys, boot-shell hardlinks)
  before being generalized.
- **Floor your iteration counts.** Guided search silently under-delivers
  planned configs unless floored and asserted (one round ran 9/20 valid
  points before the floor existed).
- **DDP row-group starvation.** Every shard needs ≥ 2 row groups per rank
  or ranks with no data hang before the first all_reduce — size row groups
  from the *smallest* shard, not the average.
- **Crash resume at experiment granularity** (rc + weight-match), not stage
  granularity — a fleet of 100+ proxy trainings survives node churn with
  ≈0 lost work.
- **Single-pass, token-capped selection keeps proxy budgets honest** —
  steps derived from tokens / total-batch-size, never from step knobs.

## Where the next gains might come from (open thread)

- Within-cluster quality selection (quality scores currently gate whole
  clusters at pruning time only; within-cluster selection is random) —
  offline-first, zero-NPU feasibility on the existing fleet data.
- Acquisition robustness: predictor ensembles + LCB to fix the
  over-exploitation pattern.
- Curriculum ordering (quality-dependent data order) — depends on the
  quality-score track above.

## Reading map

| You want | Read |
|---|---|
| The round-by-round story with every number | experiment_prod*.md (one file per round) |
| Why each deviation from the paper exists | paper_deviations.md (D1–D19) |
| The algorithm design review + audits | algorithm_review.md |
| How to launch / monitor / rehearse a round | prod5_runbook.md, scripts/diagnostics/ |
