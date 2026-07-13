# CLIMB vs Paper: Alignment TODO List

Source: arXiv:2504.13161 (Nemotron-CLIMB)

## Resolved Issues ✅

### 1. Quality score dimensions mismatch ✅
- **Resolution**: Changed `QualityFilterConfig.method` default to `"none"` — no filtering initially.
  Threshold adjustment deferred until quality scores are properly normalized or
  paper's 4-dim Nemotron-340B classifiers are retrained.

### 2. Proxy model trained from scratch vs continual pre-training ✅
- **Resolution**: Added `ProxyConfig.phase1_checkpoint_path` to load phase-1 checkpoint.
  `ProxyRunner._create_model()` loads checkpoint when path is provided.
  **Why paper does this**: measuring *incremental improvement* from data mixture
  on a pretrained model (not absolute loss from random init). A pretrained model
  has baseline knowledge; the validation gain reflects the mixture's value.
  Analogous to real pre-training where you add new data to an existing model.

### 3. Validation metric: loss vs benchmark accuracy ✅
- **Resolution**: Integrated lm-eval-harness via `benchmark_eval.py`.
  `ProxyConfig.validation_metric` default changed to `"accuracy"`.
  `ProxyResult` now has `validation_accuracy` + `per_task_accuracies`.
  `ProxyResult.score` property returns accuracy (positive) or -loss.
  Bootstrapper uses `metric_direction` ("maximize"/"minimize") for correct ranking.

### 4. Proxy training parameters mismatch ✅
- **Resolution**: Token-based batch via `ProxyConfig.batch_tokens` (default 2M).
  `SIZE_PARAMS` table maps model_size → (batch_tokens, LR, decay_LR, training_tokens).
  WSD schedule implemented (`_wsd_schedule`): warmup → stable → decay.
  `apply_size_defaults()` auto-sets params per model size.

### 5. Default quality filter strategy mismatch ✅
- **Resolution**: `QualityFilterConfig.method` default changed to `"none"`.
  Initially use FDC domain classification without quality-based pruning.
  Cluster-level and doc-level filters remain available as options.

### 6. Jitter in iterative search sampling ✅
- **Resolution**: Replaced Gaussian jitter with Dirichlet exploration.
  `DirichletSampler.sample_from_top_n()` now uses `Dir(concentration * weights)`
  centered around each top-N config. Naturally simplex-constrained, no clip+renorm.
  `exploration_concentration` param controls how close samples stay to originals.

### 7. LightGBM extra hyperparameters ✅
- **Resolution**: `subsample=1.0` (paper doesn't row-subsample).
  `colsample_bytree` dynamically computed via `_compute_colspace()`:
  `min(1.0, max(0.3, 20/num_clusters))` — scales with search parameter count.
  21 clusters → 0.95 (≈paper), 50 clusters → 0.4, 10 clusters → 1.0.

### 8. Final optimal weight selection from sampled pool vs full space ✅
- **Resolution**: `_search_full_design_space()` samples 100K candidates at
  multiple Dirichlet concentrations (1, 5, 10, 50) for wide exploration,
  then refines 5K around the top prediction with high concentration (50).
  Total: 105K candidates evaluated, much closer to "full design space".

## Remaining Work

- Retrain 4-dim Nemotron-340B fasttext classifiers (or normalize existing scores)
- Obtain actual phase-1 pretrained checkpoints for proxy models
- lm-eval integration needs real proxy model testing (requires GPU + model weights)
- GPU end-to-end test of full pipeline with WSD schedule + accuracy validation

## Already Aligned (Verified ✅)

- Embedding model: stella_en_400M_v5 ✅
- K_init=1000, K_enhanced=21 ✅
- FAISS K-means ✅
- Cluster prune threshold=3.0, merge distance=1.5 ✅
- 3 iterations with 64/32/16 configs ✅
- LightGBM predictor with L1/L2/max_depth=4/min_samples_leaf=5/early_stopping=20 ✅
- Dirichlet initialization proportional to token counts ✅
- Validation tasks: PIQA, ARC_E, HellaSwag ✅
- Predictor trained on accumulated (current + past) data ✅
- Proxy model sizes: 62M, 350M ✅
