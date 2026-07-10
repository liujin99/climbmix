# CLIMB vs Paper: Alignment TODO List

Source: arXiv:2504.13161 (Nemotron-CLIMB)

## Critical Issues

### 1. Quality score dimensions mismatch
- **Paper**: 4 dimensions (quality, educational, informational, advertisement), scored 1-5 by Nemotron-340B, trained fasttext classifiers on 1M annotated texts
- **Code**: 5 fasttext scores from Essential-Web (`qs_dclm`, `qs_fineweb_edu_approx`, `qs_english`, `qs_eai_general_math`, `qs_eai_open_web_math`) with different ranges (0-1, 0-3.9, etc.)
- **Impact**: `prune_threshold=3.0` assumes 1-5 range but actual scores are 0-1 range; threshold will prune almost everything
- **Fix**: Normalize all scores to 0-5 range OR retrain 4-dim fasttext classifiers as in paper; adjust thresholds accordingly

### 2. Proxy model trained from scratch vs continual pre-training
- **Paper**: Proxy models start from phase-1 pretrained checkpoints (trained on 10T tokens), then do continual pre-training on 40B tokens with mixture data
- **Code**: `proxy_runner.py` creates fresh `ProxyModel()` from random initialization, trains 1000 steps
- **Impact**: Validation loss doesn't reflect "improvement from mixture data" on a pretrained model; absolute loss vs relative improvement
- **Fix**: Load phase-1 pretrained checkpoint as starting point for each proxy experiment

### 3. Validation metric: loss vs benchmark accuracy
- **Paper**: Uses lm-evaluation-harness to compute benchmark accuracy on PIQA, ARC_E, HellaSwag (0-shot)
- **Code**: `ProxyRunner._run_validation()` computes cross-entropy loss on tokenized validation data
- **Impact**: Loss ≠ accuracy; optimization target is different from paper's
- **Fix**: Integrate lm-evaluation-harness for proper benchmark evaluation (computationally expensive but necessary for alignment)

## High Priority Issues

### 4. Proxy training parameters mismatch
- **Paper**: Batch size = 2M tokens; LR = 5e-5 (stable) → 1e-5 (decay); WSD schedule; AdamW
- **Code**: `ProxyConfig.batch_size=64` (sequences, not tokens); `learning_rate=4e-4`; linear warmup + cosine decay
- **Fix**: Change to token-based batch size config; implement WSD (warmup-stable-decay) schedule; adjust LR to 5e-5

### 5. Default quality filter strategy mismatch
- **Paper**: Only cluster-level pruning (part of cluster merging step); no document-level filtering
- **Code**: Default `QualityFilterConfig.method="doc_and_cluster"` — does both doc-level and cluster-level
- **Fix**: Change default to `"cluster_level"`; keep doc-level as optional but not default

## Medium Priority Issues

### 6. Jitter in iterative search sampling
- **Paper**: "randomly sample M new configurations from the top N ranked configurations" — pure random sampling from top-N
- **Code**: `DirichletSampler.sample_from_top_n()` adds Gaussian jitter (scale=0.1) then normalizes
- **Fix**: Remove jitter; implement simple random sampling from top-N ranked configs

### 7. LightGBM extra hyperparameters
- **Paper**: Only specifies L1/L2 regularization, max_depth=4, min_samples_leaf=5, early_stopping=20
- **Code**: Also sets `subsample=0.8` and `colsample_bytree=0.8` which paper doesn't mention
- **Fix**: Set `subsample=1.0` and `colsample_bytree=1.0` for strict paper alignment (or keep as enhancement with ablation)

## Low Priority Issues

### 8. Final optimal weight selection from sampled pool vs full space
- **Paper**: "selects the best configuration predicted by the final predictor" from the full design space A
- **Code**: Samples 10000 configs from Dirichlet, picks best predicted from that pool
- **Impact**: Search space limited to 10000 samples from Dirichlet distribution
- **Fix**: Acceptable for practical purposes; could increase pool size or use grid search for stricter alignment

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
