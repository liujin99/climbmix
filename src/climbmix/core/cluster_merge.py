"""
CLIMB Step 2: Cluster pruning and merging.

Paper details (Section 2.1, "Cluster merging"):
  1. Cluster-level pruning: remove low-quality clusters based on quality scores
     (threshold=3.0), retaining K_pruned clusters
  2. Merge clusters "according to the distance between centroids" into
     K_enhanced < K_pruned < K_init clusters (the paper text gives no numeric
     threshold; we use a tau-guarded band instead — see below)

Quality labels (configurable via config/quality_columns.yaml):
  STEM: stem_relevance, knowledge_value, notation_fidelity,
        rigor_coherence, noise_level (all 1-5, higher = better)
  FineWeb: qs_dclm, qs_fineweb_edu_approx, qs_english, ...
  Nemotron: qs_quality, qs_educational, qs_informational, qs_advertisement

If no quality labels are found, pruning is skipped and clusters are
merged directly to K_enhanced.

This produces the final cluster set D = {D_1, ..., D_K_enhanced}
that defines the data mixture search space.
"""

import time
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist

from climbmix.core.types import ClusterInfo


def compute_cluster_quality(
    cluster_labels: npt.NDArray[np.int64],
    quality_scores: Optional[npt.NDArray[np.float64]] = None,
    quality_columns: Optional[List[str]] = None,
    prune_threshold: float = 3.0,
) -> Dict[int, float]:
    """
    Compute per-cluster quality score for pruning.

    Quality scores are 1-5 discrete, higher = better (all dimensions
    including noise_level). Prune clusters with average quality < threshold.

    If quality_scores is None or all-zero, skip pruning (assign 5.0 to all).
    This handles data without quality labels gracefully.

    Args:
        cluster_labels: Per-document cluster labels.
        quality_scores: Per-document quality scores (num_docs, N).
                        If None or all-zero, no pruning is done.
        quality_columns: Names of quality criteria.
        prune_threshold: Quality threshold for cluster pruning.

    Returns:
        Dict mapping cluster_id -> average quality score.
    """
    cluster_quality: Dict[int, float] = {}

    if quality_scores is None or np.all(quality_scores == 0):
        unique_clusters = np.unique(cluster_labels)
        for c in unique_clusters:
            if c < 0:
                continue
            cluster_quality[int(c)] = 5.0
        if quality_scores is not None and np.all(quality_scores == 0):
            print("[ClusterQuality] All-zero scores detected, skipping pruning")
        return cluster_quality

    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        if c < 0:
            continue
        mask = cluster_labels == c
        avg_quality = float(quality_scores[mask].mean())
        cluster_quality[int(c)] = avg_quality

    return cluster_quality


def prune_clusters(
    cluster_labels: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float32],
    cluster_quality: Dict[int, float],
    threshold: float = 3.0,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], Dict[int, int]]:
    """
    Prune low-quality clusters based on quality threshold.

    Args:
        cluster_labels: Per-document cluster labels (K_init clusters).
        centroids: Cluster centroids (K_init, dim).
        cluster_quality: Per-cluster quality scores.
        threshold: Minimum quality threshold.

    Returns:
        Tuple of (pruned_labels, pruned_centroids, old_to_new_id_map).
    """
    unique_clusters = np.unique(cluster_labels)
    kept_clusters = []
    for c in unique_clusters:
        if c < 0:
            continue
        if cluster_quality.get(int(c), 0.0) >= threshold:
            kept_clusters.append(int(c))

    n_pruned = len(unique_clusters) - len(kept_clusters)
    print(f"[Prune] Pruned {n_pruned}/{len(unique_clusters)} clusters "
          f"(threshold={threshold}), keeping {len(kept_clusters)}")

    old_to_new: Dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(kept_clusters)):
        old_to_new[old_id] = new_id

    pruned_labels = np.full(len(cluster_labels), -1, dtype=np.int64)
    for old_id, new_id in old_to_new.items():
        mask = cluster_labels == old_id
        pruned_labels[mask] = new_id

    kept_indices = sorted(old_to_new.keys())
    pruned_centroids = centroids[kept_indices].copy()

    n_removed = int((pruned_labels == -1).sum())
    print(f"[Prune] Removed {n_removed} documents from pruned clusters")

    return pruned_labels, pruned_centroids, old_to_new


def natural_k_from_profile(profile, tau):
    """Largest K whose closest-pair distance already exceeds tau.

    profile: list of (K, closest_pair_dist) in merge order (K descending).
    Returns None if the distance never exceeded tau within the recorded
    range (i.e., natural structure is at or below the final K).
    """
    for k, d in profile:
        if d > tau:
            return k
    return None


def suggest_elbow(profile):
    """K after the largest consecutive distance jump (dendrogram elbow).

    The jump (k1, d1) -> (k2, d2) means merging from K=k1 down to K=k2
    crosses the biggest distance increase: k2 is where "all close pairs
    merged, everything left is far apart" — the natural cluster count.

    Returns (K, gap). None if the profile is too short.
    """
    if len(profile) < 2:
        return None, 0.0
    best_gap, best_k = 0.0, None
    for (_, d1), (k2, d2) in zip(profile, profile[1:]):
        gap = d2 - d1
        if gap > best_gap:
            best_gap, best_k = gap, k2
    return best_k, best_gap


def _sample_profile(profile, max_points=12):
    """Evenly spaced profile entries (first and last always kept) for display."""
    if len(profile) <= max_points:
        return list(profile)
    idxs = sorted(set(
        int(round(i)) for i in np.linspace(0, len(profile) - 1, max_points)))
    return [profile[i] for i in idxs]


def diagnose_merge_profile(
    profile: List[Tuple[int, float]],
    merge_distance: float,
    k_floor: Optional[int],
    k_max: Optional[int],
    k_final: int,
    stop_reason: str,
    forced: bool,
    elbow_k: Optional[int],
    natural_k_table: Dict[float, Optional[int]],
) -> List[str]:
    """Preliminary tuning advice for the user, derived from THIS pool's profile.

    Decision table (what the user should look at):
      guard stop inside band   -> parameters fine, no change needed
      forced merges            -> raise K_max if natural_K slightly above cap;
                                  otherwise keep (paper fixes K by budget too)
      floor stop               -> pool naturally coarser than the floor
      tau swing across table   -> how trustworthy natural_K is
      elbow vs natural_K(tau)  -> independent cross-check

    Returns a list of advice lines ("OK" / "!" / "info" markers included);
    printed after clustering and stored in merge_profile.json.
    """
    advice: List[str] = []
    if not profile:
        advice.append("no merges were needed (K already within the band) — nothing to tune")
        return advice

    nk_now = natural_k_from_profile(profile, merge_distance)
    vals = [k for k in natural_k_table.values() if k is not None]

    if vals:
        lo, hi = min(vals), max(vals)
        if hi - lo <= 1:
            advice.append(f"✓ natural_K is stable across tau ({lo}-{hi}): robust "
                          f"structure, the exact tau value barely matters")
        elif hi - lo <= 3:
            advice.append(f"· natural_K is mildly tau-sensitive ({lo}-{hi}): acceptable; "
                          f"tau={merge_distance} sits mid-table")
        else:
            advice.append(f"· natural_K swings {lo}-{hi} across tau: no sharp structure — "
                          f"keep tau={merge_distance} or trust the elbow (K={elbow_k})")

    if elbow_k is not None and nk_now is not None:
        if abs(elbow_k - nk_now) <= 2:
            advice.append(f"✓ elbow (K={elbow_k}) agrees with natural_K(tau)={nk_now} — "
                          f"high confidence")
        elif abs(elbow_k - nk_now) >= 5:
            advice.append(f"· elbow (K={elbow_k}) disagrees with natural_K(tau)={nk_now} — "
                          f"structure ambiguous; consider the elbow as an alternative")

    if stop_reason == "guard" and not forced:
        advice.append(f"✓ natural structure (K={k_final}) falls inside the band "
                      f"[{k_floor}, {k_max}] — parameters OK, no change needed")
    elif forced:
        if nk_now is not None and k_max is not None and nk_now <= k_max + 3:
            advice.append(f"! pool slightly richer than the cap (natural_K(tau)={nk_now} > "
                          f"K_max={k_max}): consider raising K_CLUSTER_MAX/--K-max to "
                          f"{nk_now} if the search budget allows")
        elif nk_now is not None:
            advice.append(f"! pool much richer than the cap (natural_K(tau)={nk_now} >> "
                          f"K_max={k_max}): forced merges are by design (the paper also "
                          f"merges to a budget-fixed K); raise the cap only if the search "
                          f"budget allows")
        else:
            advice.append(f"! no natural_K at tau={merge_distance} before the cap: "
                          f"structure richer than the band — forced merges are by design "
                          f"(the paper also fixes K by budget)")
    elif stop_reason == "floor":
        hint = ""
        stricter = [(t, k) for t, k in sorted(natural_k_table.items())
                    if k is not None and t < merge_distance]
        if stricter:
            t, k = stricter[-1]
            hint = f"; a stricter tau would keep more clusters (tau={t} -> K={k})"
        advice.append(f"· pool is naturally coarse: merges stayed legal all the way to "
                      f"the floor (K pinned at {k_floor}){hint}")
        if k_floor is not None and k_floor > 3:
            advice.append(f"· to follow an even coarser natural structure, lower "
                          f"K_ENHANCED/--K-enhanced below {k_floor}")

    advice.append("knobs: MERGE_DISTANCE (tau) / K_ENHANCED (floor) / K_CLUSTER_MAX "
                  "(cap) — or --merge-distance/--K-enhanced/--K-max; re-merge after "
                  "a knob change costs seconds (embeddings are pool-cached)")
    return advice


def merge_clusters_by_distance(
    cluster_labels: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float32],
    merge_distance: float = 0.9,
    target_K: Optional[int] = None,
    K_max: Optional[int] = None,
    profile_path: Optional[str] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], Dict[int, int]]:
    """
    Merge similar clusters based on centroid Euclidean distance.

    Band stop rule (pool-adaptive K, bounded by the search budget):

    1. floor:  stop when K <= target_K            (safety bound against
                degenerate collapse on coarse pools)
    2. guard:  stop when closest-pair distance > merge_distance AND K <= K_max
                (never force-merge semantically distinct clusters inside the
                band — the pool's natural structure wins)
    3. cap:    when K > K_max and distance > merge_distance, merge anyway
                (closest-pair first, logged) so heterogeneous pools still end
                within the search budget.

    Result: K_final = clamp(natural_K(merge_distance), target_K, K_max).

    On unit-normalized embeddings, L2 distance d relates to cosine
    similarity as d^2 = 2(1 - cos); the default 0.9 means clusters must be
    ~60% similar to be mergeable. The paper instead merges to a fixed
    K_enhanced regardless of distance; the band is a deliberate deviation
    (documented). The default floor is a permissive safety bound — set
    target_K to the paper's K_enhanced (e.g. 10 or 20) for paper-faithful
    fixed-K semantics.

    After merging, a diagnostics block is printed (sampled distance profile,
    natural_K at candidate taus, elbow, and tuning advice) and stored in
    merge_profile.json — the audit trail for K decisions on each data pool.

    Args:
        cluster_labels: Per-document cluster labels (after pruning).
        centroids: Cluster centroids.
        merge_distance: Max centroid distance for a merge to be legal (tau).
        target_K: Floor on the final cluster count (safety bound; pass the
               paper's K_enhanced for paper-faithful fixed-K semantics).
        K_max: Cap on the final cluster count (search-budget bound); the
               distance guard is only honored at or below K_max.
        profile_path: Optional path for merge_profile.json.

    Returns:
        Tuple of (merged_labels, merged_centroids, merge_map).
    """
    if (K_max is not None and target_K is not None and K_max < target_K):
        raise ValueError(
            f"K_max ({K_max}) must be >= target_K/K_enhanced ({target_K}): "
            f"the cluster-count band would be empty")

    K = len(np.unique(cluster_labels[cluster_labels >= 0]))
    if K == 0:
        return cluster_labels, centroids, {}

    print(f"[Merge] Starting with K={K} clusters, target_K={target_K} (floor), "
          f"K_max={K_max}, merge_distance={merge_distance}")

    unique_ids = sorted(np.unique(cluster_labels[cluster_labels >= 0]).tolist())
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    D = centroids.shape[1]
    max_id = max(unique_ids) + 1

    current_centroids = np.zeros((max_id, D), dtype=np.float32)
    for uid in unique_ids:
        current_centroids[uid] = centroids[uid]

    cluster_sizes = np.bincount(cluster_labels[cluster_labels >= 0], minlength=max_id).astype(np.int64)

    dist_matrix = cdist(current_centroids[unique_ids], current_centroids[unique_ids], metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    active = np.zeros(max_id, dtype=bool)
    for uid in unique_ids:
        active[uid] = True

    current_to_final: Dict[int, int] = {uid: uid for uid in unique_ids}
    cluster_groups: Dict[int, List[int]] = {uid: [uid] for uid in unique_ids}

    # (K_current, closest-pair distance) per merge step, K descending —
    # the dendrogram cut profile used for the diagnostics below.
    profile: List[Tuple[int, float]] = []
    stop_reason = "floor"
    forced_warning_shown = False

    iteration = 0
    while True:
        active_ids = np.where(active)[0]
        K_current = len(active_ids)

        if target_K is not None and K_current <= target_K:
            break

        sub = dist_matrix[np.ix_(active_ids, active_ids)]
        min_idx = np.argmin(sub)
        i, j = divmod(min_idx, K_current)
        min_dist = sub[i, j]

        profile.append((int(K_current), float(min_dist)))

        if min_dist > merge_distance:
            guard_active = (K_max is None) or (K_current <= K_max)
            if guard_active:
                # Natural structure reached inside the band: the closest
                # remaining pair is semantically too far apart to merge.
                stop_reason = "guard"
                print(f"[Merge] Closest pair distance {min_dist:.4f} > merge_distance "
                      f"{merge_distance:.4f}: stopping at natural K={K_current} "
                      f"(floor={target_K}, cap={K_max})")
                break
            # K_current > K_max: cap takes precedence over the guard —
            # closest-pair-first forced merge, loudly logged.
            if not forced_warning_shown:
                forced_warning_shown = True
                print(f"[Merge] WARNING: pool is more heterogeneous than K_max="
                      f"{K_max} (closest pair {min_dist:.4f} > {merge_distance:.4f} at "
                      f"K={K_current}) — force-merging closest pairs down to K_max")

        id_i = int(active_ids[i])
        id_j = int(active_ids[j])

        docs_i = int(cluster_sizes[id_i])
        docs_j = int(cluster_sizes[id_j])
        new_centroid = (current_centroids[id_i] * docs_i + current_centroids[id_j] * docs_j) / (docs_i + docs_j)

        merged_id = min(id_i, id_j)
        absorbed_id = max(id_i, id_j)

        cluster_groups[merged_id] = cluster_groups.pop(id_i) + cluster_groups.pop(id_j)

        current_centroids[merged_id] = new_centroid
        cluster_sizes[merged_id] += cluster_sizes[id_j]
        active[absorbed_id] = False

        for uid in active_ids:
            if uid == merged_id:
                continue
            d = np.linalg.norm(new_centroid - current_centroids[uid])
            dist_matrix[merged_id, uid] = d
            dist_matrix[uid, merged_id] = d
        dist_matrix[merged_id, merged_id] = np.inf

        for old_id in cluster_groups[merged_id]:
            current_to_final[old_id] = merged_id

        iteration += 1
        if iteration % 50 == 0:
            print(f"[Merge] Iteration {iteration}: K={K_current - 1}, "
                  f"merged {id_i}+{id_j} (dist={min_dist:.3f})")

    final_ids = sorted(cluster_groups.keys())
    final_to_consecutive: Dict[int, int] = {}
    for new_id, final_id in enumerate(final_ids):
        final_to_consecutive[final_id] = new_id

    merged_labels = np.full(len(cluster_labels), -1, dtype=np.int64)
    for old_id, final_id in current_to_final.items():
        mask = cluster_labels == old_id
        new_id = final_to_consecutive.get(final_id, -1)
        merged_labels[mask] = new_id

    merged_centroid_list = [current_centroids[final_id] for final_id in final_ids]
    merged_centroids = np.array(merged_centroid_list, dtype=np.float32)

    print(f"[Merge] Final K={len(final_ids)} clusters from K_init={K} "
          f"(stop={stop_reason}, forced_merges_beyond_tau={forced_warning_shown})")

    # ── Diagnostics: dendrogram cut profile + tuning advice for THIS pool ──
    natural_k_table = {tau: natural_k_from_profile(profile, tau)
                       for tau in (0.7, 0.8, 0.9, 1.0, 1.2)}
    elbow_k, elbow_gap = suggest_elbow(profile)
    if elbow_gap < 1e-3:
        elbow_k = None  # flat profile — the "jump" is noise, not a suggestion
    advice = diagnose_merge_profile(
        profile, merge_distance, target_K, K_max, len(final_ids),
        stop_reason, forced_warning_shown, elbow_k, natural_k_table)

    nk_disp = {t: (k if k is not None else f"<={len(final_ids)}")
               for t, k in natural_k_table.items()}
    print("\n" + "─" * 66)
    print(f"  Merge profile diagnostics  "
          f"(band [{target_K}, {K_max}], tau={merge_distance})")
    print("─" * 66)
    if profile:
        print("  closest-pair distance by K (merge order, sampled):")
        cells = [f"K={k}→{d:.2f}" for k, d in _sample_profile(profile)]
        width = max(len(c) for c in cells) + 2
        for r in range(0, len(cells), 4):
            print("    " + " ".join(c.ljust(width) for c in cells[r:r + 4]).rstrip())
    else:
        print("  (no merges were needed — K was already within the band)")
    print("  natural_K(tau): " + ", ".join(f"{t}→{nk_disp[t]}" for t in sorted(nk_disp)))
    if elbow_k is not None and elbow_gap > 0:
        print(f"  elbow: K={elbow_k} (largest distance jump {elbow_gap:.4f})")
    print(f"  result: K_final={len(final_ids)}  stop_reason={stop_reason}")
    print("  diagnosis:")
    for line in advice:
        print(f"    {line}")
    print("─" * 66)

    if profile_path:
        import json
        profile_data = {
            "K_pruned": K,
            "K_final": len(final_ids),
            "target_K_floor": target_K,
            "K_max": K_max,
            "merge_distance_tau": merge_distance,
            "stop_reason": stop_reason,
            "forced_merges_beyond_tau": forced_warning_shown,
            "profile": [{"K": k, "closest_pair_dist": round(d, 6)} for k, d in profile],
            "natural_K": {f"{t}": nk_disp[t] for t in sorted(nk_disp)},
            "elbow_K": elbow_k,
            "elbow_gap": round(float(elbow_gap), 6),
            "advice": advice,
        }
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2)
        print(f"[Merge] Profile written → {profile_path}")

    return merged_labels, merged_centroids, {old: final_to_consecutive[final] for old, final in current_to_final.items()}


def build_cluster_info(
    merged_labels: npt.NDArray[np.int64],
    merged_centroids: npt.NDArray[np.float32],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
) -> List[ClusterInfo]:
    """
    Build ClusterInfo objects from merged cluster results.
    """
    if token_counts is None:
        token_counts = np.ones(len(merged_labels), dtype=np.int64)

    unique_ids = np.unique(merged_labels[merged_labels >= 0])
    clusters: List[ClusterInfo] = []

    for cid in sorted(unique_ids):
        mask = merged_labels == cid
        n_docs = int(mask.sum())
        n_tokens = int(token_counts[mask].sum())
        clusters.append(ClusterInfo(
            cluster_id=int(cid),
            centroid=merged_centroids[int(cid)].astype(np.float64),
            num_docs=n_docs,
            num_tokens=n_tokens,
            label=f"C{int(cid)}",
        ))

    print(f"[ClusterInfo] Built {len(clusters)} clusters, "
          f"total docs={sum(c.num_docs for c in clusters)}, "
          f"total tokens={sum(c.num_tokens for c in clusters)}")

    return clusters


def preprocess_pipeline(
    texts: Optional[List[str]] = None,
    quality_scores: Optional[npt.NDArray[np.float64]] = None,
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    embedding_model: str = "NovaSearch/stella_en_400M_v5",
    K_init: int = 1000,
    K_enhanced: int = 3,
    K_max: Optional[int] = None,
    prune_threshold: float = 3.0,
    merge_distance: float = 0.9,
    embedding_cache: Optional[str] = None,
    kmeans_cache: Optional[str] = None,
    profile_path: Optional[str] = None,
    device: str = "cpu",
    metadata_manager: Optional[object] = None,
    embedding_truncate_len: int = 512,
) -> Tuple[List[ClusterInfo], npt.NDArray[np.int64]]:
    """
    Full CLIMB preprocessing pipeline: embed → cluster → prune → merge → build info.

    Args:
        texts: Raw document texts (for subsampled mode).
        metadata_manager: If provided (and texts is None), stream-embeds
            all texts shard-by-shard without loading everything into memory.
        quality_scores: Per-document quality scores for pruning.
        token_counts: Per-document token counts.
        embedding_model: Sentence-transformer model name.
        K_init: Initial number of clusters.
        K_enhanced: Floor on the final cluster count (paper's K_enhanced).
        K_max: Cap on the final cluster count (search-budget bound).
        prune_threshold: Quality threshold for cluster pruning.
        merge_distance: Merge legality threshold (tau) on centroid distance.
        embedding_cache: Cache path for embeddings (stable, pool-keyed).
        kmeans_cache: Cache path for K-means labels+centroids (stable,
            pool-keyed; survives K_enhanced/merge_distance changes).
        profile_path: Where to write merge_profile.json (run-level audit).
        device: Device for embedding.

    Returns:
        Tuple of (cluster_info_list, final_cluster_labels).
    """
    from climbmix.core.embedding_cluster import embed_documents, embed_texts_streaming, cluster_embeddings

    print("\n" + "=" * 70)
    print("  CLIMB Preprocessing Pipeline")
    print("=" * 70)

    t0 = time.time()

    if texts is not None:
        embeddings = embed_documents(
            texts, model_name=embedding_model,
            cache_path=embedding_cache, device=device,
        )
    elif metadata_manager is not None:
        print("[Preprocess] Streaming mode: embedding shard-by-shard (no full text load)")
        embeddings = embed_texts_streaming(
            metadata_manager, model_name=embedding_model,
            cache_path=embedding_cache, device=device,
            truncate_len=embedding_truncate_len,
        )
    else:
        raise ValueError("Either texts or metadata_manager must be provided")

    cluster_labels, centroids = cluster_embeddings(
        embeddings, K_init=K_init, cache_path=kmeans_cache,
    )

    cluster_quality = compute_cluster_quality(cluster_labels, quality_scores, prune_threshold=prune_threshold)

    pruned_labels, pruned_centroids, _ = prune_clusters(
        cluster_labels, centroids, cluster_quality, threshold=prune_threshold,
    )

    valid_mask = pruned_labels >= 0
    if valid_mask.sum() == 0:
        print("[Preprocess] WARNING: All clusters pruned, using original labels")
        merged_labels = cluster_labels
        merged_centroids = centroids
    else:
        merged_labels, merged_centroids, _ = merge_clusters_by_distance(
            pruned_labels, pruned_centroids,
            merge_distance=merge_distance, target_K=K_enhanced,
            K_max=K_max, profile_path=profile_path,
        )

    valid_mask = merged_labels >= 0
    if token_counts is None:
        if metadata_manager is not None:
            token_counts_full = metadata_manager.estimate_token_counts()
        elif texts is not None:
            token_counts_full = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int64)
        else:
            token_counts_full = np.ones(len(merged_labels), dtype=np.int64)
    else:
        token_counts_full = token_counts

    cluster_info = build_cluster_info(merged_labels, merged_centroids, token_counts_full)

    elapsed = time.time() - t0
    print(f"\n[Preprocess] Complete in {elapsed:.1f}s")
    print(f"  K_init={K_init} → K_final={len(cluster_info)} (floor={K_enhanced}, cap={K_max})")
    print(f"  Total docs: {sum(c.num_docs for c in cluster_info)}")
    print(f"  Total tokens: {sum(c.num_tokens for c in cluster_info)}")

    return cluster_info, merged_labels
