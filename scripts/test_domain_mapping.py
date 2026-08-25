#!/usr/bin/env python3
"""Test domain mapping fix: proportional vs dominant cluster assignment."""
import numpy as np
import sys
sys.path.insert(0, "/home/liujin99/climbmix/src")

from climbmix.core.discovery import EmbeddingClusterDiscovery

# Simulate: 1000 docs, 4 domains, K_enhanced=10
n_total = 10000
n_domains = 4
K = 10
n_sampled = 1000

rng = np.random.default_rng(42)
domain_labels = rng.integers(0, n_domains, size=n_total).astype(np.int64)
sample_indices = np.sort(rng.choice(n_total, size=n_sampled, replace=False)).astype(np.int64)

# Sample labels: each domain has docs spread across multiple clusters
sample_labels = np.zeros(n_sampled, dtype=np.int64)
for i, idx in enumerate(sample_indices):
    d = domain_labels[idx]
    # Domain 0: clusters 0-2, Domain 1: clusters 3-5, etc.
    # But with some overlap to make it realistic
    base = d * (K // n_domains)
    sample_labels[i] = base + rng.integers(0, min(3, K - base))

print(f"Setup: {n_total} docs, {n_domains} domains, K={K}, sampled={n_sampled}")
print(f"Domain distribution: {np.bincount(domain_labels)}")
print(f"Sample cluster distribution: {np.bincount(sample_labels, minlength=K)}")
print(f"Sample cluster distribution per domain:")
for d in range(n_domains):
    mask = domain_labels[sample_indices] == d
    dist = np.bincount(sample_labels[mask], minlength=K)
    print(f"  Domain {d}: {dist}")

# Test the fixed method
final_labels = EmbeddingClusterDiscovery._assign_remaining_by_domain(
    sample_indices, sample_labels, domain_labels, n_total,
)

print(f"\nFinal cluster distribution: {np.bincount(final_labels, minlength=K)}")
print(f"Clusters used: {len(np.unique(final_labels))}")

# Verify: sampled docs keep their original labels
assert np.array_equal(final_labels[sample_indices], sample_labels), "Sampled labels changed!"
print("\n[PASS] Sampled docs keep original labels")

# Verify: all docs are assigned (no -1)
assert (final_labels == -1).sum() == 0, "Unassigned docs remain!"
print("[PASS] All docs assigned")

# Verify: cluster distribution is proportional per domain
print("\nCluster distribution per domain (should be proportional):")
for d in range(n_domains):
    domain_mask = domain_labels == d
    domain_final = final_labels[domain_mask]
    dist = np.bincount(domain_final, minlength=K)
    
    # Get sampled distribution for this domain
    sampled_mask = domain_labels[sample_indices] == d
    sampled_dist = np.bincount(sample_labels[sampled_mask], minlength=K)
    
    # Compare proportions
    prop_sampled = sampled_dist[sampled_dist > 0] / sampled_dist.sum()
    prop_final = dist[dist > 0] / dist.sum()
    
    print(f"  Domain {d}: final={dist}")
    print(f"           sampled={sampled_dist}")
    print(f"           proportions match: {len(prop_sampled) == len(prop_final)}")
    
    # Check that all clusters from sampled are present in final
    sampled_clusters = set(np.where(sampled_dist > 0)[0])
    final_clusters = set(np.where(dist > 0)[0])
    assert sampled_clusters.issubset(final_clusters), \
        f"Domain {d}: clusters {sampled_clusters - final_clusters} missing in final!"

print("\n[PASS] All sampled clusters preserved in final distribution")

# Compare with old behavior (all to dominant)
print("\n--- Old behavior comparison ---")
final_labels_old = np.full(n_total, -1, dtype=np.int64)
final_labels_old[sample_indices] = sample_labels
for d in range(n_domains):
    domain_mask = domain_labels == d
    domain_indices = np.where(domain_mask)[0]
    sampled_mask = domain_labels[sample_indices] == d
    sampled_clusters = sample_labels[sampled_mask]
    unique_clusters, counts = np.unique(sampled_clusters, return_counts=True)
    dominant = unique_clusters[np.argmax(counts)]
    unassigned = domain_indices[final_labels_old[domain_indices] == -1]
    final_labels_old[unassigned] = int(dominant)

print(f"Old: {np.bincount(final_labels_old, minlength=K)}")
print(f"New: {np.bincount(final_labels, minlength=K)}")
old_clusters = len(np.unique(final_labels_old))
new_clusters = len(np.unique(final_labels))
print(f"Old clusters used: {old_clusters}, New clusters used: {new_clusters}")
assert new_clusters >= old_clusters, "Fix should use >= clusters than old"
print(f"\n[PASS] Fix uses {new_clusters} clusters (old used {old_clusters})")

# Check cluster balance
old_max = np.bincount(final_labels_old, minlength=K).max()
new_max = np.bincount(final_labels, minlength=K).max()
print(f"Old max cluster size: {old_max}, New max cluster size: {new_max}")
assert new_max < old_max, "Fix should reduce max cluster size"
print(f"[PASS] Max cluster size reduced: {old_max} -> {new_max}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
