"""
CLIMB Step 1: Text embedding + clustering.

Given raw documents, embed them using a sentence-transformer model,
then cluster the embeddings using FAISS K-means.

Paper details (Section 2.1):
  - Embedding model: stella_en_400M_v5
  - Clustering: FAISS K-means, K_init = 1000
  - Output: cluster labels for each document
"""

import os
import time
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple


def embed_documents(
    texts: List[str],
    model_name: str = "NovaSearch/stella_en_400M_v5",
    batch_size: int = 256,
    cache_path: Optional[str] = None,
    device: str = "cpu",
) -> npt.NDArray[np.float32]:
    """
    Embed documents using sentence-transformers.

    Args:
        texts: List of document texts.
        model_name: Sentence-transformer model name.
        batch_size: Batch size for encoding.
        cache_path: Path to cache embeddings (npz file). If exists, loads from cache.
        device: Device for encoding ('cpu', 'cuda', 'npu').
            If 'npu', attempts to use Ascend NPU via torch_npu.
            Falls back to CPU if sentence-transformers doesn't support NPU.

    Returns:
        Embeddings array of shape (num_docs, embedding_dim).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Embed] Loading cached embeddings from: {cache_path}")
        data = np.load(cache_path)
        embeddings = data["embeddings"]
        print(f"[Embed] Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        return embeddings

    actual_device = device

    if device == "npu":
        try:
            import torch
            import torch_npu
            if torch.npu.is_available():
                print("[Embed] NPU available, attempting to use Ascend NPU for embedding")
            else:
                print("[Embed] torch_npu imported but NPU not available, falling back to CPU")
                actual_device = "cpu"
        except ImportError:
            print("[Embed] torch_npu not available, falling back to CPU")
            actual_device = "cpu"

        if actual_device == "npu":
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Embed] Loading model: {model_name} (device=npu)")
                t0 = time.time()
                model = SentenceTransformer(model_name, device="npu")
                print(f"[Embed] Model loaded in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[Embed] NPU embedding failed ({e}), falling back to CPU")
                actual_device = "cpu"

    if actual_device != "npu":
        if actual_device == "cpu":
            import os
            num_threads = os.environ.get("OMP_NUM_THREADS", "")
            if not num_threads:
                import multiprocessing
                num_cpus = multiprocessing.cpu_count()
                print(f"[Embed] Using CPU with {num_cpus} threads")
                try:
                    import torch
                    torch.set_num_threads(num_cpus)
                except ImportError:
                    pass

        from sentence_transformers import SentenceTransformer
        print(f"[Embed] Loading model: {model_name} (device={actual_device})")
        t0 = time.time()
        model = SentenceTransformer(model_name, device=actual_device, trust_remote_code=True)
        print(f"[Embed] Model loaded in {time.time() - t0:.1f}s")

    print(f"[Embed] Encoding {len(texts)} documents (batch_size={batch_size})...")
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"[Embed] Encoded {len(texts)} docs in {time.time() - t1:.1f}s, dim={embeddings.shape[1]}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings)
        print(f"[Embed] Cached embeddings to: {cache_path}")

    return embeddings


def cluster_embeddings_faiss(
    embeddings: npt.NDArray[np.float32],
    K_init: int = 1000,
    n_iter: int = 20,
    n_redo: int = 5,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """
    Cluster embeddings using FAISS K-means.

    Args:
        embeddings: Document embeddings of shape (num_docs, dim).
        K_init: Initial number of clusters (paper: 1000).
        n_iter: Number of K-means iterations.
        n_redo: Number of restarts.
        seed: Random seed.
        cache_path: Path to cache cluster labels + centroids.

    Returns:
        Tuple of (cluster_labels, centroids).
        cluster_labels: shape (num_docs,)
        centroids: shape (K_init, dim)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Cluster] Loading cached clustering from: {cache_path}")
        data = np.load(cache_path)
        labels = data["labels"]
        centroids = data["centroids"]
        print(f"[Cluster] Loaded {len(labels)} labels, K={len(centroids)}")
        return labels, centroids

    import faiss

    dim = embeddings.shape[1]
    n_docs = embeddings.shape[0]

    print(f"[Cluster] FAISS K-means: K={K_init}, dim={dim}, n_docs={n_docs}")
    t0 = time.time()

    kmeans = faiss.Kmeans(
        dim,
        K_init,
        niter=n_iter,
        nredo=n_redo,
        verbose=True,
        seed=seed,
        spherical=True,
    )
    kmeans.train(embeddings)

    centroids = kmeans.centroids
    index = faiss.IndexFlatIP(dim)
    index.add(centroids)

    _, labels = index.search(embeddings, 1)
    labels = labels.reshape(-1).astype(np.int64)

    elapsed = time.time() - t0
    unique_labels = np.unique(labels)
    print(f"[Cluster] Done in {elapsed:.1f}s, {len(unique_labels)} unique clusters")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, labels=labels, centroids=centroids)
        print(f"[Cluster] Cached to: {cache_path}")

    return labels, centroids


def cluster_embeddings_sklearn(
    embeddings: npt.NDArray[np.float32],
    K_init: int = 1000,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """
    Cluster embeddings using sklearn K-means (fallback when FAISS unavailable).

    Args:
        embeddings: Document embeddings of shape (num_docs, dim).
        K_init: Initial number of clusters.
        seed: Random seed.
        cache_path: Path to cache results.

    Returns:
        Tuple of (cluster_labels, centroids).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Cluster] Loading cached clustering from: {cache_path}")
        data = np.load(cache_path)
        labels = data["labels"]
        centroids = data["centroids"]
        return labels, centroids

    from sklearn.cluster import MiniBatchKMeans

    print(f"[Cluster] sklearn MiniBatchKMeans: K={K_init}")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=K_init,
        random_state=seed,
        batch_size=1024,
        max_iter=100,
    )
    labels = kmeans.fit_predict(embeddings).astype(np.int64)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    elapsed = time.time() - t0
    print(f"[Cluster] Done in {elapsed:.1f}s, {len(np.unique(labels))} clusters")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, labels=labels, centroids=centroids)

    return labels, centroids


def cluster_embeddings(
    embeddings: npt.NDArray[np.float32],
    K_init: int = 1000,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """
    Cluster embeddings, using FAISS if available, sklearn as fallback.
    """
    try:
        return cluster_embeddings_faiss(embeddings, K_init, seed=seed, cache_path=cache_path)
    except ImportError:
        print("[Cluster] FAISS not available, falling back to sklearn")
        return cluster_embeddings_sklearn(embeddings, K_init, seed=seed, cache_path=cache_path)
