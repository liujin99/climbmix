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
import sys
import types
import time
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple

try:
    import xformers  # noqa: F401
except ImportError:
    import torch
    import torch.nn.functional as _F

    class _BlockDiagonalMask:
        """Fake BlockDiagonalMask — stores seq lens, materializes to bool tensor."""
        def __init__(self, q_seqlen, kv_seqlen=None, device=None):
            if isinstance(q_seqlen, int):
                q_seqlen = [q_seqlen]
            self.q_seqlen = list(q_seqlen)
            self.kv_seqlen = list(kv_seqlen) if kv_seqlen is not None else self.q_seqlen
            self.device = device

        @classmethod
        def from_seqlens(cls, q_seqlen, kv_seqlen=None, device=None, **kw):
            if (isinstance(q_seqlen, tuple) and len(q_seqlen) == 2
                    and isinstance(q_seqlen[0], (list, tuple))):
                q_seqlen, kv_seqlen = q_seqlen
            return cls(q_seqlen, kv_seqlen, device)

    class _LowerTriangularMask:
        def __init__(self, *a, **kw):
            pass

    def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
        # xformers layout: (B, S, H, D) — SDPA layout: (B, H, S, D)
        need_transpose = q.dim() == 4 and q.shape[1] > q.shape[2]

        def _to_sdpa(t):
            return t.transpose(1, 2) if need_transpose else t

        def _from_sdpa(t):
            return t.transpose(1, 2).contiguous() if need_transpose else t

        if isinstance(attn_bias, _BlockDiagonalMask):
            # Always use fallback path (pad + bool mask SDPA) for all docs.
            # This ensures identical computation path for batch=512 and batch=1,
            # eliminating any bias from different attention kernels.
            qs_list = attn_bias.q_seqlen
            ks_list = attn_bias.kv_seqlen
            n = len(qs_list)
            _, _, H, D = q.shape

            max_s = max(qs_list)
            max_ks = max(ks_list)
            q_pad = torch.zeros(n, max_s, H, D, dtype=q.dtype, device=q.device)
            k_pad = torch.zeros(n, max_ks, H, D, dtype=k.dtype, device=k.device)
            v_pad = torch.zeros(n, max_ks, H, D, dtype=v.dtype, device=v.device)
            kmask = torch.zeros(n, 1, max_s, max_ks, dtype=torch.bool, device=q.device)
            q_off = k_off = 0
            for i, (qs_i, ks_i) in enumerate(zip(qs_list, ks_list)):
                q_pad[i, :qs_i] = q[0, q_off:q_off + qs_i]
                k_pad[i, :ks_i] = k[0, k_off:k_off + ks_i]
                v_pad[i, :ks_i] = v[0, k_off:k_off + ks_i]
                kmask[i, 0, :, :ks_i] = True
                q_off += qs_i
                k_off += ks_i
            out = _F.scaled_dot_product_attention(
                q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
                attn_mask=kmask, dropout_p=p
            )
            out = out.transpose(1, 2)  # (n, max_s, H, D)
            chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
            return torch.cat(chunks, dim=1)  # (1, total_S, H, D)

        q_s, k_s, v_s = _to_sdpa(q), _to_sdpa(k), _to_sdpa(v)
        if isinstance(attn_bias, _LowerTriangularMask):
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, dropout_p=p)
        elif attn_bias is not None:
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_bias, dropout_p=p)
        else:
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=p)
        return _from_sdpa(out)

    # Build fake module hierarchy: xformers.ops.fmha.attn_bias
    _attn_bias_mod = types.ModuleType("xformers.ops.fmha.attn_bias")
    _attn_bias_mod.BlockDiagonalMask = _BlockDiagonalMask
    _attn_bias_mod.LowerTriangularMask = _LowerTriangularMask

    _fmha_mod = types.ModuleType("xformers.ops.fmha")
    _fmha_mod.attn_bias = _attn_bias_mod
    _fmha_mod.memory_efficient_attention = _memory_efficient_attention

    _ops = types.ModuleType("xformers.ops")
    _ops.memory_efficient_attention = _memory_efficient_attention
    _ops.fmha = _fmha_mod

    _xfm = types.ModuleType("xformers")
    _xfm.ops = _ops
    _xfm.__version__ = "0.0.0"

    sys.modules["xformers"] = _xfm
    sys.modules["xformers.ops"] = _ops
    sys.modules["xformers.ops.fmha"] = _fmha_mod
    sys.modules["xformers.ops.fmha.attn_bias"] = _attn_bias_mod


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

    def _load_model(model_name, dev):
        """Load SentenceTransformer with attention implementation fallback."""
        from sentence_transformers import SentenceTransformer
        try:
            import xformers  # noqa: F401
            has_xformers = True
        except ImportError:
            has_xformers = False

        if has_xformers:
            return SentenceTransformer(model_name, device=dev, trust_remote_code=True)

        for impl in ["sdpa", "eager"]:
            try:
                print(f"[Embed] xformers not available, trying attn_implementation={impl}")
                return SentenceTransformer(
                    model_name, device=dev, trust_remote_code=True,
                    model_kwargs={"attn_implementation": impl},
                )
            except (KeyError, AssertionError, ValueError, TypeError) as e:
                print(f"[Embed] attn_implementation={impl} failed: {e}")
        raise RuntimeError("Failed to load model: no compatible attention implementation")

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
                print(f"[Embed] Loading model: {model_name} (device=npu)")
                t0 = time.time()
                model = _load_model(model_name, "npu")
                model.eval()
                model.half()
                model.max_seq_length = 512
                batch_size = max(batch_size, 512)
                print(f"[Embed] Model loaded in {time.time() - t0:.1f}s (fp16, msl=512, bs={batch_size})")
            except Exception as e:
                print(f"[Embed] NPU embedding failed ({e}), falling back to CPU")
                actual_device = "cpu"

    if actual_device != "npu":
        if actual_device == "cpu":
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

        print(f"[Embed] Loading model: {model_name} (device={actual_device})")
        t0 = time.time()
        model = _load_model(model_name, actual_device)
        print(f"[Embed] Model loaded in {time.time() - t0:.1f}s")

    safe_texts = [t if t and len(t.strip()) > 0 else "empty" for t in texts]

    sort_order = sorted(range(len(safe_texts)), key=lambda i: len(safe_texts[i]))
    sorted_texts = [safe_texts[i] for i in sort_order]
    inv_order = [0] * len(sort_order)
    for new_pos, orig_pos in enumerate(sort_order):
        inv_order[orig_pos] = new_pos

    print(f"[Embed] Encoding {len(sorted_texts)} documents (batch_size={batch_size}, length-sorted)...")
    t1 = time.time()
    sorted_embeddings = model.encode(
        sorted_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    sorted_embeddings = np.array(sorted_embeddings, dtype=np.float32)
    embeddings = np.empty_like(sorted_embeddings)
    for orig_pos, new_pos in enumerate(inv_order):
        embeddings[orig_pos] = sorted_embeddings[new_pos]
    print(f"[Embed] Encoded {len(sorted_texts)} docs in {time.time() - t1:.1f}s, dim={embeddings.shape[1]}")

    n_nan = np.isnan(embeddings).any(axis=1).sum()
    if n_nan > 0:
        nan_indices = np.where(np.isnan(embeddings).any(axis=1))[0]
        print(f"[Embed] {n_nan}/{len(safe_texts)} docs produced NaN, retrying with bs=1...")
        for idx in nan_indices:
            single = model.encode(
                [safe_texts[idx]], batch_size=1, show_progress_bar=False,
                normalize_embeddings=True)
            embeddings[idx] = np.array(single[0], dtype=np.float32)
        still_nan = np.isnan(embeddings).any(axis=1).sum()
        if still_nan > 0:
            raise RuntimeError(
                f"{still_nan} docs still produce NaN after bs=1 retry — "
                f"cannot recover, aborting to prevent data loss")
        print(f"[Embed] All {n_nan} NaN docs recovered via bs=1 retry")
    else:
        print(f"[Embed] No NaN detected ({len(safe_texts)} docs)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings)
        print(f"[Embed] Cached embeddings to: {cache_path}")

    return embeddings


def _load_model_stream(model_name, dev):
    """Load sentence transformer model with xformers/SDPA fallback."""
    from sentence_transformers import SentenceTransformer
    try:
        import xformers  # noqa: F401
        has_xformers = True
    except ImportError:
        has_xformers = False

    if has_xformers:
        return SentenceTransformer(model_name, device=dev, trust_remote_code=True)

    for impl in ["sdpa", "eager"]:
        try:
            print(f"[Embed-Stream] xformers not available, trying attn_implementation={impl}")
            return SentenceTransformer(
                model_name, device=dev, trust_remote_code=True,
                model_kwargs={"attn_implementation": impl},
            )
        except (KeyError, AssertionError, ValueError, TypeError) as e:
            print(f"[Embed-Stream] attn_implementation={impl} failed: {e}")
    raise RuntimeError("Failed to load model: no compatible attention implementation")


def _embed_streaming_worker(worker_id, shard_indices, shard_infos, text_col,
                            model_name, batch_size, emb_dim,
                            memmap_path, total_docs, shared_done, truncate_len):
    """Worker process: embed assigned shards on NPU worker_id, write to shared memmap."""
    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    import torch
    import torch_npu
    import numpy as np
    import pyarrow.parquet as pq
    import time
    from concurrent.futures import ThreadPoolExecutor

    print(f"[NPU {worker_id}] Loading model...", flush=True)
    model = _load_model_stream(model_name, "npu")
    model.eval()
    model.max_seq_length = truncate_len
    model.half()
    print(f"[NPU {worker_id}] Model loaded (fp16, max_seq_len={truncate_len})", flush=True)

    all_emb = np.memmap(memmap_path, dtype=np.float32, mode='r+',
                        shape=(total_docs, emb_dim))

    device = "npu"

    def _to_device(features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(device)
            elif isinstance(features[key], dict):
                _to_device(features[key])
        return features

    def _read_shard_texts(sinfo):
        table = pq.read_table(sinfo["path"], columns=[text_col], use_threads=True)
        col = table.column(text_col)
        texts = [str(v) if v is not None else "" for v in col.to_pylist()]
        del table, col
        return texts

    io_pool = ThreadPoolExecutor(max_workers=1)
    tok_pool = ThreadPoolExecutor(max_workers=1)

    shard_future = io_pool.submit(_read_shard_texts, shard_infos[shard_indices[0]])

    n_ws = len(shard_indices)
    print(f"[NPU {worker_id}] Starting encoding ({n_ws} shards, batch_size={batch_size})", flush=True)

    docs_done = 0
    nan_count = 0
    t0 = time.time()

    for si_idx, si in enumerate(shard_indices):
        texts = shard_future.result()

        if si_idx + 1 < n_ws:
            shard_future = io_pool.submit(
                _read_shard_texts, shard_infos[shard_indices[si_idx + 1]])

        sinfo = shard_infos[si]
        start_idx = sinfo["start_idx"]
        num_docs = sinfo["num_docs"]

        sort_order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in sort_order]
        del texts

        next_tok_future = tok_pool.submit(model.tokenize, sorted_texts[:batch_size])

        for j in range(0, len(sorted_texts), batch_size):
            batch_len = min(batch_size, len(sorted_texts) - j)

            features = next_tok_future.result()

            next_j = j + batch_size
            if next_j < len(sorted_texts):
                next_tok_future = tok_pool.submit(
                    model.tokenize, sorted_texts[next_j:next_j + batch_size])

            features = _to_device(features)
            with torch.no_grad():
                output = model(features)
            emb = output["sentence_embedding"].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.cpu().numpy()

            nan_mask = np.isnan(emb).any(axis=1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                nan_count += n_nan
                nan_locs = np.where(nan_mask)[0]
                for k in nan_locs:
                    retry_features = model.tokenize([sorted_texts[j + k]])
                    retry_features = _to_device(retry_features)
                    with torch.no_grad():
                        retry_out = model(retry_features)
                    retry_emb = retry_out["sentence_embedding"].float()
                    retry_emb = torch.nn.functional.normalize(retry_emb, p=2, dim=1)
                    emb[k] = retry_emb.cpu().numpy()[0]
                    del retry_features, retry_out
                still_nan = np.isnan(emb).any(axis=1).sum()
                if still_nan > 0:
                    raise RuntimeError(
                        f"[NPU {worker_id}] {still_nan} docs still NaN after bs=1 retry — "
                        f"cannot recover, aborting to prevent data loss")

            del features, output

            batch_orig = np.array(sort_order[j:j + batch_len], dtype=np.int64) + start_idx
            all_emb[batch_orig] = emb
            shared_done[worker_id] = docs_done + j + batch_len

        docs_done += num_docs
        del sorted_texts

    io_pool.shutdown()
    tok_pool.shutdown()
    all_emb.flush()
    elapsed = time.time() - t0
    nan_msg = f", {nan_count} NaN recovered" if nan_count > 0 else ""
    print(f"[NPU {worker_id}] Done: {docs_done:,} docs in {elapsed:.0f}s "
          f"({docs_done/elapsed:.0f} docs/s{nan_msg})", flush=True)


def embed_texts_streaming(
    metadata_manager,
    model_name: str = "NovaSearch/stella_en_400M_v5",
    batch_size: int = 512,
    cache_path: Optional[str] = None,
    device: str = "cpu",
    truncate_len: int = 512,
) -> npt.NDArray[np.float32]:
    """
    Stream-embed texts shard by shard, avoiding loading all texts into memory.

    Reads one shard's text column at a time, embeds it, stores embeddings
    in a preallocated array, then frees the text memory.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Embed-Stream] Loading cached embeddings from: {cache_path}")
        data = np.load(cache_path)
        embeddings = data["embeddings"]
        print(f"[Embed-Stream] Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        return embeddings

    actual_device = device

    if device == "npu":
        try:
            import torch
            import torch_npu
            if torch.npu.is_available():
                print("[Embed-Stream] NPU available, using Ascend NPU")
            else:
                print("[Embed-Stream] NPU not available, falling back to CPU")
                actual_device = "cpu"
        except ImportError:
            print("[Embed-Stream] torch_npu not available, falling back to CPU")
            actual_device = "cpu"

    if actual_device == "cpu":
        num_threads = os.environ.get("OMP_NUM_THREADS", "")
        if not num_threads:
            import multiprocessing
            num_cpus = multiprocessing.cpu_count()
            print(f"[Embed-Stream] Using CPU with {num_cpus} threads")
            try:
                import torch
                torch.set_num_threads(num_cpus)
            except ImportError:
                pass

    print(f"[Embed-Stream] Loading model: {model_name} (device={actual_device})")
    t0 = time.time()
    model = _load_model_stream(model_name, actual_device)
    print(f"[Embed-Stream] Model loaded in {time.time() - t0:.1f}s")

    import pyarrow.parquet as pq

    text_col = metadata_manager.schema.text_col
    shard_infos = metadata_manager.shard_info
    total_docs = metadata_manager.num_docs
    n_shards = len(shard_infos)

    dummy_emb = model.encode(["test"], show_progress_bar=False, normalize_embeddings=True)
    emb_dim = dummy_emb.shape[1]

    # Detect NPUs for process-based parallelism
    n_npus = 0
    if actual_device == "npu":
        try:
            n_npus = torch.npu.device_count()
        except Exception:
            pass

    t1 = time.time()

    if n_npus > 1:
        print(f"[Embed-Stream] {n_npus} NPUs detected, using process-based parallelism")
        del model, dummy_emb
        import gc
        gc.collect()
        import torch_npu
        torch.npu.empty_cache()

        import multiprocessing as mp
        import threading

        ctx = mp.get_context("spawn")
        shared_done = ctx.Array('q', n_npus)

        def _monitor(shared_done, total_docs, n_npus, procs):
            t0 = time.time()
            while True:
                time.sleep(15)
                done = sum(shared_done[i] for i in range(n_npus))
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_docs - done) / speed if speed > 0 else 0
                alive = sum(1 for p in procs if p.is_alive())
                print(f"[Embed-Stream] {done:,}/{total_docs:,} docs ({done/total_docs*100:.1f}%), "
                      f"{speed:.0f} docs/s, elapsed {elapsed:.0f}s, ETA {eta:.0f}s, "
                      f"{alive}/{n_npus} workers", flush=True)
                if alive == 0:
                    break

        cache_dir = os.path.dirname(cache_path) if cache_path else "/tmp"
        os.makedirs(cache_dir, exist_ok=True)
        memmap_path = os.path.join(cache_dir, "embedding_memmap.tmp")
        if os.path.exists(memmap_path):
            os.remove(memmap_path)
        memmap_init = np.memmap(memmap_path, dtype=np.float32, mode='w+',
                                shape=(total_docs, emb_dim))
        del memmap_init

        print(f"[Embed-Stream] Preallocated ({total_docs:,}, {emb_dim}) memmap "
              f"({total_docs * emb_dim * 4 / (1024**3):.1f} GB)")

        indices = list(range(n_shards))
        chunks = np.array_split(indices, n_npus)

        procs = []
        for wid, chunk in enumerate(chunks):
            p = ctx.Process(
                target=_embed_streaming_worker,
                args=(wid, list(chunk), shard_infos, text_col,
                      model_name, batch_size, emb_dim,
                      memmap_path, total_docs, shared_done, truncate_len),
            )
            p.start()
            procs.append(p)

        monitor = threading.Thread(target=_monitor,
                                   args=(shared_done, total_docs, n_npus, procs),
                                   daemon=True)
        monitor.start()

        for p in procs:
            p.join()
        monitor.join()

        failed = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} workers failed: {failed}")

        all_embeddings = np.memmap(memmap_path, dtype=np.float32, mode='r',
                                   shape=(total_docs, emb_dim))

        elapsed = time.time() - t1
        print(f"[Embed-Stream] Encoded {total_docs:,} docs in {elapsed:.1f}s "
              f"({total_docs / elapsed:.0f} docs/s), dim={emb_dim}")

        if cache_path:
            cache_dir = os.path.dirname(cache_path) or "."
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, embeddings=np.array(all_embeddings))
            print(f"[Embed-Stream] Cached embeddings to: {cache_path}")
            os.remove(memmap_path)

        return np.array(all_embeddings)

    # Single NPU / CPU path
    all_embeddings = np.empty((total_docs, emb_dim), dtype=np.float32)
    print(f"[Embed-Stream] Preallocated ({total_docs:,}, {emb_dim}) embeddings array "
          f"({total_docs * emb_dim * 4 / (1024**3):.1f} GB)")
    print(f"[Embed-Stream] Streaming {n_shards} shards, batch_size={batch_size}")
    docs_done = 0
    batch_num = 0
    for si, sinfo in enumerate(shard_infos):
        shard_path = sinfo["path"]
        start_idx = sinfo["start_idx"]
        num_docs = sinfo["num_docs"]

        table = pq.read_table(shard_path, columns=[text_col], use_threads=True)
        col = table.column(text_col)
        shard_texts = [str(v) if v is not None else "" for v in col.to_pylist()]
        del table, col

        for j in range(0, len(shard_texts), batch_size):
            batch = shard_texts[j:j + batch_size]
            emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings[start_idx + j:start_idx + j + len(batch)] = emb

            batch_num += 1
            if batch_num <= 5 or batch_num % 50 == 0:
                elapsed = time.time() - t1
                done = docs_done + j + len(batch)
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total_docs - done) / speed if speed > 0 else 0
                print(f"  [Embed-Stream] batch {batch_num}: {done:,}/{total_docs:,} docs "
                      f"({done / total_docs * 100:.1f}%), {speed:.0f} docs/s, "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        docs_done += num_docs
        del shard_texts
        elapsed = time.time() - t1
        speed = docs_done / elapsed if elapsed > 0 else 0
        eta = (total_docs - docs_done) / speed if speed > 0 else 0
        if si % 10 == 0 or si == n_shards - 1:
            print(f"[Embed-Stream] Shard {si + 1}/{n_shards}: {docs_done:,}/{total_docs:,} docs "
                  f"({docs_done / total_docs * 100:.1f}%), {speed:.0f} docs/s, "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - t1
    print(f"[Embed-Stream] Encoded {total_docs:,} docs in {elapsed:.1f}s "
          f"({total_docs / elapsed:.0f} docs/s), dim={emb_dim}")

    if cache_path:
        cache_dir = os.path.dirname(cache_path) or "."
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, embeddings=all_embeddings)
        print(f"[Embed-Stream] Cached embeddings to: {cache_path}")

    return all_embeddings


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

    n_nan = np.isnan(embeddings).any(axis=1).sum()
    n_inf = np.isinf(embeddings).any(axis=1).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"[Cluster] WARNING: {n_nan} NaN + {n_inf} Inf rows found, replacing with zeros")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

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

    zero_mask = np.all(embeddings == 0, axis=1)
    n_zero = zero_mask.sum()
    if n_zero > 0:
        print(f"[Cluster] {n_zero} docs have zero/NaN embeddings — excluding from clusters")
        labels[zero_mask] = -1

    elapsed = time.time() - t0
    valid_labels = labels[labels >= 0]
    n_unique = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0
    print(f"[Cluster] Done in {elapsed:.1f}s, {n_unique} unique clusters, {n_zero} excluded")

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
