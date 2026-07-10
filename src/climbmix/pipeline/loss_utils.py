"""
Chunked loss utilities (adapted from quadmix).
"""

import torch
import torch.nn.functional as F


def chunked_loss_from_hidden(model, hidden, targets, chunk_size=2048):
    B, T, _ = hidden.shape
    V = model.config.vocab_size
    total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
    n_tokens = 0
    for i in range(0, T, chunk_size):
        chunk_logits = model.lm_head(hidden[:, i:i + chunk_size])
        chunk_tgt = targets[:, i:i + chunk_size]
        n = chunk_tgt.numel()
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.reshape(-1, V), chunk_tgt.reshape(-1)
        ) * n
        n_tokens += n
        del chunk_logits
    return total_loss / n_tokens


def chunked_loss_per_token_from_hidden(model, hidden, targets, chunk_size=2048):
    B, T, _ = hidden.shape
    V = model.config.vocab_size
    per_token = torch.empty(B, T, device=hidden.device, dtype=torch.float32)
    for i in range(0, T, chunk_size):
        chunk_logits = model.lm_head(hidden[:, i:i + chunk_size])
        chunk_tgt = targets[:, i:i + chunk_size]
        per_token[:, i:i + chunk_size] = F.cross_entropy(
            chunk_logits.reshape(-1, V), chunk_tgt.reshape(-1), reduction="none"
        ).view(chunk_tgt.shape)
        del chunk_logits
    return per_token
