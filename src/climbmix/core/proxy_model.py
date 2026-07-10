"""
Proxy model architecture for CLIMB.

Reuses the RegMix-style tinyllama architecture from quadmix,
with additional size variants for CLIMB (62M, 350M).

Paper: uses 350M proxy model for main experiments,
       62M for ablation studies.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:q.shape[-2], :]
    sin = sin[:q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "ModelArchitectureConfig"):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, max_seq_len=config.block_size, base=config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class LLaMAMLP(nn.Module):
    def __init__(self, config: "ModelArchitectureConfig"):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: "ModelArchitectureConfig"):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class ModelArchitectureConfig:
    """
    Proxy model architecture configurations.
    
    CLIMB paper sizes:
      - 62M: 12 layers, 12 heads, 768 hidden (GPT-2-medium-ish)
      - 350M: 24 layers, 16 heads, 1024 hidden
      - 1M: tiny proxy for quick tests (same as quadmix)
    """

    def __init__(
        self,
        n_layer: int = 24,
        n_head: int = 16,
        n_embd: int = 1024,
        vocab_size: int = 50432,
        block_size: int = 2048,
        bias: bool = False,
        norm_eps: float = 1e-5,
        rope_base: int = 10000,
        intermediate_size: int = 3072,
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.bias = bias
        self.norm_eps = norm_eps
        self.rope_base = rope_base
        self.intermediate_size = intermediate_size

    @classmethod
    def from_name(cls, name: str, block_size: Optional[int] = None) -> "ModelArchitectureConfig":
        variants = {
            "1M": cls(n_layer=2, n_head=8, n_embd=256, intermediate_size=512),
            "62M": cls(n_layer=12, n_head=12, n_embd=768, intermediate_size=2304),
            "350M": cls(n_layer=24, n_head=16, n_embd=1024, intermediate_size=3072),
            "1B": cls(n_layer=24, n_head=32, n_embd=2048, intermediate_size=6144),
        }
        if name not in variants:
            raise ValueError(f"Unknown variant {name}. Options: {list(variants.keys())}")
        config = variants[name]
        if block_size is not None:
            config.block_size = block_size
        return config


class ProxyModel(nn.Module):
    """
    GPT-style decoder for CLIMB proxy experiments.
    
    Same architecture as quadmix's tinyllama, but supports
    larger variants (62M, 350M) as per CLIMB paper.
    """

    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        B, T = input_ids.shape
        assert T <= self.config.block_size
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if return_hidden:
            return x
        logits = self.lm_head(x)
        return logits

    def count_params(self, non_embedding_only: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding_only:
            total -= self.embed.weight.numel()
        return total
