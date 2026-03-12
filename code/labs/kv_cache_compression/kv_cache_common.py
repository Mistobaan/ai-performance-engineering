"""Shared utilities for the KV-cache compression lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.harness.arch_config import prefer_sdpa_backends


def resolve_device() -> torch.device:
    """Return a CUDA device or raise if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for KV-cache compression benchmarks.")
    return torch.device("cuda")


@dataclass
class KVCache:
    cache_k: torch.Tensor
    cache_v: torch.Tensor


def allocate_kv_cache(
    batch_size: int,
    total_tokens: int,
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCache:
    """Allocate K/V cache tensors for the lab."""
    cache_shape = (batch_size, total_tokens, num_heads, head_dim)
    return KVCache(
        cache_k=torch.empty(cache_shape, device=device, dtype=dtype),
        cache_v=torch.empty(cache_shape, device=device, dtype=dtype),
    )


def build_token_batches(
    *,
    batch_size: int,
    prefill_seq: int,
    decode_seq: int,
    decode_steps: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create synthetic prefill/decode token batches."""
    prefill = [
        torch.randn(batch_size, prefill_seq, hidden_dim, device=device, dtype=dtype),
        torch.randn(batch_size, prefill_seq, hidden_dim, device=device, dtype=dtype),
    ]
    # Decode inputs are read-only; reuse one deterministic batch across steps to
    # keep setup memory bounded while preserving the per-step decode workload.
    decode_batch = torch.randn(batch_size, decode_seq, hidden_dim, device=device, dtype=dtype)
    decode = [decode_batch for _ in range(decode_steps)]
    return prefill, decode


class KVCacheAttention(nn.Module):
    """Single attention block that writes into an external KV cache."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        linear_cls: Type[nn.Module],
        layernorm_cls: Type[nn.Module],
        params_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = (self.head_dim**-0.5)
        # TELayerNorm needs params_dtype too.
        try:
            self.ln = layernorm_cls(hidden_dim, params_dtype=params_dtype, device=device)
        except TypeError as exc:
            raise TypeError("layernorm_cls must accept params_dtype and device") from exc
        # For TELinear, pass params_dtype and device to avoid dtype mismatch.
        try:
            self.qkv = linear_cls(
                hidden_dim,
                hidden_dim * 3,
                bias=True,
                params_dtype=params_dtype,
                device=device,
            )
            self.proj = linear_cls(
                hidden_dim,
                hidden_dim,
                bias=True,
                params_dtype=params_dtype,
                device=device,
            )
        except TypeError as exc:
            raise TypeError("linear_cls must accept params_dtype and device") from exc

    def forward(self, tokens: torch.Tensor, cache: KVCache, start_offset: int) -> torch.Tensor:
        """Compute attention for tokens and append K/V into cache."""
        if tokens.dim() != 3:
            raise ValueError(f"tokens must have shape [batch, seq, hidden], got {tuple(tokens.shape)}")
        batch, seq_len, _ = tokens.shape
        x = self.ln(tokens)
        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Write into cache
        cache.cache_k[:, start_offset : start_offset + seq_len].copy_(k)
        cache.cache_v[:, start_offset : start_offset + seq_len].copy_(v)

        # Attention over current cache (prefill + decode-so-far)
        k_ctx = cache.cache_k[:, : start_offset + seq_len]
        v_ctx = cache.cache_v[:, : start_offset + seq_len]

        # Use the fused SDPA path so this lab measures KV-cache tradeoffs rather
        # than materializing a giant [batch, heads, query, context] tensor.
        q_t = q.transpose(1, 2)
        k_t = k_ctx.transpose(1, 2)
        v_t = v_ctx.transpose(1, 2)
        with prefer_sdpa_backends():
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
        out = out.transpose(1, 2).contiguous().reshape(batch, seq_len, self.hidden_dim)
        return self.proj(out)


def reset_cache(cache: KVCache) -> None:
    """Zero cache contents to keep iterations independent."""
    cache.cache_k.zero_()
    cache.cache_v.zero_()


def cache_is_finite(cache: KVCache) -> bool:
    """Check for non-finite entries in cache tensors."""
    return torch.isfinite(cache.cache_k).all() and torch.isfinite(cache.cache_v).all()
