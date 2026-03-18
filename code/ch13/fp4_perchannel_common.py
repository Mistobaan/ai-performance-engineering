"""Shared initialization helpers for the FP4 per-channel benchmark pair."""

from __future__ import annotations

import torch


def build_fp4_perchannel_reference_tensors(
    *,
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build canonical weights, biases, and inputs for both benchmark variants."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    w1 = torch.randn(hidden_dim * 2, hidden_dim, device=device, dtype=dtype) * 0.02
    b1 = torch.zeros(hidden_dim * 2, device=device, dtype=dtype)
    w2 = torch.randn(hidden_dim, hidden_dim * 2, device=device, dtype=dtype) * 0.02
    b2 = torch.zeros(hidden_dim, device=device, dtype=dtype)
    inputs = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    return w1, b1, w2, b2, inputs
