"""Shared helpers for Chapter 1 training-loop performance benchmarks."""

from __future__ import annotations

from typing import Tuple

import torch


def seed_chapter1(seed: int = 42) -> None:
    """Seed CPU and CUDA deterministically for benchmark reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_mlp(hidden_dim: int) -> torch.nn.Sequential:
    """Return the small MLP used by the Chapter 1 goodput benchmarks."""
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 10),
    )


def capture_tf32_state() -> Tuple[bool, bool | None]:
    """Snapshot the current TF32 backend settings so callers can restore them."""
    cudnn_state = None
    if torch.backends.cudnn.is_available():
        cudnn_state = bool(torch.backends.cudnn.allow_tf32)
    return bool(torch.backends.cuda.matmul.allow_tf32), cudnn_state


def set_tf32_state(enabled: bool) -> None:
    """Enable or disable TF32 across CUDA matmul/cuDNN backends."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = enabled
