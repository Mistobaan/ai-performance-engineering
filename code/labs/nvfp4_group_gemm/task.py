"""Input/output type aliases for the canonical NVFP4 group GEMM lab.

These types are shared by the benchmark wrappers and the custom CUDA submission.
"""

from __future__ import annotations

from typing import List, Tuple, TypeVar

import torch

input_t = TypeVar(
    "input_t",
    bound=tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[int, int, int, int]],
    ],
)
output_t = TypeVar("output_t", bound=list[torch.Tensor])
