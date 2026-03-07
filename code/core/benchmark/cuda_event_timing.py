"""Helpers for finalizing CUDA-event timings outside benchmark_fn()."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

CudaEventPair = Tuple[torch.cuda.Event, torch.cuda.Event]


def elapsed_ms(pair: CudaEventPair) -> float:
    """Synchronize the end event and return elapsed milliseconds."""
    start_event, end_event = pair
    end_event.synchronize()
    return float(start_event.elapsed_time(end_event))


def elapsed_ms_list(pairs: Sequence[CudaEventPair]) -> List[float]:
    """Synchronize each end event and return elapsed milliseconds."""
    return [elapsed_ms(pair) for pair in pairs]


def max_elapsed_ms(pairs: Sequence[CudaEventPair]) -> float:
    """Return the slowest elapsed interval from a list of CUDA event pairs."""
    values = elapsed_ms_list(pairs)
    return max(values, default=0.0)
