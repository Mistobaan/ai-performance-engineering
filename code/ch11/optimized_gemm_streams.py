"""Optimized legacy `gemm_streams` target using overlapped copy+elementwise stream work."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedGemmStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("gemm_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedGemmStreamsBenchmark()
