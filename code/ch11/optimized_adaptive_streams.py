"""Optimized legacy `adaptive_streams` target using fixed round-robin multi-stream overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedAdaptiveStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        super().__init__("adaptive_streams")


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedAdaptiveStreamsBenchmark()
