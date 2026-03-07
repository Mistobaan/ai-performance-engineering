"""Optimized FlashAttention-4 lab target that picks the fastest correct local backend."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.flashattention4_benchmarks import (
    OptimizedBestAvailableAttentionBenchmarkBase,
)


class OptimizedBestAvailableAttentionBenchmark(OptimizedBestAvailableAttentionBenchmarkBase):
    """Select the fastest correct backend on the current machine for the chosen mode."""


def get_benchmark() -> BaseBenchmark:
    return OptimizedBestAvailableAttentionBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
