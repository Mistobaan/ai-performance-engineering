"""Optimized FlashAttention-4 lab: compiled FlexAttention on Blackwell."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.flashattention4_benchmarks import (
    OptimizedFlashAttention4BenchmarkBase,
)


class OptimizedFlashAttention4Benchmark(OptimizedFlashAttention4BenchmarkBase):
    """Compiled FlashAttention path with TMA and optional FLASH backend."""


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttention4Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
