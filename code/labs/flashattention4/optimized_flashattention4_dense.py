"""Explicit dense-mode optimized target for the educational FlashAttention-4 lab."""

from __future__ import annotations

from labs.flashattention4.optimized_flashattention4 import OptimizedFlashAttention4Benchmark
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class OptimizedFlashAttention4DenseBenchmark(
    FlashAttention4FixedConfigMixin, OptimizedFlashAttention4Benchmark
):
    fixed_mode = "dense"


def get_benchmark() -> OptimizedFlashAttention4Benchmark:
    return OptimizedFlashAttention4DenseBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
