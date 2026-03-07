"""Explicit softcap-mode optimized target for the educational FlashAttention-4 lab."""

from __future__ import annotations

from labs.flashattention4.optimized_flashattention4 import OptimizedFlashAttention4Benchmark
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class OptimizedFlashAttention4SoftcapBenchmark(
    FlashAttention4FixedConfigMixin, OptimizedFlashAttention4Benchmark
):
    fixed_mode = "softcap"


def get_benchmark() -> OptimizedFlashAttention4Benchmark:
    return OptimizedFlashAttention4SoftcapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
