"""Explicit softcap-mode optimized target for the best-available FlashAttention-4 path."""

from __future__ import annotations

from labs.flashattention4.optimized_best_available_attention import (
    OptimizedBestAvailableAttentionBenchmark,
)
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class OptimizedBestAvailableAttentionSoftcapBenchmark(
    FlashAttention4FixedConfigMixin, OptimizedBestAvailableAttentionBenchmark
):
    fixed_mode = "softcap"


def get_benchmark() -> OptimizedBestAvailableAttentionBenchmark:
    return OptimizedBestAvailableAttentionSoftcapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
