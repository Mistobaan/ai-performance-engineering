"""Explicit windowed-mode baseline target for the best-available FlashAttention-4 path."""

from __future__ import annotations

from labs.flashattention4.baseline_best_available_attention import (
    BaselineBestAvailableAttentionBenchmark,
)
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class BaselineBestAvailableAttentionWindowedBenchmark(
    FlashAttention4FixedConfigMixin, BaselineBestAvailableAttentionBenchmark
):
    fixed_mode = "windowed"


def get_benchmark() -> BaselineBestAvailableAttentionBenchmark:
    return BaselineBestAvailableAttentionWindowedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
