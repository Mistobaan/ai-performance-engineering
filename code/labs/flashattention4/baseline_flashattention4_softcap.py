"""Explicit softcap-mode baseline target for the educational FlashAttention-4 lab."""

from __future__ import annotations

from labs.flashattention4.baseline_flashattention4 import BaselineFlashAttention4Benchmark
from labs.flashattention4.target_variants import FlashAttention4FixedConfigMixin


class BaselineFlashAttention4SoftcapBenchmark(
    FlashAttention4FixedConfigMixin, BaselineFlashAttention4Benchmark
):
    fixed_mode = "softcap"


def get_benchmark() -> BaselineFlashAttention4Benchmark:
    return BaselineFlashAttention4SoftcapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
