"""Baseline FlashAttention-4 lab: eager FlexAttention with score materialization."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.flashattention4_benchmarks import (
    BaselineFlashAttention4BenchmarkBase,
)


class BaselineFlashAttention4Benchmark(BaselineFlashAttention4BenchmarkBase):
    """Uncompiled FlexAttention path for the FA4 lab."""


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttention4Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
