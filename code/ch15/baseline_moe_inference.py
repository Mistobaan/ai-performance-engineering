"""Baseline MoE inference benchmark (single GPU sequential prefill + decode)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_inference_common import BaselineMoeInferenceBenchmark, attach_benchmark_metadata


def get_benchmark() -> BaseBenchmark:
    bench = BaselineMoeInferenceBenchmark()
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
