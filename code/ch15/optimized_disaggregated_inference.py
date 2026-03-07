"""Optimized disaggregated inference benchmark (single GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.disaggregated_inference_single_common import (
    OptimizedDisaggregatedInferenceSingleGPUBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = OptimizedDisaggregatedInferenceSingleGPUBenchmark(
        label="optimized_disaggregated_inference",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
