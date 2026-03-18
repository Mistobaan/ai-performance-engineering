"""Baseline metric reduction with scalar per-output aggregation."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.training_hotpath.training_hotpath_common import MetricReductionVectorizedBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = MetricReductionVectorizedBenchmark(
        optimized=False,
        label="baseline_metric_reduction_vectorized",
    )
    return attach_benchmark_metadata(bench, __file__)
