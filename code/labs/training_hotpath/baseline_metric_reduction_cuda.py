"""Baseline segmented metric reduction using vectorized torch slices."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.training_hotpath.training_hotpath_common import MetricReductionCudaBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = MetricReductionCudaBenchmark(
        optimized=False,
        label="baseline_metric_reduction_cuda",
    )
    return attach_benchmark_metadata(bench, __file__)
