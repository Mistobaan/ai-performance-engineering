"""Baseline dense padded transformer benchmark."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.training_hotpath.training_hotpath_common import PaddingAwareTransformerBenchmark


def get_benchmark() -> BaseBenchmark:
    bench = PaddingAwareTransformerBenchmark(
        optimized=False,
        label="baseline_padding_aware_transformer",
    )
    return attach_benchmark_metadata(bench, __file__)
