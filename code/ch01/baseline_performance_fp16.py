"""baseline_performance_fp16.py - Compute-heavy FP32 baseline for the precision-only pair."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch01.baseline_performance import BaselinePerformanceBenchmark
from ch01.performance_fp16_common import PERFORMANCE_FP16_WORKLOAD


class BaselinePerformanceFP16Benchmark(BaselinePerformanceBenchmark):
    """Keep the Chapter 1 baseline logic, but use the local FP16-comparison workload."""

    def __init__(self):
        super().__init__()
        self.batch_size = PERFORMANCE_FP16_WORKLOAD.batch_size
        self.num_microbatches = PERFORMANCE_FP16_WORKLOAD.num_microbatches
        self.hidden_dim = PERFORMANCE_FP16_WORKLOAD.hidden_dim
        self.register_workload_metadata(
            samples_per_iteration=PERFORMANCE_FP16_WORKLOAD.samples_per_iteration,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselinePerformanceFP16Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

