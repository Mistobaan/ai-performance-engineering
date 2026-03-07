"""Python harness wrapper for optimized_dynamic_parallelism_host.cu."""

from __future__ import annotations
from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedDynamicParallelismHostBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dynamic_parallelism_host",
            friendly_name="Optimized Dynamic Parallelism Host",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "workload": 1,
                "dtype": 'float32',
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDynamicParallelismHostBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
