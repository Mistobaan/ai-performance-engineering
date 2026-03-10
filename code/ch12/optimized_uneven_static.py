"""Python harness wrapper for optimized_uneven_static.cu."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedUnevenStaticBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_uneven_static",
            friendly_name="Optimized Uneven Static",
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
    return OptimizedUnevenStaticBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
