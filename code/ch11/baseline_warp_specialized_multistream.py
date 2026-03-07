"""Python harness wrapper for baseline_warp_specialized_multistream.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineWarpSpecializedMultistreamBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_warp_specialized_multistream",
            friendly_name="Baseline Warp Specialized Multistream",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "TILE": 32,
                "THREADS": 96,
                "batches": 4096,
                "dtype": 'float32',
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializedMultistreamBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
