"""Python harness wrapper for baseline_warp_specialized_two_pipelines_driver.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineWarpSpecializedTwoPipelinesDriverBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_warp_specialized_two_pipelines_driver",
            friendly_name="Baseline Warp Specialized Two Pipelines Driver",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "tiles": 128,
                "tile_elems": 1024,
                "num_streams": 2,
                "dtype": "float32",
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializedTwoPipelinesDriverBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
