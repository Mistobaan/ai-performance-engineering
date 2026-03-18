"""Python harness wrapper for baseline_pipeline_3stage.cu - 2-Stage Pipeline Baseline."""

from __future__ import annotations
from typing import Optional

from pathlib import Path
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
class BaselinePipeline3StageBenchmark(CudaBinaryBenchmark):
    """Wraps the 2-stage pipeline GEMV kernel (baseline for 3-stage comparison)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_pipeline_3stage",
            friendly_name="Baseline Pipeline 3Stage",
            iterations=10,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "batch_size": 8,
                "dtype": "float32",
                "elements": 8 * 1024 * 1024,
                "segments": 128,
                "segment_size": 65536,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=65536)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the declared pipeline structure and workload params."""
        from ch10.benchmark_metrics_common import compute_pipeline_variant_metrics

        return compute_pipeline_variant_metrics(
            self._workload_params,
            num_stages=2,
        )

    def get_input_signature(self) -> dict:
        """Signature for sequential 3-stage pipeline baseline."""
        return simple_signature(
            batch_size=8,
            dtype="float32",
            elements=8 * 1024 * 1024,
            segments=128,
            segment_size=65536,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselinePipeline3StageBenchmark()
