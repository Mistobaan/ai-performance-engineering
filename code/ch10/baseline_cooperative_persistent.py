"""Python harness wrapper for baseline_cooperative_persistent.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineCooperativePersistentBenchmark(CudaBinaryBenchmark):
    """Wraps the single-launch cooperative persistent baseline kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cooperative_persistent",
            friendly_name="Baseline Cooperative Persistent",
            iterations=3,
            warmup=5,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
            workload_params={
                "batch_size": 1 << 24,
                "dtype": "float32",
                "elements": 1 << 24,
                "iterations": 40,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float((1 << 24) * 2 * 4))


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 1),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Explicit signature for the cooperative persistent baseline."""
        return simple_signature(
            batch_size=1 << 24,
            dtype="float32",
            elements=1 << 24,
            iterations=40,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

def get_benchmark() -> BaselineCooperativePersistentBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCooperativePersistentBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
