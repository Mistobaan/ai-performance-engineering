"""Python harness wrapper for optimized_cutlass_gemm_fp8.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCutlassGemmFp8Benchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cutlass_gemm_fp8",
            friendly_name="Optimized Cutlass Gemm Fp8",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "kIterations": 10,
                "kRepeats": 16,
                "dtype": "fp8_e4m3",
            },
        )
        self._selected_backend = "cutlass_sm90"

    def setup(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for CUTLASS FP8")
        major, _minor = torch.cuda.get_device_capability()
        if major != 9:
            raise RuntimeError(
                "SKIPPED: optimized_cutlass_gemm_fp8 requires SM90 Hopper hardware. "
                "This benchmark no longer falls back to CuBLASLt on other architectures."
            )
        self._selected_backend = "cutlass_sm90"
        super().setup()

    def get_custom_metrics(self) -> Optional[dict]:
        return {"fp8_backend": self._selected_backend}


def get_benchmark() -> BaseBenchmark:
    return OptimizedCutlassGemmFp8Benchmark()

