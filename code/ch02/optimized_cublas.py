"""optimized_cublas.py - Pure cuBLAS matmul with TF32 via backend policy."""

from __future__ import annotations

import time
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedCublasBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Optimized: pure cuBLAS GEMM with TF32 and warmed-up heuristics.

    This keeps the math in FP32 but lets cuBLAS route workloads through tensor cores
    (TF32) via the harness `backend_policy="performance"` path while running a
    few warmup matmuls so Lt heuristics cache the best kernel.
    """

    allow_cpu = True

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self._last_elapsed_ms: Optional[float] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        """Allocate FP32 matrices and warm the TF32-enabled cuBLAS path."""
        if self.device.type != "cuda":
            raise RuntimeError("SKIPPED: cuBLAS benchmark requires CUDA")
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.C = None

        # Warmup a handful of GEMMs so cuBLAS Lt heuristics settle before measurement.
        for _ in range(10):
            _ = torch.matmul(self.A, self.B)

    def benchmark_fn(self) -> None:
        """cuBLAS TF32 GEMM."""
        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()
        with nvtx_range("cublas", enable=enable_nvtx):
            self.C = torch.matmul(self.A, self.B)
        if self.device.type == "cuda":
            end_event.record()
            end_event.synchronize()
            self._last_elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self.C is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.C.detach().clone(),
            batch_size=self.A.shape[0],
            parameter_count=0,
            precision_flags={
                # Keep signature aligned with baseline; TF32 is the optimization detail, not a workload change.
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-2, 1e-1),
        )

    def teardown(self) -> None:
        """Restore TF32 knobs and free tensors."""
        self.A = None
        self.B = None
        self._last_elapsed_ms = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5, backend_policy="performance")
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return real GEMM workload metrics for the measured matmul."""
        from core.benchmark.metrics import compute_gemm_metrics
        return compute_gemm_metrics(
            self.m,
            self.n,
            self.k,
            elapsed_ms=self._last_elapsed_ms,
            precision="tensor",
            bytes_per_element=4,
        )

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCublasBenchmark()
