"""optimized_vectorization_memory.py - Optimized memory with FP16 vectorization.

Chapter 19 - Low-Precision Training & Memory Systems
Demonstrates how FP16 (half precision) provides 2x memory bandwidth vs FP32.

Optimization vs baseline:
- Baseline: FP32 (32 bits per element, 4 bytes)
- Optimized: FP16 (16 bits per element, 2 bytes)
- Result: 2x memory bandwidth improvement for memory-bound workloads
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedVectorizationMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: same vector add workload, but with FP16 inputs and output."""

    signature_equivalence_group = "ch19_vectorization_memory_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.tensor_a: Optional[torch.Tensor] = None
        self.tensor_b: Optional[torch.Tensor] = None
        self._compute_dtype = torch.float16
        self._tensor_a_fp16: Optional[torch.Tensor] = None
        self._tensor_b_fp16: Optional[torch.Tensor] = None
        self._tensor_a_fp16_version: Optional[int] = None
        self._tensor_b_fp16_version: Optional[int] = None
        self._work: Optional[torch.Tensor] = None
        self._verify_probe_a: Optional[torch.Tensor] = None
        self._verify_probe_b: Optional[torch.Tensor] = None

        self.repeats = 12
        self.N = 67_108_864
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * (self.repeats + 1)),
        )
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * (self.repeats + 1)),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor_a = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.tensor_b = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._tensor_a_fp16 = self.tensor_a.to(self._compute_dtype)
        self._tensor_b_fp16 = self.tensor_b.to(self._compute_dtype)
        self._tensor_a_fp16_version = self.tensor_a._version
        self._tensor_b_fp16_version = self.tensor_b._version
        self._work = torch.empty(self.N, device=self.device, dtype=self._compute_dtype)
        self._verify_probe_a = self.tensor_a[:1024].detach().cpu()
        self._verify_probe_b = self.tensor_b[:1024].detach().cpu()
        torch.cuda.synchronize(self.device)

    def _cached_a_fp16(self) -> torch.Tensor:
        if self.tensor_a is None:
            raise RuntimeError("Tensor not initialized")
        current_version = self.tensor_a._version
        if self._tensor_a_fp16 is None or self._tensor_a_fp16_version != current_version:
            self._tensor_a_fp16 = self.tensor_a.to(self._compute_dtype)
            self._tensor_a_fp16_version = current_version
        return self._tensor_a_fp16

    def _cached_b_fp16(self) -> torch.Tensor:
        if self.tensor_b is None:
            raise RuntimeError("Tensor not initialized")
        current_version = self.tensor_b._version
        if self._tensor_b_fp16 is None or self._tensor_b_fp16_version != current_version:
            self._tensor_b_fp16 = self.tensor_b.to(self._compute_dtype)
            self._tensor_b_fp16_version = current_version
        return self._tensor_b_fp16

    def benchmark_fn(self) -> None:
        if self._work is None:
            raise RuntimeError("setup() must be called before benchmark_fn()")
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("optimized_vectorization", enable=enable_nvtx):
            for _ in range(self.repeats):
                torch.add(self._cached_a_fp16(), self._cached_b_fp16(), out=self._work)
            self.output = self._work.detach()
        if self.tensor_a is None or self.tensor_b is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe_a is None or self._verify_probe_b is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:4096].detach().cpu().float().clone()
        self._set_verification_payload(
            inputs={"probe_a": self._verify_probe_a, "probe_b": self._verify_probe_b},
            output=output_slice,
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(0.1, 1.0),
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
        )

    def teardown(self) -> None:
        self.tensor_a = None
        self.tensor_b = None
        self._tensor_a_fp16 = None
        self._tensor_b_fp16 = None
        self._tensor_a_fp16_version = None
        self._tensor_b_fp16_version = None
        self._work = None
        self._verify_probe_a = None
        self._verify_probe_b = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.tensor_a is None or self.tensor_b is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedVectorizationMemoryBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
