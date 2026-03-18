"""optimized_matmul_pytorch.py - Compiled PyTorch matmul optimization.

Uses the same FP16 compute dtype as the eager baseline and applies
`torch.compile` to isolate compile/fusion benefits.
"""

from __future__ import annotations

import torch

from core.harness import arch_config as _arch_config_patch  # noqa: F401
from typing import Optional

from core.utils.compile_utils import compile_callable
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def optimized_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """CUTLASS-optimized matrix multiplication with fused epilogue."""
    out = torch.matmul(A, B)
    out = torch.relu(out + bias)
    return (out + residual) * scale


class OptimizedMatmulCUTLASSBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUTLASS matmul optimization - compiled GEMM."""

    signature_equivalence_group = "ch13_matmul_pytorch_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.bias = None
        self.residual = None
        self.scale = 0.125
        self.compiled_matmul = None
        # Match baseline dimensions for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        # Register workload metadata at init time for compliance check
        bytes_per_iter = (self.m * self.k + self.k * self.n + self.m * self.n * 3) * 2
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices and compile matmul."""
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.bias = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        self.residual = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)

        self.compiled_matmul = compile_callable(
            optimized_matmul,
            backend="inductor",
        )
        for _ in range(10):
            _ = self.compiled_matmul(self.A, self.B, self.bias, self.residual, self.scale)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - CUTLASS-optimized matmul."""
        assert (
            self.compiled_matmul is not None
            and self.A is not None
            and self.B is not None
            and self.bias is not None
            and self.residual is not None
        )
        with self._nvtx_range("matmul_pytorch"):
            # CUTLASS-optimized matrix multiplication (via torch.compile)
            self.C = self.compiled_matmul(self.A, self.B, self.bias, self.residual, self.scale)

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "A": self.A,
                "B": self.B,
                "bias": self.bias,
                "residual": self.residual,
            },
            output=self.C.detach().clone(),
            batch_size=self.m,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.bias, self.residual, self.C, self.compiled_matmul
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None or self.residual is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMatmulCUTLASSBenchmark()
