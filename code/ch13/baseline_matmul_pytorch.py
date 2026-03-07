"""baseline_matmul_pytorch.py - PyTorch matmul baseline (baseline).

Standard PyTorch matrix multiplication without CUTLASS optimization.
Good baseline but can be optimized with specialized GEMM kernels.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineMatmulPyTorchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """PyTorch matmul baseline - sequential unfused operations."""

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
        # Use larger matrices to show more difference
        self.m = 4096
        self.n = 4096
        self.k = 4096
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        # Register workload metadata at init time for compliance check
        bytes_per_iter = (self.m * self.k + self.k * self.n + self.m * self.n * 3) * 4
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Standard PyTorch matmul
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(self.m, self.n, device=self.device, dtype=torch.float32)
        self.residual = torch.randn(self.m, self.n, device=self.device, dtype=torch.float32)
        
        # Warmup
        out = torch.matmul(self.A, self.B)
        out = torch.relu(out + self.bias)
        out = (out + self.residual) * self.scale
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - PyTorch matmul."""
        assert self.A is not None and self.B is not None and self.bias is not None and self.residual is not None
        with self._nvtx_range("baseline_matmul_pytorch"):
            # Standard PyTorch matrix multiplication
            out = torch.matmul(self.A, self.B)
            out = torch.relu(out + self.bias)
            self.C = (out + self.residual) * self.scale

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B, "bias": self.bias, "residual": self.residual},
            output=self.C.detach().clone(),
            batch_size=self.m,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C, self.bias, self.residual
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None or self.residual is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulPyTorchBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
