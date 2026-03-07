"""baseline_matmul.py - Baseline FP32 matmul with serial tiling."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import ch10.arch_config  # noqa: F401
except ImportError:
    pass

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineMatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP32 matmul with serialized tiling and no tensor cores."""

    signature_equivalence_group = "ch10_matmul_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.n = 8192
        self.tile_k = 128
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 4 * 3))

    def setup(self) -> None:
        """Setup: initialize FP32 matrices and scratch buffer."""
        torch.manual_seed(42)
        self.A = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        self._chunked_matmul()
        self._synchronize()

    def _chunked_matmul(self) -> None:
        """Multiply using many FP32 tiles to emphasize poor reuse."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.n, self.tile_k):
            k_end = min(k + self.tile_k, self.n)
            a_tile = self.A[:, k:k_end]
            b_tile = self.B[k:k_end, :]
            # Repeated addmm launches mimic per-stage kernel launches.
            self.C.addmm_(a_tile, b_tile, beta=1.0, alpha=1.0)

    def benchmark_fn(self) -> None:
        """Benchmark: serialized FP32 matmul tiles."""
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("Matrices not initialized")
        with self._nvtx_range("matmul_baseline_fp32"):
            self._chunked_matmul()

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.C.detach().clone(),
            batch_size=self.A.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(5e-2, 5.0),
        )

    def teardown(self) -> None:
        """Teardown: clean up tensors."""
        self.A = None
        self.B = None
        self.C = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,  # Minimum warmup for accurate matmul timing
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulBenchmark()
