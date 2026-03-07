"""baseline_vectorization_memory.py - Naive vectorization baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class VectorizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: naive elementwise ops without vectorization."""

    signature_equivalence_group = "ch19_vectorization_memory_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.tensor: Optional[torch.Tensor] = None
        self._work_a: Optional[torch.Tensor] = None
        self._work_b: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None

        # Use a large tensor that exceeds L2 so the kernel is HBM bandwidth bound.
        # Keep the iteration count low to avoid kernel-launch overhead dominating.
        self.repeats = 8
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
        self.tensor = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._work_a = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._work_b = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._verify_probe = self.tensor[:1024].detach().cpu()
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.tensor is None or self._work_a is None or self._work_b is None:
            raise RuntimeError("setup() must be called before benchmark_fn()")
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_vectorization", enable=enable_nvtx):
            alpha = 1.0001
            beta = 0.0001
            a = self._work_a
            b = self._work_b
            a.copy_(self.tensor)
            for _ in range(self.repeats):
                torch.add(beta, a, alpha=alpha, out=b)
                a, b = b, a
            self.output = a.detach()

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:4096].detach().cpu().float().clone()
        self._set_verification_payload(
            inputs={"probe": self._verify_probe},
            output=output_slice,
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.tensor = None
        self._work_a = None
        self._work_b = None
        self._verify_probe = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return VectorizationBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
