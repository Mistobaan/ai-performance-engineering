"""NUMA-unaware baseline: copies pageable CPU tensors to GPU each step.

This benchmark demonstrates inefficient memory transfer - using non-pinned
pageable memory and blocking copies. The optimized version uses pinned
memory with async copies overlapped with compute.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


class BaselineNUMAUnawareBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Allocates pageable host memory and blocks on every copy."""

    def __init__(self):
        super().__init__()
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        # Memory copy benchmark - jitter check not applicable
        bytes_per_iter = 128_000_000 * 4  # float32 bytes
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float32)  # ~512 MB
        self.device_buffer = torch.empty_like(self.host_tensor, device=self.device)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Copy data and compute sum - blocking copy (slow)."""
        assert self.host_tensor is not None and self.device_buffer is not None
        with self._nvtx_range("baseline_numa_unaware"):
            # Blocking copy from non-pinned memory
            self.device_buffer.copy_(self.host_tensor, non_blocking=False)
            # Simple compute to overlap with in optimized version
            self.output = torch.sum(self.device_buffer).unsqueeze(0)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"host_tensor": self.host_tensor, "device_buffer": self.device_buffer},
            output=self.output.detach().clone(),
            batch_size=self.host_tensor.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffer = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNUMAUnawareBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
