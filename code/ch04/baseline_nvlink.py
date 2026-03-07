"""baseline_nvlink.py - Single-GPU baseline transfer with host staging."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin


class BaselineNVLinkBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: synchronous device copy on a single GPU."""
    
    def __init__(self):
        super().__init__()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffer = None
        self.output: Optional[torch.Tensor] = None
        self.N = 20_000_000
        # Memory transfer benchmark - jitter check not applicable
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
            bytes_per_iteration=float(self.N * 4),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data_gpu0 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data_gpu1 = torch.empty_like(self.data_gpu0)
        self.host_buffer = torch.empty(self.N, device="cpu", dtype=torch.float32)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: PCIe-based communication (no NVLink)."""
        with self._nvtx_range("baseline_nvlink"):
            self.host_buffer.copy_(self.data_gpu0, non_blocking=False)
            self.data_gpu1.copy_(self.host_buffer, non_blocking=False)

    def capture_verification_payload(self) -> None:
        if self.data_gpu0 is None or self.data_gpu1 is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.data_gpu0[: 256 * 256].view(256, 256)
        output = self.data_gpu1[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data_gpu0 is None:
            return "Data not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for memory transfer benchmark."""
        return (0.0, 0.0)



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineNVLinkBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
