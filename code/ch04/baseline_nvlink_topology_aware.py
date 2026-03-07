"""baseline_nvlink_topology_aware.py - single-GPU naive copy baseline."""

from __future__ import annotations

from typing import Optional

import torch
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch04.verification_payload_mixin import VerificationPayloadMixin


class BaselineNvlinkTopologyAwareBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive copy that uses the default stream on a single GPU."""

    def __init__(self):
        super().__init__()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.host_buffer: Optional[torch.Tensor] = None
        self.numel = 16 * 1024 * 1024  # 64 MB
        self.chunk_elems = self.numel // 8
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.numel),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        n = self.numel
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        self.src = torch.randn(n, device=self.device, dtype=torch.float16)
        self.dst = torch.empty_like(self.src)
        self.host_buffer = torch.empty(n, device="cpu", dtype=torch.float16)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.src is not None and self.dst is not None
        with self._nvtx_range("baseline_nvlink_topology_aware"):
            # Naive: default stream copy, peer access may be disabled
            for start in range(0, self.numel, self.chunk_elems):
                end = min(start + self.chunk_elems, self.numel)
                self.host_buffer[start:end].copy_(self.src[start:end], non_blocking=False)
                self.dst[start:end].copy_(self.host_buffer[start:end], non_blocking=False)

    def capture_verification_payload(self) -> None:
        if self.src is None or self.dst is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.src[: 256 * 256].view(256, 256)
        output = self.dst[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        self.host_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.src is None or self.dst is None:
            return "Buffers not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineNvlinkTopologyAwareBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(BaselineNvlinkTopologyAwareBenchmark)
