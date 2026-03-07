#!/usr/bin/env python3
"""Optimized symmetric-memory perf microbench (single GPU).

Measures copy latency/bandwidth using a preallocated destination buffer.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.cuda_event_timing import elapsed_ms
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.metrics import compute_memory_transfer_metrics
from ch04.verification_payload_mixin import VerificationPayloadMixin


class OptimizedSymmetricMemoryPerfBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized device copy benchmark with preallocated buffers."""

    def __init__(self, size_mb: float = 1.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.local_tensor: Optional[torch.Tensor] = None
        self.peer_buffer: Optional[torch.Tensor] = None
        self._last_avg_ms = 0.0
        self._bytes_transferred = 0.0
        self._pending_timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.local_tensor = torch.randn(self.numel, device=self.device, dtype=torch.float32)
        self.peer_buffer = torch.empty_like(self.local_tensor)
        self._verify_input = self.local_tensor[: 256 * 256].view(256, 256).detach()
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        if self.local_tensor is None or self.peer_buffer is None:
            raise RuntimeError("Tensors not initialized")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        self.peer_buffer.copy_(self.local_tensor, non_blocking=True)
        end.record()
        self._pending_timing_pair = (start, end)
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if self._pending_timing_pair is None or self.local_tensor is None:
            return None
        elapsed_ms_value = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
        self._last_avg_ms = elapsed_ms_value
        self._bytes_transferred = float(self.local_tensor.numel() * self.local_tensor.element_size())
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.peer_buffer is None:
            if self._verify_input is None:
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                self._verify_input = torch.randn(256, 256, device=self.device, dtype=torch.float32)
            output = self._verify_input.detach().clone()
            probe = self._verify_input
        else:
            probe = self.peer_buffer[: 256 * 256].view(256, 256).detach()
            output = probe.clone()

        self._set_verification_payload(
            inputs={"tensor": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={"world_size": 1},
        )

    def teardown(self) -> None:
        self.local_tensor = None
        self.peer_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self.peer_buffer is None:
            return "No output captured"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSymmetricMemoryPerfBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
