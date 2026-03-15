"""Optimized AI example: fuse blocks into a single inference stack without CPU sync."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class TinyBlock(nn.Module):
    """Same architecture as baseline for fair comparison."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class OptimizedAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Chains the tiny blocks without CPU sync between them (the optimization)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Sequential] = None
        self.static_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        # Must match baseline workload; tuned to keep the optimized path launch-bound
        # without making profiler runs pathological.
        self.batch = 64
        self.hidden = 32
        self.num_blocks = 256
        # Inference benchmark - jitter check not applicable
        tokens = self.batch * self.hidden * self.num_blocks
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        blocks = [TinyBlock(self.hidden) for _ in range(self.num_blocks)]
        self.model = nn.Sequential(*blocks).to(self.device).eval()
        self.static_input = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            _ = self.model(self.static_input)

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.static_input is not None
        with self._nvtx_range("optimized_ai"):
            with torch.inference_mode():
                out = self.model(self.static_input)
        self.output = out.detach()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.static_input},
            output=self.output,
            batch_size=self.batch,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.model = None
        self.static_input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.static_input is None:
            return "Model/input not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return OptimizedAIBenchmark()
