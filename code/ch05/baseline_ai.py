"""Baseline AI optimization example: repeated CPU-bound orchestration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BaselineAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs several tiny blocks sequentially with CPU sync between them."""

    def __init__(self):
        super().__init__()
        self.blocks: Optional[nn.ModuleList] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        # Keep the work per block small enough that host orchestration dominates,
        # but cap the block count so profiling stays tractable.
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
        # Initialize model weights after seeding for deterministic comparison
        self.blocks = nn.ModuleList(TinyBlock(self.hidden).to(self.device).eval() for _ in range(self.num_blocks))
        self.inputs = torch.randn(self.batch, self.hidden, device=self.device, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.inputs is not None and self.blocks is not None
        with self._nvtx_range("baseline_ai"):
            with torch.inference_mode():
                out = self.inputs
                for block in self.blocks:
                    out = block(out)
                    self._synchronize()
        self.output = out.detach()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output,
            batch_size=self.batch,
            parameter_count=sum(p.numel() for p in self.blocks.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.blocks = None
        self.inputs = None
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
        if self.inputs is None:
            return "Inputs missing"
        return None



def get_benchmark() -> BaseBenchmark:
    return BaselineAIBenchmark()
