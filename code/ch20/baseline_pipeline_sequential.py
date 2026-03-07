"""baseline_pipeline_sequential.py - Sequential pipeline baseline (baseline).

Sequential execution of pipeline stages without overlap.
Each stage waits for the previous to complete.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleStage(nn.Module):
    """Heavier pipeline stage to highlight overlap benefits."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        return self.norm(out + x)


class BaselinePipelineSequentialBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sequential pipeline - no overlap."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.stages = None
        self.inputs = None
        self.output = None
        # Larger workload so overlap benefits are measurable against sequential baseline.
        self.batch_size = 512
        self.hidden_dim = 1536
        self.num_stages = 4
        self.repeats = 6
        self.register_workload_metadata(requests_per_iteration=float(self.batch_size))
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Describe workload units processed per iteration."""
        return WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def setup(self) -> None:
        """Setup: Initialize pipeline stages."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Sequential pipeline stages
        self.stages = nn.ModuleList([
            SimpleStage(self.hidden_dim).to(self.device).half()
            for _ in range(self.num_stages)
        ]).eval()
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - sequential pipeline."""
        # Use conditional NVTX ranges - only enabled when profiling

        from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_pipeline_sequential", enable=enable_nvtx):
            # Sequential execution: each stage waits for previous
            for _ in range(self.repeats):
                x = self.inputs
                for stage in self.stages:
                    x = stage(x)  # Wait for completion before next stage
                    # Naive pipeline: copy activations back to host between stages at FP32 precision
                    host_buffer = x.detach().float().to("cpu", non_blocking=False)
                    x = host_buffer.to(self.device, non_blocking=False).half()
            # Capture output for verification
            self.output = x.detach()

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self.stages is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.stages.parameters()) if self.stages is not None else 0,
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Cleanup."""
        del self.stages, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.stages is None:
            return "Stages not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePipelineSequentialBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
