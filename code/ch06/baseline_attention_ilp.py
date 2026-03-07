"""baseline_attention_ilp.py - Baseline attention with low ILP."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


class BaselineAttentionILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: sequential attention limiting instruction-level parallelism."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.MultiheadAttention] = None
        self.input: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.batch = self.workload.attention_batch
        self.embed_dim = self.workload.attention_embed_dim
        self.tokens = self.workload.attention_tokens
        self.query_chunk = self.workload.attention_chunk_tokens
        token_count = self.batch * self.tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(token_count),
        )
        self._last_sum: Optional[torch.Tensor] = None
        # ILP benchmark: fixed dimensions for measurement

    def setup(self) -> None:
        """Setup: Initialize attention model."""
        torch.manual_seed(42)
        self.model = nn.MultiheadAttention(
            self.embed_dim,
            self.workload.attention_heads,
            batch_first=True,
        ).to(self.device).eval().half()
        self.input = torch.randn(
            self.batch,
            self.tokens,
            self.embed_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with low ILP."""
        assert self.model is not None and self.input is not None
        with self._nvtx_range("baseline_attention_ilp"):
            with torch.no_grad():
                accum = torch.zeros(1, device=self.device, dtype=self.input.dtype)
                for query in self.input.split(self.query_chunk, dim=1):
                    out = self.model(query, self.input, self.input)[0]
                    accum += out.sum()
                self._last_sum = accum
        self._payload_accum = accum

    def capture_verification_payload(self) -> None:
        accum = self._payload_accum
        self._set_verification_payload(
            inputs={"input": self.input},
            output=(self._last_sum if self._last_sum is not None else accum).float().detach(),
            batch_size=self.batch,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            output_tolerance=(1e-2, 100.0),
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self._last_sum is None:
            return "Output sum not computed"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineAttentionILPBenchmark()
