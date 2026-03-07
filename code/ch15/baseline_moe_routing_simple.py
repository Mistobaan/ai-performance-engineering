#!/usr/bin/env python3
"""baseline_moe_routing_simple.py - Uniform MoE routing + naive dispatch (Ch15).

This benchmark is a *comparable* baseline for topology-aware routing variants.

Design choice (for benchmarkability):
- All experts share the same weights (a single shared expert module).
- Routing/placement changes therefore do NOT change the final output tensor.
- The measured speedup comes purely from dispatch/routing efficiency, not from
  changing the math.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ch15.verification_payload_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP
from core.optimization.shared_expert_dispatch import dispatch_shared_expert_mask_scan


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class BaselineMoERoutingSimpleBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: uniform-ish routing and a slow mask-scan dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.ffn_size = 4096
        self.num_experts = 64
        self.batch = 64
        self.seq = 16
        self.dtype = torch.bfloat16

        tokens = self.batch * self.seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

        self.expert: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self._out_flat: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE routing benchmark")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)

        self._out_flat = torch.empty(self.batch * self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._verify_probe = self.inputs[:1, :1, :256].detach().cpu()
        self._verify_meta = torch.zeros(self.num_experts, dtype=torch.int8)

        for _ in range(3):
            with torch.no_grad():
                _ = self.expert(self.inputs.view(-1, self.hidden_size))

    def benchmark_fn(self) -> None:
        if self.expert is None or self.inputs is None or self.expert_ids is None or self._out_flat is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self.inputs.view(-1, self.hidden_size)
        expert_ids_flat = self.expert_ids.reshape(-1)

        with self._nvtx_range("baseline_moe_routing_simple"):
            with torch.no_grad():
                dispatch_shared_expert_mask_scan(
                    flat,
                    expert_ids_flat,
                    self.expert,
                    num_experts=self.num_experts,
                    out=self._out_flat,
                )
                self.output = self._out_flat.view(self.batch, self.seq, self.hidden_size)

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe is None or self._verify_meta is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:2, :2, :256].detach().cpu().float().clone()
        param_count = sum(p.numel() for p in self.expert.parameters()) if self.expert is not None else 0
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "expert_meta": self._verify_meta},
            output=output_slice,
            batch_size=int(self.batch),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.expert = None
        self.inputs = None
        self.expert_ids = None
        self._out_flat = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERoutingSimpleBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
