"""Shared single-GPU gradient fusion benchmark logic."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch04.verification_payload_mixin import VerificationPayloadMixin


class GradientFusionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU gradient fusion benchmark (fused vs unfused reductions)."""

    def __init__(
        self,
        *,
        fused: bool,
        num_tensors: int = 256,
        tensor_kb: int = 32,
        equivalence_group: str = "ch04_gradient_fusion_single",
    ) -> None:
        super().__init__()
        self.fused = bool(fused)
        self.num_tensors = int(num_tensors)
        self.tensor_kb = int(tensor_kb)
        self.signature_equivalence_group = equivalence_group
        self.signature_equivalence_ignore_fields = ("precision_flags",)
        elem_bytes = torch.tensor([], dtype=torch.float32).element_size()
        numel = max(1, (self.tensor_kb * 1024) // elem_bytes)
        total_bytes = self.num_tensors * numel * elem_bytes
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_bytes),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_bytes),
        )
        self.tensors: list[torch.Tensor] = []
        self.fused_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for gradient fusion benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        elem_bytes = torch.tensor([], dtype=torch.float32).element_size()
        numel = max(1, (self.tensor_kb * 1024) // elem_bytes)
        self.tensors = [
            torch.randn(numel, device=self.device, dtype=torch.float32)
            for _ in range(self.num_tensors)
        ]
        self.fused_tensor = torch.cat([t.view(-1) for t in self.tensors])
        self._verify_input = self.tensors[0]

    def benchmark_fn(self) -> None:
        if not self.tensors or self.fused_tensor is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        if self.fused:
            self.output = self.fused_tensor.sum()
        else:
            accum = torch.zeros((), device=self.device, dtype=torch.float32)
            for tensor in self.tensors:
                accum = accum + tensor.sum()
            self.output = accum

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self.output.detach().clone(),
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-2),
            signature_overrides={
                "world_size": 1,
                "collective_type": "all_reduce",
            },
        )

    def teardown(self) -> None:
        self.tensors = []
        self.fused_tensor = None
        self.output = None
        self._verify_input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
