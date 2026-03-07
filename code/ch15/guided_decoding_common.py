"""Shared harness logic for Chapter 15 guided decoding benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class GuidedDecodingConfig:
    """Workload configuration for the guided decoding benchmark family."""

    batch_size: int = 16
    steps: int = 64
    vocab_size: int = 32768
    allowed_count: int = 4096
    output_slice: int = 256


class GuidedDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Parameterized guided decoding benchmark with or without mask reuse."""

    def __init__(
        self,
        *,
        reuse_gpu_mask: bool,
        label: str,
        cfg: Optional[GuidedDecodingConfig] = None,
    ) -> None:
        super().__init__()
        self.reuse_gpu_mask = bool(reuse_gpu_mask)
        self.label = label
        self.cfg = cfg or GuidedDecodingConfig()
        self.batch_size = int(self.cfg.batch_size)
        self.steps = int(self.cfg.steps)
        self.vocab_size = int(self.cfg.vocab_size)
        self.allowed_count = int(self.cfg.allowed_count)
        self.output_slice = int(self.cfg.output_slice)

        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

        self.logits: Optional[torch.Tensor] = None
        self.allowed_token_ids: Optional[torch.Tensor] = None
        self.allowed_mask: Optional[torch.Tensor] = None
        self.slice_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.logits = torch.randn(
            self.batch_size,
            self.vocab_size,
            device=self.device,
            dtype=torch.float32,
        )
        self.allowed_token_ids = torch.randperm(
            self.vocab_size,
            device="cpu",
            dtype=torch.int64,
        )[: self.allowed_count]

        if self.reuse_gpu_mask:
            mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            mask[self.allowed_token_ids.to(self.device)] = True
            self.allowed_mask = mask
            self.slice_ids = self.allowed_token_ids[: self.output_slice].to(self.device)
        else:
            self.allowed_mask = None
            self.slice_ids = None
        self.output = None

    def benchmark_fn(self) -> None:
        if self.logits is None or self.allowed_token_ids is None:
            raise RuntimeError("Benchmark not initialized")

        logits = self.logits
        allowed = self.allowed_token_ids

        with self._nvtx_range(self.label):
            for _ in range(self.steps):
                if self.reuse_gpu_mask:
                    if self.allowed_mask is None or self.slice_ids is None:
                        raise RuntimeError("GPU mask state not initialized")
                    masked = logits.masked_fill(~self.allowed_mask, float("-inf"))
                    self.output = masked.index_select(1, self.slice_ids)
                    continue

                mask_cpu = torch.zeros(self.vocab_size, dtype=torch.bool, device="cpu")
                mask_cpu[allowed] = True
                mask_gpu = mask_cpu.to(self.device, non_blocking=False)
                masked = logits.masked_fill(~mask_gpu, float("-inf"))
                slice_ids = allowed[: self.output_slice].to(self.device)
                self.output = masked.index_select(1, slice_ids)

        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.logits is None or self.allowed_token_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "logits": self.logits,
                "allowed_token_ids": self.allowed_token_ids,
            },
            output=self.output,
            batch_size=int(self.batch_size),
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
        self.logits = None
        self.allowed_token_ids = None
        self.allowed_mask = None
        self.slice_ids = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None
