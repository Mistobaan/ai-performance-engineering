"""Baseline tensor-core stream workload without overlap."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import (
    canonicalize_nvtx_name,
    get_nvtx_enabled,
    nvtx_range,
)
from ch11.stream_overlap_base import resolve_device


class BaselineTensorCoresStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline tensor-core workload: sequential GEMM operations on single stream.
    
    Uses FP16/BF16 GEMM operations to demonstrate tensor core utilization,
    but processes chunks sequentially without stream overlap.
    """

    declare_all_streams = False

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.label = "baseline_tensor_cores_streams"
        self.num_segments = 16
        self.matrix_dim = 1024
        self.num_elements = self.num_segments * self.matrix_dim * self.matrix_dim
        self.stream: torch.cuda.Stream | None = None
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.host_A: torch.Tensor | None = None
        self.host_B: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self.device_A: torch.Tensor | None = None
        self.device_B: torch.Tensor | None = None
        self.device_C: torch.Tensor | None = None
        self.device_output_rows: torch.Tensor | None = None
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.stream = torch.cuda.Stream()

        self.host_A = torch.randn(
            self.num_segments,
            self.matrix_dim,
            self.matrix_dim,
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )
        self.host_B = torch.randn(
            self.num_segments,
            self.matrix_dim,
            self.matrix_dim,
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )
        self.host_output = torch.empty(
            (self.num_segments, self.matrix_dim),
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )

        self.device_A = self.host_A.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.device_B = self.host_B.to(device=self.device, dtype=self.dtype, non_blocking=True)
        self.device_C = torch.empty(
            (self.num_segments, self.matrix_dim, self.matrix_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.device_output_rows = torch.empty(
            (self.num_segments, self.matrix_dim),
            device=self.device,
            dtype=self.dtype,
        )

        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = getattr(self, "_config", None) or self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            assert self.device_A is not None
            assert self.device_B is not None
            assert self.device_C is not None
            assert self.device_output_rows is not None

            with torch.no_grad():
                for idx in range(self.num_segments):
                    with torch.cuda.stream(self.stream):
                        torch.matmul(self.device_A[idx], self.device_B[idx], out=self.device_C[idx])
                        self.device_output_rows[idx].copy_(self.device_C[idx, 0], non_blocking=True)

        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)

        assert self.host_output is not None
        self.host_output.copy_(self.device_output_rows, non_blocking=False)

        if self.host_A is None or self.host_B is None or self.host_output is None:
            raise RuntimeError("benchmark_fn() must run after setup() initializes buffers")

    def capture_verification_payload(self) -> None:
        assert self.host_A is not None
        assert self.host_B is not None
        assert self.host_output is not None
        self._set_verification_payload(
            inputs={"host_A": self.host_A, "host_B": self.host_B},
            output=self.host_output.detach().clone(),
            batch_size=self.host_output.numel(),
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def teardown(self) -> None:
        self.stream = None
        self.host_A = None
        self.host_B = None
        self.host_output = None
        self.device_A = None
        self.device_B = None
        self.device_C = None
        self.device_output_rows = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        nvtx_tag = canonicalize_nvtx_name(self.label)
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if self.host_output is None or self.host_A is None or self.host_B is None:
            return "Buffers not initialized"
        if not torch.isfinite(self.host_output).all():
            return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the baseline (sequential) path."""
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)
        return {
            f"{self.label}.elements": float(self.num_elements),
            f"{self.label}.num_segments": float(self.num_segments),
            f"{self.label}.matrix_dim": float(self.matrix_dim),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.num_streams": 1.0,
            f"{self.label}.expected_overlap_pct": 0.0,
            f"{self.label}.dtype": str(self.dtype),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.stream is None:
            return []
        return [self.stream]

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()


def get_benchmark() -> BaselineTensorCoresStreamsBenchmark:
    return BaselineTensorCoresStreamsBenchmark()
