"""Benchmark harness wrapper for NVFP4 dual GEMM optimized submission."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB_DIR = Path(__file__).resolve().parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedNvfp4DualGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the optimized NVFP4 dual GEMM submission through the benchmark harness."""

    def __init__(self) -> None:
        super().__init__()
        self.m = 256
        self.n = 3072
        self.k = 4096
        self.l = 1
        self.seed = 42
        self._input_data: Optional[Any] = None
        self._kernel_fn: Optional[Callable[[Any], torch.Tensor]] = None
        self._generate_input: Optional[Callable[..., Any]] = None
        self.output: Optional[torch.Tensor] = None
        tokens = float(self.m * self.n * self.l)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA is required for NVFP4 dual GEMM benchmarks")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        for stale in ("task", "utils", "reference_submission", "baseline_submission", "optimized_submission"):
            sys.modules.pop(stale, None)
        reference_mod = importlib.import_module("reference_submission")
        optimized_mod = importlib.import_module("optimized_submission")
        generate_input = getattr(reference_mod, "generate_input", None)
        if not callable(generate_input):
            raise RuntimeError("reference_submission.py must expose generate_input()")
        self._generate_input = generate_input
        self._input_data = self._generate_input(
            m=self.m,
            n=self.n,
            k=self.k,
            l=self.l,
            seed=self.seed,
        )
        module = optimized_mod
        kernel_fn = getattr(module, "custom_kernel", None)
        if not callable(kernel_fn):
            raise RuntimeError("optimized_submission.py does not expose callable custom_kernel(data)")
        self._kernel_fn = kernel_fn
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._input_data is None or self._kernel_fn is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("optimized_nvfp4_dual_gemm"):
            self.output = self._kernel_fn(self._input_data)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:64, :64, :1].float().detach().clone()
        self._set_verification_payload(
            inputs={
                "shape_signature": torch.tensor(
                    [self.m, self.n, self.k, self.l, self.seed],
                    dtype=torch.int64,
                )
            },
            output=verify_output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self._input_data = None
        self._kernel_fn = None
        self._generate_input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=2)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self._input_data is None or self.output is None:
            return "Benchmark output missing"
        if self.output.dtype != torch.float16:
            return f"Unexpected output dtype: {self.output.dtype}"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNvfp4DualGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
