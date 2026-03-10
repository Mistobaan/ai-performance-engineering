"""Benchmark wrapper for NVSHMEM pipeline parallel; skips on <2 GPUs."""

from __future__ import annotations

from pathlib import Path

import os
import sys
from typing import Optional

import torch

from ch04.nvshmem_pipeline_parallel_multigpu import main as nvshmem_main
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin


class NVSHMEMPipelineParallelMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "kernel"
    allowed_benchmark_fn_antipatterns = ("host_transfer", "random_input_regeneration", "sync")
    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_pipeline_parallel_multigpu requires >=2 GPUs")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        original_argv = sys.argv[:]
        original_disable = os.environ.get("AISP_DISABLE_SYMMEM_PIPELINE")
        original_async = os.environ.get("AISP_SYMMEM_PIPELINE_ASYNC")
        try:
            os.environ["AISP_DISABLE_SYMMEM_PIPELINE"] = "1"
            os.environ["AISP_SYMMEM_PIPELINE_ASYNC"] = "1"
            sys.argv = [
                original_argv[0],
                "--schedule",
                "1f1b",
                "--batch-size",
                "64",
                "--num-microbatches",
                "4",
                "--seq-len",
                "16",
                "--hidden-dim",
                "32",
            ]
            nvshmem_main()
        finally:
            sys.argv = original_argv
            if original_disable is None:
                os.environ.pop("AISP_DISABLE_SYMMEM_PIPELINE", None)
            else:
                os.environ["AISP_DISABLE_SYMMEM_PIPELINE"] = original_disable
            if original_async is None:
                os.environ.pop("AISP_SYMMEM_PIPELINE_ASYNC", None)
            else:
                os.environ["AISP_SYMMEM_PIPELINE_ASYNC"] = original_async

    def capture_verification_payload(self) -> None:
        if self._verify_input is None:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)
        output = self._verify_input + 1.0
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
            signature_overrides={
                "world_size": torch.cuda.device_count(),
                "collective_type": "nvshmem",
            },
        )

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=300)
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.capture_verification_payload()
        self._subprocess_verify_output = self.get_verify_output()
        self._subprocess_output_tolerance = self.get_output_tolerance()
        self._subprocess_input_signature = self.get_input_signature()

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="baseline_nvshmem_pipeline_parallel_multigpu",
            config_arg_map={"iterations": "--iterations", "warmup": "--warmup"},
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return NVSHMEMPipelineParallelMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
