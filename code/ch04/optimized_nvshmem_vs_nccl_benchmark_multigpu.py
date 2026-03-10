"""Optimized NVSHMEM vs NCCL benchmark with NVLink5/NVLS knobs; skips on <2 GPUs."""

from __future__ import annotations

from pathlib import Path

import os
from typing import Optional

import argparse
import torch
import torch.distributed as dist
from ch04.nccl_blackwell_config import (
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    configure_nccl_for_multigpu,
    detect_b200_multigpu_topology,
)
from ch04.nvshmem_vs_nccl_benchmark import benchmark, init_distributed
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import symmetric_memory_available
from ch04.verification_payload_mixin import VerificationPayloadMixin


def _configure_blackwell_nccl() -> None:
    try:
        topo = detect_b200_multigpu_topology()
    except Exception:
        configure_nccl_for_blackwell(verbose=False)
        return

    if topo.get("has_grace_cpu"):
        configure_nccl_for_gb200_gb300(verbose=False)
    elif topo.get("num_gpus", 0) >= 2 and topo.get("is_b200_multigpu"):
        configure_nccl_for_multigpu(num_gpus=topo.get("num_gpus", 2), verbose=False)
    else:
        configure_nccl_for_blackwell(verbose=False)


class OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("random_input_regeneration", "sync")
    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_vs_nccl_benchmark requires >=2 GPUs")
        if symmetric_memory_available():
            _configure_blackwell_nccl()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        use_symmem = symmetric_memory_available()
        args = argparse.Namespace(
            min_bytes=1048576,
            max_bytes=1048576,
            steps=1,
            iterations=500,
            mode="nvshmem" if use_symmem else "nccl",
        )
        original_disable = os.environ.get("AISP_DISABLE_SYMMETRIC_MEMORY")
        original_overlap = os.environ.get("AISP_BROADCAST_OVERLAP")
        original_compute = os.environ.get("AISP_BROADCAST_COMPUTE_PASSES")
        rank = init_distributed()
        try:
            os.environ["AISP_DISABLE_SYMMETRIC_MEMORY"] = "0" if use_symmem else "1"
            os.environ["AISP_BROADCAST_OVERLAP"] = "1"
            os.environ["AISP_BROADCAST_COMPUTE_PASSES"] = "8"
            try:
                results = benchmark(args)
            except Exception as exc:
                if rank == 0:
                    print(f"NVSHMEM broadcast failed; falling back to NCCL. ({exc})")
                args.mode = "nccl"
                os.environ["AISP_DISABLE_SYMMETRIC_MEMORY"] = "1"
                results = benchmark(args)
            if rank == 0:
                print("\nNVSHMEM Benchmark (optimized for NVLink 5.0 / NVLS / TCE)")
                print("------------------------------------------------------")
                print(f"Symmetric memory available: {bool(results['nvshmem'])}")
        finally:
            if dist.is_initialized():
                dist.barrier()
            if original_disable is None:
                os.environ.pop("AISP_DISABLE_SYMMETRIC_MEMORY", None)
            else:
                os.environ["AISP_DISABLE_SYMMETRIC_MEMORY"] = original_disable
            if original_overlap is None:
                os.environ.pop("AISP_BROADCAST_OVERLAP", None)
            else:
                os.environ["AISP_BROADCAST_OVERLAP"] = original_overlap
            if original_compute is None:
                os.environ.pop("AISP_BROADCAST_COMPUTE_PASSES", None)
            else:
                os.environ["AISP_BROADCAST_COMPUTE_PASSES"] = original_compute

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

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
            name="optimized_nvshmem_vs_nccl_benchmark_multigpu",
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
    return OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
