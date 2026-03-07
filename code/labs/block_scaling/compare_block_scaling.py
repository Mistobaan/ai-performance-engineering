"""Compare the software and hardware block scaling paths."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.block_scaling.block_scaling_common import (
    BLOCK_SCALING_SOURCE_URL,
    build_problem,
    load_lab_config_from_env,
    measure_cuda_callable,
    override_config,
    parse_int_tuple,
    parse_software_dtype,
    tflops_from_latency_ms,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for each path")
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Timed iterations for each path",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the correctness check between software and hardware outputs",
    )
    parser.add_argument("--mnkl", type=str, default=None, help="Override M,N,K,L as comma-separated ints")
    parser.add_argument(
        "--mma-tiler-mn",
        type=str,
        default=None,
        help="Override MMA tiler as M,N",
    )
    parser.add_argument(
        "--cluster-shape-mn",
        type=str,
        default=None,
        help="Override cluster shape as M,N",
    )
    parser.add_argument("--sf-vec-size", type=int, default=None, help="Override scale-factor vector size")
    parser.add_argument("--tolerance", type=float, default=None, help="Override correctness tolerance")
    parser.add_argument(
        "--software-dtype",
        type=str,
        default=None,
        help="Override software reference dtype: bf16 or fp16",
    )
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.set_defaults(lock_gpu_clocks=True)
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    args = parser.parse_args()

    config = override_config(
        load_lab_config_from_env(),
        mnkl=None if args.mnkl is None else parse_int_tuple(args.mnkl, expected_len=4, name="--mnkl"),
        mma_tiler_mn=(
            None
            if args.mma_tiler_mn is None
            else parse_int_tuple(args.mma_tiler_mn, expected_len=2, name="--mma-tiler-mn")
        ),
        cluster_shape_mn=(
            None
            if args.cluster_shape_mn is None
            else parse_int_tuple(args.cluster_shape_mn, expected_len=2, name="--cluster-shape-mn")
        ),
        sf_vec_size=args.sf_vec_size,
        tolerance=args.tolerance,
        software_dtype=(
            None if args.software_dtype is None else parse_software_dtype(args.software_dtype)
        ),
    )
    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else nullcontext()
    )
    with lock_ctx:
        torch.manual_seed(1111)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1111)
        problem = build_problem(config, compile_hardware=True)
        verification = None if args.skip_verify else problem.verify_close()

        software_ms = measure_cuda_callable(
            problem.run_software,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        prescaled_bf16_ms = measure_cuda_callable(
            problem.run_prescaled_bf16_gemm,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        hardware_ms = measure_cuda_callable(
            problem.run_hardware,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        torch.cuda.synchronize()

    payload = {
        "mnkl": config.mnkl,
        "mma_tiler_mn": config.mma_tiler_mn,
        "cluster_shape_mn": config.cluster_shape_mn,
        "sf_vec_size": config.sf_vec_size,
        "software_dtype": str(config.software_dtype).replace("torch.", ""),
        "lock_gpu_clocks": args.lock_gpu_clocks,
        "sm_clock_mhz": args.sm_clock_mhz,
        "mem_clock_mhz": args.mem_clock_mhz,
        "software_latency_ms": software_ms,
        "software_tflops": tflops_from_latency_ms(config, software_ms),
        "prescaled_bf16_gemm_latency_ms": prescaled_bf16_ms,
        "prescaled_bf16_gemm_tflops": tflops_from_latency_ms(config, prescaled_bf16_ms),
        "hardware_latency_ms": hardware_ms,
        "hardware_tflops": tflops_from_latency_ms(config, hardware_ms),
        "speedup_hardware_vs_software": software_ms / hardware_ms,
        "speedup_hardware_vs_prescaled_bf16_gemm": prescaled_bf16_ms / hardware_ms,
        "software_block_scaling_overhead_ms": software_ms - prescaled_bf16_ms,
        "software_block_scaling_overhead_pct": (
            ((software_ms / prescaled_bf16_ms) - 1.0) * 100.0
            if prescaled_bf16_ms > 0
            else 0.0
        ),
        "verification": verification,
        "source_article": BLOCK_SCALING_SOURCE_URL,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Block scaling tutorial recreation")
        print(f"shape={config.mnkl} mma_tiler={config.mma_tiler_mn} cluster={config.cluster_shape_mn}")
        print(
            f"software_dequant_bf16: {software_ms:.6f} ms "
            f"({payload['software_tflops']:.3f} TFLOP/s)"
        )
        print(
            f"prescaled_bf16_gemm: {prescaled_bf16_ms:.6f} ms "
            f"({payload['prescaled_bf16_gemm_tflops']:.3f} TFLOP/s)"
        )
        print(
            f"hardware_blockscaled: {hardware_ms:.6f} ms "
            f"({payload['hardware_tflops']:.3f} TFLOP/s)"
        )
        print(f"speedup: {payload['speedup_hardware_vs_software']:.3f}x")
        print(
            "software_overhead_vs_prescaled_gemm: "
            f"{payload['software_block_scaling_overhead_ms']:.6f} ms "
            f"({payload['software_block_scaling_overhead_pct']:.2f}%)"
        )
        if verification is not None:
            print(
                "verification: "
                f"max_abs_error={verification['max_abs_error']:.6f}, "
                f"mean_abs_error={verification['mean_abs_error']:.6f}"
            )
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
