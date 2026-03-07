"""Microbenchmark the block scaling lab against direct CUTLASS and PyTorch paths."""

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
    CUTLASS_EXAMPLE_PATH,
    build_problem,
    direct_colfax_reference_latency_ms,
    load_lab_config_from_env,
    measure_cuda_callable,
    override_config,
    parse_int_tuple,
    parse_software_dtype,
    theoretical_flops,
    tflops_from_latency_ms,
)


def _render_table(rows: list[dict[str, str]]) -> str:
    columns = ["Path", "Latency (ms)", "TFLOP/s", "Slowdown vs Lab HW", "Notes"]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join(row[column].ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for lab paths")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations for lab paths")
    parser.add_argument(
        "--colfax-warmup",
        type=int,
        default=None,
        help="Warmup iterations for the direct CUTLASS example (defaults to --warmup)",
    )
    parser.add_argument(
        "--colfax-iterations",
        type=int,
        default=None,
        help="Timed iterations for the direct CUTLASS example (defaults to --iterations)",
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
        "--skip-verify",
        action="store_true",
        help="Skip the shared software-vs-hardware correctness check",
    )
    parser.add_argument(
        "--verbose-colfax",
        action="store_true",
        help="Let the direct CUTLASS example print its own benchmark banner",
    )
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    parser.set_defaults(lock_gpu_clocks=True)
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
    colfax_warmup = args.warmup if args.colfax_warmup is None else args.colfax_warmup
    colfax_iterations = args.iterations if args.colfax_iterations is None else args.colfax_iterations

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
        lab_hardware_ms = measure_cuda_callable(
            problem.run_hardware,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        colfax_ms = direct_colfax_reference_latency_ms(
            config,
            warmup=colfax_warmup,
            iterations=colfax_iterations,
            skip_ref_check=True,
            use_cold_l2=False,
            verbose=args.verbose_colfax,
        )
        torch.cuda.synchronize()

    workload_flops = theoretical_flops(config)
    paths = {
        "software_blockscaled_reference": {
            "label": "Software blockscaled ref",
            "latency_ms": software_ms,
            "tflops": tflops_from_latency_ms(config, software_ms),
            "notes": "PyTorch scale multiply plus BF16 GEMM each iteration",
        },
        "pytorch_bf16_prescaled_gemm": {
            "label": "PyTorch BF16 GEMM",
            "latency_ms": prescaled_bf16_ms,
            "tflops": tflops_from_latency_ms(config, prescaled_bf16_ms),
            "notes": "Same GEMM after scales were applied ahead of time",
        },
        "lab_cutlass_hardware": {
            "label": "Lab CUTLASS hardware",
            "latency_ms": lab_hardware_ms,
            "tflops": tflops_from_latency_ms(config, lab_hardware_ms),
            "notes": "Compile-once lab wrapper around the Blackwell blockscaled kernel",
        },
        "colfax_cutlass_reference": {
            "label": "Colfax/CUTLASS direct",
            "latency_ms": colfax_ms,
            "tflops": tflops_from_latency_ms(config, colfax_ms),
            "notes": "Direct call into the original CUTLASS example run() path",
        },
    }

    for path in paths.values():
        path["slowdown_vs_lab_hardware"] = path["latency_ms"] / lab_hardware_ms

    payload = {
        "mnkl": config.mnkl,
        "mma_tiler_mn": config.mma_tiler_mn,
        "cluster_shape_mn": config.cluster_shape_mn,
        "sf_vec_size": config.sf_vec_size,
        "software_dtype": str(config.software_dtype).replace("torch.", ""),
        "tolerance": config.tolerance,
        "lock_gpu_clocks": args.lock_gpu_clocks,
        "sm_clock_mhz": args.sm_clock_mhz,
        "mem_clock_mhz": args.mem_clock_mhz,
        "workload_flops": workload_flops,
        "workload_tflop": workload_flops / 1e12,
        "source_article": BLOCK_SCALING_SOURCE_URL,
        "cutlass_example": str(CUTLASS_EXAMPLE_PATH.relative_to(repo_root)),
        "verification": verification,
        "paths": paths,
        "derived_metrics": {
            "lab_hardware_speedup_vs_software": software_ms / lab_hardware_ms,
            "lab_hardware_speedup_vs_prescaled_bf16_gemm": prescaled_bf16_ms / lab_hardware_ms,
            "software_block_scaling_overhead_ms": software_ms - prescaled_bf16_ms,
            "software_block_scaling_overhead_pct": (
                ((software_ms / prescaled_bf16_ms) - 1.0) * 100.0
                if prescaled_bf16_ms > 0
                else 0.0
            ),
            "lab_vs_colfax_latency_delta_ms": lab_hardware_ms - colfax_ms,
            "lab_vs_colfax_latency_delta_pct": (
                ((lab_hardware_ms / colfax_ms) - 1.0) * 100.0 if colfax_ms > 0 else 0.0
            ),
        },
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        rows = []
        for key in (
            "software_blockscaled_reference",
            "pytorch_bf16_prescaled_gemm",
            "lab_cutlass_hardware",
            "colfax_cutlass_reference",
        ):
            item = paths[key]
            rows.append(
                {
                    "Path": item["label"],
                    "Latency (ms)": f"{item['latency_ms']:.6f}",
                    "TFLOP/s": f"{item['tflops']:.3f}",
                    "Slowdown vs Lab HW": f"{item['slowdown_vs_lab_hardware']:.3f}x",
                    "Notes": item["notes"],
                }
            )
        print("Blackwell block scaling microbenchmark")
        print(
            f"shape={config.mnkl} mma_tiler={config.mma_tiler_mn} "
            f"cluster={config.cluster_shape_mn} sf_vec={config.sf_vec_size}"
        )
        print(f"work_per_iteration={payload['workload_tflop']:.6f} TFLOP")
        print(_render_table(rows))
        print(
            "lab_hardware_vs_software: "
            f"{payload['derived_metrics']['lab_hardware_speedup_vs_software']:.3f}x faster"
        )
        print(
            "lab_hardware_vs_prescaled_bf16_gemm: "
            f"{payload['derived_metrics']['lab_hardware_speedup_vs_prescaled_bf16_gemm']:.3f}x faster"
        )
        print(
            "software_block_scaling_overhead: "
            f"{payload['derived_metrics']['software_block_scaling_overhead_ms']:.6f} ms "
            f"({payload['derived_metrics']['software_block_scaling_overhead_pct']:.2f}%)"
        )
        print(
            "lab_vs_colfax_direct: "
            f"{payload['derived_metrics']['lab_vs_colfax_latency_delta_ms']:.6f} ms "
            f"({payload['derived_metrics']['lab_vs_colfax_latency_delta_pct']:.2f}%)"
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
