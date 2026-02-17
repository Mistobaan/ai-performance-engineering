"""Local parity runner for Popcorn-style NVFP4 case1 benchmarking.

This reproduces Popcorn's timing model for `nvfp4_group_gemm`:
- one `custom_kernel(data)` call per timing iteration
- same input reused across repeats
- L2 flush between repeats (optional)
- correctness precheck on a cloned input (matches Popcorn eval.py)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import lock_gpu_clocks
from core.harness.l2_cache_utils import create_l2_flush_buffer, flush_l2_cache
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_common import COMPETITION_CASES
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_inputs import generate_input


def _load_custom_kernel(path: Path) -> Callable[[Any], Any]:
    spec = importlib.util.spec_from_file_location("nvfp4_popcorn_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load submission file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "custom_kernel", None)
    if not callable(fn):
        raise RuntimeError(f"{path} does not define callable custom_kernel(data)")
    return fn


def _env_fingerprint() -> str:
    keys = [
        "AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N",
        "AISP_NVFP4_GROUP_GEMM_V2_PIPELINE_STAGES",
        "AISP_NVFP4_GROUP_GEMM_V2_TMEM_COLUMNS",
        "AISP_NVFP4_GROUP_GEMM_V2_WS_UNROLL2_MMA",
        "AISP_NVFP4_GROUP_GEMM_V2_EPILOGUE_LD_X32",
        "AISP_NVFP4_GROUP_GEMM_V2_MAXRREGCOUNT",
        "AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X",
        "AISP_NVFP4_GROUP_GEMM_V2_ENABLE_EXPERIMENTAL_CTA2",
        "AISP_NVFP4_GROUP_GEMM_V2_CTA2_PARTITION_B",
        "AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST",
        "AISP_NVFP4_GROUP_GEMM_V2_TMA_L2_PROMOTION",
        "AISP_NVFP4_GROUP_GEMM_V2_CTA_ORDER",
    ]
    material = "\n".join(f"{k}={os.getenv(k, '')}" for k in keys)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _verify_output(data, out, *, rtol: float = 1e-3, atol: float = 1e-3) -> None:
    # Popcorn test mode performs full correctness checks. Here we keep a light local gate:
    # output tensors must exist, count must match groups, and values must be finite.
    if out is None or not isinstance(out, list):
        raise RuntimeError("custom_kernel did not return list[Tensor]")
    abc_tensors = data[0]
    if len(out) != len(abc_tensors):
        raise RuntimeError(f"group count mismatch: got {len(out)} expected {len(abc_tensors)}")
    for i, t in enumerate(out):
        if not torch.isfinite(t).all():
            raise RuntimeError(f"group {i}: output contains NaN/Inf")


def _clone_data(data):
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_group_gemm_v2/popcorn_submission_case1_v2.py"),
        help="Path to single-file Popcorn submission artifact.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >=0 and --repeats must be >0")
    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")

    custom_kernel = _load_custom_kernel(args.submission_file)
    case = COMPETITION_CASES[1]
    data = generate_input(m=case.m, n=case.n, k=case.k, g=case.g, seed=case.seed)

    if args.verify:
        # Popcorn benchmark mode performs one precheck with a cloned input, then times calls on the
        # original input object. Keep this ordering for parity.
        out = custom_kernel(_clone_data(data))
        _verify_output(data, out)
        torch.cuda.synchronize()

    flush_buf = create_l2_flush_buffer() if args.flush_l2 else None
    clock_ctx = lock_gpu_clocks(device=0) if args.lock_gpu_clocks else None
    if clock_ctx is None:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        clock_ctx = _NullCtx()

    per_call_us: list[float] = []
    with clock_ctx:
        for _ in range(args.warmup):
            custom_kernel(data)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            if flush_buf is not None:
                flush_l2_cache(buffer=flush_buf)
            start.record()
            custom_kernel(data)
            end.record()
            torch.cuda.synchronize()
            per_call_us.append(start.elapsed_time(end) * 1000.0)

    sorted_vals = sorted(per_call_us)
    p50 = sorted_vals[len(sorted_vals) // 2]
    p99 = sorted_vals[max(0, int(len(sorted_vals) * 0.99) - 1)]
    mean = statistics.mean(per_call_us)
    stdev = statistics.pstdev(per_call_us) if len(per_call_us) > 1 else 0.0
    result = {
        "submission_file": str(args.submission_file),
        "case": case.name,
        "groups": case.g,
        "timing_model": "popcorn_single_call",
        "precheck_mode": "clone_then_time_original" if args.verify else "disabled",
        "config_fingerprint": _env_fingerprint(),
        "per_call_us": {
            "mean": mean,
            "p50": p50,
            "p99": p99,
            "min": min(per_call_us),
            "max": max(per_call_us),
            "stdev": stdev,
            "repeats": args.repeats,
        },
        "per_group_us": {"mean": mean / float(case.g)},
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"submission_file={result['submission_file']}")
        print(f"case={case.name} g={case.g} timing_model=popcorn_single_call cfg_fp={result['config_fingerprint']}")
        print(
            "per_call_us: "
            f"mean={mean:.3f} p50={p50:.3f} p99={p99:.3f} "
            f"min={min(per_call_us):.3f} max={max(per_call_us):.3f} stdev={stdev:.3f}"
        )
        print(f"per_group_us: mean={(mean / float(case.g)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
