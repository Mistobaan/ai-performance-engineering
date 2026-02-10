#!/usr/bin/env python3
"""
Fast torchrun connectivity probe for multi-node NCCL readiness.

This is intentionally lightweight (single all-reduce + barrier timing) so it can be used
as a required pre-benchmark gate in canonical cluster runs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_physical_gpu(logical_index: int) -> int:
    try:
        code_root = Path(__file__).resolve().parents[2]
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore

        return int(_resolve_physical_device_index(logical_index))
    except Exception:
        return logical_index


def _probe_rank(
    barrier_iters: int,
    payload_bytes: int,
    timeout_s: int,
) -> Dict[str, Any]:
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_s))
    try:
        host = socket.gethostname()
        props = torch.cuda.get_device_properties(local_rank)
        physical_gpu = _resolve_physical_gpu(local_rank)

        # Correctness check: predictable scalar all-reduce.
        value = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        allreduce_ok = abs(float(value.item()) - float(expected)) < 1e-3

        # Lightweight barrier jitter profile.
        barrier_ms: List[float] = []
        for _ in range(barrier_iters):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.barrier()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            barrier_ms.append((t1 - t0) * 1000.0)

        # Small payload collective sanity (measures comm path health, not benchmarking).
        elem_count = max(1, payload_bytes // 4)
        payload = torch.ones(elem_count, dtype=torch.float32, device="cuda")
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_s = max(1e-9, t1 - t0)
        algbw_gbps = (payload_bytes / elapsed_s) / 1e9
        busbw_gbps = algbw_gbps * (2 * (world_size - 1) / world_size) if world_size > 1 else algbw_gbps

        return {
            "status": "ok" if allreduce_ok else "error",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "hostname": host,
            "gpu": {
                "logical_index": local_rank,
                "physical_index": physical_gpu,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_gb": round(props.total_memory / (1024**3), 3),
            },
            "checks": {
                "scalar_allreduce_ok": allreduce_ok,
                "scalar_allreduce_expected": expected,
                "scalar_allreduce_observed": float(value.item()),
            },
            "barrier_ms": [round(v, 6) for v in barrier_ms],
            "payload_probe": {
                "payload_bytes": payload_bytes,
                "time_ms": round(elapsed_s * 1000.0, 6),
                "algbw_gbps": round(algbw_gbps, 6),
                "busbw_gbps": round(busbw_gbps, 6),
            },
        }
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast torchrun NCCL connectivity probe")
    parser.add_argument("--output", required=True, help="Structured JSON output path")
    parser.add_argument("--barrier-iters", type=int, default=5, help="Barrier timing iterations (default: 5)")
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Payload size in bytes for all-reduce sanity (default: 8MiB)",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Process-group timeout in seconds (default: 120)")
    args = parser.parse_args()

    if args.barrier_iters <= 0:
        raise SystemExit(f"ERROR: --barrier-iters must be > 0 (got: {args.barrier_iters})")
    if args.payload_bytes <= 0:
        raise SystemExit(f"ERROR: --payload-bytes must be > 0 (got: {args.payload_bytes})")
    if args.timeout_sec <= 0:
        raise SystemExit(f"ERROR: --timeout-sec must be > 0 (got: {args.timeout_sec})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local = _probe_rank(
        barrier_iters=args.barrier_iters,
        payload_bytes=args.payload_bytes,
        timeout_s=args.timeout_sec,
    )

    gathered: List[Dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    # Re-initialize briefly to gather rank payloads (kept separate so probe is easy to reason about).
    # torchrun guarantees env vars are still valid for this process.
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
    try:
        dist.all_gather_object(gathered, local)
    finally:
        dist.destroy_process_group()

    if rank == 0:
        failures = [r for r in gathered if (r or {}).get("status") != "ok"]
        payload = {
            "test": "torchrun_connectivity_probe",
            "run_id": os.environ.get("RUN_ID", ""),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "world_size": world_size,
            "barrier_iters": args.barrier_iters,
            "payload_bytes": args.payload_bytes,
            "status": "ok" if not failures else "error",
            "ranks": gathered,
            "failures": failures,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {output_path}")

    return 0 if local.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
