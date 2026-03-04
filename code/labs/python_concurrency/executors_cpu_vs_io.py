#!/usr/bin/env python3
"""Concurrency model comparison: asyncio vs threads vs processes.

This script provides a side-by-side benchmark harness for two workload classes:
- I/O-bound latency simulation,
- CPU-bound pure-Python computation.

It is designed to make runtime-model tradeoffs concrete with measurable data.

Quick talk track:
1) Keep workload constant and only swap runtime model.
2) Measure both I/O-bound and CPU-bound paths.
3) Compare throughput and wall time instead of assumptions.
4) Use results to justify asyncio/threads/process choice.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class BenchmarkRow:
    """One measured configuration."""

    workload: str
    model: str
    tasks: int
    workers: int
    total_seconds: float
    throughput_ops_per_sec: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Compare Python concurrency execution models")
    parser.add_argument(
        "--mode",
        choices=["all", "io", "cpu"],
        default="all",
        help="Which workload class to run.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=200,
        help="Number of tasks to execute per benchmark row.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker count for thread/process pools and asyncio semaphore.",
    )
    parser.add_argument(
        "--io-delay-ms",
        type=int,
        default=20,
        help="Per-task I/O delay for simulated I/O workload.",
    )
    parser.add_argument(
        "--cpu-n",
        type=int,
        default=70_000,
        help="Per-task CPU loop size for pure-Python CPU workload.",
    )
    return parser.parse_args()


def time_call(fn: Callable[[], None]) -> float:
    """Measure wall-clock runtime in seconds."""

    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


# ------------------------------
# I/O-bound task implementations
# ------------------------------

def io_task_blocking(delay_ms: int) -> int:
    """Thread-compatible blocking I/O simulation.

    Sleeping stands in for waiting on network or disk.
    """

    time.sleep(delay_ms / 1000.0)
    return delay_ms


async def io_task_async(delay_ms: int) -> int:
    """Async I/O simulation with cooperative scheduling."""

    await asyncio.sleep(delay_ms / 1000.0)
    return delay_ms


def run_io_threads(tasks: int, workers: int, delay_ms: int) -> float:
    """Run I/O simulation with ThreadPoolExecutor."""

    def _run() -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(io_task_blocking, [delay_ms] * tasks))

    return time_call(_run)


async def run_io_asyncio(tasks: int, workers: int, delay_ms: int) -> float:
    """Run I/O simulation with asyncio + bounded semaphore."""

    semaphore = asyncio.Semaphore(max(1, workers))

    async def _wrapped_task() -> int:
        # Semaphore bounds true in-flight operations.
        async with semaphore:
            return await io_task_async(delay_ms)

    started = time.perf_counter()
    await asyncio.gather(*(_wrapped_task() for _ in range(tasks)))
    return time.perf_counter() - started


def run_io_sequential(tasks: int, delay_ms: int) -> float:
    """Run I/O simulation serially for baseline comparison."""

    def _run() -> None:
        for _ in range(tasks):
            io_task_blocking(delay_ms)

    return time_call(_run)


# --------------------------------
# CPU-bound task implementations
# --------------------------------

def cpu_task_pure_python(n: int) -> int:
    """Pure-Python CPU work to illustrate GIL impact.

    This function intentionally keeps work in Python bytecode (loop + math).
    """

    acc = 0
    for i in range(1, n):
        # Small integer-heavy computation that remains Python-interpreted.
        acc += (i * i) % 97
        acc ^= (i << 1) & 0xFFFF
    return acc


def run_cpu_threads(tasks: int, workers: int, n: int) -> float:
    """Run CPU workload with ThreadPoolExecutor.

    For pure Python CPU work, this often scales poorly because of the GIL.
    """

    def _run() -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(cpu_task_pure_python, [n] * tasks))

    return time_call(_run)


def run_cpu_processes(tasks: int, workers: int, n: int) -> float:
    """Run CPU workload with ProcessPoolExecutor.

    Processes bypass the GIL by executing in separate Python interpreters.
    """

    def _run() -> None:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(cpu_task_pure_python, [n] * tasks))

    return time_call(_run)


def run_cpu_sequential(tasks: int, n: int) -> float:
    """Run CPU workload serially for baseline comparison."""

    def _run() -> None:
        for _ in range(tasks):
            cpu_task_pure_python(n)

    return time_call(_run)


# ------------------------------
# Reporting helpers
# ------------------------------

def to_row(workload: str, model: str, tasks: int, workers: int, total_seconds: float) -> BenchmarkRow:
    """Build one tabular output row."""

    throughput = tasks / total_seconds if total_seconds > 0 else math.inf
    return BenchmarkRow(
        workload=workload,
        model=model,
        tasks=tasks,
        workers=workers,
        total_seconds=total_seconds,
        throughput_ops_per_sec=throughput,
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    """Render a compact fixed-width table."""

    header = (
        f"{'workload':<10} {'model':<20} {'tasks':>8} {'workers':>8} "
        f"{'seconds':>10} {'ops/s':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row.workload:<10} {row.model:<20} {row.tasks:>8d} {row.workers:>8d} "
            f"{row.total_seconds:>10.4f} {row.throughput_ops_per_sec:>12.2f}"
        )


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    rows: list[BenchmarkRow] = []

    # ------------------------------
    # I/O-bound comparisons
    # ------------------------------
    if args.mode in {"all", "io"}:
        io_seq_s = run_io_sequential(tasks=args.tasks, delay_ms=args.io_delay_ms)
        rows.append(
            to_row(
                workload="io",
                model="sequential",
                tasks=args.tasks,
                workers=1,
                total_seconds=io_seq_s,
            )
        )

        io_thread_s = run_io_threads(tasks=args.tasks, workers=args.workers, delay_ms=args.io_delay_ms)
        rows.append(
            to_row(
                workload="io",
                model="ThreadPoolExecutor",
                tasks=args.tasks,
                workers=args.workers,
                total_seconds=io_thread_s,
            )
        )

        io_async_s = asyncio.run(
            run_io_asyncio(tasks=args.tasks, workers=args.workers, delay_ms=args.io_delay_ms)
        )
        rows.append(
            to_row(
                workload="io",
                model="asyncio+Semaphore",
                tasks=args.tasks,
                workers=args.workers,
                total_seconds=io_async_s,
            )
        )

    # ------------------------------
    # CPU-bound comparisons
    # ------------------------------
    if args.mode in {"all", "cpu"}:
        cpu_seq_s = run_cpu_sequential(tasks=args.tasks, n=args.cpu_n)
        rows.append(
            to_row(
                workload="cpu",
                model="sequential",
                tasks=args.tasks,
                workers=1,
                total_seconds=cpu_seq_s,
            )
        )

        cpu_thread_s = run_cpu_threads(tasks=args.tasks, workers=args.workers, n=args.cpu_n)
        rows.append(
            to_row(
                workload="cpu",
                model="ThreadPoolExecutor",
                tasks=args.tasks,
                workers=args.workers,
                total_seconds=cpu_thread_s,
            )
        )

        cpu_process_s = run_cpu_processes(tasks=args.tasks, workers=args.workers, n=args.cpu_n)
        rows.append(
            to_row(
                workload="cpu",
                model="ProcessPoolExecutor",
                tasks=args.tasks,
                workers=args.workers,
                total_seconds=cpu_process_s,
            )
        )

    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
