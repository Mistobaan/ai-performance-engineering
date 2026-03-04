#!/usr/bin/env python3
"""Empirical GIL demonstration.

This script runs two micro-benchmarks:
1) pure-Python CPU work,
2) I/O wait work.

It compares sequential, thread-based, and process/async models to show why
runtime choice depends on workload type.

Quick talk track:
1) Run identical task counts across sequential, threads, and processes/async.
2) Observe CPU-bound behavior separately from I/O-bound behavior.
3) Use measurements to explain why workload classification drives model choice.
4) Mention free-threaded runtimes as optional and environment-dependent.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(slots=True)
class Timing:
    label: str
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show GIL impact on CPU vs I/O workloads")
    parser.add_argument("--tasks", type=int, default=80, help="Number of tasks per benchmark.")
    parser.add_argument("--workers", type=int, default=8, help="Pool worker count.")
    parser.add_argument("--cpu-n", type=int, default=90_000, help="CPU loop size per task.")
    parser.add_argument("--io-delay-ms", type=int, default=25, help="Sleep duration per I/O task.")
    return parser.parse_args()


def cpu_work(n: int) -> int:
    """Pure-Python CPU loop that remains under the GIL."""

    total = 0
    for i in range(1, n):
        total += (i * i) % 193
        total ^= (i << 2) & 0xFFFF
    return total


def io_work(delay_ms: int) -> int:
    """Blocking sleep simulating I/O wait in threaded model."""

    time.sleep(delay_ms / 1000.0)
    return delay_ms


async def aio_work(delay_ms: int) -> int:
    """Cooperative sleep for asyncio model."""

    await asyncio.sleep(delay_ms / 1000.0)
    return delay_ms


def measure(fn) -> float:  # type: ignore[no-untyped-def]
    """Small helper to measure wall-clock runtime."""

    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run_cpu_suite(tasks: int, workers: int, n: int) -> list[Timing]:
    """Measure CPU workload with sequential, threads, and processes."""

    seq = measure(lambda: [cpu_work(n) for _ in range(tasks)])

    def _threads_run() -> None:
        # Context manager guarantees worker threads are joined and cleaned up.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(cpu_work, [n] * tasks))

    def _process_run() -> None:
        # Context manager guarantees process teardown and avoids orphan workers.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(cpu_work, [n] * tasks))

    thr = measure(_threads_run)
    proc = measure(_process_run)

    return [
        Timing("cpu_sequential", seq),
        Timing("cpu_threads", thr),
        Timing("cpu_processes", proc),
    ]


def run_io_suite(tasks: int, workers: int, delay_ms: int) -> list[Timing]:
    """Measure I/O workload with sequential, threads, and asyncio."""

    seq = measure(lambda: [io_work(delay_ms) for _ in range(tasks)])

    def _threads_run() -> None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(io_work, [delay_ms] * tasks))

    thr = measure(_threads_run)

    async def _aio_runner() -> None:
        semaphore = asyncio.Semaphore(max(1, workers))

        async def _one() -> int:
            async with semaphore:
                return await aio_work(delay_ms)

        await asyncio.gather(*(_one() for _ in range(tasks)))

    aio = measure(lambda: asyncio.run(_aio_runner()))

    return [
        Timing("io_sequential", seq),
        Timing("io_threads", thr),
        Timing("io_asyncio", aio),
    ]


def print_suite(title: str, timings: list[Timing]) -> None:
    """Print absolute timings and speedups versus first row."""

    print(f"\n{title}")
    print("-" * len(title))
    baseline = timings[0].seconds
    for row in timings:
        speedup = baseline / row.seconds if row.seconds > 0 else float("inf")
        print(f"{row.label:<16} {row.seconds:>9.4f}s   speedup_vs_baseline={speedup:>6.2f}x")


def main() -> int:
    args = parse_args()

    cpu_timings = run_cpu_suite(tasks=args.tasks, workers=args.workers, n=args.cpu_n)
    io_timings = run_io_suite(tasks=args.tasks, workers=args.workers, delay_ms=args.io_delay_ms)

    print_suite("CPU workload (pure Python)", cpu_timings)
    print_suite("I/O workload (simulated wait)", io_timings)

    print("\nInterpretation:")
    print("- Threads usually help I/O workloads because waiting yields execution opportunities.")
    print("- Processes usually help pure-Python CPU workloads by bypassing the single GIL.")
    print("- Always validate assumptions with measured timings on the target machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
