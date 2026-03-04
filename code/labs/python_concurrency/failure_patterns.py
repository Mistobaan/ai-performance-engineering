#!/usr/bin/env python3
"""Concurrency failure-pattern demos (broken vs fixed).

This module provides small, deterministic examples for common production bugs:
1) unbounded fanout
2) race condition on shared state
3) deadlock via inconsistent lock ordering
4) hidden blocking call in async code
5) retry storm without jitter
6) cancellation ignored

Each pair includes a broken version and a corrected pattern.

Quick talk track:
1) Name the bug class (fanout, race, deadlock, blocking call, retry storm, cancellation).
2) Show the broken pattern and the observed failure mode.
3) Apply one focused fix (bounded pool, lock order, executor offload, jitter, cancel handling).
4) Re-run and compare behavior/metrics to prove improvement.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# 1) Unbounded fanout vs bounded worker pool
# ---------------------------------------------------------------------------


async def _io_unit(value: int, delay_ms: int) -> int:
    """Small async operation used by fanout demos."""

    await asyncio.sleep(delay_ms / 1000.0)
    return value * 2


async def unbounded_fanout_broken(item_count: int, delay_ms: int) -> list[int]:
    """Broken pattern: creates one coroutine per input element at once.

    This is safe for tiny lists but dangerous for huge workloads because memory
    and scheduler overhead scale with item_count.
    """

    coroutines = [_io_unit(i, delay_ms) for i in range(item_count)]
    return await asyncio.gather(*coroutines)


async def bounded_worker_pool_fixed(item_count: int, delay_ms: int, workers: int) -> list[int]:
    """Fixed pattern: queue + fixed workers gives bounded concurrency."""

    queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
    for idx in range(item_count):
        queue.put_nowait((idx, idx))

    results: list[int | None] = [None] * item_count

    async def _worker() -> None:
        while True:
            payload = await queue.get()
            if payload is None:
                queue.task_done()
                return
            idx, value = payload
            results[idx] = await _io_unit(value, delay_ms)
            queue.task_done()

    worker_tasks = [asyncio.create_task(_worker()) for _ in range(max(1, workers))]
    await queue.join()
    for _ in worker_tasks:
        queue.put_nowait(None)
    await asyncio.gather(*worker_tasks)

    return [0 if r is None else r for r in results]


# ---------------------------------------------------------------------------
# 2) Race condition vs lock-protected critical section
# ---------------------------------------------------------------------------


async def race_condition_broken(task_count: int) -> int:
    """Broken shared-state update with deterministic lost update.

    All tasks capture the same value before writes are released.
    """

    counter = 0
    gate = asyncio.Event()

    async def _increment() -> None:
        nonlocal counter
        local_value = counter
        await gate.wait()
        counter = local_value + 1

    tasks = [asyncio.create_task(_increment()) for _ in range(task_count)]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*tasks)
    return counter


async def race_condition_fixed(task_count: int) -> int:
    """Fixed shared-state update with lock-protected read-modify-write."""

    counter = 0
    gate = asyncio.Event()
    lock = asyncio.Lock()

    async def _increment() -> None:
        nonlocal counter
        await gate.wait()
        async with lock:
            local_value = counter
            await asyncio.sleep(0)
            counter = local_value + 1

    tasks = [asyncio.create_task(_increment()) for _ in range(task_count)]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*tasks)
    return counter


# ---------------------------------------------------------------------------
# 3) Deadlock vs lock ordering
# ---------------------------------------------------------------------------


async def deadlock_detected_broken(timeout_ms: int = 40) -> bool:
    """Broken pattern: inconsistent lock order creates circular wait."""

    lock_a = asyncio.Lock()
    lock_b = asyncio.Lock()

    async def _task1() -> None:
        async with lock_a:
            await asyncio.sleep(0)
            async with lock_b:
                return

    async def _task2() -> None:
        async with lock_b:
            await asyncio.sleep(0)
            async with lock_a:
                return

    try:
        await asyncio.wait_for(
            asyncio.gather(_task1(), _task2()),
            timeout=timeout_ms / 1000.0,
        )
        return False
    except asyncio.TimeoutError:
        return True


async def deadlock_avoided_fixed(timeout_ms: int = 120) -> bool:
    """Fixed pattern: both tasks acquire locks in same order."""

    lock_a = asyncio.Lock()
    lock_b = asyncio.Lock()

    async def _ordered_task() -> None:
        async with lock_a:
            await asyncio.sleep(0)
            async with lock_b:
                return

    await asyncio.wait_for(
        asyncio.gather(_ordered_task(), _ordered_task()),
        timeout=timeout_ms / 1000.0,
    )
    return True


# ---------------------------------------------------------------------------
# 4) Hidden blocking call vs executor offload
# ---------------------------------------------------------------------------


def _blocking_sleep(delay_ms: int) -> int:
    """Deliberately blocking function for demonstration."""

    time.sleep(delay_ms / 1000.0)
    return 1


async def hidden_blocking_broken(task_count: int, delay_ms: int) -> float:
    """Broken pattern: blocking call inside async coroutine freezes loop."""

    async def _slow() -> int:
        time.sleep(delay_ms / 1000.0)
        return 1

    started = time.perf_counter()
    await asyncio.gather(*(_slow() for _ in range(task_count)))
    return time.perf_counter() - started


async def hidden_blocking_fixed(task_count: int, delay_ms: int) -> float:
    """Fixed pattern: run blocking work in executor."""

    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    await asyncio.gather(
        *(loop.run_in_executor(None, _blocking_sleep, delay_ms) for _ in range(task_count))
    )
    return time.perf_counter() - started


# ---------------------------------------------------------------------------
# 5) Retry storm vs exponential backoff + jitter
# ---------------------------------------------------------------------------


def retry_schedule_without_jitter(retries: int, base_delay_s: float = 0.05) -> list[float]:
    """Deterministic schedule that aligns clients and can create retry storms."""

    return [base_delay_s * (2**attempt) for attempt in range(max(0, retries))]


def retry_schedule_with_jitter(retries: int, seed: int, base_delay_s: float = 0.05) -> list[float]:
    """Exponential schedule with jitter to spread client retries."""

    rng = random.Random(seed)
    schedule: list[float] = []
    for attempt in range(max(0, retries)):
        backoff = base_delay_s * (2**attempt)
        schedule.append(rng.uniform(0.0, backoff))
    return schedule


def client_schedules_without_jitter(client_count: int, retries: int) -> list[list[float]]:
    """All clients get the exact same schedule (bad)."""

    return [retry_schedule_without_jitter(retries) for _ in range(max(0, client_count))]


def client_schedules_with_jitter(client_count: int, retries: int, seed: int) -> list[list[float]]:
    """Each client receives a distinct jitter stream (good)."""

    return [retry_schedule_with_jitter(retries, seed + client_idx) for client_idx in range(max(0, client_count))]


# ---------------------------------------------------------------------------
# 6) Cancellation ignored vs handled
# ---------------------------------------------------------------------------


async def cancellation_ignored_broken(timeout_ms: int = 25) -> bool:
    """Broken pattern: CancelledError is swallowed, delaying shutdown.

    Returns:
        True  -> task terminated promptly (unexpected for broken pattern)
        False -> task failed to terminate promptly (expected)
    """

    async def _worker() -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Broken: cancellation is swallowed and work continues for a while.
            await asyncio.sleep((timeout_ms * 3) / 1000.0)
            return

    task = asyncio.create_task(_worker())
    await asyncio.sleep(0)
    task.cancel()

    try:
        await asyncio.wait_for(task, timeout=timeout_ms / 1000.0)
        terminated_promptly = True
    except asyncio.TimeoutError:
        terminated_promptly = False

    # Ensure task fully exits before returning.
    with suppress(BaseException):
        await task

    return terminated_promptly


async def cancellation_handled_fixed() -> bool:
    """Fixed pattern: cleanup and re-raise CancelledError."""

    async def _worker() -> None:
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Real code would flush state/release resources here.
            raise

    task = asyncio.create_task(_worker())
    await asyncio.sleep(0)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        return True
    return False


# ---------------------------------------------------------------------------
# Small CLI runner for quick manual checks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DemoSummary:
    fanout_broken_len: int
    fanout_fixed_len: int
    race_broken: int
    race_fixed: int
    deadlock_detected: bool
    deadlock_avoided: bool
    blocking_broken_s: float
    blocking_fixed_s: float
    cancellation_broken_prompt: bool
    cancellation_fixed_prompt: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concurrency failure-pattern demos")
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--delay-ms", type=int, default=10)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    fanout_broken = await unbounded_fanout_broken(args.items, args.delay_ms)
    fanout_fixed = await bounded_worker_pool_fixed(args.items, args.delay_ms, args.workers)

    race_broken = await race_condition_broken(args.items)
    race_fixed = await race_condition_fixed(args.items)

    deadlock_detected = await deadlock_detected_broken()
    deadlock_avoided = await deadlock_avoided_fixed()

    blocking_broken_s = await hidden_blocking_broken(args.workers, args.delay_ms)
    blocking_fixed_s = await hidden_blocking_fixed(args.workers, args.delay_ms)

    _ = client_schedules_without_jitter(3, args.retries)
    _ = client_schedules_with_jitter(3, args.retries, args.seed)

    cancellation_broken_prompt = await cancellation_ignored_broken()
    cancellation_fixed_prompt = await cancellation_handled_fixed()

    summary = DemoSummary(
        fanout_broken_len=len(fanout_broken),
        fanout_fixed_len=len(fanout_fixed),
        race_broken=race_broken,
        race_fixed=race_fixed,
        deadlock_detected=deadlock_detected,
        deadlock_avoided=deadlock_avoided,
        blocking_broken_s=blocking_broken_s,
        blocking_fixed_s=blocking_fixed_s,
        cancellation_broken_prompt=cancellation_broken_prompt,
        cancellation_fixed_prompt=cancellation_fixed_prompt,
    )

    print(summary)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
