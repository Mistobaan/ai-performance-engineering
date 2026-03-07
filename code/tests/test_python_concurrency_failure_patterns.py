from __future__ import annotations

import asyncio

from labs.python_concurrency import failure_patterns as fp


def test_unbounded_and_bounded_fanout_produce_same_outputs_for_small_input() -> None:
    broken = asyncio.run(fp.unbounded_fanout_broken(item_count=25, delay_ms=1))
    fixed = asyncio.run(fp.bounded_worker_pool_fixed(item_count=25, delay_ms=1, workers=4))
    assert broken == fixed


def test_race_condition_broken_loses_updates_but_fixed_is_exact() -> None:
    broken = asyncio.run(fp.race_condition_broken(task_count=30))
    fixed = asyncio.run(fp.race_condition_fixed(task_count=30))

    # Broken path is deterministic in this demo: all tasks captured same value.
    assert broken == 1
    assert fixed == 30


def test_deadlock_broken_is_detected_and_fixed_completes() -> None:
    assert asyncio.run(fp.deadlock_detected_broken(timeout_ms=30)) is True
    assert asyncio.run(fp.deadlock_avoided_fixed(timeout_ms=80)) is True


def test_hidden_blocking_fixed_is_faster_than_broken() -> None:
    broken_s = asyncio.run(fp.hidden_blocking_broken(task_count=6, delay_ms=20))
    fixed_s = asyncio.run(fp.hidden_blocking_fixed(task_count=6, delay_ms=20))

    # Broken path tends toward serial cost (N * delay); fixed should be materially lower.
    assert fixed_s < broken_s


def test_retry_schedules_with_and_without_jitter() -> None:
    without_jitter = fp.client_schedules_without_jitter(client_count=4, retries=4)
    with_jitter = fp.client_schedules_with_jitter(client_count=4, retries=4, seed=123)

    # Without jitter every client retry schedule is identical.
    assert all(schedule == without_jitter[0] for schedule in without_jitter)

    # With jitter at least one client schedule differs.
    assert any(schedule != with_jitter[0] for schedule in with_jitter[1:])


def test_cancellation_handling_broken_vs_fixed() -> None:
    broken_terminated_promptly = asyncio.run(fp.cancellation_ignored_broken(timeout_ms=20))
    fixed_terminated_promptly = asyncio.run(fp.cancellation_handled_fixed())

    assert broken_terminated_promptly is False
    assert fixed_terminated_promptly is True
