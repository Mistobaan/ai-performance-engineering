from __future__ import annotations

import argparse
import asyncio

from labs.python_concurrency import hybrid_three_stage_pipeline as hybrid
from labs.python_concurrency import taskrun_round1_asyncio as round1
from labs.python_concurrency import taskrun_round2_controls as round2
from labs.python_concurrency import taskrun_round3_idempotency as round3


def test_round1_terminal_state_invariants_ordering_and_retries() -> None:
    items = [
        {"id": "i0", "payload": "ok", "duration_ms": 5, "should_fail": False},
        {"id": "i1", "payload": "slow", "duration_ms": 80, "should_fail": False},
        {"id": "i2", "payload": "boom", "duration_ms": 5, "should_fail": True},
    ]

    results = asyncio.run(round1.run_pipeline(items=items, max_workers=2, timeout_ms=30, retries=1))

    # Ordering invariant: result slots map 1:1 to input indexes.
    assert [r.idx for r in results] == [0, 1, 2]
    assert [r.item_id for r in results] == ["i0", "i1", "i2"]

    # Retry/timeout invariants.
    assert results[0].ok is True
    assert results[0].attempts == 1
    assert results[0].timed_out is False

    assert results[1].ok is False
    assert results[1].timed_out is True
    assert results[1].attempts == 2

    assert results[2].ok is False
    assert results[2].timed_out is False
    assert results[2].attempts == 2
    assert results[2].error is not None
    assert "RuntimeError" in results[2].error

    summary = round1.summarize(results)
    assert summary["total"] == 3
    assert summary["success"] == 1
    assert summary["failed"] == 2
    assert summary["timed_out"] == 1


def test_round2_global_timeout_preserves_order_and_marks_cancellations() -> None:
    items = [
        {"id": "u0", "delay_ms": 200, "fail_rate": 0.0},
        {"id": "u1", "delay_ms": 200, "fail_rate": 0.0},
        {"id": "u2", "delay_ms": 200, "fail_rate": 0.0},
        {"id": "u3", "delay_ms": 200, "fail_rate": 0.0},
        {"id": "u4", "delay_ms": 200, "fail_rate": 0.0},
    ]

    # Use one worker and a short global timeout so backlog is cancelled.
    args = argparse.Namespace(
        max_workers=1,
        rps=0.0,
        timeout_ms=1000,
        retries=0,
        global_timeout_ms=50,
        seed=123,
    )

    results = asyncio.run(round2.run_pipeline(args=args, items=items))

    assert [r.idx for r in results] == [0, 1, 2, 3, 4]
    assert [r.item_id for r in results] == ["u0", "u1", "u2", "u3", "u4"]

    # At least one item should complete (the in-flight one), and the backlog
    # should be cancelled explicitly.
    cancelled = [r for r in results if r.status == round2.CANCELLED]
    success = [r for r in results if r.status == round2.SUCCESS]

    assert len(success) >= 1
    assert len(cancelled) >= 1

    summary = round2.summarize(results)
    assert summary["queued"] == 5
    assert summary["completed"] == 5
    assert summary["cancelled"] == len(cancelled)
    assert summary["started"] >= len(success)

    # Terminal-state invariant: no result should remain in a non-terminal state.
    assert all(
        r.status in {round2.SUCCESS, round2.FAILED, round2.TIMED_OUT, round2.CANCELLED}
        for r in results
    )


def test_round3_dedupe_and_poison_terminal_states() -> None:
    items = [
        {
            "id": "a",
            "idempotency_key": "key-1",
            "payload": "alpha",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        {
            "id": "b",
            "idempotency_key": "key-1",
            "payload": "alpha-dup",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        {
            "id": "c",
            "idempotency_key": "key-2",
            "payload": "always-transient-fail",
            "delay_ms": 5,
            "transient_fail_rate": 1.0,
            "permanent_fail": False,
        },
    ]

    results, shared = asyncio.run(
        round3.run_pipeline(
            items=items,
            max_workers=2,
            timeout_ms=100,
            retries=1,
        )
    )

    assert [r.idx for r in results] == [0, 1, 2]

    # Dedupe invariant: exactly one success and one deduped for duplicated key-1.
    key1 = [r for r in results if r.idempotency_key == "key-1"]
    assert len(key1) == 2
    assert sorted(r.status for r in key1) == [round3.DEDUPED, round3.SUCCESS]

    # Poison invariant: always-transient item should exhaust retries and route to poison.
    key2 = [r for r in results if r.idempotency_key == "key-2"]
    assert len(key2) == 1
    assert key2[0].status == round3.POISON
    assert key2[0].attempts == 2

    summary = round3.summarize(results, shared)
    assert summary["total"] == 3
    assert summary["success"] == 1
    assert summary["deduped"] == 1
    assert summary["poison"] == 1
    assert summary["retried_attempts"] == 1

    assert len(shared.poison_items) == 1
    assert shared.poison_items[0].idempotency_key == "key-2"


def test_hybrid_pipeline_terminal_states_ordering_and_stage_failures() -> None:
    items = [
        {
            "id": "h0",
            "payload": "hello world",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail": False,
            "write_fail": False,
        },
        {
            "id": "h1",
            "payload": "fetch fail",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail": True,
            "write_fail": False,
        },
        {
            "id": "h2",
            "payload": "write fail",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail": False,
            "write_fail": True,
        },
    ]

    args = argparse.Namespace(
        stage_a_workers=2,
        stage_b_workers=2,
        stage_c_workers=2,
        queue_size=2,
        fetch_timeout_ms=100,
        write_timeout_ms=100,
        cpu_rounds=4_000,
    )

    results = asyncio.run(hybrid.run_pipeline(args=args, items=items))

    assert [r.idx for r in results] == [0, 1, 2]
    assert [r.item_id for r in results] == ["h0", "h1", "h2"]

    assert results[0].status == hybrid.SUCCESS
    assert results[0].output is not None
    assert results[0].error is None

    assert results[1].status == hybrid.FAILED_STAGE_A
    assert results[1].error is not None

    assert results[2].status == hybrid.FAILED_STAGE_C
    assert results[2].error is not None

    summary = hybrid.summarize(results)
    assert summary["total"] == 3
    assert summary["success"] == 1
    assert summary["failed_stage_a"] == 1
    assert summary["failed_stage_b"] == 0
    assert summary["failed_stage_c"] == 1
