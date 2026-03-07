from __future__ import annotations

import argparse
import asyncio
import random

from labs.python_concurrency import hybrid_three_stage_pipeline as hybrid
from labs.python_concurrency import taskrun_round2_controls as round2
from labs.python_concurrency import taskrun_round3_idempotency as round3


def test_high_volume_bounded_queue_saturation_has_complete_terminal_states() -> None:
    """Stress bounded handoff queues with many items and tiny queue capacity.

    This validates that saturated queues do not lose items and every input ends in
    a terminal record.
    """

    item_count = 220
    items = [
        {
            "id": f"hv-{i}",
            "payload": f"payload-{i}",
            "fetch_delay_ms": 1,
            "write_delay_ms": 1,
            "fetch_fail": False,
            "write_fail": False,
        }
        for i in range(item_count)
    ]

    args = argparse.Namespace(
        stage_a_workers=3,
        stage_b_workers=3,
        stage_c_workers=3,
        queue_size=2,  # intentionally tiny to force saturation/backpressure
        fetch_timeout_ms=200,
        write_timeout_ms=200,
        cpu_rounds=2_000,
    )

    results = asyncio.run(hybrid.run_pipeline(args=args, items=items))
    summary = hybrid.summarize(results)

    assert len(results) == item_count
    assert [r.idx for r in results] == list(range(item_count))
    assert all(r.status in {hybrid.SUCCESS, hybrid.FAILED_STAGE_A, hybrid.FAILED_STAGE_B, hybrid.FAILED_STAGE_C} for r in results)
    assert summary["total"] == item_count
    assert summary["success"] == item_count


def test_repeated_cancellation_races_preserve_terminal_invariants() -> None:
    """Run repeated deadline races to verify cancellation stability.

    This test stresses stop-event + queue-drain logic across multiple iterations.
    """

    race_rounds = 20
    items = [{"id": f"c-{i}", "delay_ms": 40, "fail_rate": 0.0} for i in range(35)]

    for _ in range(race_rounds):
        args = argparse.Namespace(
            max_workers=2,
            rps=0.0,
            timeout_ms=1_000,
            retries=0,
            global_timeout_ms=8,
            seed=123,
        )

        results = asyncio.run(round2.run_pipeline(args=args, items=items))
        summary = round2.summarize(results)

        assert len(results) == len(items)
        assert [r.idx for r in results] == list(range(len(items)))
        assert all(
            r.status in {round2.SUCCESS, round2.FAILED, round2.TIMED_OUT, round2.CANCELLED}
            for r in results
        )
        assert summary["queued"] == len(items)
        assert summary["completed"] == len(items)
        assert summary["success"] + summary["failed"] + summary["timed_out"] + summary["cancelled"] == len(items)
        assert summary["cancelled"] >= 1


def test_round3_cross_invariants_dedupe_poison_and_global_deadline() -> None:
    """Stress dedupe/poison logic under a global deadline in one run.

    This validates cross-invariants:
    - dedupe status appears for duplicate keys,
    - poison status appears for exhausted transient failures,
    - cancelled status appears for backlog items after global timeout.
    """

    random.seed(123)

    items = [
        # Early duplicate groups to guarantee dedupe path exercises quickly.
        {
            "id": "dup-a-0",
            "idempotency_key": "dup-a",
            "payload": "a0",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        {
            "id": "dup-a-1",
            "idempotency_key": "dup-a",
            "payload": "a1",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        # Early transient-fail keys to force poison routing.
        {
            "id": "poison-a",
            "idempotency_key": "poison-a",
            "payload": "pa",
            "delay_ms": 5,
            "transient_fail_rate": 1.0,
            "permanent_fail": False,
        },
        {
            "id": "dup-b-0",
            "idempotency_key": "dup-b",
            "payload": "b0",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        {
            "id": "dup-b-1",
            "idempotency_key": "dup-b",
            "payload": "b1",
            "delay_ms": 5,
            "transient_fail_rate": 0.0,
            "permanent_fail": False,
        },
        {
            "id": "poison-b",
            "idempotency_key": "poison-b",
            "payload": "pb",
            "delay_ms": 5,
            "transient_fail_rate": 1.0,
            "permanent_fail": False,
        },
    ]

    # Add a large backlog so global timeout cancels queued work deterministically.
    for i in range(180):
        items.append(
            {
                "id": f"tail-{i}",
                "idempotency_key": f"tail-key-{i}",
                "payload": f"tail-{i}",
                "delay_ms": 60,
                "transient_fail_rate": 0.0,
                "permanent_fail": False,
            }
        )

    results, shared = asyncio.run(
        round3.run_pipeline(
            items=items,
            max_workers=2,
            timeout_ms=20,
            retries=1,
            global_timeout_ms=90,
        )
    )
    summary = round3.summarize(results, shared)

    assert len(results) == len(items)
    assert [r.idx for r in results] == list(range(len(items)))
    assert all(
        r.status in {round3.SUCCESS, round3.DEDUPED, round3.POISON, round3.FAILED, round3.CANCELLED}
        for r in results
    )

    assert summary["total"] == len(items)
    assert summary["deduped"] >= 2
    assert summary["poison"] >= 1
    assert summary["cancelled"] >= 1
    assert summary["success"] >= 1
    assert summary["success"] + summary["deduped"] + summary["poison"] + summary["failed"] + summary["cancelled"] == len(items)
    assert len(shared.poison_items) == summary["poison"]
