from __future__ import annotations

import asyncio

from labs.python_concurrency import all_in_one_pipeline as aio_all


def test_all_in_one_terminal_states_and_ordering() -> None:
    items = [
        {
            "id": "x0",
            "idempotency_key": "k0",
            "payload": "alpha payload",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "none",
            "write_fail_mode": "none",
        },
        {
            "id": "x1",
            "idempotency_key": "k0",
            "payload": "alpha duplicate",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "none",
            "write_fail_mode": "none",
        },
        {
            "id": "x2",
            "idempotency_key": "k2",
            "payload": "fetch transient once",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "transient_once",
            "write_fail_mode": "none",
        },
        {
            "id": "x3",
            "idempotency_key": "k3",
            "payload": "write transient always",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "none",
            "write_fail_mode": "transient_always",
        },
        {
            "id": "x4",
            "idempotency_key": "k4",
            "payload": "cpu fail",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "cpu_fail": True,
            "fetch_fail_mode": "none",
            "write_fail_mode": "none",
        },
        {
            "id": "x5",
            "idempotency_key": "k5",
            "payload": "fetch permanent",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "permanent",
            "write_fail_mode": "none",
        },
        {
            "id": "x6",
            "idempotency_key": "k6",
            "payload": "write permanent",
            "fetch_delay_ms": 5,
            "write_delay_ms": 5,
            "fetch_fail_mode": "none",
            "write_fail_mode": "permanent",
        },
        {
            "id": "x7",
            "idempotency_key": "k7",
            "payload": "fetch timeout",
            "fetch_delay_ms": 1000,
            "write_delay_ms": 5,
            "fetch_fail_mode": "none",
            "write_fail_mode": "none",
        },
    ]

    results, shared = asyncio.run(
        aio_all.run_pipeline(
            items=items,
            stage_a_workers=3,
            stage_b_workers=2,
            stage_c_workers=2,
            queue_size=4,
            rps=0.0,
            fetch_inflight=3,
            write_inflight=2,
            fetch_timeout_ms=50,
            write_timeout_ms=100,
            cpu_timeout_ms=1000,
            fetch_retries=1,
            write_retries=1,
            cpu_rounds=5000,
            global_timeout_ms=0,
            seed=7,
        )
    )

    assert [r.idx for r in results] == list(range(len(items)))
    assert [r.item_id for r in results] == [item["id"] for item in items]

    key0_records = [record for record in results if record.idempotency_key == "k0"]
    assert len(key0_records) == 2
    assert sorted(record.status for record in key0_records) == [aio_all.DEDUPED, aio_all.SUCCESS]

    by_id = {record.item_id: record for record in results}
    assert by_id["x2"].status == aio_all.SUCCESS
    assert by_id["x3"].status == aio_all.POISON
    assert by_id["x4"].status == aio_all.FAILED_STAGE_B
    assert by_id["x5"].status == aio_all.FAILED_STAGE_A
    assert by_id["x6"].status == aio_all.FAILED_STAGE_C
    assert by_id["x7"].status == aio_all.TIMED_OUT

    summary = aio_all.summarize(results, shared)
    assert summary["total"] == 8
    assert summary[aio_all.SUCCESS] == 2
    assert summary[aio_all.DEDUPED] == 1
    assert summary[aio_all.POISON] == 1
    assert summary[aio_all.FAILED_STAGE_A] == 1
    assert summary[aio_all.FAILED_STAGE_B] == 1
    assert summary[aio_all.FAILED_STAGE_C] == 1
    assert summary[aio_all.TIMED_OUT] == 1
    assert summary[aio_all.CANCELLED] == 0

    # Invariant: every result is terminal.
    assert all(record.status in aio_all.TERMINAL_STATUSES for record in results)


def test_all_in_one_global_timeout_marks_cancellations() -> None:
    items = [
        {
            "id": f"c{i}",
            "idempotency_key": f"kc{i}",
            "payload": "slow",
            "fetch_delay_ms": 200,
            "write_delay_ms": 200,
            "fetch_fail_mode": "none",
            "write_fail_mode": "none",
        }
        for i in range(20)
    ]

    results, shared = asyncio.run(
        aio_all.run_pipeline(
            items=items,
            stage_a_workers=1,
            stage_b_workers=1,
            stage_c_workers=1,
            queue_size=2,
            rps=0.0,
            fetch_inflight=1,
            write_inflight=1,
            fetch_timeout_ms=1000,
            write_timeout_ms=1000,
            cpu_timeout_ms=1000,
            fetch_retries=0,
            write_retries=0,
            cpu_rounds=2000,
            global_timeout_ms=70,
            seed=5,
        )
    )

    assert len(results) == len(items)
    assert [r.idx for r in results] == list(range(len(items)))

    cancelled = [record for record in results if record.status == aio_all.CANCELLED]
    assert len(cancelled) >= 1

    summary = aio_all.summarize(results, shared)
    assert summary["completed"] == len(items)
    assert summary[aio_all.CANCELLED] == len(cancelled)

    assert all(record.status in aio_all.TERMINAL_STATUSES for record in results)
