from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import mcp.mcp_server as mcp_server
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness import run_benchmarks


def test_progress_recorder_writes_payload(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_001", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="baseline_timing",
            phase_index=1,
            total_phases=2,
            step="ch01:demo",
        )
    )
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_001"
    assert data["current"]["phase"] == "baseline_timing"
    assert data["current"]["step"] == "ch01:demo"
    assert data["current"]["run_id"] == "run_001"


def test_job_status_includes_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000000"
    progress_path = run_dir / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_002", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="optimized_timing",
            phase_index=2,
            total_phases=2,
            step="ch01:demo",
        )
    )

    store = mcp_server.JOB_STORE

    def runner():
        return {"ok": True}

    ticket = store.queue_job(
        "run_benchmarks",
        runner,
        run_metadata={
            "run_id": "run_002",
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
        },
    )
    job_id = ticket["job_id"]
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["progress"]["phase"] == "optimized_timing"
        assert payload["progress"]["run_id"] == "run_002"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_progress_phases_include_llm() -> None:
    phases = run_benchmarks.PROGRESS_PHASES
    for key in (
        "llm_analysis",
        "llm_patch_apply",
        "llm_patch_rebenchmark",
        "llm_patch_verify",
        "llm_explain",
    ):
        assert key in phases


def test_global_progress_percent_uses_run_offset_instead_of_resetting() -> None:
    corrected = run_benchmarks._compute_global_progress_percent(
        completed_benchmarks=0,
        total_benchmarks=276,
        phase_index=4,
        total_phases=run_benchmarks.PROGRESS_TOTAL_PHASES,
        benchmark_offset=236,
    )
    local_reset = ((0 + ((4 - 1) / run_benchmarks.PROGRESS_TOTAL_PHASES)) / 14) * 100.0

    assert corrected is not None
    assert corrected > 80.0
    assert local_reset < 2.0


def test_job_status_running_without_artifacts_reports_effective_queued(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "missing_run"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_missing",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "queued"
        assert payload["reported_status"] == "running"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_queued_with_progress_reports_effective_running(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000003"
    progress_path = run_dir / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_003", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="preflight",
            phase_index=1,
            total_phases=4,
            step="ch10:atomic_reduction",
        )
    )

    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "queued",
            "submitted_at": time.time(),
            "run_id": "run_003",
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
            "note": "Waiting for MCP queue runner.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "running"
        assert payload["reported_status"] == "queued"
        assert payload["progress"]["phase"] == "preflight"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_running_via_queue_runner_keeps_running_without_run_dir(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "pending_run_dir"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_pending",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
            "note": "Running via MCP queue runner.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "running"
        assert "reported_status" not in payload
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_running_without_progress_with_existing_run_dir_reports_queued(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "existing_run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_existing_dir",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
            "note": "Poll job status to track completion.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "queued"
        assert payload["reported_status"] == "running"
    finally:
        with store._lock:
            store._store.pop(job_id, None)
