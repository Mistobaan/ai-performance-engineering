from __future__ import annotations

import json
from pathlib import Path

from core.perf_core_base import PerformanceCoreBase


def _build_history_root(tmp_path: Path, run_entry: dict) -> Path:
    history_root = tmp_path / "history"
    history_root.mkdir()
    (history_root / "index.json").write_text(
        json.dumps({"suite_name": "tier1", "suite_version": 1, "runs": [run_entry]}),
        encoding="utf-8",
    )
    return history_root


def test_tier1_history_runs_surface_summary_read_warnings(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "bad_summary.json"
    summary_path.write_text("{not-json", encoding="utf-8")
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "summary_path": str(summary_path)},
    )
    core = PerformanceCoreBase(bench_root=tmp_path)
    monkeypatch.setattr(core, "_tier1_history_root", lambda: history_root)

    result = core.get_tier1_history_runs()

    assert result["total_runs"] == 0
    assert result["warnings"]
    assert any(str(summary_path) in warning for warning in result["warnings"])


def test_tier1_trends_surface_trend_snapshot_read_warnings(tmp_path: Path, monkeypatch) -> None:
    trend_path = tmp_path / "bad_trend.json"
    trend_path.write_text("[]", encoding="utf-8")
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "trend_snapshot_path": str(trend_path)},
    )
    core = PerformanceCoreBase(bench_root=tmp_path)
    monkeypatch.setattr(core, "_tier1_history_root", lambda: history_root)

    result = core.get_tier1_trends()

    assert result["warnings"]
    assert any(str(trend_path) in warning for warning in result["warnings"])


def test_tier1_target_history_surfaces_summary_read_warnings(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "bad_target_summary.json"
    summary_path.write_text("{not-json", encoding="utf-8")
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "summary_path": str(summary_path)},
    )
    core = PerformanceCoreBase(bench_root=tmp_path)
    monkeypatch.setattr(core, "_tier1_history_root", lambda: history_root)

    result = core.get_tier1_target_history(key="ch01:demo")

    assert result["run_count"] == 0
    assert result["warnings"]
    assert any(str(summary_path) in warning for warning in result["warnings"])
