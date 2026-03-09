import json
from types import SimpleNamespace

from typer.testing import CliRunner

from core.engine import get_engine, reset_engine
from dashboard.api import server


def test_configure_engine_uses_data_file(tmp_path):
    data = {
        "timestamp": "2025-01-01 00:00:00",
        "results": [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "example_a",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "baseline_gpu_metrics": {"power_draw_w": 250},
                        "optimizations": [],
                        "status": "succeeded",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "benchmark_test_results.json"
    path.write_text(json.dumps(data))

    reset_engine()
    server._configure_engine(path)
    result = get_engine().benchmark.data()

    assert result["summary"]["total_benchmarks"] == 1
    assert result["benchmarks"][0]["name"] == "example_a"

    reset_engine()


def test_dashboard_cli_has_serve_command():
    runner = CliRunner()
    result = runner.invoke(server.cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the dashboard API server." in result.output


def test_engine_exposes_tier1_history_and_trends(tmp_path, monkeypatch):
    history_root = tmp_path / "artifacts" / "history" / "tier1"
    run_dir = history_root / "20260309_010000_tier1_local"
    run_dir.mkdir(parents=True)

    summary_path = run_dir / "summary.json"
    regression_path = run_dir / "regression_summary.json"
    trend_path = run_dir / "trend_snapshot.json"
    index_path = history_root / "index.json"

    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "run_id": "20260309_010000_tier1_local",
                "generated_at": "2026-03-09T01:00:00",
                "targets": [
                    {
                        "key": "flashattention4_alibi",
                        "target": "labs/flashattention4:flashattention4_alibi",
                        "category": "attention",
                        "status": "succeeded",
                        "best_speedup": 12.5,
                        "artifacts": {
                            "baseline_nsys_rep": "artifacts/runs/demo/profiles/flash.nsys-rep",
                        },
                    }
                ],
                "summary": {
                    "target_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                    "avg_speedup": 12.5,
                    "median_speedup": 12.5,
                    "geomean_speedup": 12.5,
                    "representative_speedup": 12.5,
                    "max_speedup": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    regression_path.write_text(
        json.dumps(
            {
                "baseline_run_id": "20260308_225441_tier1_manual",
                "current_run_id": "20260309_010000_tier1_local",
                "regressions": [],
                "improvements": [{"key": "flashattention4_alibi", "delta_pct": 4.2}],
                "new_targets": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "run_count": 1,
                "latest_run_id": "20260309_010000_tier1_local",
                "best_speedup_seen": 12.5,
                "history": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "generated_at": "2026-03-09T01:00:00",
                        "avg_speedup": 12.5,
                        "median_speedup": 12.5,
                        "geomean_speedup": 12.5,
                        "representative_speedup": 12.5,
                        "max_speedup": 12.5,
                        "succeeded": 1,
                        "failed": 0,
                        "skipped": 0,
                        "missing": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "history_root": str(history_root),
                "runs": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(run_dir / "regression_summary.md"),
                        "regression_json_path": str(regression_path),
                        "trend_snapshot_path": str(trend_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "core.benchmark.suites.tier1.load_tier1_suite",
        lambda *args, **kwargs: SimpleNamespace(history_root=str(history_root)),
    )

    reset_engine()
    history = get_engine().benchmark.tier1_history()
    trends = get_engine().benchmark.tier1_trends()
    target_history = get_engine().benchmark.tier1_target_history(key="flashattention4_alibi")

    assert history["total_runs"] == 1
    assert history["latest_run_id"] == "20260309_010000_tier1_local"
    assert history["latest"]["run"]["representative_speedup"] == 12.5
    assert history["latest"]["improvements"][0]["key"] == "flashattention4_alibi"
    assert history["latest"]["run"]["regression_summary_json_path"] == str(regression_path)
    assert trends["latest_run_id"] == "20260309_010000_tier1_local"
    assert trends["best_speedup_seen"] == 12.5
    assert target_history["selected_key"] == "flashattention4_alibi"
    assert target_history["run_count"] == 1
    assert target_history["history"][0]["target"] == "labs/flashattention4:flashattention4_alibi"
    assert target_history["history"][0]["best_speedup"] == 12.5
    assert target_history["history"][0]["artifacts"]["baseline_nsys_rep"].endswith(
        "/artifacts/runs/demo/profiles/flash.nsys-rep"
    )

    reset_engine()
