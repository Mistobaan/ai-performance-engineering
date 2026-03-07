from __future__ import annotations

import json
from pathlib import Path

from cluster.analysis import build_coverage_delta


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loaders_resolve_current_run_dir_and_legacy_baseline_flat_dir(tmp_path: Path) -> None:
    cluster_dir = tmp_path / "cluster"
    current_structured = cluster_dir / "runs" / "2026-03-06_current" / "structured"
    legacy_structured = cluster_dir / "results" / "structured"

    _write_json(
        current_structured / "2026-03-06_current_benchmark_coverage_analysis.json",
        {
            "coverage_score_pct": 81,
            "advanced_coverage_score_pct": 63,
            "coverage_maturity": "good",
            "advanced_coverage": {"nvbandwidth": True, "vllm_request_rate": True},
        },
    )
    _write_json(
        current_structured / "2026-03-06_current_mlperf_alignment.json",
        {"overall_status": "aligned", "inference_track_ready": True, "training_track_ready": False},
    )
    _write_json(
        current_structured / "2026-03-06_current_cluster_scorecard.json",
        {"overall_score": 87.5, "pass_fail": "pass", "summary": {"status": "ok"}},
    )

    _write_json(
        legacy_structured / "2026-03-05_baseline_benchmark_coverage_analysis.json",
        {
            "coverage_score_pct": 74,
            "advanced_coverage_score_pct": 52,
            "coverage_maturity": "developing",
            "advanced_coverage": {"nvbandwidth": True, "vllm_request_rate": False},
        },
    )
    _write_json(
        legacy_structured / "2026-03-05_baseline_mlperf_alignment.json",
        {"overall_status": "partial", "inference_track_ready": True, "training_track_ready": False},
    )
    _write_json(
        legacy_structured / "2026-03-05_baseline_cluster_scorecard.json",
        {"overall_score": 79.0, "pass_fail": "warn", "summary": {"status": "partial"}},
    )

    current = build_coverage_delta._load_coverage(
        current_structured,
        "2026-03-06_current",
        cluster_dir=cluster_dir,
    )
    baseline = build_coverage_delta._load_coverage(
        current_structured,
        "2026-03-05_baseline",
        cluster_dir=cluster_dir,
    )
    baseline_mlperf = build_coverage_delta._load_mlperf(
        current_structured,
        "2026-03-05_baseline",
        cluster_dir=cluster_dir,
    )
    baseline_scorecard = build_coverage_delta._load_scorecard(
        current_structured,
        "2026-03-05_baseline",
        cluster_dir=cluster_dir,
    )

    assert current["path"].endswith("runs/2026-03-06_current/structured/2026-03-06_current_benchmark_coverage_analysis.json")
    assert baseline["path"].endswith("results/structured/2026-03-05_baseline_benchmark_coverage_analysis.json")
    assert baseline_mlperf["overall_status"] == "partial"
    assert baseline_scorecard["overall_score"] == 79.0


def test_missing_required_coverage_lists_all_search_paths(tmp_path: Path) -> None:
    cluster_dir = tmp_path / "cluster"
    structured_dir = cluster_dir / "runs" / "2026-03-06_current" / "structured"
    structured_dir.mkdir(parents=True)

    try:
        build_coverage_delta._load_coverage(structured_dir, "2026-03-05_missing", cluster_dir=cluster_dir)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert "runs/2026-03-05_missing/structured/2026-03-05_missing_benchmark_coverage_analysis.json" in message
    assert "results/structured/2026-03-05_missing_benchmark_coverage_analysis.json" in message
    assert "archive/runs/2026-03-05_missing/structured/2026-03-05_missing_benchmark_coverage_analysis.json" in message


def test_loaders_resolve_archived_baseline_run_dir(tmp_path: Path) -> None:
    cluster_dir = tmp_path / "cluster"
    current_structured = cluster_dir / "runs" / "2026-03-06_current" / "structured"
    archived_structured = cluster_dir / "archive" / "runs" / "2026-03-04_archived" / "structured"

    _write_json(
        current_structured / "2026-03-06_current_benchmark_coverage_analysis.json",
        {
            "coverage_score_pct": 90,
            "advanced_coverage_score_pct": 81,
            "coverage_maturity": "high",
            "advanced_coverage": {"nvbandwidth": True, "train_step_workload": True},
        },
    )
    _write_json(
        archived_structured / "2026-03-04_archived_benchmark_coverage_analysis.json",
        {
            "coverage_score_pct": 84,
            "advanced_coverage_score_pct": 45,
            "coverage_maturity": "good",
            "advanced_coverage": {"nvbandwidth": True, "train_step_workload": False},
        },
    )
    _write_json(
        archived_structured / "2026-03-04_archived_mlperf_alignment.json",
        {"overall_status": "partial", "inference_track_ready": True, "training_track_ready": False},
    )
    _write_json(
        archived_structured / "2026-03-04_archived_cluster_scorecard.json",
        {"overall_score": 61.0, "pass_fail": "warn", "summary": {"status": "partial"}},
    )

    baseline = build_coverage_delta._load_coverage(
        current_structured,
        "2026-03-04_archived",
        cluster_dir=cluster_dir,
    )
    baseline_mlperf = build_coverage_delta._load_mlperf(
        current_structured,
        "2026-03-04_archived",
        cluster_dir=cluster_dir,
    )
    baseline_scorecard = build_coverage_delta._load_scorecard(
        current_structured,
        "2026-03-04_archived",
        cluster_dir=cluster_dir,
    )

    assert baseline["path"].endswith(
        "archive/runs/2026-03-04_archived/structured/2026-03-04_archived_benchmark_coverage_analysis.json"
    )
    assert baseline_mlperf["path"].endswith(
        "archive/runs/2026-03-04_archived/structured/2026-03-04_archived_mlperf_alignment.json"
    )
    assert baseline_mlperf["overall_status"] == "partial"
    assert baseline_scorecard["overall_score"] == 61.0
