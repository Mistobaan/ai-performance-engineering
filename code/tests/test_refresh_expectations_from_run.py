import json
from pathlib import Path

from core.analysis.refresh_expectations_from_run import refresh_expectations_from_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_refresh_expectations_from_run_updates_selected_target(tmp_path: Path):
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    _write_json(
        chapter_dir / "expectations_b200.json",
        {
            "schema_version": 2,
            "hardware_key": "b200",
            "examples": {
                "gemm": {
                    "example": "gemm",
                    "type": "python",
                    "metrics": {
                        "baseline_time_ms": 10.0,
                        "best_optimized_time_ms": 5.0,
                        "best_speedup": 2.0,
                        "best_optimized_speedup": 2.0,
                        "is_regression": False,
                    },
                    "provenance": {
                        "git_commit": "oldsha",
                        "hardware_key": "b200",
                        "profile_name": "none",
                        "timestamp": "2026-03-10T00:00:00Z",
                        "iterations": 20,
                        "warmup_iterations": 5,
                    },
                    "metadata": {
                        "optimization_goal": "speed",
                        "best_optimization_speedup": 2.0,
                    },
                }
            },
        },
    )

    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "timestamp": "2026-03-11T00:00:00Z",
            "results": [
                {
                    "chapter": "ch01",
                    "status": "completed",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "python",
                            "expectation": {
                                "entry": {
                                    "example": "gemm",
                                    "type": "python",
                                    "metrics": {
                                        "baseline_time_ms": 10.0,
                                        "best_optimized_time_ms": 4.0,
                                        "best_speedup": 2.5,
                                        "best_optimized_speedup": 2.5,
                                        "is_regression": False,
                                    },
                                    "provenance": {
                                        "git_commit": "newsha",
                                        "hardware_key": "b200",
                                        "profile_name": "minimal",
                                        "timestamp": "2026-03-11T00:00:00Z",
                                        "iterations": 20,
                                        "warmup_iterations": 5,
                                    },
                                    "metadata": {
                                        "optimization_goal": "speed",
                                        "best_optimization_speedup": 2.5,
                                    },
                                }
                            },
                        }
                    ],
                }
            ],
        },
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm"},
        validity_profile="strict",
        accept_regressions=True,
        dry_run=False,
    )

    assert summary["counts"]["applied"] == 1
    assert summary["counts"]["updated"] == 1
    updated = json.loads((chapter_dir / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = updated["examples"]["gemm"]
    assert entry["metrics"]["best_speedup"] == 2.5
    assert entry["provenance"]["git_commit"] == "newsha"
    assert entry["provenance"]["validity_profile"] == "strict"
    assert entry["provenance"]["execution_environment"] in {"bare_metal", "virtualized", "unknown"}


def test_refresh_expectations_from_run_skips_missing_preview_entry(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / "ch01").mkdir(parents=True)
    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "results": [
                {
                    "chapter": "ch01",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "python",
                        }
                    ],
                }
            ]
        },
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm"},
        dry_run=True,
    )

    assert summary["counts"]["skipped"] == 1


def test_refresh_expectations_from_run_infers_validity_profile_from_run_events(tmp_path: Path):
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    _write_json(
        chapter_dir / "expectations_b200.json",
        {
            "schema_version": 2,
            "hardware_key": "b200",
            "examples": {
                "gemm_cuda": {
                    "example": "gemm",
                    "type": "cuda",
                    "metrics": {
                        "baseline_time_ms": 10.0,
                        "best_optimized_time_ms": 5.0,
                        "best_speedup": 2.0,
                        "best_optimized_speedup": 2.0,
                        "is_regression": False,
                    },
                    "provenance": {
                        "git_commit": "oldsha",
                        "hardware_key": "b200",
                        "profile_name": "none",
                        "timestamp": "2026-03-10T00:00:00Z",
                        "iterations": 20,
                        "warmup_iterations": 5,
                    },
                    "metadata": {
                        "optimization_goal": "speed",
                        "best_optimization_speedup": 2.0,
                    },
                }
            },
        },
    )

    run_dir = repo_root / "artifacts" / "runs" / "rerun"
    results_json = run_dir / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "timestamp": "2026-03-11T00:00:00Z",
            "results": [
                {
                    "chapter": "ch01",
                    "status": "completed",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "cuda",
                            "expectation": {
                                "entry": {
                                    "example": "gemm",
                                    "type": "cuda",
                                    "metrics": {
                                        "baseline_time_ms": 10.0,
                                        "best_optimized_time_ms": 5.0,
                                        "best_speedup": 2.0,
                                        "best_optimized_speedup": 2.0,
                                        "is_regression": False,
                                    },
                                    "provenance": {
                                        "git_commit": "newsha",
                                        "hardware_key": "b200",
                                        "profile_name": "minimal",
                                        "timestamp": "2026-03-11T00:00:00Z",
                                        "iterations": 20,
                                        "warmup_iterations": 5,
                                    },
                                    "metadata": {
                                        "optimization_goal": "speed",
                                        "best_optimization_speedup": 2.0,
                                    },
                                }
                            },
                        }
                    ],
                }
            ],
        },
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "benchmark_events.jsonl").write_text(
        json.dumps({"event_type": "run_start", "validity_profile": "strict"}) + "\n",
        encoding="utf-8",
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm_cuda"},
        dry_run=False,
    )

    assert summary["validity_profile"] == "strict"
    updated = json.loads((chapter_dir / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = updated["examples"]["gemm_cuda"]
    assert entry["provenance"]["validity_profile"] == "strict"
