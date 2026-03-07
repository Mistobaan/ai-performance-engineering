from __future__ import annotations

import json
import shutil
from pathlib import Path

import core.cluster.runner as cluster_runner


def test_cluster_runner_build_canonical_package_returns_explicit_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    scripts_dir = cluster_root / "scripts"
    (cluster_root / "runs").mkdir(parents=True)
    scripts_dir.mkdir(parents=True)

    canonical_run = "2026-03-05_localhost_modern_profile_r24_full20b"
    comparison_run = "2026-03-04_localhost_modern_profile_r22_fastcanon"
    historical_run = "2026-02-10_full_suite_e2e_wire_qf_mon"

    canonical_run_dir = cluster_root / "runs" / canonical_run
    (canonical_run_dir / "structured").mkdir(parents=True)
    (canonical_run_dir / "raw").mkdir(parents=True)
    (canonical_run_dir / "figures").mkdir(parents=True)
    (canonical_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (canonical_run_dir / "figures" / f"{canonical_run}_cluster_story_dashboard.png").write_text("png", encoding="utf-8")

    comparison_run_dir = cluster_root / "runs" / comparison_run
    (comparison_run_dir / "structured").mkdir(parents=True)
    (comparison_run_dir / "raw").mkdir(parents=True)
    (comparison_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (comparison_run_dir / "raw" / f"{comparison_run}_suite.log").write_text("log", encoding="utf-8")
    (cluster_root / "field-report-localhost.md").write_text("localhost report\n", encoding="utf-8")
    (cluster_root / "field-report-localhost-notes.md").write_text("localhost notes\n", encoding="utf-8")
    (cluster_root / "multi_node_harden_and_verify_next_steps.md").write_text("multinode notes\n", encoding="utf-8")
    (cluster_root / "field-report.md").write_text(f"historical `{historical_run}`\n", encoding="utf-8")

    source_script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "build_canonical_package.py"
    shutil.copy2(source_script, scripts_dir / "build_canonical_package.py")

    monkeypatch.setattr(cluster_runner, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(cluster_runner, "_cluster_root", lambda: cluster_root)

    result = cluster_runner.build_canonical_package(
        canonical_run_id=canonical_run,
        comparison_run_ids=[comparison_run],
        historical_run_ids=[historical_run],
        output_dir="cluster/canonical_package/test_pkg",
        timeout_seconds=60,
    )

    assert result["success"] is True, result.get("stderr") or result
    assert Path(result["output_dir"]) == (repo_root / "cluster" / "canonical_package" / "test_pkg").resolve()
    assert Path(result["package_manifest_path"]).exists()
    assert Path(result["package_readme_path"]).exists()
    assert Path(result["cleanup_keep_run_ids_path"]).exists()
    assert Path(result["historical_reference_path"]).exists()

    manifest = json.loads(Path(result["package_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["canonical_run_id"] == canonical_run
    assert comparison_run in manifest["comparison_run_ids"]
    assert historical_run in manifest["historical_run_ids"]
