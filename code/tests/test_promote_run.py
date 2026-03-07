from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_promotable_run(cluster_root: Path, run_id: str, label: str) -> Path:
    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    raw = run_dir / "raw" / f"{run_id}_suite"
    figures = run_dir / "figures"
    structured.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", {"run_id": run_id})
    _write_json(
        structured / f"{run_id}_suite_steps.json",
        [
            {"name": "preflight_services", "exit_code": 0, "start_time": "2026-03-05T21:08:24+00:00"},
            {"name": "discovery", "exit_code": 0, "start_time": "2026-03-05T21:08:25+00:00"},
            {"name": "hang_triage_bundle", "exit_code": 0, "start_time": "2026-03-05T21:08:28+00:00"},
            {"name": "connectivity_probe", "exit_code": 0, "start_time": "2026-03-05T21:08:29+00:00"},
            {"name": "nccl_env_sensitivity", "exit_code": 0, "start_time": "2026-03-05T21:08:47+00:00"},
            {"name": "vllm_serve_sweep", "exit_code": 0, "start_time": "2026-03-05T21:09:53+00:00"},
            {"name": "validate_required_artifacts", "exit_code": 0, "start_time": "2026-03-05T21:17:25+00:00"},
            {"name": "manifest_refresh", "exit_code": 0, "start_time": "2026-03-05T21:17:26+00:00"},
        ],
    )
    _write_json(structured / f"{run_id}_{label}_meta.json", {"commands": {"nvidia_smi_l": {"stdout": "GPU 0: Test GPU"}}})
    _write_json(structured / f"{run_id}_{label}_hang_triage_readiness.json", {"status": "ok"})
    _write_json(
        structured / f"{run_id}_torchrun_connectivity_probe.json",
        {"status": "ok", "world_size": 1, "ranks": [{"barrier_ms": [0.1], "payload_probe": {"algbw_gbps": 111.0}}]},
    )
    _write_json(structured / f"{run_id}_nccl_env_sensitivity.json", {"status": "ok", "failure_count": 0})
    _write_json(structured / f"{run_id}_node1_nccl.json", {"results": [{"algbw_gbps": 2222.0, "size_bytes": 67108864}]})
    _write_json(structured / f"{run_id}_preflight_services.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_node_parity_summary.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_fio.json", {"status": "ok"})
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text(
        "concurrency,total_token_throughput,mean_ttft_ms,p99_ttft_ms,p99_tpot_ms\n1,100.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    (structured / f"{run_id}_{label}_vllm_serve_sweep.jsonl").write_text("{}\n", encoding="utf-8")
    (raw / "suite.log").write_text("ok\n", encoding="utf-8")
    (figures / f"{run_id}_cluster_story_dashboard.png").write_text("png\n", encoding="utf-8")
    return run_dir


def test_promote_run_materializes_published_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-06_localhost_common_answer_fast_r4"
    _seed_promotable_run(cluster_root, run_id, "localhost")

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--skip-render-localhost-report",
            "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)

    assert payload["success"] is True
    assert (cluster_root / "published" / "current" / "manifest.json").exists()
    assert (cluster_root / "published" / "current" / "structured" / f"{run_id}_suite_steps.json").exists()
    assert (cluster_root / "published" / "current" / "raw" / f"{run_id}_suite" / "suite.log").exists()
    assert (cluster_root / "published" / "current" / "figures" / f"{run_id}_cluster_story_dashboard.png").exists()


def test_promote_run_renders_published_localhost_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-06_localhost_common_answer_fast_r4"
    _seed_promotable_run(cluster_root, run_id, "localhost")
    (cluster_root / "docs").mkdir(parents=True, exist_ok=True)
    (cluster_root / "docs" / "advanced-runbook.md").write_text("# runbook\n", encoding="utf-8")
    (cluster_root / "docs" / "field-report-template.md").write_text("# template\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    report_path = cluster_root / "field-report-localhost.md"
    notes_path = cluster_root / "field-report-localhost-notes.md"
    top_level_report = cluster_root / "field-report.md"
    top_level_notes = cluster_root / "field-report-notes.md"
    assert report_path.exists()
    assert notes_path.exists()
    assert top_level_report.exists()
    assert top_level_notes.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert f"published/current/structured/{run_id}_suite_steps.json" in report_text
    assert (cluster_root / "published" / "current" / "reports" / "field-report-localhost.md").exists()
    assert (cluster_root / "published" / "current" / "reports" / "field-report-localhost-notes.md").exists()
    assert (cluster_root / "published" / "current" / "reports" / "field-report.md").exists()
    assert (cluster_root / "published" / "current" / "reports" / "field-report-notes.md").exists()
    assert top_level_report.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")
    assert top_level_notes.read_text(encoding="utf-8") == notes_path.read_text(encoding="utf-8")
