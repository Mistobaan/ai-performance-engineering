from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_coverage_analysis_detects_missing_subsystems(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    run_id = "2026-03-04_cov"
    label = "node1"

    # Partial run: only GPU STREAM present.
    (structured / f"{run_id}_{label}_gpu_stream.json").write_text(
        json.dumps({"status": "ok", "operations": [{"operation": "triad", "bandwidth_gbps": 1000.0}]}) + "\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_benchmark_coverage_analysis.json"
    out_md = structured / f"{run_id}_benchmark_coverage_analysis.md"
    cmd = [
        sys.executable,
        "cluster/analysis/analyze_benchmark_coverage.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    cov = payload["subsystem_coverage"]
    assert cov["hbm_memory"] is True
    assert cov["sm_compute"] is False
    assert cov["gpu_gpu_communication"] is False
    assert cov["gpu_cpu_transfer"] is False
    assert cov["ai_workloads"] is False
    advanced = payload["advanced_coverage"]
    assert advanced["vllm_request_rate_sweep"] is False
    assert advanced["vllm_concurrency_repeat_stability"] is False
    assert advanced["vllm_request_rate_repeat_stability"] is False
    assert advanced["fio_repeat_stability"] is False
    assert advanced["allreduce_stability"] is False
    assert advanced["nccl_alltoall"] is False
    assert advanced["nccl_algo_comparison"] is False
    assert advanced["train_step_workload"] is False
    assert advanced["mlperf_alignment"] is False


def test_benchmark_coverage_analysis_marks_nccl_alltoall_as_advanced_coverage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    run_id = "2026-03-04_cov_alltoall"

    (structured / f"{run_id}_node1_alltoall_nccl_alltoall.json").write_text(
        json.dumps(
            {
                "results": [
                    {"size_bytes": 1048576, "busbw_gbps": 450.0, "algbw_gbps": 420.0},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_benchmark_coverage_analysis.json"
    out_md = structured / f"{run_id}_benchmark_coverage_analysis.md"
    cmd = [
        sys.executable,
        "cluster/analysis/analyze_benchmark_coverage.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    cov = payload["subsystem_coverage"]
    assert cov["gpu_gpu_communication"] is True
    advanced = payload["advanced_coverage"]
    assert advanced["nccl_alltoall"] is True
    assert advanced["train_step_workload"] is False
