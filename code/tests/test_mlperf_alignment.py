from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _run_alignment(repo_root: Path, run_id: str, structured: Path) -> Path:
    out_json = structured / f"{run_id}_mlperf_alignment.json"
    out_md = structured / f"{run_id}_mlperf_alignment.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_mlperf_alignment.py",
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
    assert out_json.exists()
    assert out_md.exists()
    return out_json


def test_mlperf_alignment_reports_aligned_when_inference_and_training_tracks_are_ready(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    run_id = "2026-03-05_mlperf_aligned"
    label = "node1"

    _write_text(
        structured / f"{run_id}_{label}_vllm_serve_sweep.csv",
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_power_mean_w,completed,failed\n"
        "test,1,128,64,8,80,8.0,1200.0,2400.0,40.0,39.0,55.0,8.0,7.8,10.0,450.0,80,0\n",
    )
    _write_text(
        structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv",
        "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_power_mean_w,completed,failed\n"
        "test,1,128,64,4,64,256,4.0,1000.0,2000.0,42.0,41.0,60.0,8.2,8.0,10.2,430.0,256,0\n",
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_slo_goodput.json",
        {"status": "ok", "summary": {"concurrency_points": 1, "peak_total_tok_s": 2400.0, "max_goodput_tok_s": 2000.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_request_rate_slo_goodput.json",
        {"status": "ok", "summary": {"request_rate_points": 1, "peak_total_tok_s": 2000.0, "max_goodput_tok_s": 1700.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_sweep_stability.json",
        {"summary": {"points": 1, "total_token_throughput_cv_pct_p95": 2.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep_stability.json",
        {"summary": {"points": 1, "total_token_throughput_cv_pct_p95": 3.0}},
    )

    _write_json(
        structured / f"{run_id}_{label}_torchrun_train_step.json",
        {"world_size": 2, "results": {"tokens_per_s": 100000.0}},
    )
    _write_json(
        structured / f"{run_id}_allreduce_stability.json",
        {"summary": {"busbw_mean_gbps": 700.0, "busbw_cv_pct": 3.0}},
    )
    _write_json(
        structured / f"{run_id}_node1_nccl.json",
        {"results": [{"size_bytes": 1048576, "busbw_gbps": 500.0}]},
    )
    _write_json(
        structured / f"{run_id}_node1_alltoall_nccl_alltoall.json",
        {"results": [{"size_bytes": 1048576, "busbw_gbps": 420.0}]},
    )

    out_json = _run_alignment(repo_root, run_id, structured)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "aligned"
    assert payload["inference_track_ready"] is True
    assert payload["training_track_ready"] is True


def test_mlperf_alignment_reports_inference_only_when_training_track_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    run_id = "2026-03-05_mlperf_inference_only"
    label = "node1"

    _write_text(
        structured / f"{run_id}_{label}_vllm_serve_sweep.csv",
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed\n"
        "test,1,128,64,8,80,8.0,1200.0,2400.0,40.0,39.0,55.0,8.0,7.8,10.0,80,0\n",
    )
    _write_text(
        structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv",
        "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed\n"
        "test,1,128,64,4,64,256,4.0,1000.0,2000.0,42.0,41.0,60.0,8.2,8.0,10.2,256,0\n",
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_slo_goodput.json",
        {"status": "ok", "summary": {"concurrency_points": 1, "peak_total_tok_s": 2400.0, "max_goodput_tok_s": 2000.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_request_rate_slo_goodput.json",
        {"status": "ok", "summary": {"request_rate_points": 1, "peak_total_tok_s": 2000.0, "max_goodput_tok_s": 1700.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_sweep_stability.json",
        {"summary": {"points": 1, "total_token_throughput_cv_pct_p95": 2.0}},
    )
    _write_json(
        structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep_stability.json",
        {"summary": {"points": 1, "total_token_throughput_cv_pct_p95": 3.0}},
    )

    out_json = _run_alignment(repo_root, run_id, structured)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "inference_ready_only"
    assert payload["inference_track_ready"] is True
    assert payload["training_track_ready"] is False
