from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_cluster_scorecard_detects_memory_bound(tmp_path: Path) -> None:
    run_id = "2026-03-03_scorecard_test"
    label = "node1"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    # GEMM sanity baseline.
    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
        "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
        f"{label}_gpu0,16384,16384,16384,bf16,50,10.0,10.0,11.0,650.0,640.0,600.0\n",
        encoding="utf-8",
    )

    # nvbandwidth indicates healthy copy bandwidth.
    _write_json(
        structured / f"{run_id}_{label}_nvbandwidth.json",
        {
            "status": "ok",
            "key_sum_gbps": {
                "device_to_device_memcpy_read_ce": 3000.0,
                "device_to_device_memcpy_write_ce": 2900.0,
                "host_to_device_memcpy_ce": 28.0,
                "device_to_host_memcpy_ce": 27.0,
            },
        },
    )

    # STREAM-like triad is materially below copy bandwidth -> memory inefficiency.
    _write_json(
        structured / f"{run_id}_{label}_gpu_stream.json",
        {
            "status": "ok",
            "peak_bandwidth_gbps": 1200.0,
            "operations": [
                {"operation": "copy", "bandwidth_gbps": 1200.0},
                {"operation": "scale", "bandwidth_gbps": 1100.0},
                {"operation": "add", "bandwidth_gbps": 1000.0},
                {"operation": "triad", "bandwidth_gbps": 900.0},
            ],
        },
    )

    # Healthy NCCL scaling to avoid triggering communication-bound logic.
    _write_json(
        structured / f"{run_id}_node1_nccl.json",
        {"results": [{"algbw_gbps": 500.0, "busbw_gbps": 480.0}]},
    )
    _write_json(
        structured / f"{run_id}_2nodes_nccl.json",
        {"results": [{"algbw_gbps": 410.0, "busbw_gbps": 400.0}]},
    )

    # Workload sweep without strong host-knee symptoms.
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,completed,failed\n"
        "test,1,128,64,1,10,1,100,200,10,10,12,5,5,6,50,60,1000,1200,10,0\n"
        "test,1,128,64,8,80,1,650,1200,14,13,18,8,7,9,70,80,1100,1300,80,0\n",
        encoding="utf-8",
    )
    # Request-rate sweep captures a secondary operating curve.
    (structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv").write_text(
        "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed\n"
        "test,1,128,64,1,32,200,1,100,140,15,14,20,7,6,9,200,0\n"
        "test,1,128,64,2,32,200,2,150,210,20,19,30,8,7,10,200,0\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_json.exists()
    assert out_md.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["bottleneck"]["bottleneck_type"] == "memory-bound"
    assert payload["summary"]["gpu_stream_to_hbm_ratio"] < 0.55
    assert payload["summary"]["vllm_rate_max_total_tok_s"] == 210.0


def test_cluster_scorecard_ingests_distributed_reliability_metrics(tmp_path: Path) -> None:
    run_id = "2026-03-04_scorecard_distributed"
    label = "node1"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    # Minimal base inputs to allow scorecard generation.
    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
        "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
        f"{label}_gpu0,8192,8192,8192,bf16,20,4.0,4.0,4.2,600.0,590.0,560.0\n",
        encoding="utf-8",
    )
    _write_json(
        structured / f"{run_id}_{label}_nvbandwidth.json",
        {
            "status": "ok",
            "key_sum_gbps": {
                "device_to_device_memcpy_read_ce": 2000.0,
                "device_to_device_memcpy_write_ce": 1950.0,
                "host_to_device_memcpy_ce": 40.0,
                "device_to_host_memcpy_ce": 39.0,
            },
        },
    )

    # New modern distributed reliability artifacts.
    _write_json(
        structured / f"{run_id}_allreduce_stability.json",
        {
            "summary": {
                "busbw_mean_gbps": 780.0,
                "busbw_cv_pct": 6.5,
                "p99_p50_ratio": 1.22,
                "jitter_assessment": "high_jitter",
            }
        },
    )
    _write_json(
        structured / f"{run_id}_nccl_algo_comparison.json",
        {
            "algorithms_tested": [
                {"algo": "auto", "status": "ok", "peak_busbw_gbps": 610.0},
                {"algo": "Ring", "status": "ok", "peak_busbw_gbps": 650.0},
                {"algo": "Tree", "status": "ok", "peak_busbw_gbps": 590.0},
            ]
        },
    )
    _write_json(
        structured / f"{run_id}_node1_alltoall_nccl_alltoall.json",
        {"results": [{"algbw_gbps": 520.0, "busbw_gbps": 500.0}]},
    )
    _write_json(
        structured / f"{run_id}_2nodes_alltoall_nccl_alltoall.json",
        {"results": [{"algbw_gbps": 360.0, "busbw_gbps": 340.0}]},
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["allreduce_busbw_cv_pct"] == 6.5
    assert summary["allreduce_p99_p50_ratio"] == 1.22
    assert summary["allreduce_jitter_assessment"] == "high_jitter"
    assert summary["nccl_algo_best"] == "Ring"
    assert summary["nccl_algo_peak_busbw_gbps"] == 650.0
    assert summary["nccl_algo_auto_gap_pct"] > 0
    assert summary["nccl_alltoall_single_peak_busbw_gbps"] == 500.0
    assert summary["nccl_alltoall_multi_peak_busbw_gbps"] == 340.0
    assert summary["nccl_alltoall_multi_to_single_busbw_ratio"] == 0.68


def test_cluster_scorecard_marks_single_rank_comm_metrics_not_applicable(tmp_path: Path) -> None:
    run_id = "2026-03-04_scorecard_single_rank"
    label = "node1"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
        "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
        f"{label}_gpu0,8192,8192,8192,bf16,20,4.0,4.0,4.2,600.0,590.0,560.0\n",
        encoding="utf-8",
    )

    _write_json(
        structured / f"{run_id}_{label}_nvbandwidth.json",
        {
            "status": "ok",
            "key_sum_gbps": {
                "device_to_device_memcpy_read_ce": 2000.0,
                "device_to_device_memcpy_write_ce": 1950.0,
                "host_to_device_memcpy_ce": 40.0,
                "device_to_host_memcpy_ce": 39.0,
            },
        },
    )

    # Presence of these artifacts on single-rank runs should render comm stability/algo
    # metrics as not-applicable rather than misleading zeros.
    _write_json(
        structured / f"{run_id}_allreduce_stability.json",
        {
            "world_size": 1,
            "summary": {
                "busbw_mean_gbps": 123.0,
                "busbw_cv_pct": 9.0,
                "p99_p50_ratio": 1.5,
                "jitter_assessment": "high_jitter",
            },
        },
    )
    _write_json(
        structured / f"{run_id}_nccl_algo_comparison.json",
        {
            "total_ranks": 1,
            "algorithms_tested": [
                {"algo": "auto", "status": "ok", "peak_busbw_gbps": 610.0},
                {"algo": "Ring", "status": "ok", "peak_busbw_gbps": 650.0},
            ],
        },
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["allreduce_applicable"] is False
    assert summary["allreduce_busbw_cv_pct"] is None
    assert summary["allreduce_p99_p50_ratio"] is None
    assert str(summary["allreduce_jitter_assessment"]).startswith("n/a")
    assert summary["nccl_algo_applicable"] is False
    assert str(summary["nccl_algo_best"]).startswith("n/a")
    assert summary["nccl_algo_peak_busbw_gbps"] is None


def test_cluster_scorecard_respects_primary_label_for_workload_metrics(tmp_path: Path) -> None:
    run_id = "2026-03-04_scorecard_primary_label"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    for label in ("node1", "node2"):
        (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
            "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
            f"{label}_gpu0,8192,8192,8192,bf16,20,4.0,4.0,4.2,600.0,590.0,560.0\n",
            encoding="utf-8",
        )

    # node1 has lower workload metrics than node2.
    (structured / f"{run_id}_node1_vllm_serve_sweep.csv").write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed\n"
        "test,1,128,64,1,10,1,100,200,10,10,12,5,5,6,10,0\n"
        "test,1,128,64,8,80,1,400,600,14,13,20,8,7,9,80,0\n",
        encoding="utf-8",
    )
    (structured / f"{run_id}_node2_vllm_serve_sweep.csv").write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed\n"
        "test,1,128,64,1,10,1,100,300,10,10,12,5,5,6,10,0\n"
        "test,1,128,64,8,80,1,800,1500,14,13,20,8,7,9,80,0\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--primary-label",
        "node2",
        "--structured-dir",
        str(structured),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["resolved_primary_label"] == "node2"
    # node2 max/min total_token_throughput = 1500/300 = 5.0
    assert payload["summary"]["vllm_throughput_gain_ratio"] == 5.0


def test_cluster_scorecard_computes_efficiency_and_cost_metrics(tmp_path: Path) -> None:
    run_id = "2026-03-05_scorecard_efficiency"
    label = "node1"
    structured = tmp_path / "structured"
    structured.mkdir(parents=True, exist_ok=True)

    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text(
        "label,m,n,k,dtype,iters,avg_ms,p50_ms,p99_ms,avg_tflops,p50_tflops,p99_tflops\n"
        f"{label}_gpu0,8192,8192,8192,bf16,20,4.0,4.0,4.2,600.0,590.0,560.0\n",
        encoding="utf-8",
    )

    # Concurrency curve includes power; best total tok/s is 1000 at 500W with TP=2 -> 2 tok/J.
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text(
        "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_power_mean_w,completed,failed\n"
        "test,2,128,64,8,80,1,300,600,14,13,20,8,7,9,400,80,0\n"
        "test,2,128,64,16,160,1,500,1000,20,19,30,9,8,10,500,160,0\n",
        encoding="utf-8",
    )
    # Request-rate curve includes power; best total tok/s is 1500 at 600W with TP=2 -> 2.5 tok/J.
    (structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv").write_text(
        "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_power_mean_w,completed,failed\n"
        "test,2,128,64,4,64,200,4,500,900,20,19,30,8,7,10,450,200,0\n"
        "test,2,128,64,8,64,200,8,800,1500,28,27,45,9,8,12,600,200,0\n",
        encoding="utf-8",
    )

    out_json = structured / f"{run_id}_cluster_scorecard.json"
    out_md = structured / f"{run_id}_cluster_scorecard.md"
    cmd = [
        sys.executable,
        "cluster/analysis/build_cluster_scorecard.py",
        "--run-id",
        run_id,
        "--structured-dir",
        str(structured),
        "--gpu-hourly-cost-usd",
        "2.0",
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert abs(summary["vllm_tok_per_joule_at_max_total_tok_s"] - 2.0) < 1e-9
    assert abs(summary["vllm_rate_tok_per_joule_at_max_total_tok_s"] - 2.5) < 1e-9
    # cost_per_mtok = ((hourly_cost * tp)/3600) * 1e6 / tok_s
    assert abs(summary["vllm_cost_per_mtok_usd_at_max_total_tok_s"] - 1.1111111111111112) < 1e-6
    assert abs(summary["vllm_rate_cost_per_mtok_usd_at_max_total_tok_s"] - 0.7407407407407408) < 1e-6
