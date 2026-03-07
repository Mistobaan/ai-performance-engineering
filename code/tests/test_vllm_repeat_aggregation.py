from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_aggregator(repo_root: Path, mode: str, inputs: list[Path], out_dir: Path) -> tuple[Path, Path]:
    out_csv = out_dir / f"{mode}_summary.csv"
    out_jsonl = out_dir / f"{mode}_summary.jsonl"
    out_stability = out_dir / f"{mode}_stability.json"
    out_summary = out_dir / f"{mode}_summary.txt"
    cmd = [
        sys.executable,
        "cluster/analysis/aggregate_vllm_repeat_csv.py",
        "--mode",
        mode,
        "--inputs",
        *[str(p) for p in inputs],
        "--output-csv",
        str(out_csv),
        "--output-jsonl",
        str(out_jsonl),
        "--output-stability-json",
        str(out_stability),
        "--output-summary-txt",
        str(out_summary),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.exists()
    assert out_jsonl.exists()
    assert out_stability.exists()
    assert out_summary.exists()
    return out_csv, out_stability


def test_vllm_repeat_aggregator_concurrency_and_request_rate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Concurrency mode.
    conc_in_1 = tmp_path / "conc" / "repeat_1.csv"
    conc_in_2 = tmp_path / "conc" / "repeat_2.csv"
    _write_csv(
        conc_in_1,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "concurrency": 8,
                "num_prompts": 80,
                "request_throughput": 8.0,
                "output_throughput": 1000.0,
                "total_token_throughput": 5000.0,
                "mean_ttft_ms": 40.0,
                "median_ttft_ms": 39.0,
                "p99_ttft_ms": 60.0,
                "mean_tpot_ms": 7.0,
                "median_tpot_ms": 6.9,
                "p99_tpot_ms": 8.0,
                "gpu_util_mean_pct": 70.0,
                "gpu_util_p95_pct": 85.0,
                "mem_used_mean_mb": 10000.0,
                "mem_used_max_mb": 12000.0,
                "completed": 80,
                "failed": 0,
            }
        ],
    )
    _write_csv(
        conc_in_2,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "concurrency": 8,
                "num_prompts": 80,
                "request_throughput": 8.2,
                "output_throughput": 1100.0,
                "total_token_throughput": 5500.0,
                "mean_ttft_ms": 42.0,
                "median_ttft_ms": 41.0,
                "p99_ttft_ms": 65.0,
                "mean_tpot_ms": 7.2,
                "median_tpot_ms": 7.1,
                "p99_tpot_ms": 8.5,
                "gpu_util_mean_pct": 72.0,
                "gpu_util_p95_pct": 86.0,
                "mem_used_mean_mb": 10100.0,
                "mem_used_max_mb": 12100.0,
                "completed": 80,
                "failed": 0,
            }
        ],
    )

    conc_csv, conc_stability = _run_aggregator(
        repo_root, "concurrency", [conc_in_1, conc_in_2], tmp_path / "conc_out"
    )
    with conc_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    # Median of [5000, 5500] is 5250.
    assert abs(float(rows[0]["total_token_throughput"]) - 5250.0) < 1e-6
    conc_payload = json.loads(conc_stability.read_text(encoding="utf-8"))
    assert conc_payload["mode"] == "concurrency"
    assert conc_payload["summary"]["total_token_throughput_cv_pct_p95"] is not None

    # Request-rate mode.
    rate_in_1 = tmp_path / "rate" / "repeat_1.csv"
    rate_in_2 = tmp_path / "rate" / "repeat_2.csv"
    _write_csv(
        rate_in_1,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "request_rate": 8,
                "max_concurrency": 64,
                "num_prompts": 128,
                "request_throughput": 8.0,
                "output_throughput": 1200.0,
                "total_token_throughput": 6000.0,
                "mean_ttft_ms": 30.0,
                "median_ttft_ms": 29.0,
                "p99_ttft_ms": 45.0,
                "mean_tpot_ms": 6.5,
                "median_tpot_ms": 6.4,
                "p99_tpot_ms": 7.0,
                "completed": 128,
                "failed": 0,
            }
        ],
    )
    _write_csv(
        rate_in_2,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "request_rate": 8,
                "max_concurrency": 64,
                "num_prompts": 128,
                "request_throughput": 8.1,
                "output_throughput": 1240.0,
                "total_token_throughput": 6200.0,
                "mean_ttft_ms": 31.0,
                "median_ttft_ms": 30.0,
                "p99_ttft_ms": 46.0,
                "mean_tpot_ms": 6.6,
                "median_tpot_ms": 6.5,
                "p99_tpot_ms": 7.1,
                "completed": 128,
                "failed": 0,
            }
        ],
    )

    rate_csv, rate_stability = _run_aggregator(
        repo_root, "request_rate", [rate_in_1, rate_in_2], tmp_path / "rate_out"
    )
    with rate_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert abs(float(rows[0]["total_token_throughput"]) - 6100.0) < 1e-6
    rate_payload = json.loads(rate_stability.read_text(encoding="utf-8"))
    assert rate_payload["mode"] == "request_rate"
    assert rate_payload["summary"]["total_token_throughput_cv_pct_p95"] is not None


def test_vllm_repeat_aggregator_skips_invalid_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    in_1 = tmp_path / "invalid" / "repeat_1.csv"
    in_2 = tmp_path / "invalid" / "repeat_2.csv"
    _write_csv(
        in_1,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "concurrency": 16,
                "num_prompts": 160,
                "request_throughput": 10.0,
                "output_throughput": 1600.0,
                "total_token_throughput": 8000.0,
                "mean_ttft_ms": 45.0,
                "median_ttft_ms": 44.0,
                "p99_ttft_ms": 70.0,
                "mean_tpot_ms": 7.0,
                "median_tpot_ms": 6.9,
                "p99_tpot_ms": 8.0,
                "gpu_util_mean_pct": 70.0,
                "gpu_util_p95_pct": 85.0,
                "mem_used_mean_mb": 10000.0,
                "mem_used_max_mb": 12000.0,
                "completed": 160,
                "failed": 0,
            },
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "concurrency": 16,
                "num_prompts": 160,
                "request_throughput": 0.0,
                "output_throughput": 0.0,
                "total_token_throughput": 0.0,
                "mean_ttft_ms": 0.0,
                "median_ttft_ms": 0.0,
                "p99_ttft_ms": 0.0,
                "mean_tpot_ms": 0.0,
                "median_tpot_ms": 0.0,
                "p99_tpot_ms": 0.0,
                "gpu_util_mean_pct": 10.0,
                "gpu_util_p95_pct": 20.0,
                "mem_used_mean_mb": 5000.0,
                "mem_used_max_mb": 6000.0,
                "completed": 0,
                "failed": 160,
            },
        ],
    )
    _write_csv(
        in_2,
        [
            {
                "model": "test",
                "tp": 1,
                "isl": 128,
                "osl": 64,
                "concurrency": 16,
                "num_prompts": 160,
                "request_throughput": 10.2,
                "output_throughput": 1632.0,
                "total_token_throughput": 8160.0,
                "mean_ttft_ms": 46.0,
                "median_ttft_ms": 45.0,
                "p99_ttft_ms": 72.0,
                "mean_tpot_ms": 7.1,
                "median_tpot_ms": 7.0,
                "p99_tpot_ms": 8.1,
                "gpu_util_mean_pct": 71.0,
                "gpu_util_p95_pct": 86.0,
                "mem_used_mean_mb": 10100.0,
                "mem_used_max_mb": 12100.0,
                "completed": 160,
                "failed": 0,
            }
        ],
    )

    out_csv, out_stability = _run_aggregator(repo_root, "concurrency", [in_1, in_2], tmp_path / "invalid_out")
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    # The invalid row is dropped, so the median is based on [8000, 8160].
    assert abs(float(rows[0]["total_token_throughput"]) - 8080.0) < 1e-6
    payload = json.loads(out_stability.read_text(encoding="utf-8"))
    assert payload["summary"]["invalid_rows_skipped"] == 1
