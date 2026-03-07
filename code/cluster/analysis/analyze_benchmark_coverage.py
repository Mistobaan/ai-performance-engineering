#!/usr/bin/env python3
"""Analyze benchmark subsystem coverage for a cluster run and recommend next tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _detect_labels(structured_dir: Path, run_id: str) -> List[str]:
    labels = set()
    suffixes = [
        "_gemm_gpu_sanity.csv",
        "_vllm_serve_sweep.csv",
        "_vllm_serve_sweep_stability.json",
        "_vllm_serve_request_rate_sweep.csv",
        "_vllm_serve_request_rate_sweep_stability.json",
        "_nvbandwidth.json",
        "_gpu_stream.json",
        "_fio.json",
        "_fio_stability.json",
    ]
    prefix = f"{run_id}_"
    for suffix in suffixes:
        for path in structured_dir.glob(f"{run_id}_*{suffix}"):
            name = path.name
            if not name.startswith(prefix):
                continue
            label = name[len(prefix) : -len(suffix)]
            if label:
                labels.add(label)
    return sorted(labels)


def _build_coverage(structured_dir: Path, run_id: str, labels: List[str]) -> Dict[str, Any]:
    first_label = labels[0] if labels else ""
    has_gemm = any(_exists(structured_dir / f"{run_id}_{label}_gemm_gpu_sanity.csv") for label in labels)
    has_stream = any(_exists(structured_dir / f"{run_id}_{label}_gpu_stream.json") for label in labels)
    has_nvbw = any(_exists(structured_dir / f"{run_id}_{label}_nvbandwidth.json") for label in labels)
    has_vllm_conc = any(_exists(structured_dir / f"{run_id}_{label}_vllm_serve_sweep.csv") for label in labels)
    has_vllm_conc_stability = any(
        _exists(structured_dir / f"{run_id}_{label}_vllm_serve_sweep_stability.json") for label in labels
    )
    has_vllm_rate = any(_exists(structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv") for label in labels)
    has_vllm_rate_stability = any(
        _exists(structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep_stability.json") for label in labels
    )
    has_train_step = any(_exists(path) for path in structured_dir.glob(f"{run_id}_*_torchrun_train_step.json"))
    has_fio_stability = any(_exists(structured_dir / f"{run_id}_{label}_fio_stability.json") for label in labels)
    has_nccl_single = _exists(structured_dir / f"{run_id}_node1_nccl.json")
    has_nccl_multi = _exists(structured_dir / f"{run_id}_2nodes_nccl.json")
    has_nccl_alltoall_single = _exists(structured_dir / f"{run_id}_node1_alltoall_nccl_alltoall.json")
    has_nccl_alltoall_multi = _exists(structured_dir / f"{run_id}_2nodes_alltoall_nccl_alltoall.json")
    has_allreduce_stability = _exists(structured_dir / f"{run_id}_allreduce_stability.json")
    has_allreduce_latency_comp = _exists(structured_dir / f"{run_id}_allreduce_latency_comp.json")
    has_allgather_control_plane = _exists(structured_dir / f"{run_id}_allgather_control_plane.json")
    has_nccl_algo_comparison = _exists(structured_dir / f"{run_id}_nccl_algo_comparison.json")

    mlperf_alignment_path = structured_dir / f"{run_id}_mlperf_alignment.json"
    has_mlperf_alignment = False
    if _exists(mlperf_alignment_path):
        try:
            mlperf_payload = json.loads(mlperf_alignment_path.read_text(encoding="utf-8"))
            has_mlperf_alignment = str(mlperf_payload.get("overall_status") or "").strip().lower() in {
                "aligned",
                "inference_ready_only",
                "training_ready_only",
            }
        except Exception:
            has_mlperf_alignment = False

    subsystem = {
        "sm_compute": has_gemm,
        "hbm_memory": has_stream or has_nvbw,
        "gpu_gpu_communication": has_nccl_single or has_nccl_multi or has_nccl_alltoall_single or has_nccl_alltoall_multi,
        "gpu_cpu_transfer": has_nvbw,
        "ai_workloads": has_vllm_conc or has_vllm_rate or has_train_step,
    }

    missing = [name for name, ok in subsystem.items() if not ok]
    score = int(round((sum(1 for ok in subsystem.values() if ok) / len(subsystem)) * 100))

    advanced = {
        "vllm_request_rate_sweep": has_vllm_rate,
        "vllm_concurrency_repeat_stability": has_vllm_conc_stability,
        "vllm_request_rate_repeat_stability": has_vllm_rate_stability,
        "fio_repeat_stability": has_fio_stability,
        "allreduce_stability": has_allreduce_stability,
        "allreduce_latency_comp": has_allreduce_latency_comp,
        "allgather_control_plane": has_allgather_control_plane,
        "nccl_alltoall": has_nccl_alltoall_single or has_nccl_alltoall_multi,
        "nccl_algo_comparison": has_nccl_algo_comparison,
        "train_step_workload": has_train_step,
        "mlperf_alignment": has_mlperf_alignment,
    }
    advanced_score = int(round((sum(1 for ok in advanced.values() if ok) / len(advanced)) * 100))

    recommendations: List[str] = []
    if not subsystem["sm_compute"]:
        recommendations.append("Run Benchmark C GEMM sanity to establish compute baseline.")
    if not subsystem["hbm_memory"]:
        recommendations.append("Enable GPU STREAM and/or nvbandwidth to measure memory behavior.")
    if not subsystem["gpu_gpu_communication"]:
        recommendations.append("Run NCCL all_reduce single-node (and multi-node when applicable).")
    if not subsystem["gpu_cpu_transfer"]:
        recommendations.append("Enable nvbandwidth to capture PCIe host-device transfer metrics.")
    if not subsystem["ai_workloads"]:
        recommendations.append("Run vLLM serving sweep and/or train-step workload benchmarks for real AI workload coverage.")
    if not advanced["allreduce_stability"]:
        recommendations.append("Run allreduce stability profile to track per-iteration jitter and tail behavior.")
    if not advanced["allreduce_latency_comp"]:
        recommendations.append("Run allreduce latency comparison (1x-large vs many-small payload) to expose fragmentation penalties.")
    if not advanced["allgather_control_plane"]:
        recommendations.append("Run all_gather_object vs tensor collective control-plane benchmark for scheduler/collective overhead visibility.")
    if not advanced["nccl_alltoall"]:
        recommendations.append("Run NCCL all-to-all (alltoall_perf) for MoE-style communication coverage.")
    if not advanced["nccl_algo_comparison"]:
        recommendations.append("Run NCCL algorithm comparison (Ring/Tree/NVLS/auto) to identify best collective policy.")
    if not advanced["vllm_concurrency_repeat_stability"]:
        recommendations.append("Run vLLM concurrency sweep with repeats and persist stability JSON (CV p95 metrics).")
    if has_vllm_rate and not advanced["vllm_request_rate_repeat_stability"]:
        recommendations.append("Persist vLLM request-rate sweep stability JSON to quantify tail jitter across repeats.")
    if not advanced["fio_repeat_stability"]:
        recommendations.append("Run fio with repeats and collect fio_stability.json to capture storage jitter.")
    if not advanced["train_step_workload"]:
        recommendations.append("Run torchrun train-step benchmark to cover modern LLM training behavior.")
    if not advanced["mlperf_alignment"]:
        recommendations.append("Generate MLPerf alignment artifact to track training+inference standards compliance.")

    maturity = "high" if score >= 80 else "medium" if score >= 50 else "low"
    return {
        "run_id": run_id,
        "labels": labels,
        "primary_label": first_label,
        "subsystem_coverage": subsystem,
        "coverage_score_pct": score,
        "coverage_maturity": maturity,
        "missing_subsystems": missing,
        "advanced_coverage": advanced,
        "advanced_coverage_score_pct": advanced_score,
        "recommendations": recommendations,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Benchmark Coverage Analysis: `{payload['run_id']}`")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Labels | `{', '.join(payload.get('labels', []))}` |")
    lines.append(f"| Coverage score | `{payload.get('coverage_score_pct', 0)}%` |")
    lines.append(f"| Maturity | `{payload.get('coverage_maturity', 'low')}` |")
    lines.append(f"| Advanced coverage score | `{payload.get('advanced_coverage_score_pct', 0)}%` |")
    lines.append("")
    lines.append("## Subsystem Coverage")
    lines.append("")
    lines.append("| Subsystem | Covered |")
    lines.append("|---|---|")
    for name, ok in (payload.get("subsystem_coverage") or {}).items():
        lines.append(f"| `{name}` | `{'yes' if ok else 'no'}` |")
    lines.append("")
    missing = payload.get("missing_subsystems") or []
    lines.append(f"Missing: `{', '.join(missing) if missing else 'none'}`")
    lines.append("")
    lines.append("## Advanced Coverage")
    lines.append("")
    lines.append("| Advanced signal | Covered |")
    lines.append("|---|---|")
    for name, ok in (payload.get("advanced_coverage") or {}).items():
        lines.append(f"| `{name}` | `{'yes' if ok else 'no'}` |")
    lines.append("")
    lines.append("## Recommended Next Runs")
    lines.append("")
    recs = payload.get("recommendations") or []
    if not recs:
        lines.append("- Coverage is complete across the five major subsystems.")
    else:
        for rec in recs:
            lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze benchmark subsystem coverage for a run.")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--structured-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "structured"),
    )
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    structured_dir = Path(args.structured_dir)
    run_id = args.run_id
    output_json = Path(args.output_json) if args.output_json else structured_dir / f"{run_id}_benchmark_coverage_analysis.json"
    output_md = Path(args.output_md) if args.output_md else structured_dir / f"{run_id}_benchmark_coverage_analysis.md"

    labels = _detect_labels(structured_dir, run_id)
    payload = _build_coverage(structured_dir, run_id, labels)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
