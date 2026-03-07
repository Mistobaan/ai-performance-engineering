#!/usr/bin/env python3
"""Build MLPerf-style alignment summary for cluster training + inference tracks."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _detect_labels(structured_dir: Path, run_id: str) -> List[str]:
    suffixes = [
        "_vllm_serve_sweep.csv",
        "_vllm_serve_slo_goodput.json",
        "_vllm_serve_sweep_stability.json",
        "_vllm_serve_request_rate_sweep.csv",
        "_vllm_request_rate_slo_goodput.json",
        "_vllm_serve_request_rate_sweep_stability.json",
        "_gemm_gpu_sanity.csv",
    ]
    labels: set[str] = set()
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


def _valid_vllm_csv(path: Path) -> bool:
    if not _exists(path):
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not rows:
        return False
    for row in rows:
        completed = int(_to_float(row.get("completed"), 0.0))
        failed = int(_to_float(row.get("failed"), 0.0))
        total_tok = _to_float(row.get("total_token_throughput"), 0.0)
        if completed <= 0 or failed > 0 or total_tok <= 0.0:
            return False
    return True


def _valid_slo_json(path: Path, points_key: str) -> bool:
    if not _exists(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(payload.get("status") or "").lower() != "ok":
        return False
    summary = payload.get("summary") or {}
    return int(_to_float(summary.get(points_key), 0.0)) > 0


def _valid_stability_json(path: Path) -> bool:
    if not _exists(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summary = payload.get("summary") or {}
    points = int(_to_float(summary.get("points"), 0.0))
    tok_cv = summary.get("total_token_throughput_cv_pct_p95")
    return points > 0 and tok_cv is not None


def _valid_train_step(path: Path) -> bool:
    if not _exists(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    results = payload.get("results") or {}
    return _to_float(results.get("tokens_per_s"), 0.0) > 0.0


def _build_alignment(structured_dir: Path, run_id: str, labels: List[str]) -> Dict[str, Any]:
    first_label = labels[0] if labels else ""

    # Inference track (MLPerf-style serving: throughput + latency tails + stability).
    has_conc_csv = any(_valid_vllm_csv(structured_dir / f"{run_id}_{label}_vllm_serve_sweep.csv") for label in labels)
    has_rate_csv = any(
        _valid_vllm_csv(structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv") for label in labels
    )
    has_conc_slo = any(
        _valid_slo_json(structured_dir / f"{run_id}_{label}_vllm_serve_slo_goodput.json", "concurrency_points")
        for label in labels
    )
    has_rate_slo = any(
        _valid_slo_json(structured_dir / f"{run_id}_{label}_vllm_request_rate_slo_goodput.json", "request_rate_points")
        for label in labels
    )
    has_conc_stability = any(
        _valid_stability_json(structured_dir / f"{run_id}_{label}_vllm_serve_sweep_stability.json") for label in labels
    )
    has_rate_stability = any(
        _valid_stability_json(structured_dir / f"{run_id}_{label}_vllm_serve_request_rate_sweep_stability.json")
        for label in labels
    )

    inference_readiness = {
        "concurrency_sweep": has_conc_csv,
        "request_rate_sweep": has_rate_csv,
        "concurrency_slo_goodput": has_conc_slo,
        "request_rate_slo_goodput": has_rate_slo,
        "concurrency_repeat_stability": has_conc_stability,
        "request_rate_repeat_stability": has_rate_stability,
    }
    inference_ready = all(inference_readiness.values())

    # Training track (MLPerf-style train-step + collective communication health).
    train_step_paths = sorted(structured_dir.glob(f"{run_id}_*_torchrun_train_step.json"))
    has_train_step = any(_valid_train_step(path) for path in train_step_paths)
    has_train_step_multinode = False
    for path in train_step_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _to_float(payload.get("world_size"), 0.0) > 1:
            has_train_step_multinode = True
            break

    has_nccl_single = _exists(structured_dir / f"{run_id}_node1_nccl.json")
    has_nccl_multi = _exists(structured_dir / f"{run_id}_2nodes_nccl.json")
    has_allreduce_stability = _exists(structured_dir / f"{run_id}_allreduce_stability.json")
    has_alltoall = _exists(structured_dir / f"{run_id}_node1_alltoall_nccl_alltoall.json") or _exists(
        structured_dir / f"{run_id}_2nodes_alltoall_nccl_alltoall.json"
    )

    training_readiness = {
        "train_step_workload": has_train_step,
        "nccl_collectives_single_or_multi": has_nccl_single or has_nccl_multi,
        "allreduce_stability": has_allreduce_stability,
        "alltoall_moe_coverage": has_alltoall,
        "multinode_train_step": has_train_step_multinode,
    }
    training_ready = (
        has_train_step
        and (has_nccl_single or has_nccl_multi)
        and has_allreduce_stability
        and has_alltoall
    )

    if inference_ready and training_ready:
        overall_status = "aligned"
    elif inference_ready:
        overall_status = "inference_ready_only"
    elif training_ready:
        overall_status = "training_ready_only"
    else:
        overall_status = "partial"

    recommendations: List[str] = []
    if not inference_ready:
        recommendations.append(
            "Complete inference track: concurrency + request-rate sweeps with SLO goodput and repeat stability artifacts."
        )
    if not training_ready:
        recommendations.append(
            "Complete training track: torchrun train-step + allreduce stability + NCCL alltoall coverage."
        )
    if not has_train_step_multinode:
        recommendations.append("Add multinode train-step evidence for distributed training alignment.")

    return {
        "run_id": run_id,
        "labels": labels,
        "primary_label": first_label,
        "overall_status": overall_status,
        "inference_track_ready": inference_ready,
        "training_track_ready": training_ready,
        "inference_track": inference_readiness,
        "training_track": training_readiness,
        "references": {
            "inference_track": "MLPerf Inference Datacenter (LLM-style serving: throughput + TTFT/TPOT tails)",
            "training_track": "MLPerf Training (LLM-style train-step + distributed collective behavior)",
            "future_facing_llm_set": ["llama3.1_8b", "llama3.1_405b", "gpt_oss_20b"],
        },
        "recommendations": recommendations,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# MLPerf Alignment: `{payload['run_id']}`")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Overall status | `{payload.get('overall_status')}` |")
    lines.append(f"| Inference track ready | `{payload.get('inference_track_ready')}` |")
    lines.append(f"| Training track ready | `{payload.get('training_track_ready')}` |")
    lines.append("")
    lines.append("## Inference Track")
    lines.append("")
    lines.append("| Signal | Ready |")
    lines.append("|---|---|")
    for name, ok in (payload.get("inference_track") or {}).items():
        lines.append(f"| `{name}` | `{'yes' if ok else 'no'}` |")
    lines.append("")
    lines.append("## Training Track")
    lines.append("")
    lines.append("| Signal | Ready |")
    lines.append("|---|---|")
    for name, ok in (payload.get("training_track") or {}).items():
        lines.append(f"| `{name}` | `{'yes' if ok else 'no'}` |")
    lines.append("")
    lines.append("## References")
    lines.append("")
    refs = payload.get("references") or {}
    lines.append(f"- Inference: `{refs.get('inference_track', 'n/a')}`")
    lines.append(f"- Training: `{refs.get('training_track', 'n/a')}`")
    lines.append(f"- Future-facing LLM set: `{', '.join(refs.get('future_facing_llm_set') or [])}`")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    recs = payload.get("recommendations") or []
    if not recs:
        lines.append("- Alignment is complete for current training + inference tracks.")
    else:
        for rec in recs:
            lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MLPerf-style alignment summary for cluster runs.")
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
    output_json = Path(args.output_json) if args.output_json else structured_dir / f"{run_id}_mlperf_alignment.json"
    output_md = Path(args.output_md) if args.output_md else structured_dir / f"{run_id}_mlperf_alignment.md"

    labels = _detect_labels(structured_dir, run_id)
    payload = _build_alignment(structured_dir, run_id, labels)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
