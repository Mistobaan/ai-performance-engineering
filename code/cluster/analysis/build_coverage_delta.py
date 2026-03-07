#!/usr/bin/env python3
"""Build run-to-run coverage and scorecard delta artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


CLUSTER_DIR = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _candidate_structured_dirs(structured_dir: Path, run_id: str, cluster_dir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for candidate in (
        structured_dir,
        cluster_dir / "runs" / run_id / "structured",
        cluster_dir / "results" / "structured",
        cluster_dir / "archive" / "runs" / run_id / "structured",
    ):
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        yield candidate


def _resolve_structured_artifact(
    structured_dir: Path,
    run_id: str,
    suffix: str,
    *,
    cluster_dir: Path = CLUSTER_DIR,
    required: bool,
) -> Path:
    artifact_name = f"{run_id}_{suffix}"
    candidates = [candidate / artifact_name for candidate in _candidate_structured_dirs(structured_dir, run_id, cluster_dir)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        joined = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Missing artifact {artifact_name}; checked:\n{joined}")
    preferred_optionals = (
        cluster_dir / "runs" / run_id / "structured" / artifact_name,
        cluster_dir / "archive" / "runs" / run_id / "structured" / artifact_name,
        cluster_dir / "results" / "structured" / artifact_name,
        structured_dir / artifact_name,
    )
    for candidate in preferred_optionals:
        if candidate.parent.exists():
            return candidate
    return candidates[0]


def _load_coverage(structured_dir: Path, run_id: str, *, cluster_dir: Path = CLUSTER_DIR) -> Dict[str, Any]:
    path = _resolve_structured_artifact(
        structured_dir,
        run_id,
        "benchmark_coverage_analysis.json",
        cluster_dir=cluster_dir,
        required=True,
    )
    payload = _load_json(path)
    return {
        "run_id": run_id,
        "path": str(path),
        "coverage_score_pct": int(_safe_float(payload.get("coverage_score_pct"))),
        "advanced_coverage_score_pct": int(_safe_float(payload.get("advanced_coverage_score_pct"))),
        "coverage_maturity": str(payload.get("coverage_maturity") or ""),
        "advanced_coverage": payload.get("advanced_coverage") or {},
    }


def _load_mlperf(structured_dir: Path, run_id: str, *, cluster_dir: Path = CLUSTER_DIR) -> Dict[str, Any]:
    path = _resolve_structured_artifact(
        structured_dir,
        run_id,
        "mlperf_alignment.json",
        cluster_dir=cluster_dir,
        required=False,
    )
    if not path.exists():
        return {
            "path": str(path),
            "overall_status": "",
            "inference_track_ready": False,
            "training_track_ready": False,
        }
    payload = _load_json(path)
    return {
        "path": str(path),
        "overall_status": str(payload.get("overall_status") or ""),
        "inference_track_ready": bool(payload.get("inference_track_ready")),
        "training_track_ready": bool(payload.get("training_track_ready")),
    }


def _load_scorecard(structured_dir: Path, run_id: str, *, cluster_dir: Path = CLUSTER_DIR) -> Dict[str, Any]:
    path = _resolve_structured_artifact(
        structured_dir,
        run_id,
        "cluster_scorecard.json",
        cluster_dir=cluster_dir,
        required=False,
    )
    if not path.exists():
        return {"path": str(path), "overall_score": 0.0, "pass_fail": "", "summary": {}}
    payload = _load_json(path)
    return {
        "path": str(path),
        "overall_score": _safe_float(payload.get("overall_score")),
        "pass_fail": str(payload.get("pass_fail") or ""),
        "summary": payload.get("summary") or {},
    }


def _build_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    cur_adv = current.get("advanced_coverage") or {}
    base_adv = baseline.get("advanced_coverage") or {}
    keys = sorted(set(cur_adv.keys()) | set(base_adv.keys()))
    newly_covered = [k for k in keys if bool(cur_adv.get(k)) and not bool(base_adv.get(k))]
    regressed = [k for k in keys if bool(base_adv.get(k)) and not bool(cur_adv.get(k))]

    return {
        "coverage_score_pct_delta": int(current["coverage_score_pct"]) - int(baseline["coverage_score_pct"]),
        "advanced_coverage_score_pct_delta": int(current["advanced_coverage_score_pct"])
        - int(baseline["advanced_coverage_score_pct"]),
        "newly_covered_advanced_signals": newly_covered,
        "regressed_advanced_signals": regressed,
    }


def _build_markdown(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"# Coverage Delta: `{payload['current_run_id']}` vs `{payload['baseline_run_id']}`"
    )
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Coverage score % | `{payload['baseline']['coverage_score_pct']}` | `{payload['current']['coverage_score_pct']}` | `{payload['delta']['coverage_score_pct_delta']:+d}` |"
    )
    lines.append(
        f"| Advanced coverage score % | `{payload['baseline']['advanced_coverage_score_pct']}` | `{payload['current']['advanced_coverage_score_pct']}` | `{payload['delta']['advanced_coverage_score_pct_delta']:+d}` |"
    )
    lines.append("")
    lines.append("| Metric | Baseline | Current |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Coverage maturity | `{payload['baseline']['coverage_maturity'] or 'n/a'}` | `{payload['current']['coverage_maturity'] or 'n/a'}` |"
    )
    lines.append(
        f"| MLPerf overall status | `{payload['baseline_mlperf']['overall_status'] or 'n/a'}` | `{payload['current_mlperf']['overall_status'] or 'n/a'}` |"
    )
    lines.append(
        f"| MLPerf inference track ready | `{payload['baseline_mlperf']['inference_track_ready']}` | `{payload['current_mlperf']['inference_track_ready']}` |"
    )
    lines.append(
        f"| MLPerf training track ready | `{payload['baseline_mlperf']['training_track_ready']}` | `{payload['current_mlperf']['training_track_ready']}` |"
    )
    lines.append(
        f"| Scorecard overall score | `{payload['baseline_scorecard']['overall_score']:.1f}` | `{payload['current_scorecard']['overall_score']:.1f}` |"
    )
    lines.append(
        f"| Scorecard pass/fail | `{payload['baseline_scorecard']['pass_fail'] or 'n/a'}` | `{payload['current_scorecard']['pass_fail'] or 'n/a'}` |"
    )
    lines.append("")
    lines.append("| Advanced signal changes | Value |")
    lines.append("|---|---|")
    new_signals = payload["delta"]["newly_covered_advanced_signals"]
    regressed = payload["delta"]["regressed_advanced_signals"]
    lines.append(f"| Newly covered | `{', '.join(new_signals) if new_signals else 'none'}` |")
    lines.append(f"| Regressed | `{', '.join(regressed) if regressed else 'none'}` |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run-to-run coverage + scorecard delta.")
    parser.add_argument("--run-id", required=True, help="Current run ID.")
    parser.add_argument("--baseline-run-id", required=True, help="Baseline run ID.")
    parser.add_argument(
        "--structured-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "structured"),
        help="Structured artifact directory.",
    )
    parser.add_argument("--output-json", default="", help="Output JSON path.")
    parser.add_argument("--output-md", default="", help="Output markdown path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    structured_dir = Path(args.structured_dir)
    run_id = args.run_id
    baseline_run_id = args.baseline_run_id

    current = _load_coverage(structured_dir, run_id)
    baseline = _load_coverage(structured_dir, baseline_run_id)
    current_mlperf = _load_mlperf(structured_dir, run_id)
    baseline_mlperf = _load_mlperf(structured_dir, baseline_run_id)
    current_scorecard = _load_scorecard(structured_dir, run_id)
    baseline_scorecard = _load_scorecard(structured_dir, baseline_run_id)
    delta = _build_delta(current, baseline)

    payload: Dict[str, Any] = {
        "current_run_id": run_id,
        "baseline_run_id": baseline_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current": current,
        "baseline": baseline,
        "current_mlperf": current_mlperf,
        "baseline_mlperf": baseline_mlperf,
        "current_scorecard": current_scorecard,
        "baseline_scorecard": baseline_scorecard,
        "delta": delta,
    }

    default_stem = f"{run_id}_coverage_delta_vs_{baseline_run_id}"
    output_json = Path(args.output_json) if args.output_json else structured_dir / f"{default_stem}.json"
    output_md = Path(args.output_md) if args.output_md else structured_dir / f"{default_stem}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
