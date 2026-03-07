#!/usr/bin/env python3
"""Aggregate repeated vLLM sweep CSV outputs into robust canonical artifacts.

This script supports both:
1) Concurrency sweeps (vllm_serve_sweep.csv)
2) Request-rate sweeps (vllm_serve_request_rate_sweep.csv)

For each sweep point, numeric metrics are aggregated with median values for
compatibility with existing downstream analysis, and repeat-variance metrics
are emitted in a companion stability JSON artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple


MODE_KEY_COLS = {
    "concurrency": ["model", "tp", "isl", "osl", "concurrency", "num_prompts"],
    "request_rate": ["model", "tp", "isl", "osl", "request_rate", "max_concurrency", "num_prompts"],
}

CORE_STABILITY_METRICS = ["total_token_throughput", "p99_ttft_ms", "p99_tpot_ms"]


def _to_float(v: str) -> float | None:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_sortable(v: str) -> float | str:
    f = _to_float(v)
    if f is not None:
        return f
    return v


def _percentile_nearest_rank(vals: List[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    k = max(0, min(len(xs) - 1, int(math.ceil((q / 100.0) * len(xs))) - 1))
    return xs[k]


def _stats(vals: List[float]) -> Dict[str, float | None]:
    if not vals:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "cv_pct": None, "min": None, "max": None}
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    cv = (sd / m * 100.0) if m != 0 else None
    return {
        "count": len(vals),
        "mean": m,
        "median": median(vals),
        "stdev": sd,
        "cv_pct": cv,
        "min": min(vals),
        "max": max(vals),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate repeated vLLM sweep CSVs into canonical robust outputs.")
    p.add_argument("--mode", required=True, choices=["concurrency", "request_rate"], help="Sweep mode")
    p.add_argument("--inputs", nargs="+", required=True, help="Input repeat CSVs")
    p.add_argument("--output-csv", required=True, help="Output aggregated CSV path")
    p.add_argument("--output-jsonl", required=True, help="Output aggregated JSONL path")
    p.add_argument("--output-stability-json", required=True, help="Output sweep stability JSON path")
    p.add_argument("--output-summary-txt", required=True, help="Output human-readable summary path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    key_cols = MODE_KEY_COLS[args.mode]
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"ERROR: missing input CSV: {p}")

    rows_by_key: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    first_row_by_key: Dict[Tuple[str, ...], Dict[str, str]] = {}
    numeric_cols: set[str] = set()
    fieldnames: List[str] = []
    invalid_rows_skipped = 0

    for p in inputs:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                completed = _to_float(row.get("completed", ""))
                failed = _to_float(row.get("failed", ""))
                total_tok = _to_float(row.get("total_token_throughput", ""))
                if (
                    completed is not None
                    and failed is not None
                    and total_tok is not None
                    and (completed <= 0 or failed > 0 or total_tok <= 0)
                ):
                    invalid_rows_skipped += 1
                    continue
                key = tuple((row.get(c) or "").strip() for c in key_cols)
                rows_by_key[key].append(row)
                first_row_by_key.setdefault(key, dict(row))
                for k, v in row.items():
                    if k in key_cols:
                        continue
                    if _to_float(v or "") is not None:
                        numeric_cols.add(k)

    ordered_keys = sorted(rows_by_key.keys(), key=lambda k: tuple(_to_sortable(x) for x in k))

    agg_rows: List[Dict[str, str]] = []
    jsonl_rows: List[Dict[str, object]] = []
    stability_points: List[Dict[str, object]] = []

    for key in ordered_keys:
        src_rows = rows_by_key[key]
        out_row = dict(first_row_by_key[key])
        metric_stats: Dict[str, Dict[str, float | None]] = {}
        for col in sorted(numeric_cols):
            vals = []
            for r in src_rows:
                f = _to_float(r.get(col, ""))
                if f is not None:
                    vals.append(f)
            s = _stats(vals)
            metric_stats[col] = s
            med = s["median"]
            out_row[col] = f"{med:.6f}" if isinstance(med, float) else ""

        # Keep key columns in canonical string form from first row.
        for i, c in enumerate(key_cols):
            out_row[c] = key[i]
        agg_rows.append(out_row)

        point = {c: out_row[c] for c in key_cols}
        point["repeat_count"] = len(src_rows)
        for m in CORE_STABILITY_METRICS:
            s = metric_stats.get(m) or {}
            point[f"{m}_cv_pct"] = s.get("cv_pct")
            point[f"{m}_min"] = s.get("min")
            point[f"{m}_max"] = s.get("max")
            point[f"{m}_median"] = s.get("median")
        stability_points.append(point)

        jr: Dict[str, object] = {c: out_row[c] for c in key_cols}
        for col in sorted(numeric_cols):
            f = _to_float(out_row.get(col, ""))
            jr[col] = f
        jr["repeat_count"] = len(src_rows)
        jr["repeat_source_files"] = [str(p) for p in inputs]
        jsonl_rows.append(jr)

    def _p95_cv(metric: str) -> float | None:
        vals = []
        k = f"{metric}_cv_pct"
        for p in stability_points:
            v = p.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return _percentile_nearest_rank(vals, 95.0)

    summary = {
        "mode": args.mode,
        "input_csvs": [str(p) for p in inputs],
        "repeat_count": len(inputs),
        "points": len(stability_points),
        "invalid_rows_skipped": invalid_rows_skipped,
        "total_token_throughput_cv_pct_p95": _p95_cv("total_token_throughput"),
        "p99_ttft_cv_pct_p95": _p95_cv("p99_ttft_ms"),
        "p99_tpot_cv_pct_p95": _p95_cv("p99_tpot_ms"),
    }

    stability_payload = {
        "mode": args.mode,
        "input_csvs": [str(p) for p in inputs],
        "repeat_count": len(inputs),
        "points": stability_points,
        "summary": summary,
    }

    out_csv = Path(args.output_csv)
    out_jsonl = Path(args.output_jsonl)
    out_stability = Path(args.output_stability_json)
    out_summary = Path(args.output_summary_txt)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_stability.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing schema ordering from first input where possible.
    if not fieldnames:
        fieldnames = key_cols + sorted(numeric_cols)
    for c in key_cols:
        if c not in fieldnames:
            fieldnames.append(c)
    for c in sorted(numeric_cols):
        if c not in fieldnames:
            fieldnames.append(c)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    out_stability.write_text(json.dumps(stability_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: List[str] = []
    title = "vLLM Concurrency Sweep Results" if args.mode == "concurrency" else "vLLM Request-Rate Sweep Results"
    lines.append("========================================")
    lines.append(title)
    lines.append("========================================")
    lines.append(f"Repeat count: {len(inputs)}")
    lines.append(f"Points: {len(stability_points)}")
    lines.append(f"Invalid rows skipped: {invalid_rows_skipped}")
    lines.append("")
    if args.mode == "concurrency":
        lines.append("Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 TPOT")
        lines.append("------------|--------------|-------------|-----------|-----------|----------")
        for row in agg_rows:
            lines.append(
                f"{int(float(row.get('concurrency','0') or 0)):<11d} | "
                f"{float(row.get('output_throughput','0') or 0):<12.2f} | "
                f"{float(row.get('total_token_throughput','0') or 0):<11.2f} | "
                f"{float(row.get('mean_ttft_ms','0') or 0):<9.2f} | "
                f"{float(row.get('mean_tpot_ms','0') or 0):<9.3f} | "
                f"{float(row.get('p99_tpot_ms','0') or 0):<9.3f}"
            )
    else:
        lines.append("Req/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 TTFT | P99 TPOT")
        lines.append("------|--------------|-------------|-----------|-----------|----------|----------")
        for row in agg_rows:
            lines.append(
                f"{float(row.get('request_rate','0') or 0):<6.2f} | "
                f"{float(row.get('output_throughput','0') or 0):<12.2f} | "
                f"{float(row.get('total_token_throughput','0') or 0):<11.2f} | "
                f"{float(row.get('mean_ttft_ms','0') or 0):<9.2f} | "
                f"{float(row.get('mean_tpot_ms','0') or 0):<9.3f} | "
                f"{float(row.get('p99_ttft_ms','0') or 0):<8.2f} | "
                f"{float(row.get('p99_tpot_ms','0') or 0):<8.3f}"
            )
    lines.append("")
    lines.append("Stability (p95 CV across points)")
    lines.append(
        f"  total_token_throughput: {summary['total_token_throughput_cv_pct_p95'] if summary['total_token_throughput_cv_pct_p95'] is not None else 'n/a'}%"
    )
    lines.append(f"  p99_ttft_ms: {summary['p99_ttft_cv_pct_p95'] if summary['p99_ttft_cv_pct_p95'] is not None else 'n/a'}%")
    lines.append(f"  p99_tpot_ms: {summary['p99_tpot_cv_pct_p95'] if summary['p99_tpot_cv_pct_p95'] is not None else 'n/a'}%")
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_stability}")
    print(f"Wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
