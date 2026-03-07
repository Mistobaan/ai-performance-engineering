#!/usr/bin/env python3
"""Analyze vLLM request-rate sweep results against SLOs and compute goodput."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List


@dataclass
class Point:
    request_rate: float
    request_throughput: float
    total_token_throughput: float
    p99_ttft_ms: float
    p99_tpot_ms: float
    slo_pass: bool
    goodput_req_s: float
    goodput_tok_s: float


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_mean(rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    grouped: Dict[float, Dict[str, List[float]]] = {}
    for row in rows:
        rate = _as_float(row.get("request_rate"), 0.0)
        if rate <= 0:
            continue
        completed = int(_as_float(row.get("completed"), 0))
        failed = int(_as_float(row.get("failed"), 0))
        total_tok = _as_float(row.get("total_token_throughput"), 0.0)
        if completed <= 0 or failed > 0 or total_tok <= 0.0:
            continue
        bucket = grouped.setdefault(
            rate,
            {
                "request_throughput": [],
                "total_token_throughput": [],
                "p99_ttft_ms": [],
                "p99_tpot_ms": [],
            },
        )
        bucket["request_throughput"].append(_as_float(row.get("request_throughput"), 0.0))
        bucket["total_token_throughput"].append(_as_float(row.get("total_token_throughput"), 0.0))
        bucket["p99_ttft_ms"].append(_as_float(row.get("p99_ttft_ms"), 0.0))
        bucket["p99_tpot_ms"].append(_as_float(row.get("p99_tpot_ms"), 0.0))

    out: List[Dict[str, float]] = []
    for rate in sorted(grouped):
        g = grouped[rate]
        out.append(
            {
                "request_rate": rate,
                "request_throughput": mean(g["request_throughput"]) if g["request_throughput"] else 0.0,
                "total_token_throughput": mean(g["total_token_throughput"]) if g["total_token_throughput"] else 0.0,
                "p99_ttft_ms": mean(g["p99_ttft_ms"]) if g["p99_ttft_ms"] else 0.0,
                "p99_tpot_ms": mean(g["p99_tpot_ms"]) if g["p99_tpot_ms"] else 0.0,
            }
        )
    return out


def _meets_slo(value: float, threshold_ms: float) -> bool:
    if threshold_ms <= 0:
        return True
    if value <= 0:
        return False
    return value <= threshold_ms


def _compute_points(grouped_rows: List[Dict[str, float]], slo_p99_ttft_ms: float, slo_p99_tpot_ms: float) -> List[Point]:
    out: List[Point] = []
    for row in grouped_rows:
        rate = row["request_rate"]
        req_s = row["request_throughput"]
        tok_s = row["total_token_throughput"]
        p99_ttft_ms = row["p99_ttft_ms"]
        p99_tpot_ms = row["p99_tpot_ms"]
        pass_slo = _meets_slo(p99_ttft_ms, slo_p99_ttft_ms) and _meets_slo(p99_tpot_ms, slo_p99_tpot_ms)
        out.append(
            Point(
                request_rate=rate,
                request_throughput=req_s,
                total_token_throughput=tok_s,
                p99_ttft_ms=p99_ttft_ms,
                p99_tpot_ms=p99_tpot_ms,
                slo_pass=pass_slo,
                goodput_req_s=req_s if pass_slo else 0.0,
                goodput_tok_s=tok_s if pass_slo else 0.0,
            )
        )
    return out


def _find_knee(points: List[Point]) -> float | None:
    if not points:
        return None
    saw_pass = False
    for point in points:
        if point.slo_pass:
            saw_pass = True
            continue
        if saw_pass:
            return point.request_rate
    if any(point.slo_pass for point in points):
        return None
    return points[0].request_rate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze vLLM request-rate SLO goodput.")
    p.add_argument("--input", required=True, help="Input request-rate sweep CSV")
    p.add_argument("--run-id", default="", help="Optional run id")
    p.add_argument("--label", default="", help="Optional label")
    p.add_argument("--slo-p99-ttft-ms", type=float, default=2000.0)
    p.add_argument("--slo-p99-tpot-ms", type=float, default=200.0)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    grouped = _group_mean(rows)
    points = _compute_points(grouped, args.slo_p99_ttft_ms, args.slo_p99_tpot_ms)
    if not points:
        raise SystemExit(f"No valid rows found in {args.input}")

    peak_total_tok = max((p.total_token_throughput for p in points), default=0.0)
    peak_total_req = max((p.request_throughput for p in points), default=0.0)
    max_goodput_tok = max((p.goodput_tok_s for p in points), default=0.0)
    max_goodput_req = max((p.goodput_req_s for p in points), default=0.0)

    peak_total_tok_point = max(points, key=lambda p: p.total_token_throughput)
    max_goodput_tok_point = max(points, key=lambda p: p.goodput_tok_s)

    summary = {
        "request_rate_points": len(points),
        "peak_total_tok_s": peak_total_tok,
        "peak_total_req_s": peak_total_req,
        "max_goodput_tok_s": max_goodput_tok,
        "max_goodput_req_s": max_goodput_req,
        "request_rate_at_peak_total_tok_s": peak_total_tok_point.request_rate,
        "request_rate_at_max_goodput_tok_s": max_goodput_tok_point.request_rate,
        "p99_ttft_ms_at_max_goodput_tok_s": max_goodput_tok_point.p99_ttft_ms,
        "p99_tpot_ms_at_max_goodput_tok_s": max_goodput_tok_point.p99_tpot_ms,
        "knee_request_rate": _find_knee(points),
        "goodput_efficiency_tok_ratio": (max_goodput_tok / peak_total_tok) if peak_total_tok > 0 else 0.0,
        "goodput_efficiency_req_ratio": (max_goodput_req / peak_total_req) if peak_total_req > 0 else 0.0,
        "any_point_meets_slo": any(point.slo_pass for point in points),
        "all_points_meet_slo": all(point.slo_pass for point in points),
    }

    payload = {
        "status": "ok",
        "run_id": args.run_id,
        "label": args.label,
        "input_csv": args.input,
        "slo_thresholds_ms": {
            "p99_ttft_ms": args.slo_p99_ttft_ms,
            "p99_tpot_ms": args.slo_p99_tpot_ms,
        },
        "summary": summary,
        "points": [
            {
                "request_rate": p.request_rate,
                "request_throughput": p.request_throughput,
                "total_token_throughput": p.total_token_throughput,
                "p99_ttft_ms": p.p99_ttft_ms,
                "p99_tpot_ms": p.p99_tpot_ms,
                "slo_pass": p.slo_pass,
                "goodput_req_s": p.goodput_req_s,
                "goodput_tok_s": p.goodput_tok_s,
            }
            for p in points
        ],
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_rate",
                "request_throughput",
                "total_token_throughput",
                "p99_ttft_ms",
                "p99_tpot_ms",
                "slo_pass",
                "goodput_req_s",
                "goodput_tok_s",
            ],
        )
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "request_rate": f"{p.request_rate:.6f}",
                    "request_throughput": f"{p.request_throughput:.6f}",
                    "total_token_throughput": f"{p.total_token_throughput:.6f}",
                    "p99_ttft_ms": f"{p.p99_ttft_ms:.6f}",
                    "p99_tpot_ms": f"{p.p99_tpot_ms:.6f}",
                    "slo_pass": "1" if p.slo_pass else "0",
                    "goodput_req_s": f"{p.goodput_req_s:.6f}",
                    "goodput_tok_s": f"{p.goodput_tok_s:.6f}",
                }
            )

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
