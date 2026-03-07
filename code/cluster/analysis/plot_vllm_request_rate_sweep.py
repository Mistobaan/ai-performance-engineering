#!/usr/bin/env python3
"""Plot vLLM request-rate sweep CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_is_valid(row: dict[str, str]) -> bool:
    completed = _as_float(row.get("completed"))
    failed = _as_float(row.get("failed"))
    total_tok = _as_float(row.get("total_token_throughput"))
    if completed is None or completed <= 0:
        return False
    if failed is None or failed > 0:
        return False
    if total_tok is None or total_tok <= 0:
        return False
    return True


def _group_mean(rows: list[dict[str, str]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        if not _row_is_valid(row):
            continue
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        groups.setdefault(x, []).append(y)
    xs = sorted(groups.keys())
    ys = [mean(groups[x]) for x in xs]
    return xs, ys


def _plot(xs: list[float], series: list[tuple[str, list[float]]], output: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not xs:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
    else:
        for label, ys in series:
            if ys:
                ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(series) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot vLLM request-rate sweep CSV.")
    p.add_argument("--input", required=True, help="Structured CSV from request-rate sweep")
    p.add_argument("--out-dir", required=True, help="Directory for output figures")
    p.add_argument("--run-id", default="run", help="Prefix for output files")
    args = p.parse_args()

    rows = _load_rows(Path(args.input))
    out_dir = Path(args.out_dir)

    xs, total_tok = _group_mean(rows, "request_rate", "total_token_throughput")
    _plot(
        xs,
        [("total tok/s", total_tok)],
        out_dir / f"{args.run_id}_vllm_serve_request_rate_total_tok_s.png",
        "vLLM throughput vs request rate",
        "Tokens/sec",
    )

    xs, p99_ttft = _group_mean(rows, "request_rate", "p99_ttft_ms")
    _, p99_tpot = _group_mean(rows, "request_rate", "p99_tpot_ms")
    _plot(
        xs,
        [("p99 TTFT", p99_ttft), ("p99 TPOT", p99_tpot)],
        out_dir / f"{args.run_id}_vllm_serve_request_rate_tail_latency_ms.png",
        "vLLM tail latency vs request rate",
        "Latency (ms)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
