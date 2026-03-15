#!/usr/bin/env python3
"""Update the tier-1 history index for an existing summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.analysis.history_index import update_history_index
from core.benchmark.suites.tier1 import load_tier1_suite


def _load_summary(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read tier-1 summary JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in tier-1 summary JSON {path}, got {type(payload).__name__}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Missing run_id in tier-1 summary JSON {path}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Expected summary object in tier-1 summary JSON {path}, got {type(summary).__name__}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Update tier-1 history index from an existing summary.json.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to tier-1 summary.json")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional canonical summary path override.")
    parser.add_argument("--regression-summary", type=Path, required=True, help="Path to regression_summary.md")
    parser.add_argument("--trend-snapshot", type=Path, default=None, help="Path to trend_snapshot.json")
    parser.add_argument("--history-root", type=Path, required=True, help="History root directory.")
    parser.add_argument("--config", type=Path, default=None, help="Tier-1 YAML config path.")
    args = parser.parse_args()

    suite = load_tier1_suite(args.config)
    try:
        summary = _load_summary(args.summary_json)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    updated = update_history_index(
        history_root=args.history_root,
        suite=suite,
        summary=summary,
        summary_path=args.summary_path or args.summary_json,
        regression_summary_path=args.regression_summary,
        trend_snapshot_path=args.trend_snapshot,
    )
    print(json.dumps(updated, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
