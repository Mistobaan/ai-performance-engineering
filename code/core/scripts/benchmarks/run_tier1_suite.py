#!/usr/bin/env python3
"""Run the tier-1 canonical benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.benchmark.suites.tier1 import default_tier1_config_path, run_tier1_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the tier-1 canonical benchmark suite.")
    parser.add_argument("--config", type=Path, default=default_tier1_config_path(), help="Path to tier-1 YAML config.")
    parser.add_argument("--history-root", type=Path, default=None, help="Override history root (default: artifacts/history/tier1).")
    parser.add_argument("--bench-root", type=Path, default=None, help="Repo root override for benchmark discovery.")
    parser.add_argument("--profile", default=None, help="Override profile type for the full suite.")
    parser.add_argument("--format", default=None, help="Output format override: json, markdown, or both.")
    parser.add_argument("--suite-timeout", type=int, default=14400, help="Suite timeout in seconds.")
    parser.add_argument("--iterations", type=int, default=None, help="Override benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=None, help="Override benchmark warmup iterations.")
    parser.add_argument("--artifacts-dir", default=None, help="Override benchmark artifacts root.")
    parser.add_argument("--run-id", default=None, help="Explicit run id.")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU visibility.")
    args = parser.parse_args()

    result = run_tier1_suite(
        config_path=args.config,
        history_root=args.history_root,
        bench_root=args.bench_root,
        profile_type=args.profile,
        output_format=args.format,
        suite_timeout=args.suite_timeout,
        iterations=args.iterations,
        warmup=args.warmup,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        single_gpu=args.single_gpu,
    )
    print(
        json.dumps(
            {
                "run_id": result["execution"]["run_id"],
                "summary_path": str(result["summary_path"]),
                "regression_summary_path": str(result["regression_summary_path"]),
                "trend_snapshot_path": str(result["trend_snapshot_path"]),
                "history_root": str(result["history_root"]),
                "warnings": result.get("warnings", []),
            },
            indent=2,
        )
    )
    return 1 if int(result["execution"].get("total_failed", 0) or 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
