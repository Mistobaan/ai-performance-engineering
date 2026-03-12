#!/usr/bin/env python3
"""Select auditable expectation-refresh candidate target lists from a rejection ledger."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def select_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    recommendations: List[str],
    include_non_material: bool = False,
) -> List[Dict[str, Any]]:
    allowed = set(recommendations)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("refresh_recommendation") not in allowed:
            continue
        if not include_non_material and not row.get("material_mismatch"):
            continue
        selected.append(dict(row))
    selected.sort(key=lambda row: (row.get("chapter") or "", row.get("example") or ""))
    return selected


def write_candidate_outputs(
    *,
    ledger_json: Path,
    output_dir: Path,
    recommendations: List[str],
    include_non_material: bool = False,
) -> Dict[str, Path]:
    rows = _load_rows(ledger_json)
    selected = select_candidates(
        rows,
        recommendations=recommendations,
        include_non_material=include_non_material,
    )
    recommendation_slug = "_".join(recommendations)
    suffix = "all" if include_non_material else "material"
    prefix = f"{ledger_json.stem}__{recommendation_slug}__{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "ledger_json": str(ledger_json),
        "recommendations": recommendations,
        "include_non_material": include_non_material,
        "selected_count": len(selected),
        "counts_by_recommendation": dict(Counter(row.get("refresh_recommendation") for row in selected)),
        "targets": [row["target"] for row in selected],
    }

    summary_path = output_dir / f"{prefix}.summary.json"
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    txt_path = output_dir / f"{prefix}.txt"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")

    if selected:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
    else:
        csv_path.write_text("", encoding="utf-8")

    txt_path.write_text("\n".join(row["target"] for row in selected) + ("\n" if selected else ""), encoding="utf-8")
    return {
        "summary": summary_path,
        "json": json_path,
        "csv": csv_path,
        "txt": txt_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select expectation-refresh candidates from a rejection ledger.")
    parser.add_argument("--ledger-json", type=Path, required=True, help="Ledger JSON from analyze_expectation_rejections.py")
    parser.add_argument(
        "--recommendation",
        action="append",
        required=True,
        help="refresh_recommendation value(s) to include; repeat as needed",
    )
    parser.add_argument(
        "--include-non-material",
        action="store_true",
        help="Include rows below the material mismatch threshold",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to the ledger's parent directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_candidate_outputs(
        ledger_json=args.ledger_json,
        output_dir=args.output_dir or args.ledger_json.parent,
        recommendations=args.recommendation,
        include_non_material=args.include_non_material,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
