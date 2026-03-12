#!/usr/bin/env python3
"""Refresh expectation files from a validated benchmark run artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.expectations import ExpectationEntry, ExpectationsStore
from core.harness.validity_checks import detect_execution_environment


@dataclass
class RefreshRecord:
    target: str
    expectation_file: str
    status: str
    message: str


def _load_results(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_validity_profile(results_json: Path) -> Optional[str]:
    events_path = results_json.parent.parent / "logs" / "benchmark_events.jsonl"
    if not events_path.exists():
        return None
    for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") == "run_start":
            value = event.get("validity_profile")
            if value:
                return str(value)
    return None


def _load_targets(targets: Sequence[str], targets_file: Optional[Path]) -> Optional[Set[str]]:
    loaded = set(targets)
    if targets_file:
        loaded.update(line.strip() for line in targets_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return loaded or None


def _chapter_dir(repo_root: Path, chapter: str) -> Path:
    direct = repo_root / chapter
    if direct.exists():
        return direct
    nested = repo_root / chapter.replace("_", "/")
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Could not resolve chapter directory for {chapter!r}")


def _entry_targets(chapter: str, example: str, example_type: str) -> Set[str]:
    targets = {f"{chapter}:{example}"}
    if example_type == "cuda":
        targets.add(f"{chapter}:{example}_cuda")
    return targets


def _expectation_example_key(example: str, example_type: str) -> str:
    return example if example_type == "python" else f"{example}_{example_type}"


def _prepare_entry(entry_dict: Dict, *, validity_profile: Optional[str]) -> ExpectationEntry:
    entry = ExpectationEntry.from_dict(entry_dict)
    env = detect_execution_environment()
    if not entry.provenance.execution_environment or entry.provenance.execution_environment == "unknown":
        entry.provenance.execution_environment = env.kind
    if not entry.provenance.validity_profile and validity_profile:
        entry.provenance.validity_profile = validity_profile
    if not entry.provenance.dmi_product_name:
        entry.provenance.dmi_product_name = env.dmi_product_name
    return entry


def refresh_expectations_from_run(
    *,
    results_json: Path,
    repo_root: Path,
    targets: Optional[Set[str]] = None,
    validity_profile: Optional[str] = None,
    accept_regressions: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    results = _load_results(results_json)
    effective_validity_profile = validity_profile or _infer_validity_profile(results_json)
    chapter_results = list(results.get("results") or [])
    stores: Dict[Tuple[Path, str], ExpectationsStore] = {}
    records: List[RefreshRecord] = []

    for chapter_result in chapter_results:
        chapter = str(chapter_result.get("chapter"))
        chapter_dir = _chapter_dir(repo_root, chapter)
        for bench in chapter_result.get("benchmarks") or []:
            example = str(bench.get("example"))
            example_type = str(bench.get("type") or "python")
            candidate_targets = _entry_targets(chapter, example, example_type)
            if targets is not None and candidate_targets.isdisjoint(targets):
                continue

            expectation = bench.get("expectation") or {}
            entry_dict = expectation.get("entry")
            if not isinstance(entry_dict, dict):
                records.append(
                    RefreshRecord(
                        target=sorted(candidate_targets)[0],
                        expectation_file="",
                        status="skipped",
                        message="missing expectation preview entry in run results",
                    )
                )
                continue

            entry = _prepare_entry(entry_dict, validity_profile=effective_validity_profile)
            store_key = (chapter_dir, entry.provenance.hardware_key)
            store = stores.get(store_key)
            if store is None:
                store = ExpectationsStore(
                    chapter_dir,
                    entry.provenance.hardware_key,
                    accept_regressions=accept_regressions,
                    allow_mixed_provenance=True,
                )
                stores[store_key] = store

            example_key = _expectation_example_key(entry.example, entry.type)
            update = store.update_entry(example_key, entry)
            records.append(
                RefreshRecord(
                    target=sorted(candidate_targets)[0],
                    expectation_file=str(store.path.relative_to(repo_root)),
                    status=update.status,
                    message=update.message,
                )
            )

    if not dry_run:
        for store in stores.values():
            store.save(force=True)

    summary = {
        "results_json": str(results_json),
        "dry_run": dry_run,
        "accept_regressions": accept_regressions,
        "validity_profile": effective_validity_profile,
        "updated_files": [str(store.path.relative_to(repo_root)) for store in stores.values()],
        "records": [asdict(record) for record in records],
        "counts": {
            "applied": sum(1 for r in records if r.status in {"updated", "improved", "regressed", "unchanged"}),
            "improved": sum(1 for r in records if r.status == "improved"),
            "regressed": sum(1 for r in records if r.status == "regressed"),
            "unchanged": sum(1 for r in records if r.status == "unchanged"),
            "updated": sum(1 for r in records if r.status == "updated"),
            "rejected": sum(1 for r in records if r.status == "rejected"),
            "skipped": sum(1 for r in records if r.status == "skipped"),
            "total": len(records),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh expectations from benchmark run artifacts.")
    parser.add_argument("--results-json", type=Path, required=True, help="Run results JSON (benchmark_test_results.json)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root")
    parser.add_argument("--target", action="append", default=[], help="Specific target(s) to refresh")
    parser.add_argument("--targets-file", type=Path, default=None, help="File containing one target per line")
    parser.add_argument("--validity-profile", default=None, help="Fallback validity profile when missing in run entry")
    parser.add_argument("--no-accept-regressions", action="store_true", help="Reject downward expectation updates")
    parser.add_argument("--dry-run", action="store_true", help="Compute updates without writing files")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path to write summary JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = refresh_expectations_from_run(
        results_json=args.results_json,
        repo_root=args.repo_root,
        targets=_load_targets(args.target, args.targets_file),
        validity_profile=args.validity_profile,
        accept_regressions=not args.no_accept_regressions,
        dry_run=args.dry_run,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
