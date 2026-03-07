#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_label(raw: str) -> str:
    return raw.replace(".", "_").replace(":", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a manifest JSON for a cluster eval RUN_ID.")
    p.add_argument("--root", default="", help="Cluster root (default: inferred from this script location)")
    p.add_argument("--run-id", required=True, help="RUN_ID prefix")
    p.add_argument("--run-dir", default="", help="Explicit run directory (default: <cluster_root>/runs/<run_id> when present)")
    p.add_argument("--hosts", default="", help="Comma-separated host list (optional)")
    p.add_argument("--labels", default="", help="Comma-separated labels (optional; must match host count)")
    p.add_argument(
        "--include-figures",
        action="store_true",
        help="Include figure artifacts in the manifest.",
    )
    return p.parse_args()


def _resolve_cluster_root(raw_root: str) -> Path:
    return Path(raw_root).resolve() if raw_root else Path(__file__).resolve().parents[1]


def _resolve_run_dir(cluster_root: Path, run_id: str, raw_run_dir: str) -> Path | None:
    if raw_run_dir:
        return Path(raw_run_dir).resolve()
    candidate = cluster_root / "runs" / run_id
    if candidate.exists():
        return candidate
    return None


def _collect_run_dir_paths(run_dir: Path, include_figures: bool) -> List[Path]:
    paths: List[Path] = []
    for subdir in ("structured", "raw", "reports"):
        base = run_dir / subdir
        if not base.exists():
            continue
        paths.extend(sorted(path for path in base.rglob("*") if path.is_file()))
    if include_figures:
        figures_dir = run_dir / "figures"
        if figures_dir.exists():
            paths.extend(sorted(path for path in figures_dir.rglob("*") if path.is_file()))
    return sorted(set(paths))


def _collect_legacy_paths(cluster_root: Path, run_id: str, include_figures: bool) -> List[Path]:
    struct_dir = cluster_root / "results" / "structured"
    raw_dir = cluster_root / "results" / "raw"
    fig_dir = cluster_root / "docs" / "figures"

    paths_set = set()
    if struct_dir.exists():
        for p in struct_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
    if raw_dir.exists():
        for p in raw_dir.rglob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
            elif p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file():
                        paths_set.add(fp)
    if include_figures and fig_dir.exists():
        for p in fig_dir.glob(f"{run_id}_*"):
            if p.is_file():
                paths_set.add(p)
    return sorted(paths_set)


def main() -> int:
    args = parse_args()
    cluster_root = _resolve_cluster_root(args.root)
    run_id = args.run_id
    run_dir = _resolve_run_dir(cluster_root, run_id, args.run_dir)

    if run_dir is not None:
        paths = _collect_run_dir_paths(run_dir, args.include_figures)
        files = [str(p.relative_to(run_dir)) for p in paths]
        hashes = {str(p.relative_to(run_dir)): _sha256(p) for p in paths}
        out_path = run_dir / "manifest.json"
        artifact_root = str(run_dir.relative_to(cluster_root))
        manifest_mode = "run_dir"
    else:
        paths = _collect_legacy_paths(cluster_root, run_id, args.include_figures)
        files = [str(p.relative_to(cluster_root)) for p in paths]
        hashes = {str(p.relative_to(cluster_root)): _sha256(p) for p in paths}
        out_path = cluster_root / "results" / "structured" / f"{run_id}_manifest.json"
        artifact_root = "results"
        manifest_mode = "legacy_flat"

    artifact_counts: Dict[str, int] = {}
    for p in paths:
        suffix = p.suffix.lstrip(".") or "no_ext"
        artifact_counts[suffix] = artifact_counts.get(suffix, 0) + 1

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if labels and len(labels) != len(hosts):
        raise SystemExit("--labels count must match --hosts count")

    nodes: List[Dict[str, Any]] = []
    for i, h in enumerate(hosts):
        label = labels[i] if labels else _sanitize_label(h)
        nodes.append({"label": label, "host": h})

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "run_id": run_id,
        "artifact_root": artifact_root,
        "manifest_mode": manifest_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "files": files,
        "summary": {
            "file_count": len(files),
            "artifact_counts": artifact_counts,
            "sha256": hashes,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
