#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote one run-local cluster result tree into the published cluster package."
    )
    parser.add_argument("--run-id", required=True, help="Run id under cluster/runs/<run_id>.")
    parser.add_argument("--label", default="localhost", help="Host label for localhost report rendering.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root for testing or custom promotion targets.",
    )
    parser.add_argument(
        "--publish-report-path",
        default="cluster/field-report-localhost.md",
        help="Published localhost report path (default: cluster/field-report-localhost.md).",
    )
    parser.add_argument(
        "--publish-notes-path",
        default="cluster/field-report-localhost-notes.md",
        help="Published localhost notes path (default: cluster/field-report-localhost-notes.md).",
    )
    parser.add_argument(
        "--skip-render-localhost-report",
        action="store_true",
        help="Skip rendering run-local + published localhost report markdown.",
    )
    parser.add_argument(
        "--skip-validate-localhost-report",
        action="store_true",
        help="Skip localhost package validation after promotion.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup_run_artifacts.sh after promotion using this run_id as canonical.",
    )
    parser.add_argument(
        "--allow-run-id",
        action="append",
        default=[],
        help="Additional run ids to retain during cleanup/validation hygiene checks.",
    )
    return parser.parse_args()


def _repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def _script_root() -> Path:
    return Path(__file__).resolve().parent


def _copy_overwrite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _sync_children(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_dir.iterdir()):
        _copy_overwrite(child, dst_dir / child.name)
        copied.append(child.name)
    return copied


def _run_subprocess(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }


def main() -> int:
    args = parse_args()
    repo_root = _repo_root(args)
    cluster_root = repo_root / "cluster"
    run_dir = cluster_root / "runs" / args.run_id
    structured_dir = run_dir / "structured"
    raw_dir = run_dir / "raw"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"
    manifest_path = run_dir / "manifest.json"

    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")

    published_root = cluster_root / "published" / "current"
    published_structured = published_root / "structured"
    published_raw = published_root / "raw"
    published_figures = published_root / "figures"
    published_reports = published_root / "reports"
    published_manifest = published_root / "manifest.json"
    publish_report_path = (repo_root / args.publish_report_path).resolve()
    publish_notes_path = (repo_root / args.publish_notes_path).resolve()

    summary: dict[str, Any] = {
        "success": True,
        "run_id": args.run_id,
        "label": args.label,
        "run_dir": str(run_dir),
        "published_root": str(published_root),
        "published_structured_dir": str(published_structured),
        "published_raw_dir": str(published_raw),
        "published_figures_dir": str(published_figures),
        "published_reports_dir": str(published_reports),
        "published_manifest_path": str(published_manifest),
        "published_localhost_report_path": str(publish_report_path),
        "published_localhost_notes_path": str(publish_notes_path),
        "steps": {},
    }

    for path in (published_structured, published_raw, published_figures, published_reports, published_manifest):
        if path.exists():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    published_root.mkdir(parents=True, exist_ok=True)
    copied_structured = _sync_children(structured_dir, published_structured)
    copied_raw = _sync_children(raw_dir, published_raw)
    copied_figures = _sync_children(figures_dir, published_figures)
    copied_reports = _sync_children(reports_dir, published_reports)
    if manifest_path.exists():
        _copy_overwrite(manifest_path, published_manifest)

    summary["steps"]["publish_package_sync"] = {
        "success": True,
        "structured": copied_structured,
        "raw": copied_raw,
        "figures": copied_figures,
        "reports": copied_reports,
        "manifest": published_manifest.exists(),
    }

    if not args.skip_render_localhost_report:
        render_script = _script_root() / "render_localhost_field_report_package.py"
        render_cmd = [
            sys.executable,
            str(render_script),
            "--run-id",
            args.run_id,
            "--label",
            args.label,
            "--root",
            str(cluster_root),
            "--run-dir",
            str(published_root),
            "--publish-report",
            str(publish_report_path),
            "--publish-notes",
            str(publish_notes_path),
        ]
        render_result = _run_subprocess(render_cmd, cwd=repo_root)
        summary["steps"]["render_localhost_report"] = render_result
        summary["success"] = summary["success"] and render_result["success"]
        if render_result["success"]:
            _copy_overwrite(publish_report_path, published_reports / publish_report_path.name)
            _copy_overwrite(publish_notes_path, published_reports / publish_notes_path.name)
            # For localhost canonical promotions, keep the top-level stakeholder report pair
            # aligned with the published localhost package instead of leaving stale rN docs in place.
            if args.label == "localhost":
                top_level_report = cluster_root / "field-report.md"
                top_level_notes = cluster_root / "field-report-notes.md"
                _copy_overwrite(publish_report_path, top_level_report)
                _copy_overwrite(publish_notes_path, top_level_notes)
                _copy_overwrite(top_level_report, published_reports / top_level_report.name)
                _copy_overwrite(top_level_notes, published_reports / top_level_notes.name)
                summary["steps"]["sync_top_level_reports"] = {
                    "success": True,
                    "field_report_path": str(top_level_report),
                    "field_report_notes_path": str(top_level_notes),
                }

    if args.cleanup:
        cleanup_script = _script_root() / "cleanup_run_artifacts.sh"
        cleanup_cmd = [
            "bash",
            str(cleanup_script),
            "--repo-root",
            str(repo_root),
            "--canonical-run-id",
            args.run_id,
            "--apply",
        ]
        for run_id in args.allow_run_id:
            if run_id:
                cleanup_cmd.extend(["--allow-run-id", run_id])
        cleanup_result = _run_subprocess(cleanup_cmd, cwd=repo_root)
        summary["steps"]["cleanup"] = cleanup_result
        summary["success"] = summary["success"] and cleanup_result["success"]

    if not args.skip_validate_localhost_report:
        validate_script = _script_root() / "validate_field_report_requirements.sh"
        validate_cmd = [
            "bash",
            str(validate_script),
            "--repo-root",
            str(repo_root),
            "--report",
            str(publish_report_path),
            "--notes",
            str(publish_notes_path),
            "--template",
            "cluster/docs/field-report-template.md",
            "--runbook",
            "cluster/docs/advanced-runbook.md",
            "--canonical-run-id",
            args.run_id,
        ]
        for run_id in args.allow_run_id:
            if run_id:
                validate_cmd.extend(["--allow-run-id", run_id])
        validate_result = _run_subprocess(validate_cmd, cwd=repo_root)
        summary["steps"]["validate_localhost_report"] = validate_result
        summary["success"] = summary["success"] and validate_result["success"]

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
