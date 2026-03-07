#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  cluster/scripts/cleanup_run_artifacts.sh --canonical-run-id <id> [options]

Removes superseded run artifacts from:
  cluster/runs/
  cluster/results/structured/
  cluster/results/raw/
  cluster/docs/figures/

Options:
  --repo-root <path>         Override repo root (default: auto-detect from script path)
  --canonical-run-id <id>   Canonical run id to retain (required)
  --allow-run-id <id>       Additional run id to retain (repeatable)
  --apply                   Execute deletions (default: dry-run)
  --verbose                 Print per-file actions
  -h, --help                Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANONICAL_RUN_ID=""
ALLOW_RUN_IDS=()
APPLY=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --canonical-run-id)
      CANONICAL_RUN_ID="$2"
      shift 2
      ;;
    --allow-run-id)
      ALLOW_RUN_IDS+=("$2")
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$CANONICAL_RUN_ID" ]]; then
  echo "ERROR: --canonical-run-id is required" >&2
  usage
  exit 2
fi

python3 - "$ROOT_DIR" "$CANONICAL_RUN_ID" "$APPLY" "$VERBOSE" "${ALLOW_RUN_IDS[@]}" <<'PY'
import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
canonical = sys.argv[2]
apply = sys.argv[3] == "1"
verbose = sys.argv[4] == "1"
allow = set(sys.argv[5:])

structured = root / "cluster" / "results" / "structured"
raw = root / "cluster" / "results" / "raw"
figures = root / "cluster" / "docs" / "figures"
runs = root / "cluster" / "runs"

manifest_run_pattern = re.compile(r"^(20\d{2}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9._-]*)_manifest\.json$")
date_prefixed_pattern = re.compile(r"^20\d{2}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9._-]*")
orphan_pattern = re.compile(r"^_[A-Za-z0-9].*")
mcp_smoke_pattern = re.compile(r"^mcp_cluster_smoke_test(?:_[A-Za-z0-9._-]+)?$")

keep = {canonical} | allow

run_ids: set[str] = set()
if structured.exists():
    for path in structured.glob("*_manifest.json"):
        m = manifest_run_pattern.match(path.name)
        if m:
            run_ids.add(m.group(1))
if runs.exists():
    for path in runs.iterdir():
        if path.is_dir() and date_prefixed_pattern.match(path.name):
            run_ids.add(path.name)

stale_run_ids = sorted(run_ids - keep)

stale_paths: set[Path] = set()
def belongs_to_run(path_name: str, run_id: str) -> bool:
    return (
        path_name == run_id
        or path_name.startswith(f"{run_id}_")
        or path_name.startswith(f"{run_id}.")
    )

for base in (structured, raw, figures):
    if not base.exists():
        continue
    for path in base.iterdir():
        if not (path.is_file() or path.is_dir()):
            continue
        # Explicit stale run ids discovered via manifest files.
        if any(belongs_to_run(path.name, run_id) for run_id in stale_run_ids):
            stale_paths.add(path)
            continue
        # Also prune date-prefixed leftovers not tied to canonical/allowlist run-id prefixes.
        if date_prefixed_pattern.match(path.name) and not any(
            belongs_to_run(path.name, run_id) for run_id in keep
        ):
            stale_paths.add(path)
            continue
        # Prune orphan outputs from earlier runs that were written without run ids.
        if orphan_pattern.match(path.name):
            stale_paths.add(path)
            continue
        # Prune ad-hoc smoke outputs that are not part of canonical run packaging.
        if mcp_smoke_pattern.match(path.name):
            stale_paths.add(path)

if runs.exists():
    for path in runs.iterdir():
        if not path.is_dir():
            continue
        if path.name in keep:
            continue
        if date_prefixed_pattern.match(path.name) or mcp_smoke_pattern.match(path.name):
            stale_paths.add(path)

stale_paths = sorted(stale_paths)

print(f"canonical_run_id={canonical}")
print(f"allow_run_ids={','.join(sorted(allow)) if allow else '<none>'}")
print(f"stale_run_ids={len(stale_run_ids)}")
for run_id in stale_run_ids:
    print(f"STALE_RUN {run_id}")
print(f"stale_paths={len(stale_paths)}")
print(f"stale_files={len(stale_paths)}")

if not stale_paths:
    print("No stale artifacts found.")
    raise SystemExit(0)

if not apply:
    print("Dry-run only. Re-run with --apply to delete stale artifacts.")
    raise SystemExit(0)

for path in stale_paths:
    if verbose:
        print(f"DELETE {path}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)

print(f"Deleted {len(stale_paths)} stale artifact paths.")
PY
