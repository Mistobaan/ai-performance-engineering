#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${BASE_DIR}/scripts/lib_artifact_dirs.sh"
STRUCT_DIR="${CLUSTER_RESULTS_STRUCTURED_DIR:-${BASE_DIR}/results/structured}"
RUN_ID="${RUN_ID:-}"

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/export_tcp_sysctl_json.sh --run-id <id> [--structured-dir <path>]

Exports:
  - results/structured/<run_id>_<label>_tcp_sysctl.json (per node)
  - results/structured/<run_id>_tcp_sysctl_diff.json   (keys where nodes differ)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --structured-dir) STRUCT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: --run-id is required" >&2
  usage >&2
  exit 2
fi

python3 - <<'PY' "$STRUCT_DIR" "$RUN_ID"
import json
from pathlib import Path
import sys

struct_dir = Path(sys.argv[1])
run_id = sys.argv[2]
txt_files = sorted(struct_dir.glob(f"{run_id}_*_tcp_sysctl.txt"))
if not txt_files:
    raise SystemExit(f"No tcp_sysctl txt files found for run_id={run_id} under {struct_dir}")

def label_from_path(p: Path) -> str:
    name = p.name
    prefix = f"{run_id}_"
    suffix = "_tcp_sysctl.txt"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return p.stem
    return name[len(prefix) : -len(suffix)]

# Export JSON per node
labels = []
for txt in txt_files:
    node = label_from_path(txt)
    labels.append(node)
    data = {}
    for line in txt.read_text().splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        data[key.strip()] = value.strip()
    out_path = struct_dir / f"{run_id}_{node}_tcp_sysctl.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {out_path}")

# Build diff JSON
values = {}
for node in labels:
    data = json.loads((struct_dir / f"{run_id}_{node}_tcp_sysctl.json").read_text())
    for k, v in data.items():
        values.setdefault(k, {})[node] = v

diffs = {k: v for k, v in values.items() if len(set(v.values())) > 1}

out = struct_dir / f"{run_id}_tcp_sysctl_diff.json"
out.write_text(json.dumps(diffs, indent=2, sort_keys=True))
print(f"wrote {out}")
PY
