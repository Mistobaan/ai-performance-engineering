#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_numa_mem_bw.sh [options]

Probe NUMA "memory-only" nodes (e.g., GPU-attached HBM domains exposed as NUMA
nodes on Grace systems) by allocating memory on each NUMA node and measuring
CPU memcpy bandwidth.

Outputs:
  runs/<run_id>/structured/<run_id>_<label>_numa_mem_bw.json
  runs/<run_id>/structured/<run_id>_<label>_numa_mem_bw.csv

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>        Label for output paths (default: hostname)
  --bytes <n>            Bytes per memcpy iteration (default: 1073741824)
  --iters <n>            Iterations (default: 10)
  --threads <n>          Threads (default: 16)
  --warmup <n>           Warmup iterations (default: 2)
  --nodes <csv>          Optional explicit NUMA nodes to test (default: auto)
  --cpu-node <n>         CPU node to run on (default: first CPU NUMA node)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
LABEL="${LABEL:-$(hostname)}"
BYTES="${BYTES:-1073741824}"   # 1GiB
ITERS="${ITERS:-10}"
THREADS="${THREADS:-16}"
WARMUP="${WARMUP:-2}"
NODES="${NODES:-}"
CPU_NODE="${CPU_NODE:-}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --cpu-node) CPU_NODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v numactl >/dev/null 2>&1; then
  echo "ERROR: numactl not found." >&2
  exit 1
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_STRUCT_DIR"
OUT_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_numa_mem_bw.json"
OUT_CSV="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_numa_mem_bw.csv"

SRC="${ROOT_DIR}/scripts/benchmarks/mem_bw_bench.c"
BIN_DIR="/tmp/aisp_cluster_bench"
BIN="${BIN_DIR}/mem_bw_bench"
mkdir -p "$BIN_DIR"

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found; cannot build NUMA mem bw bench." >&2
    exit 1
  fi
  echo "Building ${BIN}..."
  gcc -O3 -pthread -o "$BIN" "$SRC"
fi

NUMA_TXT="$(numactl -H)"

detect_cpu_nodes() {
  echo "$NUMA_TXT" | awk '
    /^node [0-9]+ cpus:/ {
      node=$2
      # cpu list is everything after "cpus:"
      sub(/^node [0-9]+ cpus:[[:space:]]*/, "", $0)
      if (length($0) > 0) print node
    }
  '
}

detect_mem_only_nodes() {
  # A node is "memory-only" if it has size>0 and no CPUs.
  echo "$NUMA_TXT" | awk '
    BEGIN { for(i=0;i<1024;i++){cpus[i]=0; size[i]=0;} }
    /^node [0-9]+ cpus:/ {
      node=$2
      sub(/^node [0-9]+ cpus:[[:space:]]*/, "", $0)
      if (length($0) > 0) cpus[node]=1
    }
    /^node [0-9]+ size:/ {
      node=$2
      mb=$4
      if (mb ~ /^[0-9]+$/) size[node]=mb
    }
    END {
      for (n in size) {
        if (size[n] > 0 && cpus[n] == 0) print n
      }
    }
  ' | sort -n
}

if [[ -z "$CPU_NODE" ]]; then
  CPU_NODE="$(detect_cpu_nodes | head -n 1 || true)"
fi
if [[ -z "$CPU_NODE" ]]; then
  echo "ERROR: unable to auto-detect a CPU NUMA node from numactl -H." >&2
  exit 2
fi

nodes_to_test=()
if [[ -n "$NODES" ]]; then
  IFS=',' read -r -a nodes_to_test <<<"$NODES"
else
  # Always include CPU nodes for baseline + any memory-only nodes with capacity.
  while IFS= read -r n; do nodes_to_test+=("$n"); done < <(detect_cpu_nodes)
  while IFS= read -r n; do nodes_to_test+=("$n"); done < <(detect_mem_only_nodes)
fi

if [[ "${#nodes_to_test[@]}" -eq 0 ]]; then
  echo "ERROR: no NUMA nodes selected." >&2
  exit 2
fi

tmp_jsonl="$(mktemp)"
trap 'rm -f "$tmp_jsonl"' EXIT
: >"$tmp_jsonl"

echo "========================================"
echo "NUMA Memory BW Probe"
echo "========================================"
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "CPU_NODE=${CPU_NODE}"
echo "BYTES=${BYTES} ITERS=${ITERS} THREADS=${THREADS} WARMUP=${WARMUP}"
echo "Nodes: ${nodes_to_test[*]}"
echo

for n in "${nodes_to_test[@]}"; do
  n="$(echo "$n" | xargs)"
  [[ -n "$n" ]] || continue

  echo "== node ${n} =="
  set +e
  out="$(
    numactl --cpunodebind="${CPU_NODE}" --membind="${n}" \
      "$BIN" --bytes "$BYTES" --iters "$ITERS" --threads "$THREADS" --warmup "$WARMUP" 2>/dev/null
  )"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 || -z "$out" ]]; then
    echo "  ERROR: mem_bw_bench failed for node=${n} rc=${rc}" >&2
    echo "{\"node\": ${n}, \"rc\": ${rc}, \"error\": \"mem_bw_bench_failed\"}" >>"$tmp_jsonl"
    continue
  fi

  python3 - <<'PY' "$tmp_jsonl" "$n" "$out"
import json
import sys

out_path = sys.argv[1]
node = int(sys.argv[2])
raw = sys.argv[3]
d = json.loads(raw)
d["node"] = node
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(d, sort_keys=True) + "\n")
print(f"  bw_gbps={d.get('bw_gbps')}")
PY
done

python3 - <<'PY' "$OUT_JSON" "$OUT_CSV" "$RUN_ID" "$LABEL" "$CPU_NODE" "$BYTES" "$ITERS" "$THREADS" "$WARMUP" "$tmp_jsonl"
import csv
import json
import sys
from pathlib import Path

out_json, out_csv, run_id, label = sys.argv[1:5]
cpu_node = int(sys.argv[5])
bytes_per = int(sys.argv[6])
iters = int(sys.argv[7])
threads = int(sys.argv[8])
warmup = int(sys.argv[9])
jsonl_path = Path(sys.argv[10])

rows = []
for line in jsonl_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rows.append(json.loads(line))

payload = {
    "run_id": run_id,
    "label": label,
    "cpu_node": cpu_node,
    "bytes": bytes_per,
    "iters": iters,
    "threads": threads,
    "warmup": warmup,
    "results": rows,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

cols = ["run_id", "label", "cpu_node", "node", "bytes", "iters", "threads", "warmup", "elapsed_s", "bw_gbps", "rc", "error"]
write_header = True
csv_path = Path(out_csv)
if csv_path.exists():
    try:
        if csv_path.read_text(encoding="utf-8").strip():
            write_header = False
    except Exception:
        pass

with csv_path.open("a", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(cols)
    for r in rows:
        w.writerow(
            [
                run_id,
                label,
                cpu_node,
                r.get("node", ""),
                r.get("bytes", ""),
                r.get("iters", ""),
                r.get("threads", ""),
                r.get("warmup", ""),
                r.get("elapsed_s", ""),
                r.get("bw_gbps", ""),
                r.get("rc", 0),
                r.get("error", ""),
            ]
        )

print(out_json)
print(out_csv)
PY

echo "Wrote ${OUT_JSON}"
echo "Wrote ${OUT_CSV}"
