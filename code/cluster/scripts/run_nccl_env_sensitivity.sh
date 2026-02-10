#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_nccl_env_sensitivity.sh --hosts <h1,h2,...> [options]

Runs an NCCL all-reduce sensitivity sweep over selected environment knobs:
  - NCCL_CROSS_NIC
  - NCCL_IB_QPS_PER_CONNECTION
  - NCCL_MIN_CTAS / NCCL_MAX_CTAS

Each profile runs the standard `run_nccl_all_reduce.sh` path and emits:
  - per-profile structured NCCL output:
      results/structured/<run_id>_nccl_env_<profile>_nccl.json
  - consolidated summary:
      results/structured/<run_id>_nccl_env_sensitivity.json

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --min-bytes <size>     Minimum NCCL message size (default: 1M)
  --max-bytes <size>     Maximum NCCL message size (default: 64M)
  --warmup <n>           Warmup iterations (default: 5)
  --iters <n>            Measured iterations (default: 20)
  --oob-if <iface>       OpenMPI OOB/TCP interface
  --socket-ifname <if>   NCCL socket interface
  --nccl-ib-hca <list>   NCCL IB HCA allowlist
  --ssh-key <path>       SSH key
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
GPUS_PER_NODE=""
MIN_BYTES="1M"
MAX_BYTES="64M"
WARMUP="5"
ITERS="20"
OOB_IF="${OOB_IF:-}"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
SSH_KEY="${SSH_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --min-bytes) MIN_BYTES="$2"; shift 2 ;;
    --max-bytes) MAX_BYTES="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --socket-ifname) SOCKET_IFNAME="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --gpus-per-node must be a positive integer (got: ${GPUS_PER_NODE})" >&2
  exit 2
fi
if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --warmup must be >= 0 (got: ${WARMUP})" >&2
  exit 2
fi
if ! [[ "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --iters must be a positive integer (got: ${ITERS})" >&2
  exit 2
fi

OUT_STRUCT_DIR="${ROOT_DIR}/results/structured"
mkdir -p "$OUT_STRUCT_DIR"
SUMMARY_PATH="${OUT_STRUCT_DIR}/${RUN_ID}_nccl_env_sensitivity.json"
TMP_RESULTS_FILE="$(mktemp "${OUT_STRUCT_DIR}/${RUN_ID}_nccl_env_sensitivity_profiles_XXXXXX.jsonl")"

declare -a PROFILE_NAMES=()
declare -a PROFILE_CROSS_NIC=()
declare -a PROFILE_QPS=()
declare -a PROFILE_MIN_CTAS=()
declare -a PROFILE_MAX_CTAS=()

add_profile() {
  PROFILE_NAMES+=("$1")
  PROFILE_CROSS_NIC+=("${2:-}")
  PROFILE_QPS+=("${3:-}")
  PROFILE_MIN_CTAS+=("${4:-}")
  PROFILE_MAX_CTAS+=("${5:-}")
}

add_profile "baseline_auto" "" "" "" ""
add_profile "cross_nic_0" "0" "" "" ""
add_profile "cross_nic_1" "1" "" "" ""
add_profile "cross_nic_2" "2" "" "" ""
add_profile "qps_4" "" "4" "" ""
add_profile "ctas_16_16" "" "" "16" "16"

failures=0
baseline_peak=""

echo "========================================"
echo "NCCL Env Sensitivity Sweep"
echo "RUN_ID=${RUN_ID}"
echo "HOSTS=${HOSTS}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "Profiles=${PROFILE_NAMES[*]}"
echo "========================================"

for idx in "${!PROFILE_NAMES[@]}"; do
  profile="${PROFILE_NAMES[$idx]}"
  cross_nic="${PROFILE_CROSS_NIC[$idx]}"
  qps="${PROFILE_QPS[$idx]}"
  min_ctas="${PROFILE_MIN_CTAS[$idx]}"
  max_ctas="${PROFILE_MAX_CTAS[$idx]}"

  profile_run_id="${RUN_ID}_nccl_env_${profile}"
  output_json="${OUT_STRUCT_DIR}/${profile_run_id}_nccl.json"
  args=(
    "${ROOT_DIR}/scripts/run_nccl_all_reduce.sh"
    --run-id "${profile_run_id}"
    --hosts "${HOSTS}"
    --gpus-per-node "${GPUS_PER_NODE}"
    --min-bytes "${MIN_BYTES}"
    --max-bytes "${MAX_BYTES}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --label "nccl_env_${profile}"
  )
  if [[ -n "$OOB_IF" ]]; then
    args+=(--oob-if "${OOB_IF}")
  fi
  if [[ -n "$SOCKET_IFNAME" ]]; then
    args+=(--socket-ifname "${SOCKET_IFNAME}")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    args+=(--nccl-ib-hca "${NCCL_IB_HCA}")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    args+=(--ssh-key "${SSH_KEY}")
  fi
  if [[ -n "$cross_nic" ]]; then
    args+=(--nccl-cross-nic "${cross_nic}")
  fi
  if [[ -n "$qps" ]]; then
    args+=(--nccl-ib-qps-per-connection "${qps}")
  fi
  if [[ -n "$min_ctas" ]]; then
    args+=(--nccl-min-ctas "${min_ctas}")
  fi
  if [[ -n "$max_ctas" ]]; then
    args+=(--nccl-max-ctas "${max_ctas}")
  fi

  echo ""
  echo "-- Profile: ${profile}"
  set +e
  "${args[@]}"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 || ! -f "$output_json" ]]; then
    failures=$((failures + 1))
    python3 - "$TMP_RESULTS_FILE" "$profile" "$output_json" "$rc" "$cross_nic" "$qps" "$min_ctas" "$max_ctas" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
record = {
    "profile": sys.argv[2],
    "status": "error",
    "output_json": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "env": {
        "NCCL_CROSS_NIC": sys.argv[5] or None,
        "NCCL_IB_QPS_PER_CONNECTION": sys.argv[6] or None,
        "NCCL_MIN_CTAS": sys.argv[7] or None,
        "NCCL_MAX_CTAS": sys.argv[8] or None,
    },
}
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
PY
    echo "WARNING: profile ${profile} failed (rc=${rc})."
    continue
  fi

  peak_busbw="$(python3 - "$output_json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
best = 0.0
for row in data.get("results", []):
    try:
        best = max(best, float(row.get("busbw_gbps", 0.0)))
    except Exception:
        pass
print(f"{best:.6f}")
PY
)"
  if [[ "$profile" == "baseline_auto" ]]; then
    baseline_peak="$peak_busbw"
  fi

  python3 - "$TMP_RESULTS_FILE" "$profile" "$output_json" "$peak_busbw" "$cross_nic" "$qps" "$min_ctas" "$max_ctas" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
record = {
    "profile": sys.argv[2],
    "status": "ok",
    "output_json": sys.argv[3],
    "peak_busbw_gbps": float(sys.argv[4]),
    "env": {
        "NCCL_CROSS_NIC": sys.argv[5] or None,
        "NCCL_IB_QPS_PER_CONNECTION": sys.argv[6] or None,
        "NCCL_MIN_CTAS": sys.argv[7] or None,
        "NCCL_MAX_CTAS": sys.argv[8] or None,
    },
}
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
PY
  echo "peak_busbw_gbps=${peak_busbw}"
done

python3 - "$TMP_RESULTS_FILE" "$SUMMARY_PATH" "$RUN_ID" "$HOSTS" "$GPUS_PER_NODE" "$failures" "$baseline_peak" <<'PY'
import json
import sys
from pathlib import Path

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
run_id = sys.argv[3]
hosts = [x.strip() for x in sys.argv[4].split(",") if x.strip()]
gpus_per_node = int(sys.argv[5])
failures = int(sys.argv[6])
baseline_peak_raw = sys.argv[7]

profiles = []
if in_path.exists():
    for line in in_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        profiles.append(json.loads(line))

baseline_peak = float(baseline_peak_raw) if baseline_peak_raw else None
best_profile = None
best_peak = None
for rec in profiles:
    if rec.get("status") != "ok":
        continue
    peak = rec.get("peak_busbw_gbps")
    if peak is None:
        continue
    peak = float(peak)
    if best_peak is None or peak > best_peak:
        best_peak = peak
        best_profile = rec.get("profile")

for rec in profiles:
    if rec.get("status") != "ok":
        rec["speedup_vs_baseline"] = None
        continue
    if baseline_peak is None or baseline_peak <= 0:
        rec["speedup_vs_baseline"] = None
    else:
        rec["speedup_vs_baseline"] = round(float(rec.get("peak_busbw_gbps", 0.0)) / baseline_peak, 6)

summary = {
    "run_id": run_id,
    "test": "nccl_env_sensitivity",
    "hosts": hosts,
    "gpus_per_node": gpus_per_node,
    "status": "ok" if failures == 0 else "error",
    "failure_count": failures,
    "baseline_profile": "baseline_auto",
    "baseline_peak_busbw_gbps": baseline_peak,
    "best_profile": best_profile,
    "best_peak_busbw_gbps": best_peak,
    "profiles": profiles,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY

rm -f "$TMP_RESULTS_FILE"

if [[ "$failures" -ne 0 ]]; then
  echo "ERROR: NCCL env sensitivity sweep had ${failures} failing profile(s)." >&2
  exit 1
fi

echo "Wrote ${SUMMARY_PATH}"
