#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a dedicated nvbandwidth benchmark bundle with strict GPU clock locking.

Usage:
  scripts/repro/run_nvbandwidth_bundle.sh [options]

Options:
  --run-id <id>        RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>      Label for artifact names (default: hostname -s)
  --runtime <mode>     host|container (default: host)
  --image <image>      Container image for runtime=container
                       (default: cluster_perf_orig_parity:latest or $CONTAINER_IMAGE)
  --compat-lib-dir <path>
                       Optional CUDA compat library directory for runtime=host
                       (must contain libcuda.so.1 and libnvidia-ptxjitcompiler.so.1)
  --compat-image <image>
                       Image used to source CUDA compat libs for runtime=host when
                       compat-lib-dir is not present (default: --image value)
  --nvbw-bin <path>    nvbandwidth executable path (default: nvbandwidth)
  --quick              Run reduced testcase subset with lower samples for faster turnaround

Artifacts:
  - results/raw/<run_id>_<label>_nvbandwidth/nvbandwidth.log
  - results/structured/<run_id>_<label>_nvbandwidth.json
  - results/structured/<run_id>_<label>_nvbandwidth_sums.csv
  - results/structured/<run_id>_<label>_nvbandwidth_clock_lock.json
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
# shellcheck source=../lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname -s)"
RUNTIME="host"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-cluster_perf_orig_parity:latest}"
COMPAT_IMAGE="${COMPAT_IMAGE:-$CONTAINER_IMAGE}"
NVBW_BIN="${NVBW_BIN:-nvbandwidth}"
NVBW_COMPAT_LIB_DIR="${NVBW_COMPAT_LIB_DIR:-}"
QUICK=0
QUEUE_RUNNER_LOCK_PATH="${AISP_CLUSTER_SUITE_QUEUE_LOCK_PATH:-${CODE_ROOT}/artifacts/parallel_runs/queue.runner.lock}"
QUEUE_RUNNER_LOCK_TIMEOUT_SEC="${AISP_CLUSTER_SUITE_QUEUE_LOCK_TIMEOUT_SEC:-0}"
QUEUE_RUNNER_LOCK_FD=18
QUEUE_RUNNER_LOCK_HELD=0

acquire_queue_runner_lock() {
  if [[ "${AISP_CLUSTER_SUITE_QUEUE_LOCK_HELD:-0}" == "1" ]]; then
    echo "INFO: parent suite lock is already held; skipping nested queue lock acquisition."
    QUEUE_RUNNER_LOCK_HELD=0
    return 0
  fi
  local lock_path="$QUEUE_RUNNER_LOCK_PATH"
  local timeout_sec="$QUEUE_RUNNER_LOCK_TIMEOUT_SEC"
  mkdir -p "$(dirname "$lock_path")"
  if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required for strict suite/benchmark mutual exclusion." >&2
    exit 3
  fi
  eval "exec ${QUEUE_RUNNER_LOCK_FD}>\"${lock_path}\""
  if [[ "$timeout_sec" =~ ^[0-9]+$ ]] && [[ "$timeout_sec" -gt 0 ]]; then
    if ! flock -w "$timeout_sec" "$QUEUE_RUNNER_LOCK_FD"; then
      echo "ERROR: queue lock busy at ${lock_path}; refusing overlapping run." >&2
      exit 3
    fi
  else
    if ! flock -n "$QUEUE_RUNNER_LOCK_FD"; then
      echo "ERROR: queue lock busy at ${lock_path}; refusing overlapping run." >&2
      exit 3
    fi
  fi
  QUEUE_RUNNER_LOCK_HELD=1
  printf '{"owner":"run_nvbandwidth_bundle.sh","pid":%s,"ts":"%s"}\n' "$$" "$(date -Iseconds)" >"${lock_path}" || true
}

release_queue_runner_lock() {
  if [[ "$QUEUE_RUNNER_LOCK_HELD" -eq 1 && -n "${QUEUE_RUNNER_LOCK_FD:-}" ]]; then
    flock -u "$QUEUE_RUNNER_LOCK_FD" || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --image) CONTAINER_IMAGE="${2:-}"; shift 2 ;;
    --compat-lib-dir) NVBW_COMPAT_LIB_DIR="${2:-}"; shift 2 ;;
    --compat-image) COMPAT_IMAGE="${2:-}"; shift 2 ;;
    --nvbw-bin) NVBW_BIN="${2:-}"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  exit 1
fi
if [[ "$RUNTIME" == "container" && -z "$CONTAINER_IMAGE" ]]; then
  echo "ERROR: --image is required for --runtime container." >&2
  exit 1
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${LABEL}_nvbandwidth"
STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$RAW_DIR" "$STRUCT_DIR"

RAW_LOG="${RAW_DIR}/nvbandwidth.log"
SUMMARY_JSON="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth.json"
SUMS_CSV="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth_sums.csv"
LOCK_META="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth_clock_lock.json"
NVBW_ARGS=()
if [[ "$QUICK" -eq 1 ]]; then
  NVBW_ARGS+=(
    -i 1
    -b 128
    -t host_to_device_memcpy_ce
    -t device_to_host_memcpy_ce
    -t device_to_device_memcpy_read_ce
    -t device_to_device_memcpy_write_ce
    -t device_to_device_bidirectional_memcpy_read_ce
    -t device_to_device_bidirectional_memcpy_write_ce
    -t all_to_host_memcpy_ce
    -t host_to_all_memcpy_ce
  )
fi

echo "========================================"
echo "nvbandwidth Bundle"
echo "========================================"
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "RUNTIME=${RUNTIME}"
if [[ "$RUNTIME" == "container" ]]; then
  echo "IMAGE=${CONTAINER_IMAGE}"
else
  echo "NVBW_BIN=${NVBW_BIN}"
  echo "COMPAT_IMAGE=${COMPAT_IMAGE}"
  echo "COMPAT_LIB_DIR=${NVBW_COMPAT_LIB_DIR:-<auto>}"
fi
echo "QUICK=${QUICK}"
echo "RAW_LOG=${RAW_LOG}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "SUMS_CSV=${SUMS_CSV}"
echo "LOCK_META=${LOCK_META}"
echo ""

trap release_queue_runner_lock EXIT
acquire_queue_runner_lock

run_host_nvbandwidth() {
  local -a run_cmd=()
  if [[ -n "$NVBW_COMPAT_LIB_DIR" ]]; then
    run_cmd+=(env "LD_LIBRARY_PATH=${NVBW_COMPAT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
  fi
  run_cmd+=("$NVBW_BIN")
  run_cmd+=("${NVBW_ARGS[@]}")
  RUN_ID="$RUN_ID" LABEL="$LABEL" \
    "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- "${run_cmd[@]}" 2>&1 | tee "$RAW_LOG"
  return "${PIPESTATUS[0]}"
}

run_container_nvbandwidth() {
  local -a docker_args=(
    docker run --rm
    --gpus all
    --ipc=host
    --network host
    --ulimit memlock=-1
    --ulimit stack=67108864
  )
  if [[ -d /dev/infiniband ]]; then
    docker_args+=( -v /dev/infiniband:/dev/infiniband )
  fi
  if [[ -e /dev/nvidia_imex ]]; then
    docker_args+=( -v /dev/nvidia_imex:/dev/nvidia_imex )
  fi
  NVBW_ARGS_STR="$(printf ' %q' "${NVBW_ARGS[@]}")"
  docker_args+=( "${CONTAINER_IMAGE}" bash -lc "set -euo pipefail; ${NVBW_BIN}${NVBW_ARGS_STR}" )

  RUN_ID="$RUN_ID" LABEL="$LABEL" \
    "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- "${docker_args[@]}" 2>&1 | tee "$RAW_LOG"
  return "${PIPESTATUS[0]}"
}

has_compat_libs() {
  local d="$1"
  [[ -n "$d" && -d "$d" && -f "$d/libcuda.so.1" && -f "$d/libnvidia-ptxjitcompiler.so.1" ]]
}

cache_dir_for_image() {
  local image="$1"
  local digest=""
  digest="$(docker image inspect "$image" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
  if [[ -z "$digest" ]]; then
    digest="$image"
  fi
  digest="${digest//\//_}"
  digest="${digest//:/_}"
  digest="${digest//@/_}"
  printf '%s' "${HOME}/.cache/nvbandwidth_cuda_compat/${digest}"
}

prepare_host_compat_libs() {
  if [[ "$RUNTIME" != "host" ]]; then
    return 0
  fi
  if has_compat_libs "$NVBW_COMPAT_LIB_DIR"; then
    return 0
  fi

  for candidate in \
    "/usr/local/cuda/compat/lib.real" \
    "/usr/local/cuda/compat/lib" \
    "/usr/local/cuda-13.0/compat/lib.real" \
    "/usr/local/cuda-13.0/compat/lib"; do
    if has_compat_libs "$candidate"; then
      NVBW_COMPAT_LIB_DIR="$candidate"
      return 0
    fi
  done

  if ! command -v docker >/dev/null 2>&1; then
    echo "WARNING: docker not available; continuing without CUDA compat libs for host runtime." >&2
    return 0
  fi

  if [[ -z "$COMPAT_IMAGE" ]]; then
    echo "WARNING: no compat image configured; continuing without CUDA compat libs for host runtime." >&2
    return 0
  fi

  docker pull "$COMPAT_IMAGE" >/dev/null 2>&1 || true
  local cache_root=""
  cache_root="$(cache_dir_for_image "$COMPAT_IMAGE")"
  mkdir -p "$cache_root"

  if has_compat_libs "${cache_root}/lib.real"; then
    NVBW_COMPAT_LIB_DIR="${cache_root}/lib.real"
    return 0
  fi

  local cid=""
  cid="$(docker create "$COMPAT_IMAGE" 2>/dev/null || true)"
  if [[ -z "$cid" ]]; then
    echo "WARNING: failed to create container from ${COMPAT_IMAGE}; continuing without CUDA compat libs." >&2
    return 0
  fi
  set +e
  rm -rf "${cache_root}/lib.real"
  docker cp "${cid}:/usr/local/cuda/compat/lib.real" "${cache_root}/lib.real" >/dev/null 2>&1
  cp_rc=$?
  docker rm -f "$cid" >/dev/null 2>&1 || true
  set -e
  if [[ "$cp_rc" -ne 0 ]]; then
    echo "WARNING: could not extract CUDA compat libs from ${COMPAT_IMAGE}; continuing without compat override." >&2
    return 0
  fi
  if has_compat_libs "${cache_root}/lib.real"; then
    NVBW_COMPAT_LIB_DIR="${cache_root}/lib.real"
  fi
}

REQUESTED_RUNTIME="$RUNTIME"
EFFECTIVE_RUNTIME="$RUNTIME"
run_rc=0

if [[ "$REQUESTED_RUNTIME" == "host" ]]; then
  if ! command -v "$NVBW_BIN" >/dev/null 2>&1; then
    echo "ERROR: nvbandwidth binary not found on PATH: ${NVBW_BIN}" >&2
    exit 1
  fi
  prepare_host_compat_libs
  set +e
  run_host_nvbandwidth
  run_rc=$?
  set -e
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found; required for runtime=container." >&2
    exit 1
  fi
  set +e
  run_container_nvbandwidth
  run_rc=$?
  set -e
fi

if [[ "$run_rc" -ne 0 ]]; then
  if [[ "$REQUESTED_RUNTIME" == "host" ]]; then
    echo "ERROR: nvbandwidth execution failed (runtime=${EFFECTIVE_RUNTIME}, rc=${run_rc}, compat_lib_dir=${NVBW_COMPAT_LIB_DIR:-<none>})." >&2
  else
    echo "ERROR: nvbandwidth execution failed (runtime=${EFFECTIVE_RUNTIME}, rc=${run_rc})." >&2
  fi
  exit "$run_rc"
fi

python3 - <<'PY' "$RUN_ID" "$LABEL" "$RAW_LOG" "$LOCK_META" "$SUMMARY_JSON" "$SUMS_CSV" "$REQUESTED_RUNTIME" "$EFFECTIVE_RUNTIME" "$NVBW_BIN" "$CONTAINER_IMAGE" "$QUICK" "$NVBW_COMPAT_LIB_DIR" "$COMPAT_IMAGE"
import csv
import json
import re
import sys
from pathlib import Path

(
    run_id,
    label,
    raw_log_path,
    lock_meta_path,
    summary_json_path,
    sums_csv_path,
    requested_runtime,
    effective_runtime,
    nvbw_bin,
    container_image,
    quick_flag,
    compat_lib_dir,
    compat_image,
) = sys.argv[1:]

raw_log = Path(raw_log_path)
lock_meta = Path(lock_meta_path)
summary_json = Path(summary_json_path)
sums_csv = Path(sums_csv_path)

sum_re = re.compile(r"^SUM\s+(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
gpu_re = re.compile(r"^Device\s+(\d+):\s+(.+)$")

sums = []
gpus = {}

for line in raw_log.read_text(encoding="utf-8", errors="replace").splitlines():
    m = sum_re.match(line.strip())
    if m:
        sums.append({"test": m.group(1), "sum_gbps": float(m.group(2))})
        continue
    gm = gpu_re.match(line.strip())
    if gm:
        gpus[gm.group(1)] = gm.group(2).strip()

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

lock_payload = load_json(lock_meta) or {}
locks = lock_payload.get("locks") or []
clock_summary = {
    "returncode": lock_payload.get("returncode"),
    "device_count": len(locks),
    "all_devices_locked": bool(locks) and all(bool((x or {}).get("lock", {}).get("locked")) for x in locks),
    "application_sm_mhz": sorted(
        {
            int((x or {}).get("clocks", {}).get("applications_sm_mhz"))
            for x in locks
            if (x or {}).get("clocks", {}).get("applications_sm_mhz") is not None
        }
    ),
    "application_mem_mhz": sorted(
        {
            int((x or {}).get("clocks", {}).get("applications_mem_mhz"))
            for x in locks
            if (x or {}).get("clocks", {}).get("applications_mem_mhz") is not None
        }
    ),
}

sums_by_test = {entry["test"]: entry["sum_gbps"] for entry in sums}
peak_sum = max((entry["sum_gbps"] for entry in sums), default=None)

key_tests = [
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
    "device_to_device_memcpy_read_ce",
    "device_to_device_memcpy_write_ce",
    "device_to_device_bidirectional_memcpy_read_ce_total",
    "device_to_device_bidirectional_memcpy_write_ce_total",
    "all_to_host_memcpy_ce",
    "host_to_all_memcpy_ce",
    "all_to_all_memcpy_read_ce",
    "all_to_all_memcpy_write_ce",
]
key_sums = {name: sums_by_test.get(name) for name in key_tests if name in sums_by_test}

payload = {
    "run_id": run_id,
    "label": label,
    "status": "ok" if sums and clock_summary["all_devices_locked"] else "failed",
    "requested_runtime": requested_runtime,
    "effective_runtime": effective_runtime,
    "runtime": effective_runtime,
    "image": container_image if "container" in effective_runtime else None,
    "nvbandwidth_bin": nvbw_bin if effective_runtime == "host" else None,
    "cuda_compat_lib_dir": compat_lib_dir or None,
    "cuda_compat_image": compat_image if effective_runtime == "host" else None,
    "quick": quick_flag == "1",
    "artifacts": {
        "raw_log": str(raw_log),
        "clock_lock": str(lock_meta),
        "sums_csv": str(sums_csv),
    },
    "clock_lock": clock_summary,
    "gpu_inventory": gpus,
    "sum_entries": sums,
    "sum_count": len(sums),
    "key_sum_gbps": key_sums,
    "peak_sum_gbps": peak_sum,
}

summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

with sums_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["test", "sum_gbps"])
    writer.writeheader()
    for row in sums:
        writer.writerow(row)
PY

echo "Wrote ${RAW_LOG}"
echo "Wrote ${SUMMARY_JSON}"
echo "Wrote ${SUMS_CSV}"
echo "Wrote ${LOCK_META}"
