#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_cluster_health_suite.sh [options]

Runs a 2-node "cluster health suite":
  0) container runtime + CVE checks (includes CVE-2025-23266 / CVE-2025-23267)
  1) iperf3 (TCP throughput on the OOB interface)
  2) ib_write_bw (RDMA write BW per active IB HCA)
  3) nccl-tests (all_reduce_perf, all_gather_perf, reduce_scatter_perf)
  4) torchrun fast NCCL connectivity probe (required gate)
  5) torchrun torch.distributed all-reduce sanity (timed)

Options:
  --run-id <id>             Output prefix (default: YYYY-MM-DD_HHMMSS_cluster_health_suite)
  --hosts <h1,h2>           Comma-separated host list (required)
  --gpus-per-node <n>       GPUs per node (default: all visible GPUs)
  --cuda-visible-devices <list>  Optional CUDA_VISIBLE_DEVICES list for nccl-tests + torchdist
                                (example: 1,2,3 to exclude GPU0)
  --ssh-user <user>         SSH user (default: ubuntu)
  --ssh-key <path>          SSH key for remote launch (default: $SSH_KEY)

  --oob-if <iface>          OOB interface (default: auto via route to host2; required if auto-detect fails)
  --nccl-ib-hca <list>      Comma-separated IB HCA allowlist for NCCL/torchdist (default: auto-detect active mlx5_*)

  --extended                Also run: ib_read_bw + ib_send_bw + nccl-tests alltoall_perf

  --iperf-time <sec>        iperf3 duration (default: 10)
  --iperf-parallel <n>      iperf3 parallel streams (default: 8)
  --iperf-port <port>       iperf3 server port (default: 5201)

  --ib-msg-bytes <bytes>    ib_write_bw message size (default: 1048576)
  --ib-iters <n>            ib_write_bw iters (default: 2000)
  --ib-qps <n>              ib_write_bw QPs (default: 4)
  --ib-port <n>             ib_write_bw IB port (default: 1)
  --ib-base-port <port>     Starting TCP port for ib_write_bw (default: 18515)
  --ib-lat-bytes <bytes>    ib_*_lat message size (default: 4)
  --ib-lat-iters <n>        ib_*_lat iters (default: 20000)

  --gdr                      Also run GPUDirect RDMA variants (perftest --use_cuda).
  --gdr-gpu <id>             CUDA device id for perftest --use_cuda (default: 0)
  --gdr-mem-types <csv>      Comma-separated cuda_mem_type values to test (default: 0)
                             (0=device, 1=managed, 4=malloc)
  --gdr-use-dmabuf           Also test perftest --use_cuda_dmabuf (default: off)

  --nccl-min-bytes <size>   nccl-tests min bytes (default: 64M)
  --nccl-max-ar <size>      nccl-tests max bytes for all_reduce_perf (default: 16G)
  --nccl-max-other <size>   nccl-tests max bytes for all_gather/reduce_scatter (default: 4G)
  --nccl-max-alltoall <sz>  nccl-tests max bytes for alltoall_perf (default: 1G)
  --nccl-debug <level>      NCCL_DEBUG for nccl-tests ranks (default: INFO)
  --nccl-debug-subsys <s>   NCCL_DEBUG_SUBSYS for nccl-tests ranks (default: INIT)
  --nccl-nvls-enable <0|1|2>  Export NCCL_NVLS_ENABLE to nccl-tests ranks
                            (default: unset)
                            (note: if NVLS init fails repeatedly, the suite retries after restarting IMEX;
                             after max retries it fails and records the root cause in
                             *_nvls_recovery.json; no degraded fallback is used)
  --nccl-warmup <n>         nccl-tests warmup iters (default: 5)
  --nccl-iters <n>          nccl-tests measure iters (default: 20)
  --nccl-bins <list>        Comma-separated nccl-tests bins to run
                            (default: all_reduce_perf,all_gather_perf,reduce_scatter_perf; if --extended also adds alltoall_perf)

  --mpi-map-by <spec>       mpirun --map-by spec (default: ppr:<gpus_per_node>:node)
  --mpi-bind-to <spec>      mpirun --bind-to spec (default: none)
  --mpi-rank-by <spec>      mpirun --rank-by spec (default: unset)
  --mpi-report-bindings     Add mpirun --report-bindings (useful for NUMA pinning debug)

  --torch-sizes <bytes,...> Torchdist message sizes (default: 1048576,8388608,67108864,1073741824)
  --torch-master-port <p>   Torchdist rendezvous port (default: 29500)

  --skip-iperf3             Skip iperf3
  --skip-ib                 Skip ib_write_bw
  --skip-nccl               Skip nccl-tests
  --skip-torchdist          Skip torchdist sanity
  --skip-runtime-cve-check  Skip container runtime CVE check collection
  -h, --help                Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_RAW_DIR="${ROOT_DIR}/results/raw"
OUT_STRUCT_DIR="${ROOT_DIR}/results/structured"

RUN_ID="$(date +%Y-%m-%d_%H%M%S)_cluster_health_suite"
HOSTS=""
GPUS_PER_NODE=""
CUDA_VISIBLE_DEVICES_LIST=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
EXTENDED=0

IPERF_TIME=10
IPERF_PARALLEL=8
IPERF_PORT=5201

IB_MSG_BYTES=1048576
IB_ITERS=2000
IB_QPS=4
IB_PORT=1
IB_BASE_PORT=18515
IB_LAT_BYTES=4
IB_LAT_ITERS=20000

GDR_ENABLE=0
GDR_GPU=0
GDR_MEM_TYPES="0"
GDR_USE_DMABUF=0
GDR_REQUESTED=0
GDR_DISABLE_REASON=""
GDR_PROBE_LAST_ERROR_MSG=""

NCCL_MIN_BYTES="64M"
NCCL_MAX_AR="16G"
NCCL_MAX_OTHER="4G"
NCCL_MAX_ALLTOALL="1G"
NCCL_DEBUG_LEVEL="INFO"
NCCL_DEBUG_SUBSYS="INIT"
NCCL_NVLS_ENABLE=""
NCCL_WARMUP=5
NCCL_ITERS=20
NCCL_BINS=""

MPI_MAP_BY=""
MPI_BIND_TO="none"
MPI_RANK_BY=""
MPI_REPORT_BINDINGS=0

TORCH_SIZES="1048576,8388608,67108864,1073741824"
TORCH_MASTER_PORT=29500

SKIP_IPERF3=0
SKIP_IB=0
SKIP_NCCL=0
SKIP_TORCHDIST=0
SKIP_RUNTIME_CVE_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_LIST="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --oob-if) OOB_IF="$2"; shift 2 ;;
    --nccl-ib-hca) NCCL_IB_HCA="$2"; shift 2 ;;
    --extended) EXTENDED=1; shift ;;
    --iperf-time) IPERF_TIME="$2"; shift 2 ;;
    --iperf-parallel) IPERF_PARALLEL="$2"; shift 2 ;;
    --iperf-port) IPERF_PORT="$2"; shift 2 ;;
    --ib-msg-bytes) IB_MSG_BYTES="$2"; shift 2 ;;
    --ib-iters) IB_ITERS="$2"; shift 2 ;;
    --ib-qps) IB_QPS="$2"; shift 2 ;;
    --ib-port) IB_PORT="$2"; shift 2 ;;
    --ib-base-port) IB_BASE_PORT="$2"; shift 2 ;;
    --ib-lat-bytes) IB_LAT_BYTES="$2"; shift 2 ;;
    --ib-lat-iters) IB_LAT_ITERS="$2"; shift 2 ;;
    --gdr) GDR_ENABLE=1; shift ;;
    --gdr-gpu) GDR_GPU="$2"; shift 2 ;;
    --gdr-mem-types) GDR_MEM_TYPES="$2"; shift 2 ;;
    --gdr-use-dmabuf) GDR_USE_DMABUF=1; shift ;;
    --nccl-min-bytes) NCCL_MIN_BYTES="$2"; shift 2 ;;
    --nccl-max-ar) NCCL_MAX_AR="$2"; shift 2 ;;
    --nccl-max-other) NCCL_MAX_OTHER="$2"; shift 2 ;;
    --nccl-max-alltoall) NCCL_MAX_ALLTOALL="$2"; shift 2 ;;
    --nccl-debug) NCCL_DEBUG_LEVEL="$2"; shift 2 ;;
    --nccl-debug-subsys) NCCL_DEBUG_SUBSYS="$2"; shift 2 ;;
    --nccl-nvls-enable) NCCL_NVLS_ENABLE="$2"; shift 2 ;;
    --nccl-warmup) NCCL_WARMUP="$2"; shift 2 ;;
    --nccl-iters) NCCL_ITERS="$2"; shift 2 ;;
    --nccl-bins) NCCL_BINS="$2"; shift 2 ;;
    --mpi-map-by) MPI_MAP_BY="$2"; shift 2 ;;
    --mpi-bind-to) MPI_BIND_TO="$2"; shift 2 ;;
    --mpi-rank-by) MPI_RANK_BY="$2"; shift 2 ;;
    --mpi-report-bindings) MPI_REPORT_BINDINGS=1; shift ;;
    --torch-sizes) TORCH_SIZES="$2"; shift 2 ;;
    --torch-master-port) TORCH_MASTER_PORT="$2"; shift 2 ;;
    --skip-iperf3) SKIP_IPERF3=1; shift ;;
    --skip-ib) SKIP_IB=1; shift ;;
    --skip-nccl) SKIP_NCCL=1; shift ;;
    --skip-torchdist) SKIP_TORCHDIST=1; shift ;;
    --skip-runtime-cve-check) SKIP_RUNTIME_CVE_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

if [[ -n "$NCCL_NVLS_ENABLE" && "$NCCL_NVLS_ENABLE" != "0" && "$NCCL_NVLS_ENABLE" != "1" && "$NCCL_NVLS_ENABLE" != "2" ]]; then
  echo "ERROR: --nccl-nvls-enable must be 0, 1, or 2 (got: ${NCCL_NVLS_ENABLE})" >&2
  exit 2
fi

if [[ "$GDR_ENABLE" -ne 0 ]]; then
  if ! [[ "$GDR_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --gdr-gpu must be a non-negative integer (got: ${GDR_GPU})" >&2
    exit 2
  fi
  if [[ -z "$GDR_MEM_TYPES" ]]; then
    echo "ERROR: --gdr-mem-types must not be empty when --gdr is enabled" >&2
    exit 2
  fi
  IFS=',' read -r -a _gdr_mem_types_tmp <<<"$GDR_MEM_TYPES"
  for _mt in "${_gdr_mem_types_tmp[@]}"; do
    _mt="$(echo "$_mt" | xargs)"
    [[ -n "$_mt" ]] || continue
    if [[ "$_mt" != "0" && "$_mt" != "1" && "$_mt" != "4" ]]; then
      echo "ERROR: --gdr-mem-types supports only 0,1,4 (got: ${_mt})" >&2
      exit 2
    fi
  done
fi
if [[ "$GDR_ENABLE" -eq 1 && "$SKIP_IB" -eq 1 ]]; then
  echo "ERROR: --gdr requires IB perftest runs; do not combine with --skip-ib." >&2
  exit 2
fi
if [[ "$GDR_USE_DMABUF" -eq 1 && "$GDR_ENABLE" -ne 1 ]]; then
  echo "ERROR: --gdr-use-dmabuf requires --gdr." >&2
  exit 2
fi

GDR_REQUESTED="$GDR_ENABLE"

# NVLS (NCCL NVLink SHARP) retry/degrade state. "requested" is what the user asked for;
# "effective" may be forced to 0 if NVLS init is repeatedly failing.
NCCL_NVLS_ENABLE_REQUESTED="${NCCL_NVLS_ENABLE}"
NCCL_NVLS_ENABLE_EFFECTIVE="${NCCL_NVLS_ENABLE}"
NVLS_DEGRADED=0
NVLS_MAX_ATTEMPTS=3
NVLS_LAST_RESET_RUN_ID=""
NVLS_LAST_RESET_PREFLIGHT_RC=""
NVLS_LAST_RESET_PREFLIGHT_JSON=""
NVLS_LAST_RESET_EXCERPT=""

resolve_ipv4() {
  local host="$1"
  getent ahostsv4 "$host" | awk '{print $1; exit}'
}

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if [[ "${#HOST_ARR[@]}" -ne 2 ]]; then
  echo "ERROR: --hosts must have exactly 2 hosts (got: $HOSTS)" >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot auto-detect --gpus-per-node" >&2
    exit 2
  fi
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

if [[ "$GDR_ENABLE" -eq 1 ]]; then
  if [[ "$GDR_GPU" -ge "$GPUS_PER_NODE" ]]; then
    echo "ERROR: --gdr-gpu=${GDR_GPU} is out of range for --gpus-per-node=${GPUS_PER_NODE}" >&2
    exit 2
  fi
fi

HOST0="${HOST_ARR[0]}"
HOST1="${HOST_ARR[1]}"
HOST1_IP="$(resolve_ipv4 "$HOST1")"
if [[ -z "$HOST1_IP" ]]; then
  echo "ERROR: unable to resolve IPv4 for host1=$HOST1" >&2
  exit 2
fi

if [[ -z "$OOB_IF" ]]; then
  OOB_IF="$(ip route get "$HOST1_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}' || true)"
fi
if [[ -z "$OOB_IF" ]]; then
  echo "ERROR: unable to auto-detect --oob-if; pass it explicitly." >&2
  exit 2
fi

LOCAL_IP="$(ip -o -4 addr show dev "$OOB_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
if [[ -z "$LOCAL_IP" ]]; then
  echo "ERROR: unable to determine LOCAL_IP for iface=$OOB_IF" >&2
  exit 2
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=8
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

preflight_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$SSH_KEY" ]]; then
  preflight_args+=(--ssh-key "$SSH_KEY")
fi
"${ROOT_DIR}/scripts/preflight_cluster_services.sh" "${preflight_args[@]}"

remote() {
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST1}" "$@"
}

remote_port_listening() {
  local port="$1"
  # ss filters require quoting; avoid locale-dependent output parsing.
  remote "ss -ltn \"sport = :${port}\" | tail -n +2 | grep -q LISTEN" >/dev/null 2>&1
}

find_free_port_remote() {
  local start_port="$1"
  local tries="${2:-100}"
  local port="$start_port"
  local i
  for ((i=0; i<tries; i++)); do
    if ! remote_port_listening "$port"; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done
  return 1
}

detect_active_ib_hcas() {
  local dev state_file
  if [[ ! -d /sys/class/infiniband ]]; then
    return 0
  fi
  for dev_path in /sys/class/infiniband/*; do
    dev="$(basename "$dev_path")"
    [[ "$dev" == mlx5_* ]] || continue
    state_file="/sys/class/infiniband/${dev}/ports/${IB_PORT}/state"
    if [[ -f "$state_file" ]] && grep -q "ACTIVE" "$state_file"; then
      echo "$dev"
    fi
  done
}

if [[ -z "$NCCL_IB_HCA" ]]; then
  mapfile -t IB_HCAS < <(detect_active_ib_hcas)
  if [[ "${#IB_HCAS[@]}" -gt 0 ]]; then
    NCCL_IB_HCA="$(IFS=','; echo "${IB_HCAS[*]}")"
  fi
fi

LABEL="${HOSTS//,/}"
LABEL="${LABEL//./_}"

if [[ -z "$MPI_MAP_BY" ]]; then
  MPI_MAP_BY="ppr:${GPUS_PER_NODE}:node"
fi
if [[ -z "$NCCL_BINS" ]]; then
  NCCL_BINS="all_reduce_perf,all_gather_perf,reduce_scatter_perf"
  if [[ "$EXTENDED" -eq 1 ]]; then
    NCCL_BINS="${NCCL_BINS},alltoall_perf"
  fi
fi

SUITE_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_cluster_health_suite.log"
LOCKFILE="${OUT_RAW_DIR}/cluster_health_suite.lock"

NVLS_RECOVERY_JSONL="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_nvls_recovery.jsonl"
NVLS_RECOVERY_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_nvls_recovery.json"
: > "${NVLS_RECOVERY_JSONL}"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "ERROR: another cluster health suite is running (lock: $LOCKFILE)" >&2
  exit 3
fi

ts() { date -Is; }
log() { echo "[$(ts)] $*" | tee -a "$SUITE_LOG"; }

declare -A STEP_RC=()
declare -a STEP_ORDER=()

run_step() {
  local name="$1"
  shift
  STEP_ORDER+=("$name")
  log "== START ${name} =="
  set +e
  "$@"
  local rc=$?
  set -e
  STEP_RC["$name"]="$rc"
  log "== END ${name} rc=${rc} =="
  return 0
}

nvls_log_has_init_failure() {
  local log_path="$1"
  # Typical signature when NVLS is attempted but the fabric/services are unhealthy.
  grep -E -i -q "transport/nvls\\.cc|Cuda failure 801|NVLS.*operation not supported" "$log_path"
}

write_nvls_error_excerpt() {
  local log_path="$1"
  local out_path="$2"
  local match lineno start end
  match="$(grep -n -E -i "transport/nvls\\.cc|Cuda failure 801|NVLS.*operation not supported" "$log_path" | head -n 1 || true)"
  if [[ -z "$match" ]]; then
    return 1
  fi
  lineno="${match%%:*}"
  start=$((lineno - 20))
  if [[ "$start" -lt 1 ]]; then
    start=1
  fi
  end=$((lineno + 40))
  {
    echo "NVLS init failure excerpt"
    echo "log: ${log_path}"
    echo "match: ${match}"
    echo "context: lines ${start}-${end}"
    echo ""
    sed -n "${start},${end}p" "$log_path"
  } >"$out_path"
  return 0
}

append_nvls_recovery_record() {
  local phase="$1"         # run
  local bin_name="$2"
  local attempt="$3"
  local nvls_enable_env="$4"  # "" means unset
  local rc="$5"
  local nvls_init_failure="$6" # 0|1
  local raw_log="$7"
  local cmd_log="$8"
  local structured_json="${9:-}"
  local error_excerpt="${10:-}"
  local reset_excerpt="${11:-}"
  local reset_preflight_json="${12:-}"
  local reset_preflight_rc="${13:-}"

  python3 - <<'PY' \
    "$NVLS_RECOVERY_JSONL" \
    "$phase" "$bin_name" "$attempt" "$nvls_enable_env" "$rc" "$nvls_init_failure" \
    "$raw_log" "$cmd_log" "$structured_json" "$error_excerpt" \
    "$reset_excerpt" "$reset_preflight_json" "$reset_preflight_rc"
import json
import sys
import time
from pathlib import Path

out_jsonl = Path(sys.argv[1])
phase = sys.argv[2]
bin_name = sys.argv[3]
attempt = int(sys.argv[4])
nvls_enable_env = sys.argv[5]
rc = int(sys.argv[6])
nvls_init_failure = bool(int(sys.argv[7]))
raw_log = sys.argv[8]
cmd_log = sys.argv[9]
structured_json = sys.argv[10]
error_excerpt = sys.argv[11]
reset_excerpt = sys.argv[12]
reset_preflight_json = sys.argv[13]
reset_preflight_rc = sys.argv[14]

rec = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "phase": phase,
    "bin": bin_name,
    "attempt": attempt,
    "nccl_nvls_enable_env": nvls_enable_env,
    "rc": rc,
    "nvls_init_failure": nvls_init_failure,
    "raw_log": raw_log,
    "cmd_log": cmd_log,
    "structured_json": structured_json or None,
    "error_excerpt": error_excerpt or None,
    "reset": None,
}
if reset_excerpt or reset_preflight_json or reset_preflight_rc:
    rec["reset"] = {
        "excerpt": reset_excerpt or None,
        "preflight_services_json": reset_preflight_json or None,
        "preflight_rc": int(reset_preflight_rc) if reset_preflight_rc else None,
    }

out_jsonl.parent.mkdir(parents=True, exist_ok=True)
with out_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
PY
}

nvls_reset_services() {
  local bin_name="$1"
  local attempt="$2"
  local reset_id="${RUN_ID}_${LABEL}_nvls_reset_${bin_name}_attempt${attempt}"
  local excerpt="${OUT_STRUCT_DIR}/${reset_id}.txt"
  local preflight_rc=0
  local local_restart_rc=0
  local remote_restart_rc=0

  {
    echo "NVLS reset attempt: ${attempt}"
    echo "run_id: ${RUN_ID}"
    echo "label: ${LABEL}"
    echo "bin: ${bin_name}"
    echo "hosts: ${HOSTS}"
    echo "time: $(date -Is)"
    echo ""
  } >"$excerpt"

  echo "== Restart nvidia-imex (local) ==" >>"$excerpt"
  if sudo -n systemctl restart nvidia-imex >>"$excerpt" 2>&1; then
    local_restart_rc=0
  else
    local_restart_rc=$?
  fi
  echo "local_restart_rc=${local_restart_rc}" >>"$excerpt"
  echo "" >>"$excerpt"

  echo "== Restart nvidia-imex (remote: ${HOST1}) ==" >>"$excerpt"
  if remote "sudo -n systemctl restart nvidia-imex" >>"$excerpt" 2>&1; then
    remote_restart_rc=0
  else
    remote_restart_rc=$?
  fi
  echo "remote_restart_rc=${remote_restart_rc}" >>"$excerpt"
  echo "" >>"$excerpt"

  echo "== Preflight services (strict) ==" >>"$excerpt"
  local -a pf_args=(--run-id "$reset_id" --hosts "$HOSTS" --ssh-user "$SSH_USER")
  if [[ -n "$SSH_KEY" ]]; then
    pf_args+=(--ssh-key "$SSH_KEY")
  fi
  if "${ROOT_DIR}/scripts/preflight_cluster_services.sh" "${pf_args[@]}" >>"$excerpt" 2>&1; then
    preflight_rc=0
  else
    preflight_rc=$?
  fi
  echo "preflight_rc=${preflight_rc}" >>"$excerpt"

  NVLS_LAST_RESET_RUN_ID="$reset_id"
  NVLS_LAST_RESET_PREFLIGHT_RC="$preflight_rc"
  NVLS_LAST_RESET_PREFLIGHT_JSON="${OUT_STRUCT_DIR}/${reset_id}_preflight_services.json"
  NVLS_LAST_RESET_EXCERPT="$excerpt"
  return 0
}

write_nvls_recovery_json() {
  python3 - <<'PY' \
    "$NVLS_RECOVERY_JSON" "$NVLS_RECOVERY_JSONL" \
    "$RUN_ID" "$LABEL" "$HOSTS" \
    "$NCCL_NVLS_ENABLE_REQUESTED" "$NCCL_NVLS_ENABLE_EFFECTIVE" \
    "$NVLS_DEGRADED" "$NVLS_MAX_ATTEMPTS"
import json
import sys
import time
from pathlib import Path

out_json = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
run_id = sys.argv[3]
label = sys.argv[4]
hosts = [h.strip() for h in sys.argv[5].split(",") if h.strip()]
requested = sys.argv[6]
effective = sys.argv[7]
degraded = bool(int(sys.argv[8]))
max_attempts = int(sys.argv[9])

attempts = []
if jsonl_path.exists():
    for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            attempts.append(json.loads(line))
        except json.JSONDecodeError:
            attempts.append({"raw": line, "error": "json_decode"})

payload = {
    "run_id": run_id,
    "label": label,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hosts": hosts,
    "requested_nccl_nvls_enable": requested if requested else "",
    "effective_nccl_nvls_enable": effective if effective else "",
    "degraded": degraded,
    "max_attempts": max_attempts,
    "attempts_jsonl": str(jsonl_path),
    "attempts": attempts,
    "remediation": [
        "Ensure nvidia-imex is active on all nodes and IMEX Domain State is UP (scripts/preflight_cluster_services.sh).",
        "If NVLS init fails with transport/nvls.cc + Cuda failure 801, try restarting nvidia-imex on all nodes and re-running.",
        "If NVLS continues to fail after retries, treat the run as invalid and fix service state before rerun.",
    ],
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(out_json)
PY
}

preflight() {
  local ok=1
  local gdr_usable=1
  local gdr_reason_parts=()
  if [[ "$SKIP_NCCL" -eq 0 || "$SKIP_TORCHDIST" -eq 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "Missing local dependency: sudo (required for GPU clock locking)" >&2
      ok=0
    elif ! sudo -n true >/dev/null 2>&1; then
      echo "ERROR: GPU clock locking is required, but passwordless sudo is not available on the local host (sudo -n true)." >&2
      ok=0
    fi
  fi
  if [[ "$SKIP_IPERF3" -eq 0 ]]; then
    if ! command -v iperf3 >/dev/null 2>&1; then
      echo "Missing local dependency: iperf3" >&2
      ok=0
    fi
  fi
  if [[ "$SKIP_IB" -eq 0 ]]; then
    for tool in ib_write_bw; do
      if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing local dependency: ${tool}" >&2
        ok=0
      fi
    done
    if [[ "$GDR_ENABLE" -eq 1 ]]; then
      if ! command -v ib_read_lat >/dev/null 2>&1; then
        echo "WARNING: Missing local dependency: ib_read_lat (required for --gdr latency checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("local missing ib_read_lat")
      fi
    fi
    if [[ "$EXTENDED" -eq 1 ]]; then
      for tool in ib_read_bw ib_send_bw; do
        if ! command -v "$tool" >/dev/null 2>&1; then
          echo "Missing local dependency: ${tool}" >&2
          ok=0
        fi
      done
      if [[ "$GDR_ENABLE" -eq 1 ]]; then
        if ! command -v ib_send_lat >/dev/null 2>&1; then
          echo "WARNING: Missing local dependency: ib_send_lat (required for --gdr + --extended latency checks)" >&2
          gdr_usable=0
          gdr_reason_parts+=("local missing ib_send_lat")
        fi
      fi
    fi
    if [[ "$GDR_ENABLE" -eq 1 ]]; then
      if ! (ib_write_bw --help 2>/dev/null || true) | grep -q -- "--use_cuda"; then
        echo "WARNING: Local ib_write_bw does not expose --use_cuda (required for --gdr checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("local ib_write_bw lacks --use_cuda")
      fi
      if ! (ib_read_lat --help 2>/dev/null || true) | grep -q -- "--use_cuda"; then
        echo "WARNING: Local ib_read_lat does not expose --use_cuda (required for --gdr checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("local ib_read_lat lacks --use_cuda")
      fi
      if [[ "$EXTENDED" -eq 1 ]]; then
        if ! (ib_send_lat --help 2>/dev/null || true) | grep -q -- "--use_cuda"; then
          echo "WARNING: Local ib_send_lat does not expose --use_cuda (required for --gdr + --extended checks)" >&2
          gdr_usable=0
          gdr_reason_parts+=("local ib_send_lat lacks --use_cuda")
        fi
      fi
    fi
  fi
  if [[ "$SKIP_NCCL" -eq 0 ]]; then
    if ! command -v mpirun >/dev/null 2>&1; then
      echo "Missing local dependency: mpirun" >&2
      ok=0
    fi
  fi
  if [[ "$ok" -eq 0 ]]; then
    return 1
  fi
  if [[ "$SKIP_NCCL" -eq 0 || "$SKIP_TORCHDIST" -eq 0 ]]; then
    remote "command -v sudo >/dev/null" || return 1
    remote "sudo -n true >/dev/null" || return 1
  fi
  if [[ "$SKIP_IPERF3" -eq 0 ]]; then
    remote "command -v iperf3 >/dev/null" || return 1
  fi
  if [[ "$SKIP_IB" -eq 0 ]]; then
    remote "command -v ib_write_bw >/dev/null" || return 1
    if [[ "$GDR_ENABLE" -eq 1 ]]; then
      if ! remote "command -v ib_read_lat >/dev/null"; then
        echo "WARNING: Missing remote dependency: ib_read_lat on ${HOST1} (required for --gdr latency checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("remote missing ib_read_lat")
      fi
    fi
    if [[ "$EXTENDED" -eq 1 ]]; then
      remote "command -v ib_read_bw >/dev/null" || return 1
      remote "command -v ib_send_bw >/dev/null" || return 1
      if [[ "$GDR_ENABLE" -eq 1 ]]; then
        if ! remote "command -v ib_send_lat >/dev/null"; then
          echo "WARNING: Missing remote dependency: ib_send_lat on ${HOST1} (required for --gdr + --extended latency checks)" >&2
          gdr_usable=0
          gdr_reason_parts+=("remote missing ib_send_lat")
        fi
      fi
    fi
    if [[ "$GDR_ENABLE" -eq 1 ]]; then
      if ! remote "(ib_write_bw --help 2>/dev/null || true) | grep -q -- '--use_cuda'"; then
        echo "WARNING: Remote ib_write_bw does not expose --use_cuda on ${HOST1} (required for --gdr checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("remote ib_write_bw lacks --use_cuda")
      fi
      if ! remote "(ib_read_lat --help 2>/dev/null || true) | grep -q -- '--use_cuda'"; then
        echo "WARNING: Remote ib_read_lat does not expose --use_cuda on ${HOST1} (required for --gdr checks)" >&2
        gdr_usable=0
        gdr_reason_parts+=("remote ib_read_lat lacks --use_cuda")
      fi
      if [[ "$EXTENDED" -eq 1 ]]; then
        if ! remote "(ib_send_lat --help 2>/dev/null || true) | grep -q -- '--use_cuda'"; then
          echo "WARNING: Remote ib_send_lat does not expose --use_cuda on ${HOST1} (required for --gdr + --extended checks)" >&2
          gdr_usable=0
          gdr_reason_parts+=("remote ib_send_lat lacks --use_cuda")
        fi
      fi
    fi
  fi

  if [[ "$GDR_ENABLE" -eq 1 && "$gdr_usable" -eq 0 ]]; then
    if [[ "${#gdr_reason_parts[@]}" -gt 0 ]]; then
      GDR_DISABLE_REASON="$(IFS='; '; echo "${gdr_reason_parts[*]}")"
    else
      GDR_DISABLE_REASON="gdr requested but perftest --use_cuda checks failed"
    fi
    echo "ERROR: --gdr requested but prerequisites are not met. reason=${GDR_DISABLE_REASON}" >&2
    return 1
  fi

  if [[ "$GDR_ENABLE" -eq 1 ]]; then
    if ! validate_requested_gdr_modes; then
      if [[ -z "$GDR_DISABLE_REASON" ]]; then
        GDR_DISABLE_REASON="${GDR_PROBE_LAST_ERROR_MSG:-gdr mem-type validation failed}"
      fi
      echo "ERROR: --gdr requested but cuda mem-type probe failed. reason=${GDR_DISABLE_REASON}" >&2
      return 1
    fi
  fi
  return 0
}

gdr_probe_log_has_unsupported_memtype() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    return 1
  fi
  grep -qiE "CUDA Memory type is not supported|unsupported with no odp MR|Parser function exited with Error" "$log_path"
}

run_gdr_mem_type_probe() {
  local tool="$1"
  local cuda_mem_type="$2"
  local use_dmabuf="${3:-0}"
  local tag="${4:-probe}"
  local hca=""
  local h=""
  local -a hcas=()
  local -a gdr_args=()
  local -a perf_args=()
  local base_port=19700
  local port=""
  local server_log=""
  local client_log=""
  local server_pid=0
  local ready=0
  local i
  local dmabuf_desc=""
  GDR_PROBE_LAST_ERROR_MSG=""
  if [[ "$use_dmabuf" -eq 1 ]]; then
    dmabuf_desc=" with dmabuf"
  fi

  case "$tool" in
    ib_write_bw)
      perf_args=(-q 1 -s 4096 -n 64 --report_gbits)
      ;;
    ib_read_lat)
      perf_args=(-s 4 -n 200)
      ;;
    *)
      GDR_PROBE_LAST_ERROR_MSG="unsupported gdr probe tool: ${tool}"
      return 1
      ;;
  esac

  if [[ -z "$NCCL_IB_HCA" ]]; then
    GDR_PROBE_LAST_ERROR_MSG="no active IB HCA allowlist available for GDR probe"
    return 1
  fi
  IFS=',' read -r -a hcas <<<"$NCCL_IB_HCA"
  for h in "${hcas[@]}"; do
    h="$(echo "$h" | xargs)"
    if [[ -n "$h" ]]; then
      hca="$h"
      break
    fi
  done
  if [[ -z "$hca" ]]; then
    GDR_PROBE_LAST_ERROR_MSG="failed to select IB HCA from NCCL_IB_HCA='${NCCL_IB_HCA}'"
    return 1
  fi

  port="$(find_free_port_remote "$base_port" 200)" || {
    GDR_PROBE_LAST_ERROR_MSG="unable to allocate free probe port on ${HOST1} (start=${base_port})"
    return 1
  }

  gdr_args=(--use_cuda "${GDR_GPU}" --cuda_mem_type "${cuda_mem_type}")
  if [[ "$use_dmabuf" -eq 1 ]]; then
    gdr_args+=(--use_cuda_dmabuf)
  fi

  server_log="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_gdr_probe_${tag}_${hca}_server.log"
  client_log="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_gdr_probe_${tag}_${hca}_client.log"

  remote "timeout 90s ${tool} -d ${hca} -i ${IB_PORT} ${perf_args[*]} ${gdr_args[*]} -p ${port}" 2>&1 | tee "$server_log" &
  server_pid=$!

  ready=0
  for ((i=0; i<40; i++)); do
    if remote_port_listening "$port"; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  if [[ "$ready" -ne 1 ]]; then
    GDR_PROBE_LAST_ERROR_MSG="${tool} probe server did not become ready on ${HOST1}:${port} (tag=${tag}, hca=${hca})"
    wait "$server_pid" || true
    if gdr_probe_log_has_unsupported_memtype "$server_log"; then
      GDR_PROBE_LAST_ERROR_MSG="cuda_mem_type=${cuda_mem_type} unsupported for gdr probe on hca=${hca}${dmabuf_desc}"
    fi
    return 1
  fi

  set +e
  "${tool}" -d "$hca" -i "$IB_PORT" "${perf_args[@]}" \
    --bind_source_ip "${LOCAL_IP}" \
    "${gdr_args[@]}" \
    -p "$port" "$HOST1_IP" 2>&1 | tee "$client_log"
  local client_rc=${PIPESTATUS[0]}
  set -e
  wait "$server_pid" || true

  if [[ "$client_rc" -ne 0 ]]; then
    if gdr_probe_log_has_unsupported_memtype "$server_log" || gdr_probe_log_has_unsupported_memtype "$client_log"; then
      GDR_PROBE_LAST_ERROR_MSG="cuda_mem_type=${cuda_mem_type} unsupported for gdr probe on hca=${hca}${dmabuf_desc}"
    else
      GDR_PROBE_LAST_ERROR_MSG="${tool} gdr probe failed rc=${client_rc} tag=${tag} hca=${hca} port=${port}"
    fi
    return "$client_rc"
  fi
  return 0
}

validate_requested_gdr_modes() {
  local -a mem_types=()
  local mt=""
  if [[ "$GDR_ENABLE" -ne 1 ]]; then
    return 0
  fi

  IFS=',' read -r -a mem_types <<<"$GDR_MEM_TYPES"
  if [[ "${#mem_types[@]}" -eq 0 ]]; then
    GDR_DISABLE_REASON="--gdr enabled but --gdr-mem-types is empty"
    return 1
  fi

  for mt in "${mem_types[@]}"; do
    mt="$(echo "$mt" | xargs)"
    [[ -n "$mt" ]] || continue
    if ! run_gdr_mem_type_probe ib_write_bw "$mt" 0 "mem${mt}_write_bw"; then
      GDR_DISABLE_REASON="${GDR_PROBE_LAST_ERROR_MSG:-gdr probe failed for cuda_mem_type=${mt} (ib_write_bw)}"
      return 1
    fi
    if ! run_gdr_mem_type_probe ib_read_lat "$mt" 0 "mem${mt}_read_lat"; then
      GDR_DISABLE_REASON="${GDR_PROBE_LAST_ERROR_MSG:-gdr probe failed for cuda_mem_type=${mt}}"
      return 1
    fi
  done

  if [[ "$GDR_USE_DMABUF" -eq 1 ]]; then
    if ! run_gdr_mem_type_probe ib_write_bw "0" 1 "mem0_dmabuf_write_bw"; then
      GDR_DISABLE_REASON="${GDR_PROBE_LAST_ERROR_MSG:-gdr probe failed for cuda_mem_type=0 with dmabuf (ib_write_bw)}"
      return 1
    fi
    if ! run_gdr_mem_type_probe ib_read_lat "0" 1 "mem0_dmabuf_read_lat"; then
      GDR_DISABLE_REASON="${GDR_PROBE_LAST_ERROR_MSG:-gdr probe failed for cuda_mem_type=0 with dmabuf (ib_read_lat)}"
      return 1
    fi
  fi

  return 0
}

run_runtime_cve_check() {
  local -a runtime_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
  if [[ -n "$SSH_KEY" ]]; then
    runtime_args+=(--ssh-key "$SSH_KEY")
  fi
  "${ROOT_DIR}/scripts/collect_container_runtime_all_nodes.sh" "${runtime_args[@]}"
}

run_iperf3_pair() {
  local direction="$1" # fwd|rev
  local extra=()
  if [[ "$direction" == "rev" ]]; then
    extra=(-R)
  fi

  local port
  port="$(find_free_port_remote "$IPERF_PORT" 50)" || {
    log "ERROR: unable to find a free iperf3 port on ${HOST1} (start=${IPERF_PORT})"
    return 1
  }

  local server_log="${OUT_RAW_DIR}/${RUN_ID}_iperf3_${LABEL}_${direction}_server.log"
  local client_json="${OUT_RAW_DIR}/${RUN_ID}_iperf3_${LABEL}_${direction}_client.json"

  log "iperf3 ${direction}: server=${HOST1}:${port} client_bind=${LOCAL_IP} streams=${IPERF_PARALLEL} time=${IPERF_TIME}s"

  remote "timeout $((IPERF_TIME + 30))s iperf3 -s -1 -p ${port} -B ${HOST1_IP}" 2>&1 | tee "$server_log" &
  local server_pid=$!
  local waited=0
  while ! remote_port_listening "$port"; do
    sleep 0.2
    waited=$((waited + 1))
    if [[ "$waited" -ge 50 ]]; then
      log "ERROR: iperf3 server did not become ready on ${HOST1}:${port}"
      wait "$server_pid" || true
      return 1
    fi
  done

  timeout "$((IPERF_TIME + 30))s" iperf3 -c "$HOST1_IP" -p "$port" -t "$IPERF_TIME" -P "$IPERF_PARALLEL" -B "$LOCAL_IP" --json "${extra[@]}" >"$client_json"
  local client_rc=$?
  wait "$server_pid" || true
  return "$client_rc"
}

run_ib_bw_variant() {
  local mode="$1"            # write|read|send
  local port_base_offset="$2"
  local tag="${3:-}"         # optional prefix tag in filenames
  local gdr_enable="${4:-0}" # 0|1
  local cuda_mem_type="${5:-}"
  local use_dmabuf="${6:-0}" # 0|1

  local tool="ib_${mode}_bw"

  if [[ -z "$NCCL_IB_HCA" ]]; then
    log "No active mlx5_* devices detected; skipping ${tool}"
    return 0
  fi

  local -a hcas=()
  IFS=',' read -r -a hcas <<<"$NCCL_IB_HCA"

  local -a gdr_args=()
  if [[ "$gdr_enable" -eq 1 ]]; then
    gdr_args+=(--use_cuda "${GDR_GPU}")
    if [[ -n "$cuda_mem_type" ]]; then
      gdr_args+=(--cuda_mem_type "${cuda_mem_type}")
    fi
    if [[ "$use_dmabuf" -eq 1 ]]; then
      gdr_args+=(--use_cuda_dmabuf)
    fi
  fi

  local idx=0
  for hca in "${hcas[@]}"; do
    hca="$(echo "$hca" | xargs)"
    [[ -n "$hca" ]] || continue

    local desired_port=$((IB_BASE_PORT + port_base_offset + idx))
    local port
    port="$(find_free_port_remote "$desired_port" 200)" || {
      log "ERROR: unable to find a free ${tool} port on ${HOST1} (start=${desired_port})"
      return 1
    }

    local stem="${RUN_ID}_ib_${mode}_bw_${hca}_p${port}"
    if [[ -n "$tag" ]]; then
      stem="${RUN_ID}_${tag}_ib_${mode}_bw_${hca}_p${port}"
    fi
    local server_log="${OUT_RAW_DIR}/${stem}_server.log"
    local client_log="${OUT_RAW_DIR}/${stem}_client.log"
    local client_json="${OUT_RAW_DIR}/${stem}_client.json"

    local gdr_desc="cpu"
    if [[ "$gdr_enable" -eq 1 ]]; then
      gdr_desc="gdr_gpu=${GDR_GPU}"
      if [[ -n "$cuda_mem_type" ]]; then
        gdr_desc="${gdr_desc},cuda_mem_type=${cuda_mem_type}"
      fi
      if [[ "$use_dmabuf" -eq 1 ]]; then
        gdr_desc="${gdr_desc},dmabuf=1"
      fi
    fi
    log "${tool} hca=${hca} ib_port=${IB_PORT} tcp_port=${port} msg=${IB_MSG_BYTES}B qps=${IB_QPS} iters=${IB_ITERS} mode=${gdr_desc} tag=${tag:-base}"

    remote "timeout 180s ${tool} -d ${hca} -i ${IB_PORT} -q ${IB_QPS} -s ${IB_MSG_BYTES} -n ${IB_ITERS} --report_gbits ${gdr_args[*]} -p ${port}" 2>&1 | tee "$server_log" &
    local server_pid=$!
    local waited=0
    while ! remote_port_listening "$port"; do
      sleep 0.2
      waited=$((waited + 1))
      if [[ "$waited" -ge 100 ]]; then
        log "ERROR: ${tool} server did not become ready on ${HOST1}:${port}"
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" || true
        return 1
      fi
    done

    timeout 180s "${tool}" -d "${hca}" -i "${IB_PORT}" -q "${IB_QPS}" -s "${IB_MSG_BYTES}" -n "${IB_ITERS}" --report_gbits \
      "${gdr_args[@]}" \
      --bind_source_ip "${LOCAL_IP}" \
      --out_json --out_json_file "${client_json}" \
      -p "${port}" "${HOST1_IP}" 2>&1 | tee "$client_log"
    local client_rc=${PIPESTATUS[0]}

    wait "$server_pid" || true
    if [[ "$client_rc" -ne 0 ]]; then
      log "${tool} client failed for ${hca} rc=${client_rc}"
      # Best-effort cleanup in case the server is still hanging around.
      remote "pkill -f \"${tool}.*-p ${port}\" >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" || true
      return "$client_rc"
    fi
    idx=$((idx + 1))
  done
}

run_ib_lat_variant() {
  local mode="$1"            # write|read|send
  local port_base_offset="$2"
  local tag="${3:-}"         # optional prefix tag in filenames
  local gdr_enable="${4:-0}" # 0|1
  local cuda_mem_type="${5:-}"
  local use_dmabuf="${6:-0}" # 0|1

  local tool="ib_${mode}_lat"

  if [[ -z "$NCCL_IB_HCA" ]]; then
    log "No active mlx5_* devices detected; skipping ${tool}"
    return 0
  fi

  local -a hcas=()
  IFS=',' read -r -a hcas <<<"$NCCL_IB_HCA"

  local -a gdr_args=()
  if [[ "$gdr_enable" -eq 1 ]]; then
    gdr_args+=(--use_cuda "${GDR_GPU}")
    if [[ -n "$cuda_mem_type" ]]; then
      gdr_args+=(--cuda_mem_type "${cuda_mem_type}")
    fi
    if [[ "$use_dmabuf" -eq 1 ]]; then
      gdr_args+=(--use_cuda_dmabuf)
    fi
  fi

  local idx=0
  for hca in "${hcas[@]}"; do
    hca="$(echo "$hca" | xargs)"
    [[ -n "$hca" ]] || continue

    local desired_port=$((IB_BASE_PORT + port_base_offset + idx))
    local port
    port="$(find_free_port_remote "$desired_port" 200)" || {
      log "ERROR: unable to find a free ${tool} port on ${HOST1} (start=${desired_port})"
      return 1
    }

    local stem="${RUN_ID}_ib_${mode}_lat_${hca}_p${port}"
    if [[ -n "$tag" ]]; then
      stem="${RUN_ID}_${tag}_ib_${mode}_lat_${hca}_p${port}"
    fi
    local server_log="${OUT_RAW_DIR}/${stem}_server.log"
    local client_log="${OUT_RAW_DIR}/${stem}_client.log"
    local client_json="${OUT_RAW_DIR}/${stem}_client.json"

    local gdr_desc="cpu"
    if [[ "$gdr_enable" -eq 1 ]]; then
      gdr_desc="gdr_gpu=${GDR_GPU}"
      if [[ -n "$cuda_mem_type" ]]; then
        gdr_desc="${gdr_desc},cuda_mem_type=${cuda_mem_type}"
      fi
      if [[ "$use_dmabuf" -eq 1 ]]; then
        gdr_desc="${gdr_desc},dmabuf=1"
      fi
    fi
    log "${tool} hca=${hca} ib_port=${IB_PORT} tcp_port=${port} msg=${IB_LAT_BYTES}B iters=${IB_LAT_ITERS} mode=${gdr_desc} tag=${tag:-base}"

    remote "timeout 180s ${tool} -d ${hca} -i ${IB_PORT} -s ${IB_LAT_BYTES} -n ${IB_LAT_ITERS} ${gdr_args[*]} -p ${port}" 2>&1 | tee "$server_log" &
    local server_pid=$!
    local waited=0
    while ! remote_port_listening "$port"; do
      sleep 0.2
      waited=$((waited + 1))
      if [[ "$waited" -ge 100 ]]; then
        log "ERROR: ${tool} server did not become ready on ${HOST1}:${port}"
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" || true
        return 1
      fi
    done

    timeout 180s "${tool}" -d "${hca}" -i "${IB_PORT}" -s "${IB_LAT_BYTES}" -n "${IB_LAT_ITERS}" \
      "${gdr_args[@]}" \
      --bind_source_ip "${LOCAL_IP}" \
      --out_json --out_json_file "${client_json}" \
      -p "${port}" "${HOST1_IP}" 2>&1 | tee "$client_log"
    local client_rc=${PIPESTATUS[0]}

    wait "$server_pid" || true
    if [[ "$client_rc" -ne 0 ]]; then
      log "${tool} client failed for ${hca} rc=${client_rc}"
      remote "pkill -f \"${tool}.*-p ${port}\" >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" || true
      return "$client_rc"
    fi
    idx=$((idx + 1))
  done
}

run_ib_write_bw() { run_ib_bw_variant write 0 ""; }
run_ib_read_bw() { run_ib_bw_variant read 100 ""; }
run_ib_send_bw() { run_ib_bw_variant send 200 ""; }
run_ib_write_lat() { run_ib_lat_variant write 300 ""; }
run_ib_read_lat() { run_ib_lat_variant read 400 ""; }
run_ib_send_lat() { run_ib_lat_variant send 500 ""; }

run_ib_gdr_suite() {
  if [[ "$GDR_ENABLE" -ne 1 ]]; then
    return 0
  fi

  local -a mem_types=()
  IFS=',' read -r -a mem_types <<<"$GDR_MEM_TYPES"
  if [[ "${#mem_types[@]}" -eq 0 ]]; then
    log "ERROR: --gdr enabled but --gdr-mem-types is empty"
    return 1
  fi

  local mt idx mode
  idx=0
  for mt in "${mem_types[@]}"; do
    mt="$(echo "$mt" | xargs)"
    [[ -n "$mt" ]] || continue
    local tag="gdr_gpu${GDR_GPU}_mem${mt}"
    local base=$((1000 + idx * 300))

    if ! run_ib_bw_variant write "$base" "$tag" 1 "$mt" 0; then
      log "ERROR: GDR subtest failed: tag=${tag} step=ib_write_bw"
      return 1
    fi
    if ! run_ib_lat_variant read "$((base + 120))" "$tag" 1 "$mt" 0; then
      log "ERROR: GDR subtest failed: tag=${tag} step=ib_read_lat"
      return 1
    fi
    if [[ "$EXTENDED" -eq 1 ]]; then
      if ! run_ib_bw_variant read "$((base + 20))" "$tag" 1 "$mt" 0; then
        log "ERROR: GDR subtest failed: tag=${tag} step=ib_read_bw"
        return 1
      fi
      if ! run_ib_bw_variant send "$((base + 40))" "$tag" 1 "$mt" 0; then
        log "ERROR: GDR subtest failed: tag=${tag} step=ib_send_bw"
        return 1
      fi
      if ! run_ib_lat_variant send "$((base + 140))" "$tag" 1 "$mt" 0; then
        log "ERROR: GDR subtest failed: tag=${tag} step=ib_send_lat"
        return 1
      fi
    fi
    idx=$((idx + 1))
  done

  if [[ "$GDR_USE_DMABUF" -eq 1 ]]; then
    local dmabuf_tag="gdr_gpu${GDR_GPU}_mem0_dmabuf"
    if ! run_ib_bw_variant write 5000 "$dmabuf_tag" 1 "0" 1; then
      log "ERROR: GDR subtest failed: tag=${dmabuf_tag} step=ib_write_bw"
      return 1
    fi
    if ! run_ib_lat_variant read 5120 "$dmabuf_tag" 1 "0" 1; then
      log "ERROR: GDR subtest failed: tag=${dmabuf_tag} step=ib_read_lat"
      return 1
    fi
    if [[ "$EXTENDED" -eq 1 ]]; then
      if ! run_ib_bw_variant read 5020 "$dmabuf_tag" 1 "0" 1; then
        log "ERROR: GDR subtest failed: tag=${dmabuf_tag} step=ib_read_bw"
        return 1
      fi
      if ! run_ib_bw_variant send 5040 "$dmabuf_tag" 1 "0" 1; then
        log "ERROR: GDR subtest failed: tag=${dmabuf_tag} step=ib_send_bw"
        return 1
      fi
      if ! run_ib_lat_variant send 5140 "$dmabuf_tag" 1 "0" 1; then
        log "ERROR: GDR subtest failed: tag=${dmabuf_tag} step=ib_send_lat"
        return 1
      fi
    fi
  fi
  return 0
}

run_nccl_collective() {
  local bin_name="$1"       # all_reduce_perf
  local min_bytes="$2"      # 64M
  local max_bytes="$3"      # 16G

  local hostfile="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_hosts.txt"
  printf "%s\n" "${HOST_ARR[@]}" >"$hostfile"

  local total_ranks=$((GPUS_PER_NODE * 2))
  local mpi_py="${ROOT_DIR}/env/venv/bin/python3"
  local wrapper="${ROOT_DIR}/scripts/nccl_lock_wrapper.py"
  local nccl_bin="${ROOT_DIR}/tools/nccl-tests/build/${bin_name}"

  if [[ ! -x "$mpi_py" ]]; then
    echo "Missing python venv at $mpi_py" >&2
    return 2
  fi
  if [[ ! -x "$nccl_bin" ]]; then
    echo "Missing nccl-tests binary at $nccl_bin" >&2
    return 2
  fi

  local out_json="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_nccl_${bin_name}.json"

  local attempt
  local max_attempts=1
  if [[ "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" != "0" ]]; then
    max_attempts="$NVLS_MAX_ATTEMPTS"
  fi

  local saw_nvls_init_failure=0

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    local attempt_tag="attempt${attempt}"
    local raw_log="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_nccl_${bin_name}_${attempt_tag}.log"
    local cmd_log="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_nccl_${bin_name}_${attempt_tag}.cmd.txt"

    local -a mpirun_cmd=(
      mpirun
      --hostfile "$hostfile"
      --map-by "$MPI_MAP_BY"
      --bind-to "$MPI_BIND_TO"
      --mca routed direct
    )
    if [[ "$MPI_REPORT_BINDINGS" -eq 1 ]]; then
      mpirun_cmd+=(--report-bindings)
    fi
    if [[ -n "$MPI_RANK_BY" ]]; then
      mpirun_cmd+=(--rank-by "$MPI_RANK_BY")
    fi

    if [[ -n "$SSH_KEY" ]]; then
      mpirun_cmd+=(
        --mca plm_rsh_args
        "-i ${SSH_KEY} -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -o ConnectionAttempts=3 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
      )
    fi
    if [[ -n "$OOB_IF" ]]; then
      mpirun_cmd+=(--mca oob_tcp_if_include "$OOB_IF" --mca btl_tcp_if_include "$OOB_IF")
    fi

    mpirun_cmd+=(
      -np "$total_ranks"
      -x "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"
      -x "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
      -x "NCCL_SOCKET_IFNAME=${OOB_IF}"
      -x "PATH=$PATH"
      -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    )

    if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
      mpirun_cmd+=(-x "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}")
    fi

    if [[ -n "$NCCL_IB_HCA" ]]; then
      mpirun_cmd+=(-x "NCCL_IB_HCA=${NCCL_IB_HCA}")
    fi

    if [[ -n "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" ]]; then
      mpirun_cmd+=(-x "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE_EFFECTIVE}")
    fi

    mpirun_cmd+=(
      "$mpi_py"
      "$wrapper"
      --
      "$nccl_bin"
      -b "$min_bytes"
      -e "$max_bytes"
      -f 2
      -g 1
      -w "$NCCL_WARMUP"
      -n "$NCCL_ITERS"
    )

    printf "%q " "${mpirun_cmd[@]}" >"$cmd_log"
    log "nccl-tests ${bin_name}: attempt=${attempt}/${max_attempts} ranks=${total_ranks} min=${min_bytes} max=${max_bytes} oob_if=${OOB_IF} NCCL_IB_HCA=${NCCL_IB_HCA:-<auto>} NCCL_NVLS_ENABLE(effective)=${NCCL_NVLS_ENABLE_EFFECTIVE:-<unset>} map_by=${MPI_MAP_BY} bind_to=${MPI_BIND_TO} rank_by=${MPI_RANK_BY:-<unset>}"

    "${mpirun_cmd[@]}" 2>&1 | tee "$raw_log"
    local rc=${PIPESTATUS[0]}

    if [[ "$rc" -eq 0 ]]; then
      local cmd_string
      cmd_string="$(cat "$cmd_log")"
      "${ROOT_DIR}/scripts/parse_nccl_log.py" \
        --input "$raw_log" \
        --output "$out_json" \
        --run-id "$RUN_ID" \
        --hosts "$HOSTS" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --command "$cmd_string"
      log "Wrote ${out_json}"
      append_nvls_recovery_record "run" "$bin_name" "$attempt" "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" 0 0 "$raw_log" "$cmd_log" "$out_json"
      return 0
    fi

    # Failure path.
    local nvls_init_failure=0
    local error_excerpt=""
    if [[ "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" != "0" ]] && nvls_log_has_init_failure "$raw_log"; then
      nvls_init_failure=1
      saw_nvls_init_failure=1
      error_excerpt="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_nccl_${bin_name}_${attempt_tag}_nvls_error_excerpt.txt"
      write_nvls_error_excerpt "$raw_log" "$error_excerpt" || true
    fi

    if [[ "$nvls_init_failure" -eq 1 && "$attempt" -lt "$max_attempts" ]]; then
      log "WARNING: NVLS init failure detected (attempt ${attempt}/${max_attempts}) for ${bin_name}; resetting IMEX and retrying."
      nvls_reset_services "$bin_name" "$attempt"
      append_nvls_recovery_record \
        "run" "$bin_name" "$attempt" "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" "$rc" 1 \
        "$raw_log" "$cmd_log" "" "$error_excerpt" \
        "$NVLS_LAST_RESET_EXCERPT" "$NVLS_LAST_RESET_PREFLIGHT_JSON" "$NVLS_LAST_RESET_PREFLIGHT_RC"
      continue
    fi

    append_nvls_recovery_record "run" "$bin_name" "$attempt" "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" "$rc" "$nvls_init_failure" "$raw_log" "$cmd_log" "" "$error_excerpt"
    log "nccl-tests ${bin_name} failed rc=${rc}"
    break
  done

  if [[ "$saw_nvls_init_failure" -eq 1 && "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" != "0" ]]; then
    log "ERROR: NVLS init failed after ${NVLS_MAX_ATTEMPTS} attempts; refusing degraded fallback."
    log "ERROR: Fix path: ensure nvidia-imex is active on all nodes and IMEX Domain State is UP (scripts/preflight_cluster_services.sh)."
  fi

  return 1
}

run_connectivity_probe() {
  local probe_port=$((TORCH_MASTER_PORT + 4))
  local -a probe_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --ssh-user "$SSH_USER"
    --gpus-per-node "$GPUS_PER_NODE"
    --master-addr "$LOCAL_IP"
    --master-port "$probe_port"
    --barrier-iters 5
    --payload-bytes 8388608
    --timeout-sec 120
  )
  if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
    probe_args+=(--cuda-visible-devices "$CUDA_VISIBLE_DEVICES_LIST")
  fi
  if [[ -n "$OOB_IF" ]]; then
    probe_args+=(--oob-if "$OOB_IF")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    probe_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    probe_args+=(--ssh-key "$SSH_KEY")
  fi
  "${ROOT_DIR}/scripts/run_torchrun_connectivity_probe.sh" "${probe_args[@]}"
}

write_summary() {
  local out="${OUT_STRUCT_DIR}/${RUN_ID}_${LABEL}_cluster_health_suite_summary.json"
  local require_clock_lock=0
  if [[ "$SKIP_NCCL" -eq 0 || "$SKIP_TORCHDIST" -eq 0 ]]; then
    require_clock_lock=1
  fi

  # Ensure the NVLS recovery artifact exists before summary generation.
  write_nvls_recovery_json >/dev/null

  CUDA_VISIBLE_DEVICES_LIST="$CUDA_VISIBLE_DEVICES_LIST" "${ROOT_DIR}/env/venv/bin/python3" - <<'PY' "$out" "$RUN_ID" "$LABEL" "$HOSTS" "$GPUS_PER_NODE" "$OOB_IF" "$NCCL_IB_HCA" "$require_clock_lock" "$NCCL_DEBUG_LEVEL" "$NCCL_DEBUG_SUBSYS" "$NCCL_NVLS_ENABLE" "$MPI_MAP_BY" "$MPI_BIND_TO" "$MPI_RANK_BY" "$MPI_REPORT_BINDINGS" "$SUITE_LOG" "$OUT_RAW_DIR" "$OUT_STRUCT_DIR" "$GDR_REQUESTED" "$GDR_ENABLE" "$GDR_GPU" "$GDR_MEM_TYPES" "$GDR_USE_DMABUF" "$GDR_DISABLE_REASON" "$SKIP_RUNTIME_CVE_CHECK" "${STEP_ORDER[*]}"
import json
import os
from typing import Optional
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
label = sys.argv[3]
hosts = sys.argv[4].split(",")
gpus_per_node = int(sys.argv[5])
oob_if = sys.argv[6]
nccl_ib_hca = sys.argv[7]
require_clock_lock = bool(int(sys.argv[8]))
nccl_debug = sys.argv[9]
nccl_debug_subsys = sys.argv[10]
nccl_nvls_enable = sys.argv[11]
mpi_map_by = sys.argv[12]
mpi_bind_to = sys.argv[13]
mpi_rank_by = sys.argv[14]
mpi_report_bindings = bool(int(sys.argv[15]))
suite_log = sys.argv[16]
out_raw_dir = Path(sys.argv[17])
out_struct_dir = Path(sys.argv[18])
gdr_requested = bool(int(sys.argv[19]))
gdr_enable = bool(int(sys.argv[20]))
gdr_gpu = int(sys.argv[21])
gdr_mem_types = [x.strip() for x in sys.argv[22].split(",") if x.strip()]
gdr_use_dmabuf = bool(int(sys.argv[23]))
gdr_disable_reason = sys.argv[24]
skip_runtime_cve_check = bool(int(sys.argv[25]))
step_order = sys.argv[26].split()

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def iperf_gbps(path: Path):
    d = read_json(path)
    if not d:
        return None
    end = d.get("end", {})
    # Prefer received; fall back to sent.
    for key in ("sum_received", "sum_sent", "sum"):
        obj = end.get(key)
        if isinstance(obj, dict) and "bits_per_second" in obj:
            return float(obj["bits_per_second"]) / 1e9
    return None

def perftest_avg_gbps(path: Path):
    d = read_json(path)
    if not d:
        return None
    # perftest json is typically: {"results": {"BW_average": <Gb/s>, ...}, ...}
    res = d.get("results")
    if isinstance(res, dict):
        for k in ("BW_average", "BW_peak"):
            if k in res:
                try:
                    return float(res[k])
                except Exception:
                    pass
    # Fallbacks for other shapes.
    for k in ("bw_avg_gbps", "bw_average_gbps", "average_bw_gbps"):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return None

def perftest_avg_latency_us(path: Path):
    d = read_json(path)
    if not d:
        return None
    res = d.get("results")
    if isinstance(res, dict):
        for k in (
            "latency_avg",
            "Latency_avg",
            "lat_avg",
            "t_avg",
            "average_latency",
            "avg_latency",
        ):
            if k in res:
                try:
                    return float(res[k])
                except Exception:
                    pass
    for k in ("latency_avg_us", "avg_latency_us", "latency_us"):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return None

def nccl_max_busbw(path: Path):
    d = read_json(path)
    if not d:
        return None
    best = None
    for row in d.get("results", []):
        try:
            busbw = float(row.get("busbw_gbps"))
            size = int(row.get("size_bytes"))
        except Exception:
            continue
        if best is None or busbw > best["busbw_gbps"]:
            best = {"busbw_gbps": busbw, "size_bytes": size}
    return best

def torch_max_busbw(path: Path):
    d = read_json(path)
    if not d:
        return None
    best = None
    for row in d.get("results", []):
        try:
            busbw = float(row.get("busbw_gbps"))
            size = int(row.get("size_bytes"))
        except Exception:
            continue
        if best is None or busbw > best["busbw_gbps"]:
            best = {"busbw_gbps": busbw, "size_bytes": size}
    return best

def parse_runtime_cve_fields(path: Path):
    fields = {}
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.startswith("cve_") or k.startswith("gpu_operator_"):
                fields[k] = v.strip()
    except Exception:
        return {}
    return fields

def sanitize_label(raw: str) -> str:
    return raw.replace(".", "_").replace(":", "_")

payload = {
    "run_id": run_id,
    "label": label,
    "hosts": hosts,
    "gpus_per_node": gpus_per_node,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES_LIST", ""),
    "oob_if": oob_if,
    "nccl_ib_hca": nccl_ib_hca,
    "nccl_debug": nccl_debug,
    "nccl_debug_subsys": nccl_debug_subsys,
    "nccl_nvls_enable": nccl_nvls_enable if nccl_nvls_enable else "",
    "mpi": {
        "map_by": mpi_map_by,
        "bind_to": mpi_bind_to,
        "rank_by": mpi_rank_by,
        "report_bindings": mpi_report_bindings,
    },
    "require_clock_lock": require_clock_lock,
    "suite_log": suite_log,
    "steps": {"order": step_order},
    "security": {
        "runtime_cve_check_enabled": (not skip_runtime_cve_check),
        "container_runtime_checks": {},
    },
    "gdr": {
        "requested": gdr_requested,
        "enabled": gdr_enable,
        "effective_enabled": gdr_enable,
        "gpu": gdr_gpu,
        "mem_types": gdr_mem_types,
        "use_dmabuf": gdr_use_dmabuf,
        "disabled_reason": gdr_disable_reason or None,
    },
    "iperf3": {},
    "ib_write_bw": {},
    "ib_write_lat": {},
    "ib_read_bw": {},
    "ib_read_lat": {},
    "ib_send_bw": {},
    "ib_send_lat": {},
    "nccl": {},
    "connectivity_probe": {},
    "torchdist": {},
}

for direction in ("fwd", "rev"):
    p = out_raw_dir / f"{run_id}_iperf3_{label}_{direction}_client.json"
    if p.exists():
        payload["iperf3"][direction] = {"client_json": str(p), "gbps": iperf_gbps(p)}

import re
run_esc = re.escape(run_id)

def parse_gdr_tag(tag: str):
    m = re.match(r"^gdr_gpu(\d+)_mem(\d+)(?:_dmabuf)?$", tag or "")
    if not m:
        return {}
    return {
        "gpu": int(m.group(1)),
        "cuda_mem_type": int(m.group(2)),
        "dmabuf": tag.endswith("_dmabuf"),
    }

def add_ib_entry(kind: str, mode: str, hca: str, port: int, path: Path, tag: Optional[str]):
    value_key = "avg_gbps" if kind == "bw" else "avg_latency_us"
    value = perftest_avg_gbps(path) if kind == "bw" else perftest_avg_latency_us(path)
    rec = {"client_json": str(path), "port": port, value_key: value}

    top_key = f"ib_{mode}_{kind}"
    if tag:
        gdr = payload.setdefault("ib_gdr", {})
        tag_obj = gdr.setdefault(
            tag,
            {
                "tag": tag,
                "tag_meta": parse_gdr_tag(tag),
                "write_bw": {},
                "read_bw": {},
                "send_bw": {},
                "write_lat": {},
                "read_lat": {},
                "send_lat": {},
            },
        )
        tag_obj[f"{mode}_{kind}"][hca] = rec
    else:
        payload.setdefault(top_key, {})
        payload[top_key][hca] = rec

for kind in ("bw", "lat"):
    for mode in ("write", "read", "send"):
        pat = re.compile(rf"^{run_esc}_(?:(.+?)_)?ib_{mode}_{kind}_(.+?)_p(\d+)_client\.json$")
        for p in sorted(out_raw_dir.glob(f"{run_id}_*ib_{mode}_{kind}_*_p*_client.json")):
            m = pat.match(p.name)
            if not m:
                continue
            tag = m.group(1) or None
            hca = m.group(2)
            port = int(m.group(3))
            add_ib_entry(kind=kind, mode=mode, hca=hca, port=port, path=p, tag=tag)

for bin_name in ("all_reduce_perf", "all_gather_perf", "reduce_scatter_perf", "alltoall_perf"):
    p = out_struct_dir / f"{run_id}_{label}_nccl_{bin_name}.json"
    if p.exists():
        payload["nccl"][bin_name] = {"structured_json": str(p), "max_busbw": nccl_max_busbw(p)}

probe_p = out_struct_dir / f"{run_id}_torchrun_connectivity_probe.json"
if probe_p.exists():
    probe_d = read_json(probe_p)
    probe_rec = {"structured_json": str(probe_p)}
    if isinstance(probe_d, dict):
        probe_rec["status"] = probe_d.get("status")
        probe_rec["world_size"] = probe_d.get("world_size")
        ranks = probe_d.get("ranks") or []
        best = None
        for rank_rec in ranks:
            try:
                busbw = float((rank_rec or {}).get("payload_probe", {}).get("busbw_gbps", 0.0))
            except Exception:
                continue
            best = busbw if best is None else max(best, busbw)
        if best is not None:
            probe_rec["max_payload_busbw_gbps"] = best
    payload["connectivity_probe"] = probe_rec

torch_p = out_struct_dir / f"{run_id}_torchrun_allreduce.json"
if torch_p.exists():
    payload["torchdist"]["structured_json"] = str(torch_p)
    payload["torchdist"]["max_busbw"] = torch_max_busbw(torch_p)

nvls_p = out_struct_dir / f"{run_id}_{label}_nvls_recovery.json"
nvls_d = read_json(nvls_p) if nvls_p.exists() else None
if isinstance(nvls_d, dict):
    payload["nvls"] = {
        "structured_json": str(nvls_p),
        "requested_nccl_nvls_enable": nvls_d.get("requested_nccl_nvls_enable", ""),
        "effective_nccl_nvls_enable": nvls_d.get("effective_nccl_nvls_enable", ""),
        "degraded": bool(nvls_d.get("degraded", False)),
    }

for host in hosts:
    label_key = sanitize_label(host)
    runtime_path = out_struct_dir / f"{run_id}_{label_key}_container_runtime.txt"
    if runtime_path.exists():
        payload["security"]["container_runtime_checks"][label_key] = {
            "host": host,
            "artifact": str(runtime_path),
            **parse_runtime_cve_fields(runtime_path),
        }

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(out_path)
PY
  log "Wrote ${out}"
}

log "RUN_ID=${RUN_ID} LABEL=${LABEL}"
log "HOSTS=${HOSTS} GPUS_PER_NODE=${GPUS_PER_NODE}"
if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}"
fi
log "SSH_USER=${SSH_USER} SSH_KEY=${SSH_KEY:-<agent>}"
log "OOB_IF=${OOB_IF} LOCAL_IP=${LOCAL_IP} HOST1=${HOST1} HOST1_IP=${HOST1_IP}"
log "NCCL_IB_HCA=${NCCL_IB_HCA:-<auto>}"
log "IB: msg_bytes=${IB_MSG_BYTES} iters=${IB_ITERS} qps=${IB_QPS} port=${IB_PORT} base_port=${IB_BASE_PORT} lat_bytes=${IB_LAT_BYTES} lat_iters=${IB_LAT_ITERS}"
log "GDR: requested=${GDR_REQUESTED} enabled=${GDR_ENABLE} gpu=${GDR_GPU} mem_types=${GDR_MEM_TYPES} use_dmabuf=${GDR_USE_DMABUF} reason=${GDR_DISABLE_REASON:-<none>}"
log "NCCL_DEBUG=${NCCL_DEBUG_LEVEL} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
log "NCCL_NVLS_ENABLE(requested)=${NCCL_NVLS_ENABLE_REQUESTED:-<unset>} effective=${NCCL_NVLS_ENABLE_EFFECTIVE:-<unset>} degraded=${NVLS_DEGRADED}"
log "NCCL_BINS=${NCCL_BINS}"
log "MPI: map_by=${MPI_MAP_BY} bind_to=${MPI_BIND_TO} rank_by=${MPI_RANK_BY:-<unset>} report_bindings=${MPI_REPORT_BINDINGS}"
log "RUNTIME_CVE_CHECK_ENABLED=$((1 - SKIP_RUNTIME_CVE_CHECK))"
require_clock_lock=0
if [[ "$SKIP_NCCL" -eq 0 || "$SKIP_TORCHDIST" -eq 0 ]]; then
  require_clock_lock=1
fi
log "CLOCK_LOCK_REQUIRED=${require_clock_lock}"

run_step preflight preflight
if [[ "$SKIP_RUNTIME_CVE_CHECK" -eq 0 ]]; then
  run_step runtime_cve_check run_runtime_cve_check
fi
if [[ "${STEP_RC[preflight]:-0}" -ne 0 ]]; then
  log "ERROR: preflight failed; aborting remaining benchmark steps."
  run_step summary write_summary
  log "Done. Suite log: ${SUITE_LOG}"
  exit 1
fi

if [[ "$SKIP_IPERF3" -eq 0 ]]; then
  run_step iperf3_fwd run_iperf3_pair fwd
  run_step iperf3_rev run_iperf3_pair rev
fi
if [[ "$SKIP_IB" -eq 0 ]]; then
  run_step ib_write_bw run_ib_write_bw
  if [[ "$GDR_ENABLE" -eq 1 ]]; then
    run_step ib_write_lat run_ib_write_lat
  fi
  if [[ "$EXTENDED" -eq 1 ]]; then
    run_step ib_read_bw run_ib_read_bw
    run_step ib_send_bw run_ib_send_bw
    if [[ "$GDR_ENABLE" -eq 1 ]]; then
      run_step ib_read_lat run_ib_read_lat
      run_step ib_send_lat run_ib_send_lat
    fi
  fi
  if [[ "$GDR_ENABLE" -eq 1 ]]; then
    run_step ib_gdr run_ib_gdr_suite
  fi
fi
if [[ "$SKIP_NCCL" -eq 0 ]]; then
  IFS=',' read -r -a NCCL_BIN_ARR <<<"$NCCL_BINS"
  for bin in "${NCCL_BIN_ARR[@]}"; do
    bin="$(echo "$bin" | xargs)"
    [[ -n "$bin" ]] || continue
    case "$bin" in
      all_reduce_perf)
        run_step nccl_all_reduce run_nccl_collective all_reduce_perf "$NCCL_MIN_BYTES" "$NCCL_MAX_AR"
        ;;
      all_gather_perf)
        run_step nccl_all_gather run_nccl_collective all_gather_perf "$NCCL_MIN_BYTES" "$NCCL_MAX_OTHER"
        ;;
      reduce_scatter_perf)
        run_step nccl_reduce_scatter run_nccl_collective reduce_scatter_perf "$NCCL_MIN_BYTES" "$NCCL_MAX_OTHER"
        ;;
      alltoall_perf)
        run_step nccl_alltoall run_nccl_collective alltoall_perf "$NCCL_MIN_BYTES" "$NCCL_MAX_ALLTOALL"
        ;;
      *)
        log "ERROR: unknown --nccl-bins entry: ${bin}"
        STEP_ORDER+=("nccl_bins_error")
        STEP_RC["nccl_bins_error"]=2
        ;;
    esac
  done
fi

run_step connectivity_probe run_connectivity_probe

if [[ "$SKIP_TORCHDIST" -eq 0 ]]; then
  torch_args=(
    --run-id "$RUN_ID"
    --hosts "$HOSTS"
    --gpus-per-node "$GPUS_PER_NODE"
    --master-addr "$LOCAL_IP"
    --master-port "$TORCH_MASTER_PORT"
    --warmup 5
    --iters 20
    --sizes "$TORCH_SIZES"
    --oob-if "$OOB_IF"
  )
  if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
    torch_args+=(--cuda-visible-devices "$CUDA_VISIBLE_DEVICES_LIST")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    torch_args+=(--nccl-ib-hca "$NCCL_IB_HCA")
  fi
  if [[ -n "${NCCL_NVLS_ENABLE_EFFECTIVE:-}" ]]; then
    torch_args+=(--nccl-nvls-enable "$NCCL_NVLS_ENABLE_EFFECTIVE")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    torch_args+=(--ssh-key "$SSH_KEY")
  fi
  run_step torchdist "${ROOT_DIR}/scripts/run_torchrun_allreduce.sh" "${torch_args[@]}"
fi

run_step summary write_summary

log "Done. Suite log: ${SUITE_LOG}"

# Don't fail-fast; report failures via exit code at the end.
exit_code=0
for name in "${STEP_ORDER[@]}"; do
  rc="${STEP_RC[$name]}"
  if [[ "${rc}" -ne 0 ]]; then
    exit_code=1
  fi
done
exit "$exit_code"
