#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check_ib_sharp.sh [options]

Checks whether the InfiniBand SHARP userspace stack is present and whether the
runtime is wired up to use it (sharp_am service, OpenMPI/HCOLL, NCCL net plugin).
Optionally runs a forced NCCL CollNet all-reduce as an integration check.

Options:
  --run-id <id>             Output prefix (default: YYYY-MM-DD_HHMMSS_ib_sharp_check)
  --hosts <h1,h2,...>       Comma-separated host list (required)
  --gpus-per-node <n>       GPUs per node (default: all visible GPUs)
  --cuda-visible-devices <list>  Optional CUDA_VISIBLE_DEVICES list for nccl-tests
  --ssh-user <user>         SSH user (default: ubuntu)
  --ssh-key <path>          SSH key for remote launch (default: $SSH_KEY)
  --oob-if <iface>          OOB interface (pins OpenMPI OOB/TCP and NCCL bootstrap)
  --nccl-ib-hca <list>      Comma-separated IB HCA allowlist for NCCL ranks

  --skip-nccl               Skip the forced NCCL CollNet check
  --attempt-start-sharp-am  Best-effort: install/start sharp_am systemd service on --sharp-am-host
                            WARNING: starting sharp_am resets SHARP trees and can affect running jobs.
  --sharp-am-host <host>    Host to start sharp_am on (default: first host in --hosts)

  --nccl-min-bytes <size>   nccl-tests min bytes (default: 64M)
  --nccl-max-bytes <size>   nccl-tests max bytes (default: 1G)
  --nccl-warmup <n>         nccl-tests warmup iters (default: 5)
  --nccl-iters <n>          nccl-tests measure iters (default: 20)

Outputs (runs/<run_id>/structured/):
  - ${RUN_ID}_ib_sharp_check.json (summary)
  - ${RUN_ID}_ib_sharp_stack_<host>.txt (per-host evidence)
  - ${RUN_ID}_nccl_collnet_all_reduce_<label>.json (forced CollNet NCCL parse)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_host_runtime_env.sh
source "${ROOT_DIR}/scripts/lib_host_runtime_env.sh"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
source_host_runtime_env_if_present "$ROOT_DIR"

RUN_ID="$(date +%Y-%m-%d_%H%M%S)_ib_sharp_check"
HOSTS=""
GPUS_PER_NODE=""
CUDA_VISIBLE_DEVICES_LIST=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"

SKIP_NCCL=0
ATTEMPT_START_SHARP_AM=0
SHARP_AM_HOST=""

NCCL_MIN_BYTES="64M"
NCCL_MAX_BYTES="1G"
NCCL_WARMUP=5
NCCL_ITERS=20

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
    --skip-nccl) SKIP_NCCL=1; shift ;;
    --attempt-start-sharp-am) ATTEMPT_START_SHARP_AM=1; shift ;;
    --sharp-am-host) SHARP_AM_HOST="$2"; shift 2 ;;
    --nccl-min-bytes) NCCL_MIN_BYTES="$2"; shift 2 ;;
    --nccl-max-bytes) NCCL_MAX_BYTES="$2"; shift 2 ;;
    --nccl-warmup) NCCL_WARMUP="$2"; shift 2 ;;
    --nccl-iters) NCCL_ITERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_RAW_DIR="${CLUSTER_RAW_DIR_EFFECTIVE}"
OUT_STRUCT_DIR="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -lt 1 ]]; then
  echo "ERROR: no hosts specified" >&2
  exit 2
fi

if [[ "${#HOST_ARR[@]}" -gt 1 && -z "$OOB_IF" ]]; then
  echo "ERROR: multi-node SHARP/NCCL checks require --oob-if (pin OpenMPI OOB + NCCL bootstrap)." >&2
  exit 2
fi

if [[ -z "$SHARP_AM_HOST" ]]; then
  SHARP_AM_HOST="${HOST_ARR[0]}"
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; cannot auto-detect --gpus-per-node" >&2
    exit 2
  fi
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=8
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

is_local_host() {
  local h="$1"
  local hn hns ip
  hn="$(hostname)"
  hns="$(hostname -s)"
  if [[ "$h" == "localhost" || "$h" == "127.0.0.1" || "$h" == "::1" || "$h" == "$hn" || "$h" == "$hns" ]]; then
    return 0
  fi
  while IFS= read -r ip; do
    ip="${ip%/*}"
    [[ -n "$ip" ]] || continue
    if [[ "$h" == "$ip" ]]; then
      return 0
    fi
  done < <(ip -o -4 addr 2>/dev/null | awk '{print $4}')
  return 1
}

run_on_host() {
  local host="$1"
  shift
  if is_local_host "$host"; then
    bash -lc "$*"
  else
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$*")"
  fi
}

ts() { date -Is; }
log() { echo "[$(ts)] $*"; }

HOST_STATUS_JSONL="$(mktemp)"
trap 'rm -f "$HOST_STATUS_JSONL"' EXIT

collect_host_stack() {
  local host="$1"
  local out_txt="${OUT_STRUCT_DIR}/${RUN_ID}_ib_sharp_stack_${host}.txt"

  log "Collecting SHARP stack evidence on ${host} -> ${out_txt}"

  run_on_host "$host" "
set -euo pipefail
echo \"IB SHARP (software stack) quick check on this node\"
date -Is
echo

echo \"## /opt/mellanox/sharp present?\"
if [[ -d /opt/mellanox/sharp ]]; then
  ls -ld /opt/mellanox/sharp
else
  echo \"missing\"
fi
echo

echo \"## SHARP binaries\"
if [[ -d /opt/mellanox/sharp/bin ]]; then
  ls -1 /opt/mellanox/sharp/bin | head -n 200
else
  echo \"missing\"
fi
echo

echo \"## SHARP libs\"
if [[ -d /opt/mellanox/sharp/lib ]]; then
  ls -1 /opt/mellanox/sharp/lib | head -n 200
else
  echo \"missing\"
fi
echo

echo \"## sharp_am systemd unit installed?\"
if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files --type=service 2>/dev/null | awk '{print \$1}' | grep -q '^sharp_am\\.service$'; then
    echo \"installed\"
  else
    echo \"not installed\"
  fi
else
  echo \"systemctl missing\"
fi
echo

echo \"## sharp_am systemd state\"
if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active sharp_am 2>/dev/null || true
else
  echo \"systemctl missing\"
fi
echo

echo \"## sharp_am unit file present (but maybe not installed into systemd)?\"
if [[ -f /opt/mellanox/sharp/systemd/system/sharp_am.service ]]; then
  ls -l /opt/mellanox/sharp/systemd/system/sharp_am.service
else
  echo \"missing\"
fi
echo

echo \"## OpenMPI version\"
mpirun --version 2>/dev/null | head -n 5 || true
echo

echo \"## HCOLL + NCCL net plugin presence (ldconfig)\"
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -E 'libhcoll|libnccl-net|libsharp_coll' || true
else
  echo \"ldconfig missing\"
fi
echo

echo \"## NCCL net plugin presence (common /opt paths)\"
find /opt/mellanox -maxdepth 6 -type f \\( -name 'libnccl-net*.so*' -o -name 'libhcoll*.so*' \\) 2>/dev/null | head -n 200 || true
echo

echo \"## sharp_daemons_setup.sh present?\"
if [[ -x /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh ]]; then
  echo \"present\"
  /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh -h 2>&1 | head -n 120 || true
else
  echo \"missing\"
fi
" >"$out_txt" 2>&1 || true

  local kv
  kv="$(run_on_host "$host" "
set -euo pipefail
sharp_dir_present=0; [[ -d /opt/mellanox/sharp ]] && sharp_dir_present=1
sharp_am_bin_present=0; [[ -x /opt/mellanox/sharp/bin/sharp_am ]] && sharp_am_bin_present=1
sharp_am_unit_installed=0
sharp_am_active=
if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files --type=service 2>/dev/null | awk '{print \$1}' | grep -q '^sharp_am\\.service$'; then
    sharp_am_unit_installed=1
  fi
  sharp_am_active=\"\$(systemctl is-active sharp_am 2>/dev/null || true)\"
fi
sharp_am_unit_file_present=0; [[ -f /opt/mellanox/sharp/systemd/system/sharp_am.service ]] && sharp_am_unit_file_present=1
sharp_daemons_setup_present=0; [[ -x /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh ]] && sharp_daemons_setup_present=1
libhcoll_present=0; command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libhcoll' && libhcoll_present=1
libnccl_net_present=0; command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libnccl-net' && libnccl_net_present=1
openmpi_version=\"\$(mpirun --version 2>/dev/null | head -n 1 || true)\"
printf '%s\n' \
  \"sharp_dir_present=\${sharp_dir_present}\" \
  \"sharp_am_bin_present=\${sharp_am_bin_present}\" \
  \"sharp_am_unit_installed=\${sharp_am_unit_installed}\" \
  \"sharp_am_active=\${sharp_am_active}\" \
  \"sharp_am_unit_file_present=\${sharp_am_unit_file_present}\" \
  \"sharp_daemons_setup_present=\${sharp_daemons_setup_present}\" \
  \"libhcoll_present=\${libhcoll_present}\" \
  \"libnccl_net_present=\${libnccl_net_present}\" \
  \"openmpi_version=\${openmpi_version}\"
")"

  python3 - <<'PY' "$HOST_STATUS_JSONL" "$host" "$out_txt" "$kv"
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
host = sys.argv[2]
evidence = sys.argv[3]
kv_raw = sys.argv[4]

fields = {}
for line in kv_raw.splitlines():
    if "=" not in line:
        continue
    k, v = line.split("=", 1)
    fields[k.strip()] = v.strip()

def b(k: str) -> bool:
    return fields.get(k, "0").strip() == "1"

rec = {
    "host": host,
    "evidence_path": evidence,
    "sharp_dir_present": b("sharp_dir_present"),
    "sharp_am_bin_present": b("sharp_am_bin_present"),
    "sharp_am_unit_installed": b("sharp_am_unit_installed"),
    "sharp_am_active": fields.get("sharp_am_active") or None,
    "sharp_am_unit_file_present": b("sharp_am_unit_file_present"),
    "sharp_daemons_setup_present": b("sharp_daemons_setup_present"),
    "libhcoll_present": b("libhcoll_present"),
    "libnccl_net_present": b("libnccl_net_present"),
    "openmpi_version": fields.get("openmpi_version") or None,
}

jsonl_path.write_text(
    jsonl_path.read_text(encoding="utf-8") + json.dumps(rec, sort_keys=True) + "\n"
    if jsonl_path.exists()
    else json.dumps(rec, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

attempt_start_sharp_am() {
  local host="$1"
  local out_txt="${OUT_STRUCT_DIR}/${RUN_ID}_sharp_am_start_${host}.txt"
  log "Attempting sharp_am install/start on ${host} -> ${out_txt}" >&2
  run_on_host "$host" "
set -euo pipefail
echo \"sharp_am install/start attempt\"
date -Is
echo \"host: ${host}\"
echo

if ! sudo -n true >/dev/null 2>&1; then
  echo \"ERROR: passwordless sudo not available\"
  exit 3
fi

if [[ ! -x /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh ]]; then
  echo \"ERROR: /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh missing\"
  exit 3
fi

echo \"== Install systemd unit (sharp_daemons_setup.sh -s) ==\"
sudo -n /opt/mellanox/sharp/sbin/sharp_daemons_setup.sh -s 2>&1 || true
echo

echo \"== systemctl daemon-reload ==\"
sudo -n systemctl daemon-reload 2>&1 || true
echo

echo \"== systemctl start sharp_am ==\"
sudo -n systemctl start sharp_am 2>&1 || true
echo

echo \"== systemctl is-active sharp_am ==\"
systemctl is-active sharp_am 2>&1 || true
echo

echo \"== systemctl status sharp_am (last 50 lines) ==\"
systemctl status sharp_am --no-pager 2>&1 | tail -n 50 || true
" >"$out_txt" 2>&1 || true
  echo "$out_txt"
}

run_nccl_collnet_all_reduce() {
  local label="$1"
  local hostfile="${OUT_RAW_DIR}/${RUN_ID}_hosts.txt"
  local raw_log="${OUT_RAW_DIR}/${RUN_ID}_nccl_collnet_all_reduce_${label}.log"
  local cmd_log="${OUT_RAW_DIR}/${RUN_ID}_nccl_collnet_all_reduce_${label}.cmd.txt"
  local out_json="${OUT_STRUCT_DIR}/${RUN_ID}_nccl_collnet_all_reduce_${label}.json"

  printf "%s\n" "${HOST_ARR[@]}" >"$hostfile"
  local total_ranks=$((GPUS_PER_NODE * ${#HOST_ARR[@]}))

  local mpi_py="${ROOT_DIR}/env/venv/bin/python3"
  local wrapper="${ROOT_DIR}/scripts/nccl_lock_wrapper.py"
  local nccl_bin="${ROOT_DIR}/tools/nccl-tests/build/all_reduce_perf"

  if [[ ! -x "$mpi_py" ]]; then
    echo "Missing python venv at $mpi_py" >&2
    return 2
  fi
  if [[ ! -x "$nccl_bin" ]]; then
    echo "Missing nccl-tests binary at $nccl_bin" >&2
    return 2
  fi

  local -a mpirun_cmd=(
    mpirun
    --hostfile "$hostfile"
    --map-by "ppr:${GPUS_PER_NODE}:node"
    --bind-to none
    --mca routed direct
  )
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
    -x "NCCL_DEBUG=INFO"
    -x "NCCL_DEBUG_SUBSYS=INIT,COLL,NET"
    # NCCL >=2.14 exposes CollNet algos as CollnetDirect/CollnetChain. NCCL >=2.24
    # fails hard on unknown NCCL_ALGO tokens, so avoid the older "CollNet" token.
    -x "NCCL_ALGO=allreduce:CollnetDirect,CollnetChain"
    -x "NCCL_COLLNET_ENABLE=1"
    -x "PATH=$PATH"
    -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  )
  if [[ -n "$CUDA_VISIBLE_DEVICES_LIST" ]]; then
    mpirun_cmd+=(-x "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}")
  fi
  if [[ -n "$OOB_IF" ]]; then
    mpirun_cmd+=(-x "NCCL_SOCKET_IFNAME=${OOB_IF}")
  fi
  if [[ -n "$NCCL_IB_HCA" ]]; then
    mpirun_cmd+=(-x "NCCL_IB_HCA=${NCCL_IB_HCA}")
  fi

  mpirun_cmd+=(
    "$mpi_py"
    "$wrapper"
    --
    "$nccl_bin"
    -b "$NCCL_MIN_BYTES"
    -e "$NCCL_MAX_BYTES"
    -f 2
    -g 1
    -w "$NCCL_WARMUP"
    -n "$NCCL_ITERS"
  )

  printf "%q " "${mpirun_cmd[@]}" >"$cmd_log"
  log "NCCL forced CollNet all-reduce: label=${label} ranks=${total_ranks} min=${NCCL_MIN_BYTES} max=${NCCL_MAX_BYTES} OOB_IF=${OOB_IF:-<unset>} NCCL_IB_HCA=${NCCL_IB_HCA:-<auto>}" >&2

  "${mpirun_cmd[@]}" 2>&1 | tee "$raw_log"
  local rc=${PIPESTATUS[0]}
  if [[ "$rc" -ne 0 ]]; then
    log "Forced CollNet NCCL run failed rc=${rc}" >&2
    local error_excerpt="${OUT_STRUCT_DIR}/${RUN_ID}_nccl_collnet_all_reduce_${label}_error_excerpt.txt"
    {
      echo "NCCL forced CollNet all-reduce error excerpt"
      echo "time: $(date -Is)"
      echo "run_id: ${RUN_ID}"
      echo "label: ${label}"
      echo "rc: ${rc}"
      echo "raw_log: ${raw_log}"
      echo "cmd_log: ${cmd_log}"
      echo ""
      grep -n -E -i "NCCL WARN|NCCL ERROR|invalid usage|Test NCCL failure|collnet|sharp|libnccl-net|hcoll" "$raw_log" | head -n 250 || true
    } >"$error_excerpt"
    log "Wrote ${error_excerpt}" >&2
    return "$rc"
  fi

  local cmd_string
  cmd_string="$(cat "$cmd_log")"
  "${ROOT_DIR}/scripts/parse_nccl_log.py" \
    --input "$raw_log" \
    --output "$out_json" \
    --run-id "$RUN_ID" \
    --hosts "$HOSTS" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --command "$cmd_string"
  log "Wrote ${out_json}" >&2
  return 0
}

log "RUN_ID=${RUN_ID}"
log "HOSTS=${HOSTS} GPUS_PER_NODE=${GPUS_PER_NODE}"
log "OOB_IF=${OOB_IF:-<unset>} NCCL_IB_HCA=${NCCL_IB_HCA:-<auto>}"
log "SKIP_NCCL=${SKIP_NCCL} ATTEMPT_START_SHARP_AM=${ATTEMPT_START_SHARP_AM} SHARP_AM_HOST=${SHARP_AM_HOST}"

for host in "${HOST_ARR[@]}"; do
  host="$(echo "$host" | xargs)"
  [[ -n "$host" ]] || continue
  collect_host_stack "$host"
done

sharp_start_path=""
sharp_am_active_after_start=""
collnet_before_rc=""
collnet_after_rc=""

if [[ "$SKIP_NCCL" -eq 0 ]]; then
  if run_nccl_collnet_all_reduce before_start; then
    collnet_before_rc="0"
  else
    collnet_before_rc="$?"
  fi
fi

if [[ "$ATTEMPT_START_SHARP_AM" -eq 1 ]]; then
  sharp_start_path="$(attempt_start_sharp_am "$SHARP_AM_HOST" || true)"
  sharp_am_active_after_start="$(run_on_host "$SHARP_AM_HOST" "systemctl is-active sharp_am 2>/dev/null || true" 2>/dev/null || true)"
  if [[ "$SKIP_NCCL" -eq 0 ]]; then
    if run_nccl_collnet_all_reduce after_start; then
      collnet_after_rc="0"
    else
      collnet_after_rc="$?"
    fi
  fi
fi

SUMMARY_JSON="${OUT_STRUCT_DIR}/${RUN_ID}_ib_sharp_check.json"
python3 - <<'PY' \
  "$SUMMARY_JSON" "$RUN_ID" "$HOSTS" "$GPUS_PER_NODE" "$OOB_IF" "$NCCL_IB_HCA" "$SHARP_AM_HOST" \
  "$HOST_STATUS_JSONL" "$sharp_start_path" "$sharp_am_active_after_start" "$collnet_before_rc" "$collnet_after_rc"
import json
import sys
import time
from pathlib import Path

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
hosts = [h.strip() for h in sys.argv[3].split(",") if h.strip()]
gpus_per_node = int(sys.argv[4])
oob_if = sys.argv[5]
nccl_ib_hca = sys.argv[6]
sharp_am_host = sys.argv[7]
host_jsonl = Path(sys.argv[8])
sharp_start_path = sys.argv[9]
sharp_am_active_after = sys.argv[10]
collnet_before_rc = sys.argv[11]
collnet_after_rc = sys.argv[12]

per_host = []
if host_jsonl.exists():
    for line in host_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        per_host.append(json.loads(line))

def read_json(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

struct_dir = out_path.parent
raw_dir = struct_dir.parent / "raw"

def coll_run(label: str, rc_str: str):
    if not rc_str:
        return None
    try:
        rc = int(rc_str)
    except ValueError:
        rc = None
    raw_log = raw_dir / f"{run_id}_nccl_collnet_all_reduce_{label}.log"
    cmd_log = raw_dir / f"{run_id}_nccl_collnet_all_reduce_{label}.cmd.txt"
    out_json = struct_dir / f"{run_id}_nccl_collnet_all_reduce_{label}.json"
    err_excerpt = struct_dir / f"{run_id}_nccl_collnet_all_reduce_{label}_error_excerpt.txt"
    d = read_json(str(out_json))
    return {
        "label": label,
        "rc": rc,
        "raw_log": str(raw_log) if raw_log.exists() else None,
        "cmd_log": str(cmd_log) if cmd_log.exists() else None,
        "structured_json": str(out_json) if out_json.exists() else None,
        "error_excerpt": str(err_excerpt) if err_excerpt.exists() else None,
        "log_summary": (d.get("log_summary") if isinstance(d, dict) else None),
    }

collnet_runs = []
for label, rc_str in (("before_start", collnet_before_rc), ("after_start", collnet_after_rc)):
    r = coll_run(label, rc_str)
    if r:
        collnet_runs.append(r)

payload = {
    "run_id": run_id,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hosts": hosts,
    "gpus_per_node": gpus_per_node,
    "oob_if": oob_if or None,
    "nccl_ib_hca": nccl_ib_hca or None,
    "per_host": per_host,
    "sharp_am_start_attempt": {
        "host": sharp_am_host or None,
        "evidence_path": sharp_start_path or None,
        "sharp_am_active_after_start": sharp_am_active_after.strip() or None,
    },
    "nccl_forced_collnet_all_reduce": {
        "ran": bool(collnet_runs),
        "env": {
            "NCCL_ALGO": "allreduce:CollnetDirect,CollnetChain",
            "NCCL_COLLNET_ENABLE": "1",
            "NCCL_DEBUG_SUBSYS": "INIT,COLL,NET",
        },
        "runs": collnet_runs,
    },
    "interpretation": [
        "This checks IB SHARP (in-network reduction) readiness, not NCCL NVLS (NVLink SHARP).",
        "If the SHARP userspace stack exists but sharp_am is inactive and no NCCL net plugin/HCOLL is present, expect SHARP not to engage for NCCL/MPI collectives.",
        "The forced CollNet run is an integration hint: look for CollNet/Collnet* + SHARP mentions in the NCCL log_summary.",
    ],
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY

log "Wrote ${SUMMARY_JSON}"
