#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: preflight_cluster_services.sh --run-id <id> --hosts <h1,h2,...> [options]

Starts/validates required NVIDIA services for cluster eval runs and writes a
structured preflight artifact under results/structured/.

This script is intentionally strict:
  - nvidia-persistenced must be active (needed for Docker GPU containers).
  - nvidia-dcgm must be active (required for DCGM-backed monitoring/health checks).
  - If nvidia-imex-ctl is available and this is multi-node, IMEX Domain State
    must be UP (required for MNNVL-capable NCCL setups).

Options:
  --run-id <id>        RUN_ID prefix (required; used for output filenames)
  --hosts <h1,h2,...>  Comma-separated host list (required)
  --ssh-user <user>    SSH user (default: ubuntu)
  --ssh-key <path>     SSH key (default: $SSH_KEY)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"

RUN_ID=""
HOSTS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: --run-id is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -eq 0 ]]; then
  echo "ERROR: no hosts specified" >&2
  exit 2
fi

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"

OUT_STRUCT="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
mkdir -p "$OUT_STRUCT"
OUT_JSON="${OUT_STRUCT}/${RUN_ID}_preflight_services.json"
IMEX_N_OUT="${OUT_STRUCT}/${RUN_ID}_imex_ctl_N.txt"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
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
  # Treat local IPs as local to avoid self-SSH hangs on some clusters.
  while IFS= read -r ip; do
    ip="${ip%/*}"
    [[ -n "$ip" ]] || continue
    if [[ "$h" == "$ip" ]]; then
      return 0
    fi
  done < <(ip -o -4 addr 2>/dev/null | awk '{print $4}')
  return 1
}

run_remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
}

start_service_if_present() {
  local svc="$1"
  if command -v systemctl >/dev/null 2>&1; then
    if service_unit_present "$svc"; then
      sudo -n systemctl start "${svc}" >/dev/null 2>&1 || true
    fi
  fi
}

service_unit_present() {
  local svc="$1"
  local unit="${svc}.service"
  if command -v systemctl >/dev/null 2>&1; then
    local load_state
    load_state="$(systemctl show -p LoadState --value "$unit" 2>/dev/null || true)"
    [[ -n "$load_state" && "$load_state" != "not-found" ]]
    return $?
  fi
  return 1
}

get_is_active() {
  local svc="$1"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl is-active "${svc}" 2>/dev/null || true
  fi
}

sanitize_host_for_path() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  raw="${raw//\\//_}"
  echo "$raw"
}

host_status_jsonl="$(mktemp)"
trap 'rm -f "$host_status_jsonl"' EXIT

multi_node=0
if [[ "${#HOST_ARR[@]}" -gt 1 ]]; then
  multi_node=1
fi

for raw_host in "${HOST_ARR[@]}"; do
  host="$(echo "$raw_host" | xargs)"
  [[ -n "$host" ]] || continue

  if is_local_host "$host"; then
    if ! sudo -n true >/dev/null 2>&1; then
      echo "ERROR: passwordless sudo is required for cluster eval preflight (local host)." >&2
      echo "Fix: ensure \`sudo -n true\` succeeds for this user." >&2
      exit 3
    fi
    if ! service_unit_present nvidia-dcgm; then
      echo "ERROR: nvidia-dcgm.service is not installed/present on the local host." >&2
      echo "Fix: install/enable DCGM, then re-run." >&2
      exit 3
    fi
    dcgm_restart_policy="$(systemctl show -p Restart --value nvidia-dcgm 2>/dev/null || true)"
    dcgm_active_before="$(get_is_active nvidia-dcgm)"
    start_service_if_present nvidia-persistenced
    start_service_if_present nvidia-dcgm
    if [[ "$multi_node" -eq 1 ]]; then
      start_service_if_present nvidia-imex
    fi
    persistenced_active="$(get_is_active nvidia-persistenced)"
    dcgm_active_after="$(get_is_active nvidia-dcgm)"
    imex_active="$(get_is_active nvidia-imex)"
    persistenced_socket="missing"
    if [[ -S "/run/nvidia-persistenced/socket" ]]; then
      persistenced_socket="present"
    fi
  else
remote_cmd=$(
      cat <<'BASH'
set -euo pipefail
svc_present() {
  local svc="$1"
  local unit="${svc}.service"
  if ! command -v systemctl >/dev/null 2>&1; then
    return 1
  fi
  local load_state
  load_state="$(systemctl show -p LoadState --value "$unit" 2>/dev/null || true)"
  [[ -n "$load_state" && "$load_state" != "not-found" ]]
}
start_if_present() { if command -v systemctl >/dev/null 2>&1 && svc_present "$1"; then sudo -n systemctl start "$1" >/dev/null 2>&1 || true; fi; }
is_active() { if command -v systemctl >/dev/null 2>&1; then systemctl is-active "$1" 2>/dev/null || true; fi; }
get_restart() { if command -v systemctl >/dev/null 2>&1; then systemctl show -p Restart --value "$1" 2>/dev/null || true; fi; }

if ! sudo -n true >/dev/null 2>&1; then
  echo "ERROR: passwordless sudo is required for cluster eval preflight (remote host)." >&2
  exit 3
fi

dcgm_present="missing"
if svc_present nvidia-dcgm; then
  dcgm_present="present"
fi

dcgm_restart="$(get_restart nvidia-dcgm)"
dcgm_active_before="$(is_active nvidia-dcgm)"

start_if_present nvidia-persistenced
start_if_present nvidia-dcgm
if [[ "${MULTI_NODE:-0}" == "1" ]]; then
  start_if_present nvidia-imex
fi

persistenced_active="$(is_active nvidia-persistenced)"
dcgm_active_after="$(is_active nvidia-dcgm)"
imex_active="$(is_active nvidia-imex)"
sock="missing"
if [[ -S "/run/nvidia-persistenced/socket" ]]; then
  sock="present"
fi

printf '%s\n' "persistenced_active=${persistenced_active}" "persistenced_socket=${sock}" "dcgm_present=${dcgm_present}" "dcgm_restart=${dcgm_restart}" "dcgm_active_before=${dcgm_active_before}" "dcgm_active_after=${dcgm_active_after}" "imex_active=${imex_active}"
BASH
    )
    # Run as a single token on the remote side.
    remote_out="$(run_remote "$host" "bash -lc $(printf '%q' "MULTI_NODE=${multi_node} ${remote_cmd}")")"
    persistenced_active="$(echo "$remote_out" | awk -F= '$1=="persistenced_active"{print $2}' | tail -n 1)"
    persistenced_socket="$(echo "$remote_out" | awk -F= '$1=="persistenced_socket"{print $2}' | tail -n 1)"
    dcgm_present="$(echo "$remote_out" | awk -F= '$1=="dcgm_present"{print $2}' | tail -n 1)"
    dcgm_restart_policy="$(echo "$remote_out" | awk -F= '$1=="dcgm_restart"{print $2}' | tail -n 1)"
    dcgm_active_before="$(echo "$remote_out" | awk -F= '$1=="dcgm_active_before"{print $2}' | tail -n 1)"
    dcgm_active_after="$(echo "$remote_out" | awk -F= '$1=="dcgm_active_after"{print $2}' | tail -n 1)"
    imex_active="$(echo "$remote_out" | awk -F= '$1=="imex_active"{print $2}' | tail -n 1)"
  fi

  dcgm_started_by_preflight=0
  if [[ "${dcgm_active_before:-}" != "active" && "${dcgm_active_after:-}" == "active" ]]; then
    dcgm_started_by_preflight=1
    echo "=======================================================================" >&2
    echo "WARNING: nvidia-dcgm was NOT active on host ${host}" >&2
    echo "WARNING: before='${dcgm_active_before:-<missing>}' restart_policy='${dcgm_restart_policy:-<missing>}'" >&2
    echo "WARNING: preflight started DCGM (after='${dcgm_active_after}')" >&2
    echo "WARNING: this indicates DCGM can silently disappear and break monitoring" >&2
    echo "=======================================================================" >&2
  fi

  host_tag="$(sanitize_host_for_path "$host")"
  dcgmi_rel="results/structured/${RUN_ID}_dcgmi_discovery_${host_tag}.txt"
  dcgmi_out="${OUT_STRUCT}/${RUN_ID}_dcgmi_discovery_${host_tag}.txt"
  dcgmi_rc=0
  if is_local_host "$host"; then
    set +e
    timeout 10s dcgmi discovery -l >"$dcgmi_out" 2>&1
    dcgmi_rc=$?
    set -e
  else
    set +e
    run_remote "$host" "bash -lc $(printf '%q' "timeout 10s dcgmi discovery -l")" >"$dcgmi_out" 2>&1
    dcgmi_rc=$?
    set -e
  fi

  python3 - <<'PY' "$host_status_jsonl" "$host" "$persistenced_active" "$persistenced_socket" "${dcgm_present:-present}" "${dcgm_active_before:-}" "${dcgm_active_after:-}" "${dcgm_restart_policy:-}" "$dcgm_started_by_preflight" "$dcgmi_rel" "$dcgmi_rc" "$imex_active"
import json
import sys
from pathlib import Path

out_path, host, persist_act, persist_sock, dcgm_present, dcgm_before, dcgm_after, dcgm_restart, dcgm_started, dcgmi_path, dcgmi_rc, imex_act = sys.argv[1:]
rec = {
    "host": host,
    "nvidia_persistenced": {"active": persist_act or None, "socket": persist_sock or None},
    # Redundant top-level DCGM fields make per-host before/after audits obvious in JSON.
    "dcgm_active_before": dcgm_before or None,
    "dcgm_active_after": dcgm_after or None,
    "dcgm_started_by_preflight": (dcgm_started == "1"),
    "nvidia_dcgm": {
        "present": (dcgm_present == "present"),
        "restart": dcgm_restart or None,
        "active_before": dcgm_before or None,
        "active_after": dcgm_after or None,
        "started_by_preflight": (dcgm_started == "1"),
    },
    "dcgmi": {
        "discovery_list_path": dcgmi_path,
        "discovery_list_rc": int(dcgmi_rc),
    },
    "nvidia_imex": {"active": imex_act or None},
}
Path(out_path).write_text(Path(out_path).read_text() + json.dumps(rec) + "\n" if Path(out_path).exists() else json.dumps(rec) + "\n")
PY

  if [[ "$persistenced_active" != "active" ]]; then
    echo "ERROR: nvidia-persistenced is not active on host ${host} (status='${persistenced_active}')." >&2
    echo "Fix: sudo systemctl start nvidia-persistenced" >&2
    exit 3
  fi
  if is_local_host "$host"; then
    if [[ "$persistenced_socket" != "present" ]]; then
      echo "ERROR: /run/nvidia-persistenced/socket is missing on the local host." >&2
      echo "Fix: sudo systemctl start nvidia-persistenced" >&2
      exit 3
    fi
  fi

  if [[ "${dcgm_present:-present}" != "present" ]]; then
    echo "ERROR: nvidia-dcgm.service is not installed/present on host ${host}." >&2
    echo "Fix: install/enable DCGM, then re-run." >&2
    exit 3
  fi
  if [[ "${dcgm_active_after:-}" != "active" ]]; then
    echo "ERROR: nvidia-dcgm is not active on host ${host} (before='${dcgm_active_before:-<missing>}', after='${dcgm_active_after:-<missing>}', restart_policy='${dcgm_restart_policy:-<missing>}')." >&2
    echo "Fix: sudo systemctl start nvidia-dcgm" >&2
    exit 3
  fi
  if [[ "${dcgmi_rc:-0}" -ne 0 ]]; then
    echo "ERROR: dcgmi failed on host ${host} (rc=${dcgmi_rc})." >&2
    echo "See: ${dcgmi_rel}" >&2
    echo "Fix: ensure nv-hostengine is running and reachable (systemctl start nvidia-dcgm), then re-run." >&2
    exit 3
  fi
done

imex_domain_state=""
if [[ "$multi_node" -eq 1 && -x "$(command -v nvidia-imex-ctl || true)" ]]; then
  # Capture IMEX domain health (used by MNNVL-capable NCCL setups).
  nvidia-imex-ctl -N >"$IMEX_N_OUT" 2>&1 || true
  imex_domain_state="$(awk -F': ' '/Domain State:/{print $2; exit}' "$IMEX_N_OUT" | tr -d '\r')"
  if [[ "$imex_domain_state" != "UP" ]]; then
    echo "ERROR: IMEX domain is not UP (Domain State: ${imex_domain_state:-<missing>})." >&2
    echo "See: ${IMEX_N_OUT}" >&2
    echo "Fix: sudo systemctl start nvidia-imex (on all nodes), then re-run." >&2
    exit 3
  fi
fi

python3 - <<'PY' "$OUT_JSON" "$RUN_ID" "$HOSTS" "$SSH_USER" "$imex_domain_state" "$host_status_jsonl"
import json
import sys
import time
from pathlib import Path

out_path, run_id, hosts, ssh_user, domain_state, jsonl_path = sys.argv[1:]

records = []
for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    records.append(json.loads(line))

payload = {
    "run_id": run_id,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hosts": [h.strip() for h in hosts.split(",") if h.strip()],
    "ssh_user": ssh_user,
    "imex": {
        "checked": bool(domain_state),
        "domain_state": domain_state or None,
        "imex_ctl_N_path": (f"results/structured/{run_id}_imex_ctl_N.txt" if domain_state else None),
    },
    "per_host": records,
}

Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote {out_path}")
PY

if [[ -f "$IMEX_N_OUT" ]]; then
  echo "Wrote ${IMEX_N_OUT}"
fi
