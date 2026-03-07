#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
LABEL="${LABEL:-$(hostname)}"
ORIGINAL_ARGS="$*"

APPLY=0
SET_HOSTNAME=""
REGEN_MACHINE_ID=0
REGEN_SSH_KEYS=0
PEERS=""
LOG_OPS=0
OPS_LOG=""

usage() {
  cat <<'USAGE'
Usage: setup.sh [options]

Identity snapshot (always):
  --label <label>            Label for output files (default: hostname)
  --run-id <run_id>          Run ID prefix (default: YYYY-MM-DD)
  --peers <ip_list>          Comma-separated peer IPs for ping checks
  --log-ops                  Append high-level actions to operator log

Identity changes (only with --apply):
  --set-hostname <name>      Set a unique hostname via hostnamectl
  --regenerate-machine-id    Regenerate /etc/machine-id and dbus machine-id (blocked unless ALLOW_ID_ROTATION=1)
  --regenerate-ssh-hostkeys  Regenerate SSH host keys and restart ssh (blocked unless ALLOW_SSH_KEY_ROTATION=1)
  --apply                    Actually perform changes (otherwise dry-run)

Examples:
  scripts/setup.sh --label node1
  scripts/setup.sh --label node1 --peers <peer_ip1,peer_ip2>
  scripts/setup.sh --label node2 --set-hostname node2 --regenerate-machine-id --regenerate-ssh-hostkeys --apply
  scripts/setup.sh --label node1 --log-ops
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --peers)
      PEERS="$2"
      shift 2
      ;;
    --log-ops)
      LOG_OPS=1
      shift 1
      ;;
    --set-hostname)
      SET_HOSTNAME="$2"
      shift 2
      ;;
    --regenerate-machine-id)
      REGEN_MACHINE_ID=1
      shift 1
      ;;
    --regenerate-ssh-hostkeys)
      REGEN_SSH_KEYS=1
      shift 1
      ;;
    --apply)
      APPLY=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
OUT_STRUCTURED="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"
OUT_RAW="${CLUSTER_RAW_DIR_EFFECTIVE}"
mkdir -p "$OUT_STRUCTURED" "$OUT_RAW"

LOG_FILE="${OUT_RAW}/${RUN_ID}_${LABEL}_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -Iseconds
}

if [[ "$LOG_OPS" -eq 1 ]]; then
  OPS_LOG="${OPS_LOG:-${OUT_RAW}/${RUN_ID}_operator_actions.jsonl}"
  touch "$OPS_LOG"
fi

ops() {
  if [[ "$LOG_OPS" -eq 1 ]]; then
    local ts
    local action
    local command
    local result
    local notes
    ts="$(timestamp)"
    action="${1:-}"
    command="${2:-}"
    result="${3:-ok}"
    notes="${4:-}"
    python3 - <<'PY' "$OPS_LOG" "$LABEL" "$ts" "$action" "$command" "$result" "$notes"
import json
import sys

path = sys.argv[1]
label = sys.argv[2]
ts = sys.argv[3]
action = sys.argv[4]
command = sys.argv[5]
result = sys.argv[6]
notes = sys.argv[7]

entry = {
    "timestamp": ts,
    "node": label,
    "action": action,
    "command": command,
    "result": result,
}
if notes:
    entry["notes"] = notes

with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, sort_keys=True) + "\n")
PY
  fi
}

echo "=== setup.sh ==="
echo "Run ID: ${RUN_ID}"
echo "Label: ${LABEL}"
echo "Log: ${LOG_FILE}"
echo "Peers: ${PEERS:-<none>}"
if [[ "$LOG_OPS" -eq 1 ]]; then
  echo "Ops log: ${OPS_LOG}"
fi
ops "setup_start" "setup.sh ${ORIGINAL_ARGS}" "ok" "run_id=${RUN_ID} hostname=$(hostname) apply=${APPLY} set_hostname=${SET_HOSTNAME:-<none>} regen_machine_id=${REGEN_MACHINE_ID} regen_ssh_keys=${REGEN_SSH_KEYS} peers=${PEERS:-<none>} log=${LOG_FILE}"

write_identity_json() {
  local out_path="$1"
  python3 - <<'PY' "$out_path" "$LABEL"
import glob
import json
import os
import subprocess
import sys
import time

out_path = sys.argv[1]
label = sys.argv[2]

def read_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        return f"<error: {exc}>"

def stat_file(path: str):
    try:
        st = os.stat(path)
        return {
            "path": path,
            "mode": oct(st.st_mode),
            "uid": st.st_uid,
            "gid": st.st_gid,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "inode": st.st_ino,
        }
    except Exception as exc:
        return {"path": path, "error": str(exc)}

def fingerprint(path: str):
    try:
        proc = subprocess.run([
            "ssh-keygen", "-l", "-f", path
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {
            "path": path,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"path": path, "error": str(exc)}

ssh_pub_keys = sorted(glob.glob("/etc/ssh/ssh_host_*_key.pub"))

payload = {
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": read_file("/etc/hostname"),
    "hostname_runtime": subprocess.getoutput("hostname"),
    "machine_id": read_file("/etc/machine-id"),
    "dbus_machine_id": read_file("/var/lib/dbus/machine-id"),
    "files": {
        "hostname": stat_file("/etc/hostname"),
        "machine_id": stat_file("/etc/machine-id"),
        "dbus_machine_id": stat_file("/var/lib/dbus/machine-id"),
    },
    "ssh_host_keys": [fingerprint(p) for p in ssh_pub_keys],
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY
}

PRE_JSON="${OUT_STRUCTURED}/${RUN_ID}_${LABEL}_identity_pre.json"
POST_JSON="${OUT_STRUCTURED}/${RUN_ID}_${LABEL}_identity_post.json"
READINESS_JSON="${OUT_STRUCTURED}/${RUN_ID}_${LABEL}_readiness.json"

echo "[$(timestamp)] Capturing identity snapshot (pre)"
ops "capture_identity_pre" "write_identity_json ${PRE_JSON}" "ok"
write_identity_json "$PRE_JSON"

if [[ "$APPLY" -eq 1 ]]; then
  if [[ -n "$SET_HOSTNAME" ]]; then
    echo "[$(timestamp)] Setting hostname to '${SET_HOSTNAME}'"
    ops "set_hostname" "hostnamectl set-hostname ${SET_HOSTNAME}" "ok"
    sudo hostnamectl set-hostname "$SET_HOSTNAME"
  fi

  if [[ "$REGEN_MACHINE_ID" -eq 1 ]]; then
    if [[ "${ALLOW_ID_ROTATION:-0}" != "1" ]]; then
      echo "[$(timestamp)] Refusing to regenerate machine-id (set ALLOW_ID_ROTATION=1 to override)"
      ops "regenerate_machine_id" "systemd-machine-id-setup" "skipped" "blocked: ALLOW_ID_ROTATION not set"
    else
    echo "[$(timestamp)] Regenerating machine-id"
    ops "regenerate_machine_id" "systemd-machine-id-setup" "ok"
    sudo rm -f /etc/machine-id /var/lib/dbus/machine-id
    sudo systemd-machine-id-setup
    if [[ -e /etc/machine-id ]]; then
      sudo ln -sf /etc/machine-id /var/lib/dbus/machine-id
    fi
    fi
  fi

  if [[ "$REGEN_SSH_KEYS" -eq 1 ]]; then
    if [[ "${ALLOW_SSH_KEY_ROTATION:-0}" != "1" ]]; then
      echo "[$(timestamp)] Refusing to regenerate SSH host keys (set ALLOW_SSH_KEY_ROTATION=1 to override)"
      ops "regenerate_ssh_hostkeys" "ssh-keygen -A" "skipped" "blocked: ALLOW_SSH_KEY_ROTATION not set"
    else
      echo "[$(timestamp)] Regenerating SSH host keys"
      ops "regenerate_ssh_hostkeys" "ssh-keygen -A" "ok"
      sudo rm -f /etc/ssh/ssh_host_*key /etc/ssh/ssh_host_*key.pub
      sudo ssh-keygen -A
      sudo systemctl restart ssh || sudo systemctl restart sshd || true
    fi
  fi

  echo "[$(timestamp)] Capturing identity snapshot (post)"
  ops "capture_identity_post" "write_identity_json ${POST_JSON}" "ok"
  write_identity_json "$POST_JSON"
else
  if [[ -n "$SET_HOSTNAME" || "$REGEN_MACHINE_ID" -eq 1 || "$REGEN_SSH_KEYS" -eq 1 ]]; then
    echo "[$(timestamp)] Changes requested but not applied. Re-run with --apply."
    ops "apply_changes" "setup.sh" "skipped" "changes requested but not applied"
  else
    echo "[$(timestamp)] No changes requested."
    ops "apply_changes" "setup.sh" "skipped" "no changes requested"
  fi
fi

run_readiness_checks() {
  local out_path="$1"
  local peers_normalized="${PEERS//,/ }"
  export PEERS="${peers_normalized}"
  python3 - <<'PY' "$out_path" "$LABEL"
import json
import os
import subprocess
import sys
import time

out_path = sys.argv[1]
label = sys.argv[2]
peers = [p for p in os.environ.get("PEERS", "").split() if p]

def run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

cmds = [
    ("date", "date -Iseconds"),
    ("hostname", "hostname"),
    ("uname", "uname -a"),
    ("nvidia_smi", "nvidia-smi"),
    ("nvidia_smi_l", "nvidia-smi -L"),
    ("nvidia_smi_query", "nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm,clocks.mem,clocks.gr --format=csv"),
    ("nvidia_smi_q", "nvidia-smi -q -d ECC,POWER,CLOCK,PERFORMANCE,UTILIZATION,TEMPERATURE,PCIE,ERROR,NVLINK"),
    ("nvidia_smi_nvlink", "nvidia-smi nvlink -s"),
    ("nvidia_smi_topo", "nvidia-smi topo -m"),
    ("lsblk", "lsblk -o NAME,TYPE,SIZE,MODEL,FSTYPE,MOUNTPOINTS"),
    ("df", "df -hT"),
    ("mount", "mount"),
    ("nvme_list", "nvme list"),
    ("ip_link", "ip -o link"),
    ("ip_addr", "ip -o addr"),
    ("ethtool", "for i in $(ls /sys/class/net); do echo ==== $i ===; ethtool $i 2>/dev/null; done"),
    ("ibstat", "ibstat"),
    ("rdma_link", "rdma link"),
    ("ibv_devinfo", "ibv_devinfo"),
    ("systemctl_nvidia_persistenced", "systemctl is-active nvidia-persistenced"),
    ("systemctl_nvidia_fabricmanager", "systemctl is-active nvidia-fabricmanager"),
    ("systemctl_nvidia_dcgm", "systemctl is-active nvidia-dcgm"),
]

for peer in peers:
    cmds.append((f"ping_{peer}", f"ping -c 3 -W 1 {peer}"))

results = {
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "peers": peers,
    "commands": {},
}

for name, cmd in cmds:
    results["commands"][name] = run(cmd)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY
}

echo "[$(timestamp)] Running readiness checks (disk, network, GPU)"
ops "run_readiness_checks" "write_readiness_json ${READINESS_JSON}" "ok"
run_readiness_checks "$READINESS_JSON"

ops "setup_complete" "setup.sh" "ok"
echo "[$(timestamp)] setup.sh complete"
