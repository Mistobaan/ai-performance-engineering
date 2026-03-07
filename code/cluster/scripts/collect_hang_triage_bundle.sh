#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/collect_hang_triage_bundle.sh --hosts <h1,h2,...> [options]

Collects a per-node hang-triage readiness bundle:
  - py-spy availability/version (venv path)
  - strace availability/version
  - ptrace_scope
  - process snapshots for likely distributed workloads
  - ready-to-run debug command templates

Artifacts:
  results/structured/<run_id>_<label>_hang_triage_readiness.json
  results/raw/<run_id>_<label>_hang_triage_readiness.log

This script is strict by design:
  it returns non-zero if py-spy or strace is unavailable on any host.

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo root)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./lib_artifact_dirs.sh
source "${ROOT_DIR}/scripts/lib_artifact_dirs.sh"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

is_local_host() {
  local host="$1"
  local h_full
  h_full="$(hostname -f 2>/dev/null || hostname)"
  [[ "$host" == "localhost" || "$host" == "127.0.0.1" || "$host" == "$(hostname)" || "$host" == "$(hostname -s)" || "$host" == "$h_full" ]]
}

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

resolve_cluster_artifact_dirs "$ROOT_DIR" "$RUN_ID"
mkdir -p "${CLUSTER_RAW_DIR_EFFECTIVE}" "${CLUSTER_STRUCTURED_DIR_EFFECTIVE}"

fail=0
for idx in "${!HOST_ARR[@]}"; do
  host="$(echo "${HOST_ARR[$idx]}" | xargs)"
  label=""
  if [[ -n "$LABELS" ]]; then
    label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
  fi
  if [[ -z "$label" ]]; then
    label="$(sanitize_label "$host")"
  fi

  out_json="${CLUSTER_STRUCTURED_DIR_EFFECTIVE}/${RUN_ID}_${label}_hang_triage_readiness.json"
  out_log="${CLUSTER_RAW_DIR_EFFECTIVE}/${RUN_ID}_${label}_hang_triage_readiness.log"

  read -r -d '' PY_PAYLOAD <<'PY' || true
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

remote_root = Path(sys.argv[1])

def run(cmd):
    if isinstance(cmd, str):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
        "rc": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }

py_spy_bin = remote_root / "env" / "venv" / "bin" / "py-spy"
strace_bin = shutil.which("strace")
ptrace_scope = ""
try:
    ptrace_scope = Path("/proc/sys/kernel/yama/ptrace_scope").read_text(encoding="utf-8").strip()
except Exception:
    ptrace_scope = ""

py_spy_check = run([str(py_spy_bin), "--version"]) if py_spy_bin.exists() else {"cmd": str(py_spy_bin), "rc": 127, "stdout": "", "stderr": "missing"}
strace_check = run([strace_bin, "-V"]) if strace_bin else {"cmd": "strace -V", "rc": 127, "stdout": "", "stderr": "missing"}

ps_snapshot = run("ps -eo pid,ppid,etime,stat,comm,args --sort=-etime | head -n 120")
pgrep_snapshot = run("pgrep -a -f 'python|torchrun|ray|vllm|nccl' || true")

status = "ok" if (py_spy_check["rc"] == 0 and strace_check["rc"] == 0) else "error"
payload = {
    "test": "hang_triage_readiness",
    "run_id": os.environ.get("RUN_ID", ""),
    "host": socket.gethostname(),
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": status,
    "tools": {
        "py_spy": {
            "path": str(py_spy_bin),
            "available": py_spy_check["rc"] == 0,
            "version": py_spy_check["stdout"],
            "error": py_spy_check["stderr"] or None,
        },
        "strace": {
            "path": strace_bin or "",
            "available": strace_check["rc"] == 0,
            "version": strace_check["stdout"].splitlines()[0] if strace_check["stdout"] else "",
            "error": strace_check["stderr"] or None,
        },
        "ptrace_scope": ptrace_scope,
    },
    "process_snapshots": {
        "ps_top120": ps_snapshot["stdout"],
        "pgrep_match": pgrep_snapshot["stdout"],
    },
    "debug_templates": {
        "py_spy_dump_by_pid": f"{py_spy_bin} dump -n -p <PID>",
        "py_spy_dump_python_children": "pgrep -P $(pgrep -o python) | xargs -I {} "
                                      + f"{py_spy_bin} dump --pid {{}}",
        "strace_attach": "strace --pid <PID>",
        "strace_follow_forks": "strace -f -o /tmp/strace.log python -m torch.distributed.run --nproc_per_node=<N> --nnodes=<M> <script.py>",
    },
}
print(json.dumps(payload))
PY

  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && RUN_ID=$(printf '%q' "${RUN_ID}") python3 -c $(printf '%q' "${PY_PAYLOAD}") $(printf '%q' "${REMOTE_ROOT}")"

  echo "Collecting hang triage readiness: host=${host} label=${label}"
  if is_local_host "$host"; then
    set +e
    probe_json="$(bash -lc "$remote_cmd" 2>"$out_log")"
    rc=$?
    set -e
  else
    set +e
    probe_json="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "bash -lc $(printf '%q' "$remote_cmd")" 2>"$out_log")"
    rc=$?
    set -e
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: remote triage probe failed on ${host}; see ${out_log}" >&2
    fail=1
    continue
  fi

  python3 - "$probe_json" "$out_json" "$out_log" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[1])
out_json = Path(sys.argv[2])
out_log = Path(sys.argv[3])
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

log_lines = [
    f"status={payload.get('status')}",
    f"host={payload.get('host')}",
    f"timestamp_utc={payload.get('timestamp_utc')}",
    f"py_spy_available={payload.get('tools', {}).get('py_spy', {}).get('available')}",
    f"py_spy_version={payload.get('tools', {}).get('py_spy', {}).get('version')}",
    f"strace_available={payload.get('tools', {}).get('strace', {}).get('available')}",
    f"strace_version={payload.get('tools', {}).get('strace', {}).get('version')}",
    f"ptrace_scope={payload.get('tools', {}).get('ptrace_scope')}",
    "",
    "[pgrep_match]",
    payload.get("process_snapshots", {}).get("pgrep_match", ""),
    "",
    "[ps_top120]",
    payload.get("process_snapshots", {}).get("ps_top120", ""),
]
out_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
print(out_json)
PY

  status="$(python3 - "$out_json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status", "error"))
PY
)"
  if [[ "$status" != "ok" ]]; then
    echo "ERROR: triage readiness failed on ${host}; missing required tools (see ${out_json})" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "Hang triage readiness bundle complete."
