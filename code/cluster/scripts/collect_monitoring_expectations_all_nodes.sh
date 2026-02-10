#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/collect_monitoring_expectations_all_nodes.sh --hosts <h1,h2,...> [options]

Collects monitoring evidence expected by stakeholders and writes:
  results/structured/<run_id>_<label>_monitoring_expectations.json
  results/raw/<run_id>_<label>_monitoring_expectations.log

Checks (default):
  kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo root)
  --checks <csv>         Checks to run (default: all)
  --sample-count <n>     Sample count for dmon checks (default: 20)
  --dmesg-lines <n>      dmesg tail lines to capture (default: 400)
  --timeout-sec <n>      Per-check timeout seconds (default: 180)
  --strict               Return non-zero if any host status is not "ok"
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"
CHECKS="kubectl_pods,kubectl_top_nodes,kubectl_top_pods,nvidia_dmon,nvidia_nvlink,dcgmi_discovery,dcgmi_dmon,dmesg_tail"
SAMPLE_COUNT="20"
DMESG_LINES="400"
TIMEOUT_SEC="180"
STRICT=0

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --checks) CHECKS="$2"; shift 2 ;;
    --sample-count) SAMPLE_COUNT="$2"; shift 2 ;;
    --dmesg-lines) DMESG_LINES="$2"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if ! [[ "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --sample-count must be a positive integer (got: ${SAMPLE_COUNT})" >&2
  exit 2
fi
if ! [[ "$DMESG_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --dmesg-lines must be a positive integer (got: ${DMESG_LINES})" >&2
  exit 2
fi
if ! [[ "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --timeout-sec must be a positive integer (got: ${TIMEOUT_SEC})" >&2
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

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/structured"

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

  out_json="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_monitoring_expectations.json"
  out_log="${ROOT_DIR}/results/raw/${RUN_ID}_${label}_monitoring_expectations.log"

  read -r -d '' PY_PAYLOAD <<'PY' || true
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time

run_id = sys.argv[1]
checks_csv = sys.argv[2]
sample_count = int(sys.argv[3])
dmesg_lines = int(sys.argv[4])
timeout_sec = int(sys.argv[5])

allowed = {
    "kubectl_pods",
    "kubectl_top_nodes",
    "kubectl_top_pods",
    "nvidia_dmon",
    "nvidia_nvlink",
    "dcgmi_discovery",
    "dcgmi_dmon",
    "dmesg_tail",
}
checks_selected = [c.strip() for c in checks_csv.split(",") if c.strip()]
for check_name in checks_selected:
    if check_name not in allowed:
        raise SystemExit(f"unsupported check: {check_name}")

hostname = socket.gethostname()
kubectl_bin = shutil.which("kubectl")
nvidia_smi_bin = shutil.which("nvidia-smi")
dcgmi_bin = shutil.which("dcgmi")


def _tail(text: str, limit: int = 10000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _record(name: str, command, rc: int, elapsed: float, stdout: str, stderr: str, timed_out: bool = False, details=None):
    return {
        "name": name,
        "command": command if isinstance(command, str) else " ".join(shlex.quote(x) for x in command),
        "status": "ok" if rc == 0 else "error",
        "exit_code": int(rc),
        "duration_sec": round(float(elapsed), 6),
        "timed_out": bool(timed_out),
        "stdout_excerpt": _tail(stdout or ""),
        "stderr_excerpt": _tail(stderr or ""),
        "details": details or {},
    }


def _missing_tool(name: str, tool: str, message: str):
    return _record(
        name=name,
        command=f"<missing:{tool}>",
        rc=127,
        elapsed=0.0,
        stdout="",
        stderr=message,
        details={"required_tool": tool},
    )


def _run(name: str, cmd, per_check_timeout: int):
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=per_check_timeout)
        return _record(
            name=name,
            command=cmd,
            rc=proc.returncode,
            elapsed=time.monotonic() - start,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return _record(
            name=name,
            command=cmd,
            rc=124,
            elapsed=time.monotonic() - start,
            stdout=stdout,
            stderr=stderr or f"timeout after {per_check_timeout}s",
            timed_out=True,
        )


records = []
check_map = {}

if "kubectl_pods" in checks_selected:
    if not kubectl_bin:
        rec = _missing_tool("kubectl_pods", "kubectl", "kubectl is not available")
    else:
        rec = _run("kubectl_pods", [kubectl_bin, "get", "pods", "-A", "-o", "wide"], min(timeout_sec, 120))
    records.append(rec)
    check_map["kubectl_pods"] = rec

if "kubectl_top_nodes" in checks_selected:
    if not kubectl_bin:
        rec = _missing_tool("kubectl_top_nodes", "kubectl", "kubectl is not available")
    else:
        rec = _run("kubectl_top_nodes", [kubectl_bin, "top", "nodes"], min(timeout_sec, 120))
    records.append(rec)
    check_map["kubectl_top_nodes"] = rec

if "kubectl_top_pods" in checks_selected:
    if not kubectl_bin:
        rec = _missing_tool("kubectl_top_pods", "kubectl", "kubectl is not available")
    else:
        rec = _run("kubectl_top_pods", [kubectl_bin, "top", "pods", "-A"], min(timeout_sec, 120))
    records.append(rec)
    check_map["kubectl_top_pods"] = rec

if "nvidia_dmon" in checks_selected:
    if not nvidia_smi_bin:
        rec = _missing_tool("nvidia_dmon", "nvidia-smi", "nvidia-smi is not available")
    else:
        dmon_timeout = max(timeout_sec, sample_count + 30)
        rec = _run(
            "nvidia_dmon",
            [nvidia_smi_bin, "dmon", "-s", "pucm", "-d", "1", "-c", str(sample_count)],
            dmon_timeout,
        )
    records.append(rec)
    check_map["nvidia_dmon"] = rec

if "nvidia_nvlink" in checks_selected:
    if not nvidia_smi_bin:
        rec = _missing_tool("nvidia_nvlink", "nvidia-smi", "nvidia-smi is not available")
    else:
        rec = _run("nvidia_nvlink", [nvidia_smi_bin, "nvlink", "-s"], min(timeout_sec, 120))
    records.append(rec)
    check_map["nvidia_nvlink"] = rec

if "dcgmi_discovery" in checks_selected:
    if not dcgmi_bin:
        rec = _missing_tool("dcgmi_discovery", "dcgmi", "dcgmi is not available")
    else:
        rec = _run("dcgmi_discovery", [dcgmi_bin, "discovery", "-l"], min(timeout_sec, 120))
    records.append(rec)
    check_map["dcgmi_discovery"] = rec

if "dcgmi_dmon" in checks_selected:
    if not dcgmi_bin:
        rec = _missing_tool("dcgmi_dmon", "dcgmi", "dcgmi is not available")
    else:
        dcgmi_timeout = max(timeout_sec, sample_count + 30)
        rec = _run(
            "dcgmi_dmon",
            [dcgmi_bin, "dmon", "-e", "1002,1003,1004", "-c", str(sample_count)],
            dcgmi_timeout,
        )
    records.append(rec)
    check_map["dcgmi_dmon"] = rec

if "dmesg_tail" in checks_selected:
    rec = _run(
        "dmesg_tail",
        ["bash", "-lc", f"set -o pipefail; dmesg -T | tail -n {dmesg_lines}"],
        min(timeout_sec, 120),
    )
    records.append(rec)
    check_map["dmesg_tail"] = rec

category_checks = {
    "control_plane": ["kubectl_pods", "kubectl_top_nodes", "kubectl_top_pods"],
    "gpu_telemetry": ["nvidia_dmon", "nvidia_nvlink", "dcgmi_discovery", "dcgmi_dmon"],
    "system_signals": ["dmesg_tail"],
}

categories = {}
requested_ok_values = []
for category, names in category_checks.items():
    requested = [n for n in names if n in checks_selected]
    if not requested:
        categories[category] = {"status": "not_requested", "checks": []}
        continue
    ok_checks = [n for n in requested if (check_map.get(n) or {}).get("status") == "ok"]
    category_ok = len(ok_checks) > 0
    categories[category] = {
        "status": "ok" if category_ok else "error",
        "checks": requested,
        "ok_checks": ok_checks,
    }
    requested_ok_values.append(category_ok)

if not requested_ok_values:
    overall_status = "error"
elif all(requested_ok_values):
    overall_status = "ok"
elif any(requested_ok_values):
    overall_status = "degraded"
else:
    overall_status = "error"

failed_checks = [r["name"] for r in records if r.get("status") != "ok"]
missing_categories = [c for c, rec in categories.items() if rec.get("status") == "error"]

payload = {
    "test": "monitoring_expectations",
    "run_id": run_id,
    "host": hostname,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": overall_status,
    "selected_checks": checks_selected,
    "failed_checks": failed_checks,
    "missing_categories": missing_categories,
    "categories": categories,
    "checks": records,
    "config": {
        "sample_count": sample_count,
        "dmesg_lines": dmesg_lines,
        "timeout_sec": timeout_sec,
    },
    "tool_paths": {
        "kubectl": kubectl_bin,
        "nvidia_smi": nvidia_smi_bin,
        "dcgmi": dcgmi_bin,
    },
}

print(json.dumps(payload))
PY

  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && python3 -c $(printf '%q' "${PY_PAYLOAD}") $(printf '%q' "${RUN_ID}") $(printf '%q' "${CHECKS}") $(printf '%q' "${SAMPLE_COUNT}") $(printf '%q' "${DMESG_LINES}") $(printf '%q' "${TIMEOUT_SEC}")"

  echo "Collecting monitoring expectations: host=${host} label=${label}"
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
    echo "ERROR: monitoring expectations probe failed on ${host}; see ${out_log}" >&2
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

lines = [
    f"status={payload.get('status')}",
    f"host={payload.get('host')}",
    f"timestamp_utc={payload.get('timestamp_utc')}",
    f"selected_checks={','.join(payload.get('selected_checks') or [])}",
    f"failed_checks={','.join(payload.get('failed_checks') or [])}",
    f"missing_categories={','.join(payload.get('missing_categories') or [])}",
    "",
    "[categories]",
    json.dumps(payload.get("categories") or {}, sort_keys=True),
    "",
]
for rec in payload.get("checks") or []:
    lines.extend(
        [
            f"[{rec.get('name')}]",
            f"status={rec.get('status')}",
            f"exit_code={rec.get('exit_code')}",
            f"duration_sec={rec.get('duration_sec')}",
            f"command={rec.get('command')}",
            f"details={json.dumps(rec.get('details') or {}, sort_keys=True)}",
            "--- stdout_excerpt ---",
            rec.get("stdout_excerpt") or "",
            "--- stderr_excerpt ---",
            rec.get("stderr_excerpt") or "",
            "",
        ]
    )

out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
  if [[ "$STRICT" -eq 1 && "$status" != "ok" ]]; then
    echo "ERROR: monitoring expectations status is not ok for ${host}; see ${out_json}" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "Monitoring expectations collection complete."
