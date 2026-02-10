#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_quick_friction_all_nodes.sh --hosts <h1,h2,...> [options]

Runs a provider friction check on each host and writes:
  results/structured/<run_id>_<label>_quick_friction.json
  results/raw/<run_id>_<label>_quick_friction.log

Checks (default):
  uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo root)
  --checks <csv>         Checks to run (default: all)
  --timeout-sec <n>      Per-check timeout seconds (default: 900)
  --torch-version <ver>  Torch version for install checks (default: 2.5.1)
  --torch-index-url <u>  Torch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --ngc-image <ref>      NGC image for pull check (default: nvcr.io/nvidia/pytorch:24.05-py3)
  --hf-model <id>        HF model for download check (default: openai-community/gpt2)
  --hf-local-dir-base <path>  Base dir for temporary HF downloads (default: /tmp)
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
CHECKS="uv_torch_install,pip_torch_install,ngc_pull,torch_import,hf_download,ip_owner,speedtest"
TIMEOUT_SEC="900"
TORCH_VERSION="2.5.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
NGC_IMAGE="nvcr.io/nvidia/pytorch:24.05-py3"
HF_MODEL="openai-community/gpt2"
HF_LOCAL_DIR_BASE="/tmp"
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
    --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
    --torch-version) TORCH_VERSION="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --ngc-image) NGC_IMAGE="$2"; shift 2 ;;
    --hf-model) HF_MODEL="$2"; shift 2 ;;
    --hf-local-dir-base) HF_LOCAL_DIR_BASE="$2"; shift 2 ;;
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

  out_json="${ROOT_DIR}/results/structured/${RUN_ID}_${label}_quick_friction.json"
  out_log="${ROOT_DIR}/results/raw/${RUN_ID}_${label}_quick_friction.log"

  read -r -d '' PY_PAYLOAD <<'PY' || true
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

run_id = sys.argv[1]
checks_csv = sys.argv[2]
timeout_sec = int(sys.argv[3])
torch_version = sys.argv[4]
torch_index_url = sys.argv[5]
ngc_image = sys.argv[6]
hf_model = sys.argv[7]
hf_local_dir_base = sys.argv[8]

allowed = {
    "uv_torch_install",
    "pip_torch_install",
    "ngc_pull",
    "torch_import",
    "hf_download",
    "ip_owner",
    "speedtest",
}
checks_selected = [c.strip() for c in checks_csv.split(",") if c.strip()]
for check_name in checks_selected:
    if check_name not in allowed:
        raise SystemExit(f"unsupported check: {check_name}")

hostname = socket.gethostname()
tmp_root = tempfile.mkdtemp(prefix=f"quick_friction_{run_id}_{hostname}_")
python_bin = shutil.which("python3") or sys.executable
uv_bin = shutil.which("uv")
docker_bin = shutil.which("docker")
hf_bin = shutil.which("huggingface-cli")
curl_bin = shutil.which("curl")
whois_bin = shutil.which("whois")
speedtest_bin = shutil.which("speedtest-cli") or shutil.which("speedtest")


def _tail(text: str, limit: int = 6000) -> str:
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

uv_env_python = ""
pip_env_python = ""

if "uv_torch_install" in checks_selected:
    if not python_bin:
        rec = _missing_tool("uv_torch_install", "python3", "python3 is not available")
    elif not uv_bin:
        rec = _missing_tool("uv_torch_install", "uv", "uv is not available")
    else:
        uv_env_dir = Path(tmp_root) / "uv_env"
        venv_start = time.monotonic()
        venv_proc = subprocess.run([python_bin, "-m", "venv", str(uv_env_dir)], capture_output=True, text=True)
        venv_elapsed = time.monotonic() - venv_start
        if venv_proc.returncode != 0:
            rec = _record(
                "uv_torch_install",
                [python_bin, "-m", "venv", str(uv_env_dir)],
                venv_proc.returncode,
                venv_elapsed,
                venv_proc.stdout,
                venv_proc.stderr,
                details={"phase": "venv_create"},
            )
        else:
            uv_env_python = str(uv_env_dir / "bin" / "python")
            rec = _run(
                "uv_torch_install",
                [
                    uv_bin,
                    "pip",
                    "install",
                    "--python",
                    uv_env_python,
                    f"torch=={torch_version}",
                    "--index-url",
                    torch_index_url,
                ],
                timeout_sec,
            )
            rec["details"]["venv_create_sec"] = round(venv_elapsed, 6)
            rec["details"]["venv_python"] = uv_env_python
    records.append(rec)
    check_map["uv_torch_install"] = rec

if "pip_torch_install" in checks_selected:
    if not python_bin:
        rec = _missing_tool("pip_torch_install", "python3", "python3 is not available")
    else:
        pip_env_dir = Path(tmp_root) / "pip_env"
        venv_start = time.monotonic()
        venv_proc = subprocess.run([python_bin, "-m", "venv", str(pip_env_dir)], capture_output=True, text=True)
        venv_elapsed = time.monotonic() - venv_start
        if venv_proc.returncode != 0:
            rec = _record(
                "pip_torch_install",
                [python_bin, "-m", "venv", str(pip_env_dir)],
                venv_proc.returncode,
                venv_elapsed,
                venv_proc.stdout,
                venv_proc.stderr,
                details={"phase": "venv_create"},
            )
        else:
            pip_bin = str(pip_env_dir / "bin" / "pip")
            pip_env_python = str(pip_env_dir / "bin" / "python")
            rec = _run(
                "pip_torch_install",
                [
                    pip_bin,
                    "install",
                    f"torch=={torch_version}",
                    "--index-url",
                    torch_index_url,
                ],
                timeout_sec,
            )
            rec["details"]["venv_create_sec"] = round(venv_elapsed, 6)
            rec["details"]["venv_python"] = pip_env_python
    records.append(rec)
    check_map["pip_torch_install"] = rec

if "ngc_pull" in checks_selected:
    if not docker_bin:
        rec = _missing_tool("ngc_pull", "docker", "docker is not available")
    else:
        rec = _run("ngc_pull", [docker_bin, "pull", ngc_image], timeout_sec)
    records.append(rec)
    check_map["ngc_pull"] = rec

if "torch_import" in checks_selected:
    import_python = ""
    if pip_env_python and (check_map.get("pip_torch_install") or {}).get("status") == "ok":
        import_python = pip_env_python
    elif uv_env_python and (check_map.get("uv_torch_install") or {}).get("status") == "ok":
        import_python = uv_env_python
    else:
        import_python = python_bin
    if not import_python:
        rec = _missing_tool("torch_import", "python3", "python is not available for torch import check")
    else:
        rec = _run(
            "torch_import",
            [import_python, "-c", "import torch; print(torch.__version__)"],
            min(timeout_sec, 180),
        )
        rec["details"]["python_executable"] = import_python
    records.append(rec)
    check_map["torch_import"] = rec

if "hf_download" in checks_selected:
    if not hf_bin:
        rec = _missing_tool("hf_download", "huggingface-cli", "huggingface-cli is not available")
    else:
        hf_dir = Path(hf_local_dir_base) / f"{run_id}_{hostname}_hf_download"
        rec = _run(
            "hf_download",
            [hf_bin, "download", hf_model, "--local-dir", str(hf_dir)],
            timeout_sec,
        )
        rec["details"]["model_id"] = hf_model
        rec["details"]["local_dir"] = str(hf_dir)
        shutil.rmtree(hf_dir, ignore_errors=True)
    records.append(rec)
    check_map["hf_download"] = rec

if "ip_owner" in checks_selected:
    if not curl_bin:
        rec = _missing_tool("ip_owner", "curl", "curl is not available")
    else:
        ip_rec = _run("ip_owner", [curl_bin, "-fsS", "--max-time", "20", "https://ifconfig.me"], min(timeout_sec, 120))
        public_ip = (ip_rec.get("stdout_excerpt") or "").strip().splitlines()[-1] if ip_rec.get("stdout_excerpt") else ""
        details = {"public_ip": public_ip}
        if ip_rec["status"] != "ok" or not public_ip:
            rec = ip_rec
            rec["status"] = "error"
            rec["details"] = details
            if not public_ip and rec["stderr_excerpt"] == "":
                rec["stderr_excerpt"] = "public IP lookup returned empty result"
        elif not whois_bin:
            rec = _missing_tool("ip_owner", "whois", "whois is not available")
            rec["details"] = details
        else:
            whois_rec = _run("ip_owner", [whois_bin, public_ip], min(timeout_sec, 120))
            whois_rec["details"] = details
            rec = whois_rec
    records.append(rec)
    check_map["ip_owner"] = rec

if "speedtest" in checks_selected:
    if not speedtest_bin:
        rec = _missing_tool("speedtest", "speedtest-cli/speedtest", "speedtest-cli or speedtest is not available")
    else:
        if os.path.basename(speedtest_bin) == "speedtest-cli":
            cmd = [speedtest_bin, "--secure", "--json"]
        else:
            cmd = [speedtest_bin, "--accept-license", "--accept-gdpr", "-f", "json"]
        rec = _run("speedtest", cmd, min(timeout_sec, 300))
        rec["details"]["binary"] = speedtest_bin
    records.append(rec)
    check_map["speedtest"] = rec

failed_checks = [r["name"] for r in records if r.get("status") != "ok"]
passed_checks = [r["name"] for r in records if r.get("status") == "ok"]
if len(records) == 0:
    overall_status = "error"
elif len(failed_checks) == 0:
    overall_status = "ok"
elif len(passed_checks) > 0:
    overall_status = "degraded"
else:
    overall_status = "error"

payload = {
    "test": "quick_friction",
    "run_id": run_id,
    "host": hostname,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": overall_status,
    "selected_checks": checks_selected,
    "failed_checks": failed_checks,
    "passed_checks": passed_checks,
    "checks": records,
    "config": {
        "timeout_sec": timeout_sec,
        "torch_version": torch_version,
        "torch_index_url": torch_index_url,
        "ngc_image": ngc_image,
        "hf_model": hf_model,
        "hf_local_dir_base": hf_local_dir_base,
    },
    "tool_paths": {
        "python3": python_bin,
        "uv": uv_bin,
        "docker": docker_bin,
        "huggingface_cli": hf_bin,
        "curl": curl_bin,
        "whois": whois_bin,
        "speedtest": speedtest_bin,
    },
}

shutil.rmtree(tmp_root, ignore_errors=True)
print(json.dumps(payload))
PY

  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && python3 -c $(printf '%q' "${PY_PAYLOAD}") $(printf '%q' "${RUN_ID}") $(printf '%q' "${CHECKS}") $(printf '%q' "${TIMEOUT_SEC}") $(printf '%q' "${TORCH_VERSION}") $(printf '%q' "${TORCH_INDEX_URL}") $(printf '%q' "${NGC_IMAGE}") $(printf '%q' "${HF_MODEL}") $(printf '%q' "${HF_LOCAL_DIR_BASE}")"

  echo "Collecting quick friction battery: host=${host} label=${label}"
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
    echo "ERROR: quick friction probe failed on ${host}; see ${out_log}" >&2
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
    echo "ERROR: quick friction status is not ok for ${host}; see ${out_json}" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "Quick friction battery collection complete."
