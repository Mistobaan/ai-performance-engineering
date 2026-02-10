#!/usr/bin/env bash
set -euo pipefail

OUTPUT=""
LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "Usage: $0 --output <path> [--label <node_label>]" >&2
  exit 1
fi

python3 - <<'PY' "$OUTPUT" "$LABEL"
import json
import os
import platform
import subprocess
import sys
import time

out_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""

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
    ("os_release", "cat /etc/os-release"),
    ("kernel_cmdline", "cat /proc/cmdline"),
    ("uptime", "uptime"),
    ("whoami", "whoami"),
    ("ulimit", "ulimit -a"),
    ("limits_conf", "grep -v '^#' /etc/security/limits.conf | sed '/^$/d'"),
    ("sysctl_core_net", "sysctl net.core.rmem_max net.core.wmem_max net.core.rmem_default net.core.wmem_default net.core.netdev_max_backlog"),
    ("sysctl_tcp", "sysctl net.ipv4.tcp_rmem net.ipv4.tcp_wmem net.ipv4.tcp_congestion_control net.ipv4.tcp_mtu_probing net.ipv4.tcp_sack net.ipv4.tcp_timestamps net.ipv4.tcp_window_scaling"),
    ("nvidia_smi", "nvidia-smi"),
    ("nvidia_smi_l", "nvidia-smi -L"),
    ("nvidia_smi_topo", "nvidia-smi topo -m"),
    ("nvidia_smi_clocks_power", "nvidia-smi -q -d CLOCK,POWER"),
    ("nvidia_smi_query_telemetry", "nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,clocks.applications.graphics,clocks.applications.memory,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown,fan.speed --format=csv"),
    ("nvcc_version", "nvcc --version"),
    ("lspci_nvidia", "lspci -nn | grep -i nvidia"),
    ("lscpu", "lscpu"),
    ("numactl", "numactl -H"),
    ("free", "free -h"),
    ("lsblk", "lsblk"),
    ("df", "df -hT"),
    ("mount", "mount"),
    ("ip_link", "ip -o link"),
    ("ip_addr", "ip -o addr"),
    ("ethtool", "for i in $(ls /sys/class/net); do echo ==== $i ===; ethtool $i 2>/dev/null; done"),
    ("ibstat", "ibstat"),
    ("rdma_link", "rdma link"),
    ("ibv_devinfo", "ibv_devinfo"),
    ("env_nccl", "env | sort | grep -E '^NCCL_'"),
    ("systemctl_nvidia_dcgm", "systemctl is-active nvidia-dcgm"),
    ("systemctl_nvidia_persistenced", "systemctl is-active nvidia-persistenced"),
    ("systemctl_nvidia_imex", "systemctl is-active nvidia-imex"),
    ("systemctl_nvidia_fabricmanager", "systemctl is-active nvidia-fabricmanager"),
    ("nvidia_imex_ctl_q", "nvidia-imex-ctl -q"),
    ("nvidia_imex_ctl_N", "nvidia-imex-ctl -N"),
    ("dcgmi_version", "dcgmi --version"),
    ("dcgmi_discovery_list", "timeout 5s dcgmi discovery -l"),
    ("dmesg_tail", "timeout 5s sudo -n dmesg -T 2>/dev/null | tail -n 200 || timeout 5s dmesg -T 2>/dev/null | tail -n 200"),
    ("dmesg_xid_sxid_aer_tail", "timeout 5s sudo -n dmesg -T 2>/dev/null | egrep -i 'xid|sxid|aer|pcie bus error|nvrm|nvswitch' | tail -n 200 || timeout 5s dmesg -T 2>/dev/null | egrep -i 'xid|sxid|aer|pcie bus error|nvrm|nvswitch' | tail -n 200"),
    ("journalctl_kern_xid_sxid_aer_tail", "timeout 5s sudo -n journalctl -k -n 800 --no-pager 2>/dev/null | egrep -i 'xid|sxid|aer|pcie bus error|nvrm|nvswitch' | tail -n 200 || timeout 5s journalctl -k -n 800 --no-pager 2>/dev/null | egrep -i 'xid|sxid|aer|pcie bus error|nvrm|nvswitch' | tail -n 200"),
    ("sensors", "sensors"),
    ("ipmitool_sdr", "timeout 8s sudo -n ipmitool sdr elist 2>/dev/null | head -n 200 || timeout 8s ipmitool sdr elist 2>/dev/null | head -n 200"),
    ("systemctl_kubelet", "systemctl is-active kubelet"),
    ("systemctl_containerd", "systemctl is-active containerd"),
    ("systemctl_slurmd", "systemctl is-active slurmd"),
    ("systemctl_slurmctld", "systemctl is-active slurmctld"),
    ("kubectl_version", "kubectl version --client --short"),
    ("kubectl_nodes", "kubectl get nodes -o wide"),
    ("sinfo", "sinfo"),
    ("scontrol_show_config", "scontrol show config"),
]

results = {
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    },
    "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
    "commands": {},
}

for name, cmd in cmds:
    results["commands"][name] = run(cmd)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY
