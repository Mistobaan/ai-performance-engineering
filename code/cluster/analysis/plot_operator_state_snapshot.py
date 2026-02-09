#!/usr/bin/env python3
"""Build a compact operator-state visualization from structured artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "docs/figures/2026-02-08_operator_state_snapshot.png"
JSON_OUT = ROOT / "results/structured/2026-02-08_operator_state_snapshot.json"


NODE_META = {
    "node1": ROOT / "results/structured/2026-02-08_032814_cloud_eval_full_fixed_node1_meta.json",
    "node2": ROOT / "results/structured/2026-02-08_032814_cloud_eval_full_fixed_node2_meta.json",
}
NUMACTL_EVIDENCE = {
    "node1": ROOT / "results/structured/2026-02-08_numactl_numa_evidence_node1.txt",
    "node2": ROOT / "results/structured/2026-02-08_numactl_numa_evidence_node2.txt",
}
IBSTAT_EVIDENCE = {
    "node1": ROOT / "results/structured/2026-02-08_ibstat_evidence_node1.txt",
    "node2": ROOT / "results/structured/2026-02-08_ibstat_evidence_node2.txt",
}
STORAGE_META = {
    "node1": ROOT / "results/structured/2026-02-08_032814_cloud_eval_full_fixed_node1_storage.json",
    "node2": ROOT / "results/structured/2026-02-08_032814_cloud_eval_full_fixed_node2_storage.json",
}
SHARP_JSON = ROOT / "results/structured/2026-02-08_082000_ib_sharp_check_v3_ib_sharp_check.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def command_stdout(meta_path: Path, key: str) -> str:
    meta = load_json(meta_path)
    commands = meta.get("commands") or {}
    payload = commands.get(key) or {}
    out = payload.get("stdout")
    if isinstance(out, str):
        return out
    return ""


def command_or_fallback(meta_path: Path, key: str, fallback_path: Path) -> str:
    text = command_stdout(meta_path, key)
    if text.strip():
        return text
    if fallback_path.exists():
        return fallback_path.read_text()
    raise FileNotFoundError(
        f"Missing command output and fallback for {key}: meta={meta_path}, fallback={fallback_path}"
    )


def parse_service_states(meta_path: Path) -> dict[str, str]:
    meta = load_json(meta_path)
    commands = meta["commands"]
    keys = {
        "dcgm": "systemctl_nvidia_dcgm",
        "imex": "systemctl_nvidia_imex",
        "fabricmanager": "systemctl_nvidia_fabricmanager",
        "persistenced": "systemctl_nvidia_persistenced",
    }
    out: dict[str, str] = {}
    for short, key in keys.items():
        value = commands.get(key, {}).get("stdout", "").strip().splitlines()
        out[short] = value[0].strip() if value else "unknown"
    return out


def parse_numa_types(numactl_text: str) -> dict[str, int]:
    cpus: dict[int, str] = {}
    sizes: dict[int, int] = {}
    for line in numactl_text.splitlines():
        m_cpu = re.match(r"node\s+(\d+)\s+cpus:\s*(.*)$", line)
        if m_cpu:
            cpus[int(m_cpu.group(1))] = m_cpu.group(2).strip()
            continue
        m_size = re.match(r"node\s+(\d+)\s+size:\s+(\d+)\s+MB$", line)
        if m_size:
            sizes[int(m_size.group(1))] = int(m_size.group(2))

    cpu_nodes = 0
    memory_only_nodes = 0
    memoryless_nodes = 0
    for node_id, size_mb in sizes.items():
        cpu_text = cpus.get(node_id, "")
        has_cpu = bool(cpu_text)
        if has_cpu:
            cpu_nodes += 1
        elif size_mb > 0:
            memory_only_nodes += 1
        else:
            memoryless_nodes += 1

    return {
        "cpu_nodes": cpu_nodes,
        "memory_only_nodes": memory_only_nodes,
        "memoryless_nodes": memoryless_nodes,
    }


def parse_ib_links(text: str) -> dict[str, int]:
    blocks = re.split(r"(?=CA 'mlx5_\d+')", text)
    active_ib = 0
    down_eth = 0
    other = 0
    for block in blocks:
        if "CA 'mlx5_" not in block:
            continue
        state = re.search(r"State:\s*(\w+)", block)
        layer = re.search(r"Link layer:\s*(\w+)", block)
        state_val = state.group(1) if state else "Unknown"
        layer_val = layer.group(1) if layer else "Unknown"
        if layer_val == "InfiniBand" and state_val == "Active":
            active_ib += 1
        elif layer_val == "Ethernet" and state_val == "Down":
            down_eth += 1
        else:
            other += 1
    return {"active_ib": active_ib, "down_eth": down_eth, "other": other}


def parse_storage(storage_path: Path) -> dict[str, int]:
    storage = load_json(storage_path)
    lsblk_stdout = storage["commands"]["lsblk_json"]["stdout"]
    block = json.loads(lsblk_stdout)
    mounted = 0
    unmounted = 0
    total = 0
    for dev in block["blockdevices"]:
        if dev.get("type") != "disk":
            continue
        model = (dev.get("model") or "").upper()
        if "SAMSUNG" not in model:
            continue
        children = dev.get("children") or []
        if children:
            for child in children:
                if child.get("type") != "part":
                    continue
                total += 1
                if child.get("mountpoint"):
                    mounted += 1
                else:
                    unmounted += 1
        else:
            total += 1
            if dev.get("mountpoint"):
                mounted += 1
            else:
                unmounted += 1
    return {
        "samsung_partitions_total": total,
        "samsung_partitions_mounted": mounted,
        "samsung_partitions_unmounted": unmounted,
    }


def parse_sharp(sharp_path: Path) -> dict:
    sharp = load_json(sharp_path)
    per_host = {}
    for host_record in sharp["per_host"]:
        host = host_record["host"]
        per_host[host] = {
            "sharp_dir_present": bool(host_record.get("sharp_dir_present")),
            "sharp_am_active": host_record.get("sharp_am_active", "unknown"),
            "libhcoll_present": bool(host_record.get("libhcoll_present")),
            "libnccl_net_present": bool(host_record.get("libnccl_net_present")),
        }
    collnet_rc = {run["label"]: run["rc"] for run in sharp["nccl_forced_collnet_all_reduce"]["runs"]}
    return {"per_host": per_host, "forced_collnet_rc": collnet_rc}


def build() -> None:
    apply_plot_style()
    nodes = ["node1", "node2"]
    services = {n: parse_service_states(NODE_META[n]) for n in nodes}
    numa = {
        n: parse_numa_types(command_or_fallback(NODE_META[n], "numactl", NUMACTL_EVIDENCE[n]))
        for n in nodes
    }
    ib = {
        n: parse_ib_links(command_or_fallback(NODE_META[n], "ibstat", IBSTAT_EVIDENCE[n]))
        for n in nodes
    }
    storage = {n: parse_storage(STORAGE_META[n]) for n in nodes}
    sharp = parse_sharp(SHARP_JSON)

    output = {
        "sources": {
            "meta": {n: str(p) for n, p in NODE_META.items()},
            "numactl_evidence": {n: f"{NODE_META[n]}::commands.numactl (fallback: {NUMACTL_EVIDENCE[n]})" for n in nodes},
            "ibstat_evidence": {n: f"{NODE_META[n]}::commands.ibstat (fallback: {IBSTAT_EVIDENCE[n]})" for n in nodes},
            "storage": {n: str(p) for n, p in STORAGE_META.items()},
            "sharp": str(SHARP_JSON),
        },
        "services": services,
        "numa": numa,
        "ib_links": ib,
        "storage": storage,
        "sharp": sharp,
    }
    JSON_OUT.write_text(json.dumps(output, indent=2))

    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Operator State Snapshot (node1/node2)", fontsize=16, y=0.995)

    service_cols = ["dcgm", "imex", "fabricmanager", "persistenced"]
    status_map = {"active": 1, "inactive": 0, "failed": -1, "unknown": -2}
    mat = np.array([[status_map.get(services[n][c], -2) for c in service_cols] for n in nodes], dtype=int)
    display = mat + 2
    cmap = ListedColormap(["#8c8c8c", "#d62728", "#ffcc00", "#2ca02c"])
    ax = axs[0, 0]
    ax.imshow(display, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(service_cols)))
    ax.set_xticklabels(service_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)
    for i in range(len(nodes)):
        for j in range(len(service_cols)):
            ax.text(j, i, services[nodes[i]][service_cols[j]], ha="center", va="center", fontsize=9, color="black")
    ax.set_title("NVIDIA Service States")

    ax = axs[0, 1]
    x = np.arange(len(nodes))
    ib_active = [ib[n]["active_ib"] for n in nodes]
    eth_down = [ib[n]["down_eth"] for n in nodes]
    other = [ib[n]["other"] for n in nodes]
    ax.bar(x, ib_active, label="Active IB (400G)", color="#1f77b4")
    ax.bar(x, eth_down, bottom=ib_active, label="Down Ethernet", color="#ff7f0e")
    ax.bar(x, other, bottom=np.array(ib_active) + np.array(eth_down), label="Other", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.set_ylabel("HCAs")
    ax.set_title("ConnectX Link States")
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1, 0]
    categories = ["cpu_nodes", "memory_only_nodes", "memoryless_nodes"]
    width = 0.25
    xx = np.arange(len(categories))
    node1_vals = [numa["node1"][c] for c in categories]
    node2_vals = [numa["node2"][c] for c in categories]
    if sum(node1_vals) == 0 and sum(node2_vals) == 0:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "NUMA node-type counts unavailable\n(check captured numactl output)",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title("NUMA Node Types")
    else:
        ax.bar(xx - width / 2, node1_vals, width, label="node1", color="#1f77b4")
        ax.bar(xx + width / 2, node2_vals, width, label="node2", color="#ff7f0e")
        ax.set_xticks(xx)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("NUMA Node Types")
        ax.legend(fontsize=8)

    ax = axs[1, 1]
    total = [storage[n]["samsung_partitions_total"] for n in nodes]
    unmounted = [storage[n]["samsung_partitions_unmounted"] for n in nodes]
    mounted = [storage[n]["samsung_partitions_mounted"] for n in nodes]
    ax.bar(x, unmounted, label="Unmounted Samsung NVMe parts", color="#d62728")
    ax.bar(x, mounted, bottom=unmounted, label="Mounted Samsung NVMe parts", color="#2ca02c")
    for i, t in enumerate(total):
        ax.text(i, t + 0.1, f"total={t}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.set_ylabel("Partitions")
    ax.set_title("Local Scratch Mount Status")
    ax.legend(fontsize=8)

    ax = axs[2, 0]
    sharp_cols = ["sharp_dir_present", "sharp_am_active", "libhcoll_present", "libnccl_net_present"]
    sharp_mat = []
    for node in nodes:
        rec = sharp["per_host"][node]
        sharp_mat.append(
            [
                1 if rec["sharp_dir_present"] else 0,
                1 if rec["sharp_am_active"] == "active" else 0,
                1 if rec["libhcoll_present"] else 0,
                1 if rec["libnccl_net_present"] else 0,
            ]
        )
    sharp_mat = np.array(sharp_mat)
    ax.imshow(sharp_mat, cmap=ListedColormap(["#d62728", "#2ca02c"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sharp_cols)))
    ax.set_xticklabels(sharp_cols, rotation=20, ha="right")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)
    for i in range(len(nodes)):
        rec = sharp["per_host"][nodes[i]]
        labels = [
            "yes" if rec["sharp_dir_present"] else "no",
            rec["sharp_am_active"],
            "yes" if rec["libhcoll_present"] else "no",
            "yes" if rec["libnccl_net_present"] else "no",
        ]
        for j in range(len(sharp_cols)):
            ax.text(j, i, labels[j], ha="center", va="center", fontsize=8, color="black")
    rc_before = sharp["forced_collnet_rc"].get("before_start", "n/a")
    rc_after = sharp["forced_collnet_rc"].get("after_start", "n/a")
    ax.set_title(f"IB SHARP Readiness (forced CollNet rc: before={rc_before}, after={rc_after})")

    ax = axs[2, 1]
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "Coverage:\n"
        "- NUMA layout (Finding 2)\n"
        "- Service split/monitoring risk (Findings 5, 6, 7)\n"
        "- Ethernet-mode port state (Finding 8)\n"
        "- SHARP stack readiness (Finding 9)\n"
        "- Scratch mount status (Finding 10)\n\n"
        f"Data: {JSON_OUT.relative_to(ROOT)}",
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=160)
    plt.close(fig)
    print(f"Wrote: {FIG_OUT}")
    print(f"Wrote: {JSON_OUT}")


if __name__ == "__main__":
    build()
