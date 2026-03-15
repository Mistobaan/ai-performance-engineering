"""Lightweight GPU↔NUMA topology helpers for the dynamic router lab."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _available_numa_nodes() -> List[int]:
    node_root = Path("/sys/devices/system/node")
    nodes = []
    if not node_root.exists():
        return nodes
    for child in node_root.iterdir():
        if child.name.startswith("node") and child.name[4:].isdigit():
            nodes.append(int(child.name[4:]))
    return sorted(nodes)


def _distance_matrix() -> Dict[int, List[int]]:
    matrix: Dict[int, List[int]] = {}
    for node in _available_numa_nodes():
        path = Path(f"/sys/devices/system/node/node{node}/distance")
        if not path.exists():
            continue
        try:
            parts = [int(x) for x in path.read_text().split()]
        except Exception:
            continue
        matrix[node] = parts
    return matrix


def _normalized_bus_id(bus_id: str) -> str:
    # NVML can emit 00000000:17:00.0; sysfs expects 0000:17:00.0
    bus = bus_id.strip().replace("\x00", "").lower()
    if bus.count(":") >= 2:
        parts = bus.split(":")
        return f"{parts[-3]}:{parts[-2]}:{parts[-1]}"
    return bus


def _sysfs_numa_for_bus(bus_id: str) -> Optional[int]:
    bus_norm = _normalized_bus_id(bus_id)
    path = Path(f"/sys/bus/pci/devices/{bus_norm}/numa_node")
    if not path.exists():
        return None
    return _read_int(path)


def _nvml_gpu_bus_and_numa(max_gpus: Optional[int] = None) -> Dict[int, Dict[str, Optional[int]]]:
    mapping: Dict[int, Dict[str, Optional[int]]] = {}
    try:
        import pynvml  # type: ignore
    except Exception:
        return mapping
    try:
        pynvml.nvmlInit()
    except Exception:
        return mapping

    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        count = 0
    limit = count if max_gpus is None else min(count, max_gpus)
    for idx in range(limit):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            bus_id = pci.busId.decode() if hasattr(pci.busId, "decode") else str(pci.busId)
            try:
                numa_id = pynvml.nvmlDeviceGetNumaNodeId(handle)
            except Exception:
                numa_id = None
            mapping[idx] = {"bus_id": bus_id, "numa_node": None if numa_id is None or numa_id < 0 else numa_id}
        except Exception:
            continue

        pynvml.nvmlShutdown()
    return mapping


@dataclass
class TopologySnapshot:
    gpu_numa: Dict[int, Optional[int]]
    distance: Dict[int, List[int]]
    timestamp: float

    def to_json(self) -> Dict[str, object]:
        return {
            "gpu_numa": self.gpu_numa,
            "distance": self.distance,
            "timestamp": self.timestamp,
        }


def detect_topology(max_gpus: Optional[int] = None) -> TopologySnapshot:
    """Best-effort GPU→NUMA map plus NUMA distance matrix."""
    gpu_map: Dict[int, Optional[int]] = {}

    nvml_info = _nvml_gpu_bus_and_numa(max_gpus=max_gpus)
    for idx, info in nvml_info.items():
        numa_guess = info.get("numa_node")
        if numa_guess is None:
            bus = info.get("bus_id")
            if bus:
                numa_guess = _sysfs_numa_for_bus(bus)
        gpu_map[idx] = numa_guess

    # Fallback: attempt sysfs based on CUDA_VISIBLE_DEVICES order
    if not gpu_map:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
        for idx, dev in enumerate(visible):
            dev = dev.strip()
            if dev.isdigit():
                gpu_map[idx] = None

    distance = _distance_matrix()
    return TopologySnapshot(gpu_numa=gpu_map, distance=distance, timestamp=time.time())


def default_topology_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "artifacts" / "topology" / "topology.json"


def write_topology(snapshot: TopologySnapshot, path: Optional[Path] = None) -> Path:
    target = path or default_topology_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(snapshot.to_json(), f, indent=2)
    return target
