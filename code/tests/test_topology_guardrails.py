from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_TOPOLOGY_FILES = [
    Path("ch02/cpu_gpu_topology_aware.py"),
    Path("ch02/optimized_grace_coherent_memory.py"),
    Path("ch03/bind_numa_affinity.py"),
    Path("ch03/numa_topology_script.sh"),
    Path("ch03/cpu_gpu_numa_optimizations.sh"),
    Path("ch03/kubernetes_topology_pod.yaml"),
    Path("ch04/gb200_grace_numa_optimization.py"),
    Path("core/optimization/parallelism_planner/topology_detector.py"),
    Path("labs/dynamic_router/driver.py"),
    Path("labs/dynamic_router/topology.py"),
]

SYSTEM_CONFIG_METRIC_FILES = [
    Path("ch03/baseline_numa_unaware.py"),
    Path("ch03/baseline_kubernetes.py"),
    Path("ch03/baseline_gemm.py"),
    Path("ch03/baseline_docker.py"),
    Path("ch03/optimized_numa_unaware.py"),
    Path("ch03/optimized_docker.py"),
    Path("ch03/optimized_gemm.py"),
    Path("ch03/optimized_kubernetes.py"),
    Path("ch03/optimized_rack_prep.py"),
    Path("core/scripts/update_custom_metrics.py"),
]

TOPOLOGY_REPORT_FILES = [
    Path("core/analysis/advanced_analysis.py"),
    Path("labs/dynamic_router/topology_probe.py"),
]

BANNED_LITERALS = {
    "nvmlDeviceGetNUMANodeId": "Wrong NVML API symbol casing.",
    "topo -m -i": "nvidia-smi topo -m does not support -i.",
}

BANNED_HEURISTIC_SNIPPETS = {
    "numa_node = gpu_id": "Do not infer GPU locality from matching indices.",
    "gpu_id % len(topology)": "Do not infer NUMA locality from topology length modulo.",
    "i // gpus_per_numa": "Do not synthesize GPU->NUMA mappings from GPU counts.",
    "numa_nodes\"] = 1  # numactl not installed": "Do not turn unavailable CPU NUMA topology into a fake single-node result.",
    "numa_node: int = 0": "Do not default runtime GPU locality to NUMA node 0.",
}

BANNED_SYSTEM_CONFIG_SNIPPETS = {
    "getattr(self, 'numa_nodes', 1)": "Do not default benchmark NUMA metadata to a fake single-node topology.",
}

BANNED_TOPOLOGY_REPORT_SNIPPETS = {
    "NUMA: Not available or single-node system": "Keep unknown NUMA availability distinct from single-node topology.",
    "numa_nodes_known": "Report explicit NUMA status instead of an inferred known-node count.",
}


def _read(relpath: Path) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_runtime_topology_files_do_not_reintroduce_banned_literals() -> None:
    for relpath in RUNTIME_TOPOLOGY_FILES:
        text = _read(relpath)
        for literal, reason in BANNED_LITERALS.items():
            assert literal not in text, f"{relpath}: {reason}"


def test_runtime_topology_files_do_not_guess_gpu_numa_locality() -> None:
    for relpath in RUNTIME_TOPOLOGY_FILES:
        text = _read(relpath)
        for snippet, reason in BANNED_HEURISTIC_SNIPPETS.items():
            assert snippet not in text, f"{relpath}: {reason}"


def test_system_config_metric_files_do_not_default_unknown_numa_to_one() -> None:
    for relpath in SYSTEM_CONFIG_METRIC_FILES:
        text = _read(relpath)
        for snippet, reason in BANNED_SYSTEM_CONFIG_SNIPPETS.items():
            assert snippet not in text, f"{relpath}: {reason}"


def test_topology_reporting_files_keep_unknown_state_explicit() -> None:
    for relpath in TOPOLOGY_REPORT_FILES:
        text = _read(relpath)
        for snippet, reason in BANNED_TOPOLOGY_REPORT_SNIPPETS.items():
            assert snippet not in text, f"{relpath}: {reason}"


def test_ch03_topology_shell_helpers_are_shell_syntax_clean() -> None:
    for relpath in (
        Path("ch03/numa_topology_script.sh"),
        Path("ch03/cpu_gpu_numa_optimizations.sh"),
    ):
        subprocess.run(
            ["bash", "-n", str(REPO_ROOT / relpath)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )


def test_kubernetes_topology_manifest_uses_pci_bus_query_not_topo_index_hack() -> None:
    text = _read(Path("ch03/kubernetes_topology_pod.yaml"))
    assert "--query-gpu=pci.bus_id" in text
    assert "topo -m -i" not in text
