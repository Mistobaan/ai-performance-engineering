from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

import ch03.bind_numa_affinity as bind_numa_affinity
import labs.dynamic_router.topology as dynamic_router_topology


def _configure_bind_nvml(monkeypatch: pytest.MonkeyPatch, fake_nvml: object) -> None:
    monkeypatch.setattr(bind_numa_affinity, "_HAS_NVML", True)
    monkeypatch.setattr(bind_numa_affinity, "_NVML_INIT_DONE", True)
    monkeypatch.setattr(bind_numa_affinity, "_ensure_nvml_initialized", lambda: None)
    monkeypatch.setattr(bind_numa_affinity, "_gpu_pci_bus", lambda _: "0000:17:00.0")
    monkeypatch.setattr(bind_numa_affinity, "nvml", fake_nvml)


def test_bind_numa_affinity_prefers_explicit_nvml_numa_node(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByPciBusId_v2=lambda pci: "handle",
        nvmlDeviceGetNumaNodeId=lambda handle: 3,
    )

    _configure_bind_nvml(monkeypatch, fake_nvml)

    assert bind_numa_affinity._gpu_node_from_nvml(0) == 3


def test_bind_numa_affinity_formats_pci_bus_id_when_torch_exposes_integer_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bind_numa_affinity, "_libcudart", None)
    monkeypatch.setattr(
        bind_numa_affinity.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(pci_bus_id=5, pci_domain_id=0, pci_device_id=0),
    )

    assert bind_numa_affinity._gpu_pci_bus(0) == "0000:05:00.0"


def test_bind_numa_affinity_falls_back_when_explicit_nvml_numa_query_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NotSupported(RuntimeError):
        pass

    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByPciBusId_v2=lambda pci: "handle",
        nvmlDeviceGetNumaNodeId=lambda handle: (_ for _ in ()).throw(_NotSupported("not supported")),
        nvmlDeviceGetCpuAffinity=lambda handle, elems: [0b1111],
        NVMLError_NotSupported=_NotSupported,
    )

    _configure_bind_nvml(monkeypatch, fake_nvml)
    monkeypatch.setattr(bind_numa_affinity.psutil, "cpu_count", lambda logical=True: 4)
    monkeypatch.setattr(bind_numa_affinity.glob, "glob", lambda pattern: ["/sys/devices/system/node/node2"])
    monkeypatch.setattr("builtins.open", mock_open(read_data="0-3"))

    assert bind_numa_affinity._gpu_node_from_nvml(0) == 2


def test_dynamic_router_topology_uses_nvml_numa_node_api(monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_calls: list[bool] = []
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda idx: "handle",
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(busId=b"00000000:17:00.0"),
        nvmlDeviceGetNumaNodeId=lambda handle: 4,
        nvmlShutdown=lambda: shutdown_calls.append(True),
    )

    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    assert dynamic_router_topology._nvml_gpu_bus_and_numa(max_gpus=1) == {
        0: {"bus_id": "00000000:17:00.0", "numa_node": 4}
    }
    assert shutdown_calls == [True]
