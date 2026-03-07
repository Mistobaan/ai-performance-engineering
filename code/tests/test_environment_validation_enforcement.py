"""Harness-level enforcement tests for validate_environment().

These tests use a synthetic /proc + /sys snapshot via EnvironmentProbe and verify
that BenchmarkHarness FAILS for chapter/lab benchmarks when the environment is
invalid (i.e., validate_environment() returns errors).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import pytest
import torch

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from core.harness.validity_checks import EnvironmentProbe, validate_environment


def _write_file(root: Path, relpath: str, content: str) -> None:
    path = root / relpath.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_base_env(root: Path, *, governor: str = "performance") -> None:
    _write_file(root, "/proc/swaps", "Filename\tType\tSize\tUsed\tPriority\n")
    _write_file(root, "/proc/sys/vm/swappiness", "0\n")
    _write_file(root, "/proc/cpuinfo", "processor\t: 0\n")
    _write_file(root, "/sys/devices/virtual/dmi/id/product_name", "BareMetal\n")
    _write_file(root, "/sys/devices/system/node/node0/cpulist", "0-3\n")
    _write_file(root, "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", governor + "\n")


def _load_ch_fake_benchmark(module_dir: Path) -> object:
    """Load a benchmark class from a file path containing '/ch' so the harness enforces."""
    module_path = module_dir / "ch_fake_env_bench.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import torch
            from core.harness.benchmark_harness import BaseBenchmark

            class EnvBench(BaseBenchmark):
                allow_cpu = True

                def __init__(self):
                    super().__init__()
                    self.x = None

                def setup(self) -> None:
                    self.x = torch.ones(1, device=self.device)

                def benchmark_fn(self) -> None:
                    self.x.add_(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ch_fake_env_bench", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EnvBench()


def _run_harness(env_root: Path, *, probe: EnvironmentProbe, allow_virtualization: Optional[bool] = None) -> list[str]:
    bench_dir = env_root / "bench_mod"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench = _load_ch_fake_benchmark(bench_dir)
    harness = BenchmarkHarness(environment_probe=probe)
    config_kwargs = dict(iterations=1, warmup=5, use_subprocess=False)
    if allow_virtualization is not None:
        config_kwargs["allow_virtualization"] = allow_virtualization
    config = BenchmarkConfig(**config_kwargs)
    result = harness._benchmark_with_threading(bench, config)
    return list(result.errors)


def test_environment_enforcement_cpu_governor_mismatch() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root, governor="powersave")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "CPU governor mismatch" in e for e in errors), errors


def test_environment_enforcement_cgroup_cpu_quota() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "100000 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "max\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "cpu.max" in e for e in errors), errors


def test_environment_enforcement_cgroup_memory_limit() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "max 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "1073741824\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "memory.max" in e for e in errors), errors


def test_environment_enforcement_numa_affinity_spans_nodes() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/sys/devices/system/node/node0/cpulist", "0-1\n")
        _write_file(env_root, "/sys/devices/system/node/node1/cpulist", "2-3\n")
        probe = EnvironmentProbe(root=env_root, env={}, cpu_affinity={0, 2})
        result = validate_environment(probe=probe)
        assert result.is_valid, result
        assert any("CPU affinity spans multiple NUMA nodes" in w for w in result.warnings), result.warnings


def test_environment_enforcement_swap_enabled() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(
            env_root,
            "/proc/swaps",
            "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t1024\t0\t-2\n",
        )
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "Swap is enabled" in e for e in errors), errors


def test_environment_virtualization_strict_is_warning_only() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}), allow_virtualization=False)
        assert not any("ENVIRONMENT INVALID" in e and "Virtualization detected" in e for e in errors), errors


def test_environment_warning_virtualization_portable_message() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        result = validate_environment(
            probe=EnvironmentProbe(root=env_root, env={}),
            allow_virtualization=True,
        )
        assert result.is_valid, result
        assert not result.errors, result.errors
        assert any("validity_profile=portable" in warning for warning in result.warnings), result.warnings


def test_environment_warning_virtualization_strict_loud_message() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        result = validate_environment(
            probe=EnvironmentProbe(root=env_root, env={}),
            allow_virtualization=False,
        )
        assert result.is_valid, result
        assert not result.errors, result.errors
        assert any("STRICT VALIDITY NOTICE [VIRTUALIZATION]" in notice for notice in result.notices), result.notices


def test_environment_enforcement_foreign_gpu_processes_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 4321, "process_name": "vllm", "used_memory_mb": 32768.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors


def test_environment_enforcement_foreign_gpu_processes_allow_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 9876, "process_name": "vllm", "used_memory_mb": 12288.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_ALLOW_FOREIGN_GPU_PROCESSES": "1",
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert any("Foreign CUDA compute process(es) detected on benchmark GPU" in warning for warning in result.warnings), result.warnings


def test_environment_enforcement_foreign_gpu_processes_ignore_owned_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid()), 1111},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 1111, "process_name": "VLLM::EngineCore", "used_memory_mb": 6114.0}],
            None,
        ),
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_owned_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid()), 2222},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 2222, "process_name": "python", "used_memory_mb": 612.0}],
            None,
        ),
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_dead_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 3333, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): False,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details
