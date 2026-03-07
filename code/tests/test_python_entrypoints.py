from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from core.utils.python_entrypoints import (
    build_python_entry_command,
    build_repo_python_env,
    build_torchrun_entry_command,
)


def test_build_repo_python_env_prepends_repo_root_and_dedupes() -> None:
    repo_root = Path("/tmp/repo")
    env = build_repo_python_env(
        repo_root,
        base_env={"PYTHONPATH": f"/tmp/repo{os.pathsep}/existing"},
        extra_pythonpath=["/extra", "/existing"],
    )

    assert env["PYTHONPATH"].split(os.pathsep) == ["/tmp/repo", "/extra", "/existing"]


def test_build_python_entry_command_supports_module_launch() -> None:
    cmd = build_python_entry_command(module_name="ch11.stream_overlap_demo", argv=["--help"])
    assert cmd == [sys.executable, "-m", "ch11.stream_overlap_demo", "--help"]


def test_build_python_entry_command_supports_script_launch(tmp_path: Path) -> None:
    script_path = tmp_path / "tool.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    cmd = build_python_entry_command(script_path=script_path, argv=["--flag"])
    assert cmd == [sys.executable, str(script_path.resolve()), "--flag"]


def test_build_python_entry_command_requires_exactly_one_target(tmp_path: Path) -> None:
    script_path = tmp_path / "tool.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_python_entry_command()
    with pytest.raises(ValueError):
        build_python_entry_command(module_name="json.tool", script_path=script_path)


def test_build_torchrun_entry_command_supports_module_launch() -> None:
    cmd = build_torchrun_entry_command(
        "torchrun",
        module_name="ch15.tensor_parallel_demo",
        argv=["--batch", "1"],
        nproc_per_node=2,
        nnodes=1,
    )
    assert cmd == [
        "torchrun",
        "--nproc_per_node",
        "2",
        "--nnodes",
        "1",
        "-m",
        "ch15.tensor_parallel_demo",
        "--batch",
        "1",
    ]


def test_ch11_ch12_ch15_ch16_ch17_and_ch20_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for chapter in ("ch11", "ch12", "ch15", "ch16", "ch17", "ch20"):
        for path in sorted((repo_root / chapter).glob("*.py")):
            if "sys.path.insert" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_selected_public_entrypoints_no_longer_carry_local_arch_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "ch05/gpudirect_storage_example.py",
        "ch09/fusion_pytorch.py",
        "ch16/compare.py",
        "ch16/gpt_large_benchmark.py",
        "ch16/inference_profiling.py",
        "ch16/inference_server_load_test.py",
        "ch16/inference_serving_multigpu.py",
        "ch16/inference_optimizations_blackwell.py",
        "ch16/multi_gpu_validation.py",
        "ch16/perplexity_eval.py",
        "ch16/radix_attention_example.py",
        "ch16/synthetic_moe_inference_benchmark.py",
        "ch16/test_fp8_quantization_real.py",
        "ch16/vllm_monitoring.py",
        "ch17/blackwell_profiling_guide.py",
        "ch17/compare.py",
        "ch17/dynamic_routing.py",
        "ch17/early_rejection.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
        assert "from arch_config import ArchitectureConfig" not in text
