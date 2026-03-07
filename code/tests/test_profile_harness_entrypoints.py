from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO_ROOT / "core" / "scripts" / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from example_registry import EXAMPLE_BY_NAME, _example  # type: ignore  # noqa: E402
import profile_harness  # type: ignore  # noqa: E402


def test_example_run_command_prefers_module_name_for_python_examples() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
    )

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch15.tensor_parallel_demo"]


def test_base_env_includes_repo_root_on_pythonpath() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
        tags=["ch15"],
    )

    env = profile_harness.base_env(example)

    assert str(REPO_ROOT) in env["PYTHONPATH"].split(os.pathsep)


def test_ch20_example_registry_uses_module_launch_for_ai_kernel_generator() -> None:
    example = EXAMPLE_BY_NAME["ch20_ai_kernel_generator"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch20.ai_kernel_generator"]
