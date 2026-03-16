# Runtime configuration helpers for Chapter 1 examples targeting NVIDIA Blackwell.
# Applies PyTorch knobs that keep us on Tensor Core fast paths and exposes a
# reusable torch.compile wrapper with safe defaults for static workloads.

from __future__ import annotations

import os
from pathlib import Path

from core.utils.compile_utils import enable_tf32
from core.utils.warning_filters import (
    suppress_benchmark_import_warnings,
    warn_optional_component_unavailable,
)

with suppress_benchmark_import_warnings(context="ch01.arch_config torch import"):
    import torch


def _configure_torch_defaults() -> None:
    """Enable TF32 Tensor Core math and cuDNN autotune."""
    with suppress_benchmark_import_warnings(context="ch01.arch_config configure defaults"):
        enable_tf32()
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def _configure_environment() -> None:
    """Set default TorchInductor knobs that play nicely on Blackwell."""
    # Set cache directory and ensure it exists with required subdirectories
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor")
    if not os.path.isabs(cache_dir):
        # Convert relative paths to absolute paths to avoid working directory issues
        cache_dir = str(Path.cwd() / cache_dir)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories needed by PyTorch inductor for C++ compilation
        # 'od' is for output directory, 'tk' is for temporary kernel files
        (cache_path / "od").mkdir(parents=True, exist_ok=True)
        (cache_path / "tk").mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        warn_optional_component_unavailable(
            "TorchInductor cache precreation",
            exc,
            impact="TorchInductor will create cache directories lazily instead.",
            context="ch01.arch_config",
        )

    os.environ.setdefault("TORCHINDUCTOR_FUSE_TRANSPOSE", "1")
    os.environ.setdefault("TORCHINDUCTOR_FUSE_ROTARY", "1")
    os.environ.setdefault("TORCHINDUCTOR_SCHEDULING", "1")


def compile_model(module: torch.nn.Module, *, mode: str = "reduce-overhead",
                  fullgraph: bool = False, dynamic: bool = False) -> torch.nn.Module:
    """
    Compile a model with torch.compile when available.

    Defaults target steady-state inference/training loops with static shapes.
    """
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    with suppress_benchmark_import_warnings(context="ch01.arch_config torch.compile"):
        return compile_fn(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


_configure_environment()
_configure_torch_defaults()
