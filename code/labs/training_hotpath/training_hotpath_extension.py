"""Local CUDA extension loader for the training-hotpath lab."""

from __future__ import annotations

from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension_v2

_EXTENSION_DIR = Path(__file__).parent


def load_training_hotpath_extension():
    """Load segmented reduction and row-pack CUDA kernels for this lab."""

    return load_cuda_extension_v2(
        name="training_hotpath_kernels",
        sources=[_EXTENSION_DIR / "training_hotpath_kernels.cu"],
        extra_cuda_cflags=["-lineinfo"],
    )
