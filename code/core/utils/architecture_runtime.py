from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from core.harness.arch_config import ArchitectureConfig


@lru_cache(maxsize=1)
def get_arch_config() -> ArchitectureConfig:
    return ArchitectureConfig()


def get_architecture() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return get_arch_config().arch


def get_architecture_info() -> dict[str, Any]:
    arch_cfg = get_arch_config()
    return {
        "name": arch_cfg.get_architecture_name(),
        "compute_capability": arch_cfg.config.get("compute_capability", "Unknown"),
        "sm_version": arch_cfg.config.get("sm_version", "sm_unknown"),
        "memory_bandwidth": arch_cfg.config.get("memory_bandwidth", "Unknown"),
        "tensor_cores": arch_cfg.config.get("tensor_cores", "Unknown"),
        "features": arch_cfg.config.get("features", []),
    }
