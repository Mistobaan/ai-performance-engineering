"""Shared serving stack pin resolution for benchmark hosts.

Single source of truth:
- Prefer exact pins from requirements_latest.txt.
- Fall back to stable defaults if requirements file is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REQUIREMENTS = _REPO_ROOT / "requirements_latest.txt"

_DEFAULT_TORCH = "2.10.0.dev20251213+cu130"
_DEFAULT_VLLM = "0.15.0+cu130"
_DEFAULT_FLASHINFER = "0.6.2"


@dataclass(frozen=True)
class ServingStackPins:
    torch_version: str
    vllm_version: str
    flashinfer_version: str

    @property
    def vllm_pip_spec(self) -> str:
        return f"vllm=={self.vllm_version}"

    @property
    def pinned_stack_str(self) -> str:
        return (
            f"torch=={self.torch_version}, "
            f"vllm=={self.vllm_version}, "
            f"flashinfer-python=={self.flashinfer_version}"
        )


def _read_pinned_version(package_name: str, requirements_path: Path) -> Optional[str]:
    if not requirements_path.exists():
        return None
    pattern = re.compile(rf"^\s*{re.escape(package_name)}==([^\s#]+)\s*(?:#.*)?$")
    for raw_line in requirements_path.read_text().splitlines():
        match = pattern.match(raw_line)
        if match:
            return match.group(1).strip()
    return None


def get_serving_stack_pins(requirements_path: Optional[Path] = None) -> ServingStackPins:
    req_path = (requirements_path or _DEFAULT_REQUIREMENTS).resolve()
    torch_version = _read_pinned_version("torch", req_path) or _DEFAULT_TORCH
    vllm_version = _read_pinned_version("vllm", req_path) or _DEFAULT_VLLM
    flashinfer_version = (
        _read_pinned_version("flashinfer-python", req_path) or _DEFAULT_FLASHINFER
    )
    return ServingStackPins(
        torch_version=torch_version,
        vllm_version=vllm_version,
        flashinfer_version=flashinfer_version,
    )

