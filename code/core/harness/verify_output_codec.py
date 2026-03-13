from __future__ import annotations

from typing import Any, Dict

import torch


def _parse_torch_dtype(dtype_name: str | None) -> torch.dtype:
    if not dtype_name:
        return torch.float32
    normalized = str(dtype_name).strip()
    if normalized.startswith("torch."):
        normalized = normalized[len("torch.") :]
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype '{dtype_name}' in verify_output payload")
    return dtype


def serialize_verify_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    cpu_tensor = tensor.detach().cpu()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "data": cpu_tensor.tolist(),
    }


def deserialize_verify_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    data = payload.get("data")
    shape = payload.get("shape")
    dtype = _parse_torch_dtype(payload.get("dtype"))
    if data is None:
        raise ValueError("verify_output missing 'data'")
    tensor = torch.tensor(data, dtype=dtype)
    if shape is not None:
        tensor = tensor.reshape(tuple(int(dim) for dim in shape))
    return tensor
