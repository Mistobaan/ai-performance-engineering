"""Exploratory helpers for tcgen05 CUTLASS cluster-shape comparisons."""

from __future__ import annotations

from pathlib import Path


def cutlass_sm100_supports_4sm() -> bool:
    dispatch_policy = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "cutlass"
        / "include"
        / "cutlass"
        / "gemm"
        / "dispatch_policy.hpp"
    )
    if not dispatch_policy.exists():
        return False
    return "KernelTmaWarpSpecialized4SmSm100" in dispatch_policy.read_text(encoding="utf-8", errors="ignore")
