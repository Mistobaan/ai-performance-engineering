"""CUDA extension loader for CUTLASS SM100 NVFP4 block-scaled grouped GEMM."""

from __future__ import annotations

import os
import sys
import builtins
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.utils.extension_loader_template import load_cuda_extension

_BASE_EXT_NAME = "nvfp4_group_gemm_cutlass_sm100"
_DEFAULT_S5_RESERVE_BYTES = 8 * 1024
_DEFAULT_CASE23_S5_RESERVE_BYTES = 16 * 1024
_DEFAULT_CASE2_RESERVE_BYTES = 12 * 1024
_DEFAULT_CASE2_S5_RESERVE_BYTES = 20 * 1024
_DEFAULT_CASE2_NVF4_RESERVE_BYTES = 8 * 1024
_DEFAULT_CASE3_RESERVE_BYTES = 6 * 1024
_DEFAULT_N192_CASE2_RESERVE_BYTES = 10 * 1024
_DEFAULT_N192_CASE3_RESERVE_BYTES = 6 * 1024
_EXT_BY_NAME: dict[str, object] = {}


def _get_process_extension_cache() -> dict[str, object]:
    """Return a process-global cache shared across import paths."""
    cache_name = "_AISP_EXT_PROCESS_CACHE"
    cache = getattr(builtins, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(builtins, cache_name, cache)
    return cache


def _read_nonnegative_int_env(name: str, default_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default_value)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _sanitize_extension_suffix(raw: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw.strip())
    token = token.strip("_")
    return token


def _resolve_extension_name(
    s5_reserve_bytes: int,
    case23_s5_reserve_bytes: int,
    case2_reserve_bytes: int,
    case2_s5_reserve_bytes: int,
    case2_nvf4_reserve_bytes: int,
    case3_reserve_bytes: int,
    n192_case2_reserve_bytes: int,
    n192_case3_reserve_bytes: int,
) -> str:
    parts = [_BASE_EXT_NAME]
    if int(s5_reserve_bytes) != int(_DEFAULT_S5_RESERVE_BYTES):
        parts.append(f"s5r{int(s5_reserve_bytes)}")
    if int(case23_s5_reserve_bytes) != int(_DEFAULT_CASE23_S5_RESERVE_BYTES):
        parts.append(f"c23s5r{int(case23_s5_reserve_bytes)}")
    if int(case2_reserve_bytes) != int(_DEFAULT_CASE2_RESERVE_BYTES):
        parts.append(f"c2r{int(case2_reserve_bytes)}")
    if int(case2_s5_reserve_bytes) != int(_DEFAULT_CASE2_S5_RESERVE_BYTES):
        parts.append(f"c2s5r{int(case2_s5_reserve_bytes)}")
    if int(case2_nvf4_reserve_bytes) != int(_DEFAULT_CASE2_NVF4_RESERVE_BYTES):
        parts.append(f"c2nvf4r{int(case2_nvf4_reserve_bytes)}")
    if int(case3_reserve_bytes) != int(_DEFAULT_CASE3_RESERVE_BYTES):
        parts.append(f"c3r{int(case3_reserve_bytes)}")
    if int(n192_case2_reserve_bytes) != int(_DEFAULT_N192_CASE2_RESERVE_BYTES):
        parts.append(f"n192c2r{int(n192_case2_reserve_bytes)}")
    if int(n192_case3_reserve_bytes) != int(_DEFAULT_N192_CASE3_RESERVE_BYTES):
        parts.append(f"n192c3r{int(n192_case3_reserve_bytes)}")
    user_suffix = _sanitize_extension_suffix(os.environ.get("AISP_NVFP4_GROUP_GEMM_CUTLASS_EXT_SUFFIX", ""))
    if user_suffix:
        parts.append(user_suffix)
    return "_".join(parts)


def load_cutlass_nvfp4_grouped_gemm_sm100(*, verbose: bool = False) -> object:
    """Load (and JIT-build if needed) the CUTLASS NVFP4 grouped GEMM extension."""
    s5_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_S5_RESERVE_BYTES",
        _DEFAULT_S5_RESERVE_BYTES,
    )
    case23_s5_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE23_S5_RESERVE_BYTES",
        _DEFAULT_CASE23_S5_RESERVE_BYTES,
    )
    case2_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_RESERVE_BYTES",
        _DEFAULT_CASE2_RESERVE_BYTES,
    )
    case2_s5_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_S5_RESERVE_BYTES",
        _DEFAULT_CASE2_S5_RESERVE_BYTES,
    )
    case2_nvf4_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_NVF4_RESERVE_BYTES",
        _DEFAULT_CASE2_NVF4_RESERVE_BYTES,
    )
    case3_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE3_RESERVE_BYTES",
        _DEFAULT_CASE3_RESERVE_BYTES,
    )
    n192_case2_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N192_CASE2_RESERVE_BYTES",
        _DEFAULT_N192_CASE2_RESERVE_BYTES,
    )
    n192_case3_reserve_bytes = _read_nonnegative_int_env(
        "AISP_NVFP4_GROUP_GEMM_1SM_N192_CASE3_RESERVE_BYTES",
        _DEFAULT_N192_CASE3_RESERVE_BYTES,
    )
    ext_name = _resolve_extension_name(
        s5_reserve_bytes,
        case23_s5_reserve_bytes,
        case2_reserve_bytes,
        case2_s5_reserve_bytes,
        case2_nvf4_reserve_bytes,
        case3_reserve_bytes,
        n192_case2_reserve_bytes,
        n192_case3_reserve_bytes,
    )
    cached_local = _EXT_BY_NAME.get(ext_name)
    if cached_local is not None:
        return cached_local

    process_cache = _get_process_extension_cache()
    cached = process_cache.get(ext_name)
    if cached is not None:
        _EXT_BY_NAME[ext_name] = cached
        return cached
    if ext_name in sys.modules:
        loaded = sys.modules[ext_name]
        process_cache[ext_name] = loaded
        _EXT_BY_NAME[ext_name] = loaded
        return loaded

    lab_dir = Path(__file__).resolve().parent
    source = lab_dir / "cutlass_nvfp4_grouped_gemm_sm100.cu"

    cutlass_util_inc = REPO_ROOT / "third_party" / "cutlass" / "tools" / "util" / "include"
    if not cutlass_util_inc.exists():
        raise FileNotFoundError(f"Missing CUTLASS util include dir: {cutlass_util_inc}")

    extra_cuda_cflags = [
        "--std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        # B200 (SM100). We keep this narrow to reduce compile time.
        "-gencode=arch=compute_100a,code=sm_100a",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_S5_RESERVE_BYTES={int(s5_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_CASE23_S5_RESERVE_BYTES={int(case23_s5_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_RESERVE_BYTES={int(case2_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_S5_RESERVE_BYTES={int(case2_s5_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_NVF4_RESERVE_BYTES={int(case2_nvf4_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N128_CASE3_RESERVE_BYTES={int(case3_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N192_CASE2_RESERVE_BYTES={int(n192_case2_reserve_bytes)}",
        f"-DAISP_NVFP4_GROUP_GEMM_1SM_N192_CASE3_RESERVE_BYTES={int(n192_case3_reserve_bytes)}",
    ]

    ext = load_cuda_extension(
        extension_name=ext_name,
        cuda_source_file=str(source),
        include_dirs=[cutlass_util_inc],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )
    # Keep cache coherent even when module is imported through different paths.
    process_cache[ext_name] = ext
    _EXT_BY_NAME[ext_name] = ext
    sys.modules[ext_name] = ext
    return ext


__all__ = ["load_cutlass_nvfp4_grouped_gemm_sm100"]
