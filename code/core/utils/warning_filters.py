"""Scoped warning suppression helpers for known noisy runtime imports."""

from __future__ import annotations

from contextlib import contextmanager
import warnings
from typing import Iterator, Sequence

_CUDA_CAPABILITY_WARNING_PATTERNS: tuple[str, ...] = (
    ".*Found GPU.*cuda capability.*",
    ".*Found GPU.*which is of cuda capability.*",
    ".*Minimum and Maximum cuda capability supported.*",
)

_BENCHMARK_IMPORT_WARNING_PATTERNS: tuple[str, ...] = (
    ".*Please use the new API settings to control TF32.*",
    ".*TensorFloat32 tensor cores.*available but not enabled.*",
    ".*Overriding a previously registered kernel.*",
    ".*Warning only once for all operators.*",
)


@contextmanager
def suppress_user_warnings(patterns: Sequence[str]) -> Iterator[None]:
    """Suppress a narrow set of known-noisy user warnings for one operation."""
    with warnings.catch_warnings():
        for pattern in patterns:
            warnings.filterwarnings("ignore", message=pattern, category=UserWarning)
        yield


@contextmanager
def suppress_known_cuda_capability_warnings() -> Iterator[None]:
    """Suppress PyTorch capability warnings only around targeted imports/probes."""
    with suppress_user_warnings(_CUDA_CAPABILITY_WARNING_PATTERNS):
        yield


@contextmanager
def suppress_benchmark_import_warnings() -> Iterator[None]:
    """Suppress known import-time benchmark noise without mutating global filters."""
    with suppress_user_warnings(
        (*_CUDA_CAPABILITY_WARNING_PATTERNS, *_BENCHMARK_IMPORT_WARNING_PATTERNS)
    ):
        yield
