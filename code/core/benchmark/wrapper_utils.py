"""Utilities for thin benchmark wrapper modules."""

from __future__ import annotations

from typing import Any


def attach_benchmark_metadata(bench: Any, module_file: str) -> Any:
    """Annotate a benchmark so subprocess runners re-import the wrapper module."""
    bench._module_file_override = module_file
    bench._factory_name_override = "get_benchmark"
    return bench
