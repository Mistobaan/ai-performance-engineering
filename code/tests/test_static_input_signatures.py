from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.scripts.validate_benchmark_pairs import get_input_signature_safe, load_benchmark_class


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark instantiation")
@pytest.mark.parametrize(
    "path_str",
    [
        "ch10/baseline_tcgen05_cluster_pipeline.py",
        "ch10/optimized_tcgen05_cluster_pipeline.py",
        "ch12/baseline_nvfp4_mlp.py",
        "ch12/optimized_nvfp4_mlp.py",
        "ch19/baseline_mxfp8_moe.py",
        "ch19/optimized_mxfp8_moe.py",
        "ch19/baseline_nvfp4_training.py",
        "ch19/optimized_nvfp4_training.py",
    ],
)
def test_static_signature_available_without_running_setup(path_str: str) -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark, error = load_benchmark_class(root / path_str)
    assert benchmark is not None, error

    signature, signature_error = get_input_signature_safe(benchmark)
    assert signature is not None, signature_error
