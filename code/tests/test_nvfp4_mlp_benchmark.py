from __future__ import annotations

from core.benchmark.nvfp4_mlp import NVFP4MLPBenchmark


def test_nvfp4_mlp_reports_memory_goal() -> None:
    bench = NVFP4MLPBenchmark(use_nvfp4=True)
    assert bench.get_optimization_goal() == "memory"
