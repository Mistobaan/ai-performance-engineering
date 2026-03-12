from core.harness.benchmark_harness import ExecutionMode
from labs.decode_optimization.baseline_decode import get_benchmark as get_baseline_decode
from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig
from labs.decode_optimization.optimized_decode_graph import (
    get_benchmark as get_optimized_decode_graph,
)
from labs.decode_optimization.optimized_decode_ultimate import (
    get_benchmark as get_optimized_decode_ultimate,
)


def test_decode_benchmark_uses_subprocess_execution() -> None:
    bench = DecodeBenchmark(DecodeConfig())

    config = bench.get_config()

    assert config.use_subprocess is True
    assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_decode_variants_inherit_subprocess_execution() -> None:
    for factory in (
        get_baseline_decode,
        get_optimized_decode_graph,
        get_optimized_decode_ultimate,
    ):
        config = factory().get_config()
        assert config.use_subprocess is True
        assert config.execution_mode == ExecutionMode.SUBPROCESS
