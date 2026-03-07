"""Chapter 16: MoE Routing - Compare baseline vs optimized implementations using formal harness.

Uses the BaseBenchmark - benchmarks provide get_benchmark() function,
harness measures directly (no subprocess, no output parsing).
"""

from pathlib import Path
from typing import Dict, Any

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from core.utils.chapter_compare_template import (
    profile_template,
)


def profile() -> Dict[str, Any]:
    """Compare all baseline/optimized pairs using formal harness."""
    chapter_dir = Path(__file__).parent
    
    return profile_template(
        chapter='ch16',
        chapter_dir=chapter_dir,
        harness_config=BenchmarkConfig(iterations=20, warmup=5),
    )


if __name__ == '__main__':
    result = profile()
    print("\nMetrics:", result)
