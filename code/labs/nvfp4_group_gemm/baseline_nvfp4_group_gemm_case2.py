"""Baseline NVFP4 grouped GEMM (competition case 2, custom CUDA tcgen05 kernel).

This baseline keeps the conservative kernel routing (no cluster/cta_group::2, UnrollN=1),
but runs with the fast runtime path (fused inputs + iter-graph replay) by default.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep baseline behavior stable by default, but allow explicit overrides for experiments.
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_UNROLL_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ENABLE_EXPERIMENTAL_CTA2", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS_COMPRESS_LIST", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH", "1")
os.environ.setdefault(
    "AISP_NVFP4_GROUP_GEMM_EXT_NAME",
    "nvfp4_group_gemm_tcgen05_baseline",
)

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.custom_cuda_submission import (
    custom_kernel_custom_cuda,
    prepare_custom_cuda,
)
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[2]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_custom_cuda,
        prepare=prepare_custom_cuda,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=f"nvfp4_group_gemm_{case.name}_baseline_custom_cuda",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
