"""Optimized NVFP4 grouped GEMM (competition case 1, custom CUDA path).

This is the integration point for the from-scratch Blackwell kernel work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tuning knobs for the custom CUDA extension (kept explicit so harness defaults stay unchanged).
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_BLOCK_M", "8")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_BLOCK_N", "32")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_KPACK_TILE", "64")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_UNROLL_N", "2")

# Compile-time kernel knobs (require rebuild under a unique AISP_NVFP4_GROUP_GEMM_EXT_NAME).
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_WS_UNROLL2_MMA", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_WS_TMA_PRODUCER", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_EPILOGUE_LD_X32", "1")

# Runtime tuning knobs.
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS_COMPRESS_LIST", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CTA_ORDER", "tm_major")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_TMA_L2_PROMOTION", "3")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_ASSUME_NO_N_TAIL", "1")

# Keep the build namespace stable for this optimized configuration so it doesn't collide
# with the conservative baseline extension build.
os.environ.setdefault(
    "AISP_NVFP4_GROUP_GEMM_EXT_NAME",
    "nvfp4_group_gemm_tcgen05_opt_u2_tp1_epi1_tm",
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
    case = COMPETITION_CASES[1]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_custom_cuda,
        prepare=prepare_custom_cuda,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=f"nvfp4_group_gemm_{case.name}_optimized_custom_cuda",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
