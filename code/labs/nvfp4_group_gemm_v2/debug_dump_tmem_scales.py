"""Debug helper: dump TMEM scale fragments written by UTCCP and compare vs packed scale tiles.

This is a developer tool (NOT a harness-comparable baseline/optimized benchmark).

It relies on the kernel's debug path:
  - AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_DUMP>=10 dumps raw TMEM scale bytes into C (as FP16-coded bytes).
  - AISP_NVFP4_GROUP_GEMM_V2_DEBUG_STAGE=3 compiles an early-exit after UTCCP (no MMA/epilogue),
    so the dumped TMEM contents correspond to k_tile=0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.nvfp4_group_gemm_v2.custom_cuda_submission import (
    custom_kernel_v2_custom_cuda_tcgen05,
    prepare_v2_custom_cuda_tcgen05,
)
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_common import COMPETITION_CASES
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_inputs import generate_input


def _extract_dump_region(
    c_out_mn: torch.Tensor,
    *,
    tile_m: int,
    tile_n: int,
    rank: int,
    cluster: int,
    enable_cta2: int,
    partition_b: int,
    rows: int = 32,
    cols: int = 64,
) -> torch.Tensor:
    """Return the region where the kernel writes the DP32 x (seg*16) dump for a given rank."""
    # Kernel dumps at gm = m_offset + dp.
    #
    # In the current cta_group::2 bring-up, mode `partition_b=1` partitions N across ranks
    # (both ranks compute the same M rows), so `m_offset` does NOT include rank*64.
    # Other experimental modes may partition M across ranks.
    if int(cluster) == 2 and int(enable_cta2) == 1 and int(partition_b) == 1:
        gm_start = tile_m * 128
    else:
        gm_start = tile_m * 128 + rank * 64
    gn_start = tile_n * 128
    return c_out_mn[gm_start : gm_start + rows, gn_start : gn_start + cols]


def _expected_scale_tile_bytes(
    scale_packed: torch.Tensor, *, tile_mn: int, k_tile: int, sfb_k_major: bool
) -> torch.Tensor:
    """Return expected packed [128,16] scale tile bytes for SFA/SFB at (tile_mn, k_tile)."""
    if scale_packed.dtype != torch.uint8:
        raise ValueError(f"expected packed scales uint8, got {scale_packed.dtype}")
    if scale_packed.ndim != 4:
        raise ValueError(f"expected packed scales 4D, got shape={tuple(scale_packed.shape)}")
    if sfb_k_major:
        # [k_tiles, n_tiles, 128, 16]
        return scale_packed[k_tile, tile_mn]
    # [mn_tiles, k_tiles, 128, 16]
    return scale_packed[tile_mn, k_tile]


def _compare_dump_to_expected(
    dump_fp16: torch.Tensor, expected_u8_tile: torch.Tensor
) -> Tuple[int, int, int]:
    """Return (mismatch_count, total, max_abs_diff)."""
    if dump_fp16.dtype != torch.float16:
        raise ValueError(f"dump must be float16-coded bytes, got {dump_fp16.dtype}")
    if expected_u8_tile.dtype != torch.uint8:
        raise ValueError(f"expected tile must be uint8, got {expected_u8_tile.dtype}")
    # dump is [32,64] with col = seg*16 + byte_idx, row = dp (0..31).
    # expected tile is [128,16] with row = seg*32 + dp, col = byte_idx.
    dump_u8 = dump_fp16.to(torch.int32).clamp(0, 255).to(torch.uint8)
    exp = torch.empty_like(dump_u8)
    for seg in range(4):
        exp[:, seg * 16 : (seg + 1) * 16] = expected_u8_tile[seg * 32 : (seg + 1) * 32, :]
    diff = (dump_u8.to(torch.int16) - exp.to(torch.int16)).abs().to(torch.int32)
    mismatches = int((diff != 0).sum().item())
    total = int(diff.numel())
    max_abs = int(diff.max().item()) if total else 0
    return mismatches, total, max_abs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=1, choices=range(4))
    parser.add_argument("--group", type=int, default=0, help="Group index to dump (0-based within the case).")
    parser.add_argument("--rank", type=int, default=-1, choices=[-1, 0, 1], help="Which cluster rank to dump (-1=both).")
    parser.add_argument("--tile-m", type=int, default=0)
    parser.add_argument("--tile-n", type=int, default=0)
    parser.add_argument("--dump", type=str, default="sfa", choices=["sfa", "sfb_u0", "sfb_u1"])
    parser.add_argument(
        "--debug-stage",
        type=int,
        default=3,
        choices=[3, 4],
        help="Kernel compile-time debug stage (3=UTCCP-only early-exit, 4=UTCCP+MMA then break).",
    )
    parser.add_argument(
        "--clear-dp-base",
        action="store_true",
        help=(
            "Use debug dump modes 20/21/22 instead of 10/11/12 (clears dp bits in the TMEM base pointer "
            "before reading). This helps detect whether UTCCP ignores dp bit7 (dp_add=128)."
        ),
    )
    parser.add_argument("--cluster", type=int, default=2, choices=[1, 2], help="Cluster dim x (1=no cluster, 2=cta2).")
    parser.add_argument("--enable-cta2", type=int, default=1, choices=[0, 1])
    parser.add_argument("--unroll-n", type=int, default=1, choices=[1, 2])
    parser.add_argument("--partition-b", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    def _env_tag(name: str, default: str = "d") -> str:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = raw.strip()
        if value == "":
            return default
        # Keep tags filename-safe for torch-extension build directories.
        return value.replace("/", "_").replace(" ", "")

    case = COMPETITION_CASES[int(args.case)]
    group_idx = int(args.group)
    if group_idx < 0 or group_idx >= int(case.g):
        raise ValueError(f"--group out of range for {case.name}: {group_idx} (g={case.g})")

    # Reduce to a single-group input so the kernel's debug dump (hard-coded to group_idx==0)
    # captures the requested group without needing kernel-side selection knobs.
    m_sel = int(case.m[group_idx])
    n_sel = int(case.n[group_idx])
    k_sel = int(case.k[group_idx])

    # Force a unique extension name so compile-time knobs (DEBUG_STAGE) rebuild cleanly.
    # Include any compile-time UTCCP knobs from the environment so we don't accidentally reuse
    # a previously-built extension with different constexpr settings.
    ext_name = (
        "nvfp4_group_gemm_v2_dbg_dump"
        f"_case{args.case}"
        f"_c{args.cluster}"
        f"_u{args.unroll_n}"
        f"_pb{args.partition_b}"
        f"_s{args.debug_stage}"
        f"_cut{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_USE_CUTLASS_TMEM_SF_FRG', 'd')}"
        f"_ut64{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_USE_UTCCP_64X128B', 'd')}"
        f"_ut64sfa{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_USE_UTCCP_64X128B_SFA', 'd')}"
        f"_ut64sfb{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_USE_UTCCP_64X128B_SFB', 'd')}"
        f"_ut64sch{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_UTCCP_64X128B_SCHEDULE', 'd')}"
        f"_sfdp{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_CTA2_SF_DP_BANK', 'd')}"
        f"_sfbam{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_CTA2_SFB_ALLOC_MODE', 'd')}"
        f"_u1{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_CTA2_SFB_POPULATE_U1', 'd')}"
        f"_skA{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_CTA2_SKIP_UTCCP_SFA', 'd')}"
        f"_skB{_env_tag('AISP_NVFP4_GROUP_GEMM_V2_CTA2_SKIP_UTCCP_SFB', 'd')}"
    )
    os.environ["AISP_NVFP4_GROUP_GEMM_V2_EXT_NAME"] = ext_name

    os.environ["AISP_NVFP4_GROUP_GEMM_V2_DEBUG_STAGE"] = str(int(args.debug_stage))

    # Runtime knobs for the kernel debug dump.
    dump_base = 20 if args.clear_dp_base else 10
    if args.dump == "sfa":
        os.environ["AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_DUMP"] = str(dump_base + 0)
    elif args.dump == "sfb_u0":
        os.environ["AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_DUMP"] = str(dump_base + 1)
    else:
        os.environ["AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_DUMP"] = str(dump_base + 2)

    os.environ["AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X"] = str(int(args.cluster))
    os.environ["AISP_NVFP4_GROUP_GEMM_V2_ENABLE_EXPERIMENTAL_CTA2"] = str(int(args.enable_cta2))
    os.environ["AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N"] = str(int(args.unroll_n))
    os.environ["AISP_NVFP4_GROUP_GEMM_V2_CTA2_PARTITION_B"] = str(int(args.partition_b))
    os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_PRINT_PTRS", "1")

    # One input, one group set.
    data_raw = generate_input(m=(m_sel,), n=(n_sel,), k=(k_sel,), g=1, seed=int(case.seed))
    prepared_seq = prepare_v2_custom_cuda_tcgen05([data_raw])
    if prepared_seq is None:
        raise RuntimeError("prepare_v2_custom_cuda_tcgen05 returned None")
    prepared = prepared_seq[0]

    # Pull expected packed scale tensors for group0 (single-group input).
    ctx = prepared[4]
    sfa_packed = ctx["sfa_packed_tensors"][0]
    sfb_packed = ctx["sfb_packed_tensors"][0]
    sfb_k_major = bool(args.unroll_n == 2)

    # Determine which tile index is being loaded. DEBUG_STAGE=3 breaks after first UTCCP => k_tile=0.
    k_tile = 0

    # For cta_group::2 dumps, avoid dumping both ranks into C in a single run: depending on
    # the active partitioning mode, both ranks may target the same (gm,gn) region and race.
    if int(args.cluster) == 2 and int(args.enable_cta2) == 1:
        ranks = [int(args.rank)] if args.rank >= 0 else [0, 1]
    else:
        ranks = [0]

    for r in ranks:
        if int(args.cluster) == 2 and int(args.enable_cta2) == 1:
            os.environ["AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_ONLY_RANK"] = str(int(r))
        else:
            os.environ.pop("AISP_NVFP4_GROUP_GEMM_V2_DEBUG_TMEM_ONLY_RANK", None)

        # Run once per rank: kernel will overwrite C with the debug dump.
        out = custom_kernel_v2_custom_cuda_tcgen05(prepared)
        torch.cuda.synchronize()

        # Output tensors are [M,N,1] list per group.
        c0 = out[0][:, :, 0]

        dump = _extract_dump_region(
            c0,
            tile_m=int(args.tile_m),
            tile_n=int(args.tile_n),
            rank=int(r),
            cluster=int(args.cluster),
            enable_cta2=int(args.enable_cta2),
            partition_b=int(args.partition_b),
        )
        if dump.numel() == 0:
            print(f"rank={r} dump_shape={tuple(dump.shape)} (no valid rows written; likely m < rank*64)")
            continue
        print(
            f"rank={r} dump_shape={tuple(dump.shape)} dump_min={float(dump.min().item()):.1f} dump_max={float(dump.max().item()):.1f}"
        )

        if args.dump == "sfa":
            exp_tile = _expected_scale_tile_bytes(sfa_packed, tile_mn=int(args.tile_m), k_tile=k_tile, sfb_k_major=False)
        else:
            # For UnrollN=2, the kernel loads two adjacent N tiles (u=0 -> tile_n, u=1 -> tile_n+1).
            # The TMEM dump mode selects which `u` we read back, so expectations must match that tile.
            tile_n_exp = int(args.tile_n) + (1 if args.dump == "sfb_u1" else 0)
            exp_tile = _expected_scale_tile_bytes(sfb_packed, tile_mn=tile_n_exp, k_tile=k_tile, sfb_k_major=sfb_k_major)

        mismatches, total, max_abs = _compare_dump_to_expected(dump, exp_tile)
        print(f"rank={r} compare: mismatches={mismatches}/{total} max_abs_byte_diff={max_abs}")

        # Print a small human-scannable slice: first 8 rows, 32 cols.
        dump_u8 = dump.to(torch.int32).clamp(0, 255).to(torch.uint8).cpu()
        head = dump_u8[:8, :32].numpy()
        print("rank={} dump_u8[0:8,0:32]:".format(r))
        for row in head:
            print(" ".join(f"{int(x):3d}" for x in row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
