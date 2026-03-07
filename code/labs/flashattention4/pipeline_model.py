"""Coarse pipeline model for FlashAttention-4 style overlap.

This is not a cycle-accurate simulator. It is a compact teaching tool that
captures the core idea from the Colfax FlashAttention-4 article: when tensor
core throughput scales faster than scalar/SFU throughput, overlapping the
scalar softmax path with tensor-core work matters more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json


@dataclass(frozen=True)
class PipelineStageProfile:
    """Approximate per-tile stage costs in microseconds."""

    qk_gemm_us: float = 1.00
    pv_gemm_us: float = 0.85
    softmax_us: float = 0.90
    scalar_fixup_us: float = 0.20
    tma_load_us: float = 0.15


@dataclass(frozen=True)
class PipelineEstimate:
    """Serial versus overlapped latency summary."""

    tile_count: int
    tensor_core_path_us: float
    scalar_path_us: float
    serial_tile_us: float
    overlapped_tile_us: float
    serial_total_us: float
    overlapped_total_us: float
    speedup: float


def estimate_pipeline(tile_count: int, profile: PipelineStageProfile) -> PipelineEstimate:
    """Estimate serial and overlapped execution for a tiled attention loop."""
    if tile_count < 1:
        raise ValueError("tile_count must be >= 1")

    tensor_core_path_us = profile.qk_gemm_us + profile.pv_gemm_us
    scalar_path_us = profile.softmax_us + profile.scalar_fixup_us + profile.tma_load_us
    serial_tile_us = tensor_core_path_us + scalar_path_us
    overlapped_tile_us = max(tensor_core_path_us, scalar_path_us)

    serial_total_us = tile_count * serial_tile_us
    overlapped_total_us = serial_tile_us + max(0, tile_count - 1) * overlapped_tile_us
    speedup = serial_total_us / overlapped_total_us if overlapped_total_us else 1.0

    return PipelineEstimate(
        tile_count=tile_count,
        tensor_core_path_us=tensor_core_path_us,
        scalar_path_us=scalar_path_us,
        serial_tile_us=serial_tile_us,
        overlapped_tile_us=overlapped_tile_us,
        serial_total_us=serial_total_us,
        overlapped_total_us=overlapped_total_us,
        speedup=speedup,
    )


def project_scaled_profile(
    profile: PipelineStageProfile,
    *,
    tensor_core_scale: float,
    scalar_scale: float,
) -> PipelineStageProfile:
    """Project the stage costs under asymmetric hardware scaling."""
    if tensor_core_scale <= 0 or scalar_scale <= 0:
        raise ValueError("scale factors must be > 0")

    return PipelineStageProfile(
        qk_gemm_us=profile.qk_gemm_us / tensor_core_scale,
        pv_gemm_us=profile.pv_gemm_us / tensor_core_scale,
        softmax_us=profile.softmax_us / scalar_scale,
        scalar_fixup_us=profile.scalar_fixup_us / scalar_scale,
        tma_load_us=profile.tma_load_us / scalar_scale,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate FlashAttention-4 pipeline overlap gains.")
    parser.add_argument("--tiles", type=int, default=32, help="Number of K/V tiles processed per Q tile.")
    parser.add_argument("--qk-us", type=float, default=1.00, help="QK GEMM time per tile (microseconds).")
    parser.add_argument("--pv-us", type=float, default=0.85, help="PV GEMM time per tile (microseconds).")
    parser.add_argument("--softmax-us", type=float, default=0.90, help="Softmax reduction time per tile (microseconds).")
    parser.add_argument("--scalar-fixup-us", type=float, default=0.20, help="Scalar rescale/fixup time per tile (microseconds).")
    parser.add_argument("--tma-us", type=float, default=0.15, help="TMA / data movement time per tile (microseconds).")
    parser.add_argument("--tensor-core-scale", type=float, default=1.0, help="Tensor-core throughput multiplier relative to the base profile.")
    parser.add_argument("--scalar-scale", type=float, default=1.0, help="Scalar/SFU throughput multiplier relative to the base profile.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text summary.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    profile = PipelineStageProfile(
        qk_gemm_us=args.qk_us,
        pv_gemm_us=args.pv_us,
        softmax_us=args.softmax_us,
        scalar_fixup_us=args.scalar_fixup_us,
        tma_load_us=args.tma_us,
    )
    projected = project_scaled_profile(
        profile,
        tensor_core_scale=args.tensor_core_scale,
        scalar_scale=args.scalar_scale,
    )
    estimate = estimate_pipeline(args.tiles, projected)

    payload = {
        "base_profile": asdict(profile),
        "projected_profile": asdict(projected),
        "estimate": asdict(estimate),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("FlashAttention-4 Pipeline Model")
    print(f"tiles                : {estimate.tile_count}")
    print(f"tensor_core_path_us  : {estimate.tensor_core_path_us:.4f}")
    print(f"scalar_path_us       : {estimate.scalar_path_us:.4f}")
    print(f"serial_total_us      : {estimate.serial_total_us:.4f}")
    print(f"overlapped_total_us  : {estimate.overlapped_total_us:.4f}")
    print(f"speedup              : {estimate.speedup:.4f}x")


if __name__ == "__main__":
    main()
