from __future__ import annotations

from labs.flashattention4.pipeline_model import (
    PipelineStageProfile,
    estimate_pipeline,
    project_scaled_profile,
)


def test_pipeline_overlap_reduces_total_latency() -> None:
    profile = PipelineStageProfile(
        qk_gemm_us=1.0,
        pv_gemm_us=0.8,
        softmax_us=0.9,
        scalar_fixup_us=0.2,
        tma_load_us=0.1,
    )
    estimate = estimate_pipeline(32, profile)
    assert estimate.overlapped_total_us < estimate.serial_total_us
    assert estimate.speedup > 1.0


def test_asymmetric_scaling_increases_overlap_value() -> None:
    base = PipelineStageProfile(
        qk_gemm_us=1.0,
        pv_gemm_us=0.8,
        softmax_us=0.9,
        scalar_fixup_us=0.2,
        tma_load_us=0.1,
    )
    balanced = estimate_pipeline(
        48,
        project_scaled_profile(base, tensor_core_scale=2.0, scalar_scale=2.0),
    )
    asymmetric = estimate_pipeline(
        48,
        project_scaled_profile(base, tensor_core_scale=4.0, scalar_scale=2.0),
    )
    assert asymmetric.speedup > balanced.speedup
