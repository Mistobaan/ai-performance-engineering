"""Tests for run_benchmarks config merge invariants (no mocks)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.defaults import BenchmarkDefaults
from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia
from core.harness.run_benchmarks import (
    _compute_locked_fields,
    _merge_benchmark_config,
    _resolve_expectation_validation_policy,
)


@dataclass
class _DummyBenchmark:
    override: BenchmarkConfig

    def get_config(self) -> BenchmarkConfig:
        return self.override


def _base_config(**overrides) -> BenchmarkConfig:
    # Keep defaults explicit for fields that interact with locking/merging.
    kwargs = dict(
        timeout_multiplier=1.0,
        enforce_environment_validation=True,
        enable_memory_tracking=True,
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
        profile_type="none",
    )
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def test_merge_respects_cli_iterations_and_warmup_locks() -> None:
    base = _base_config(iterations=20, warmup=5)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=True,
        cli_warmup_provided=True,
        enable_profiling=False,
    )
    override = BenchmarkConfig(iterations=999, warmup=99, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.iterations == base.iterations
    assert merged.warmup == base.warmup


def test_merge_locks_runner_protections_when_enabled() -> None:
    base = _base_config()
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(enable_memory_tracking=False, detect_setup_precomputation=False, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.enable_memory_tracking is True
    assert merged.detect_setup_precomputation is True


def test_merge_locks_profiling_knobs_when_cli_enabled() -> None:
    base = _base_config(enable_profiling=True, enable_nsys=True, enable_ncu=True, profile_type="minimal")
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=True,
    )
    override = BenchmarkConfig(
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
        enable_nvtx=False,
        profile_type="none",
        timeout_multiplier=1.0,
    )

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.enable_profiling is True
    assert merged.enable_nsys is True
    assert merged.enable_ncu is True
    assert merged.enable_nvtx is True
    assert merged.profile_type == "minimal"


def test_merge_preserves_cli_launch_via_when_benchmark_supplies_default() -> None:
    defaults_obj = BenchmarkDefaults(launch_via="python")
    base = _base_config(launch_via="torchrun")
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(launch_via="python", timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=defaults_obj,
        locked_fields=locked,
    )

    assert merged.launch_via == LaunchVia.TORCHRUN


def test_merge_target_extra_args_merges_dicts() -> None:
    base = _base_config(target_extra_args={"a": ["--foo"], "b": ["--bar"]})
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(target_extra_args={"b": ["--override"], "c": ["--baz"]}, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.target_extra_args == {"a": ["--foo"], "b": ["--override"], "c": ["--baz"]}


def test_merge_env_passthrough_ignores_empty_override() -> None:
    base = _base_config(env_passthrough=["CUDA_VISIBLE_DEVICES", "FOO"])
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(env_passthrough=[], timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.env_passthrough == ["CUDA_VISIBLE_DEVICES", "FOO"]


def test_merge_clamps_timeouts_to_base() -> None:
    base = _base_config(measurement_timeout_seconds=10, setup_timeout_seconds=10)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(measurement_timeout_seconds=999, setup_timeout_seconds=999, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.measurement_timeout_seconds == base.measurement_timeout_seconds
    assert merged.setup_timeout_seconds == base.setup_timeout_seconds


def test_merge_enforces_policy_invariants() -> None:
    base = _base_config(timeout_multiplier=1.0, enforce_environment_validation=True)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(timeout_multiplier=5.0, enforce_environment_validation=False)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.timeout_multiplier == base.timeout_multiplier
    assert merged.enforce_environment_validation == base.enforce_environment_validation


def test_expectation_policy_strict_without_write_flags_is_preview_only() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="strict",
        update_expectations=False,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is True
    assert writes_enabled is False


def test_expectation_policy_strict_with_update_flag_enables_writes() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="strict",
        update_expectations=True,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is True
    assert writes_enabled is True


def test_expectation_policy_portable_requires_explicit_override_for_validation() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="portable",
        update_expectations=True,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is False
    assert writes_enabled is False
