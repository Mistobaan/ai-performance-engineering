from core.harness.run_benchmarks import (
    _collect_required_profiler_failures,
    _format_required_profiler_failure,
)


def test_collect_required_profiler_failures_captures_baseline_and_optimized_failures() -> None:
    result_entry = {
        "baseline_profiler_statuses": {
            "nsys": "succeeded",
            "ncu": "failed",
            "torch": "skipped",
        }
    }
    best_opt = {
        "optimized_profiler_statuses": {
            "nsys": "failed",
            "ncu": "succeeded",
        }
    }

    failures = _collect_required_profiler_failures(
        result_entry,
        best_opt,
        profiling_requested=True,
    )

    assert failures == [
        "baseline:ncu:failed",
        "baseline:torch:skipped",
        "optimized:nsys:failed",
    ]


def test_collect_required_profiler_failures_ignores_disabled_profiling() -> None:
    failures = _collect_required_profiler_failures(
        {"baseline_profiler_statuses": {"nsys": "failed"}},
        {"optimized_profiler_statuses": {"ncu": "failed"}},
        profiling_requested=False,
    )

    assert failures == []


def test_format_required_profiler_failure_is_explicit() -> None:
    message = _format_required_profiler_failure(
        ["baseline:torch:failed", "optimized:nsys:skipped"]
    )

    assert message == (
        "Required profilers did not succeed: "
        "baseline:torch:failed, optimized:nsys:skipped"
    )
