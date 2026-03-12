"""Regression tests for expectation update event payloads."""

import json
import logging

from core.benchmark.expectations import ExpectationEntry, RunProvenance, UpdateResult, ValidationIssue
from core.harness.run_benchmarks import BenchmarkEventLogger, log_expectation_delta


def _make_entry(example: str, baseline_ms: float, optimized_ms: float) -> ExpectationEntry:
    return ExpectationEntry(
        example=example,
        type="python",
        optimization_goal="speed",
        baseline_time_ms=baseline_ms,
        best_optimized_time_ms=optimized_ms,
        provenance=RunProvenance(
            git_commit="deadbeef",
            hardware_key="b200",
            profile_name="minimal",
            timestamp="2026-03-11T00:00:00Z",
            iterations=10,
            warmup_iterations=5,
        ),
    )


def test_expectation_update_event_includes_rejection_reason(tmp_path):
    event_path = tmp_path / "events.jsonl"
    logger = logging.getLogger("test.expectation_event_logging")
    logger.addHandler(logging.NullHandler())

    old_entry = _make_entry("gemm_cuda", 10.0, 5.0)
    new_entry = _make_entry("gemm_cuda", 10.0, 6.0)
    new_entry.provenance.git_commit = "feedface"
    update_result = UpdateResult(
        status="rejected",
        message="Provenance mismatch: git_commit differs",
        validation_issues=[
            ValidationIssue(
                example_key="gemm_cuda",
                issue_type="provenance_mismatch",
                message="Mixed provenance update rejected",
                stored_value={"git_commit": "old"},
                expected_value={"git_commit": "new"},
            )
        ],
    )

    event_logger = BenchmarkEventLogger(event_path, run_id="test_run", logger=logger)
    try:
        log_expectation_delta(
            logger,
            example_key="gemm_cuda",
            goal="speed",
            old_entry=old_entry,
            new_entry=new_entry,
            update_result=update_result,
            event_logger=event_logger,
            chapter="ch01",
        )
    finally:
        event_logger.close()

    event = json.loads(event_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "expectation_update"
    assert event["status"] == "rejected"
    assert event["update_message"] == "Provenance mismatch: git_commit differs"
    assert event["validation_issue_types"] == ["provenance_mismatch"]
    assert event["old_provenance"] == old_entry.provenance.to_dict()
    assert event["new_provenance"] == new_entry.provenance.to_dict()
    assert event["provenance_mismatch_fields"] == ["git_commit"]
    assert event["validation_issues"] == [
        {
            "example_key": "gemm_cuda",
            "issue_type": "provenance_mismatch",
            "message": "Mixed provenance update rejected",
            "stored_value": {"git_commit": "old"},
            "expected_value": {"git_commit": "new"},
            "delta_pct": None,
        }
    ]
