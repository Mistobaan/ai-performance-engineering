from __future__ import annotations

import copy
from pathlib import Path

import yaml

from core.scripts.validate_benchmark_run import validate_benchmark_run_document


def _load_template() -> dict:
    path = Path("templates/benchmark_run.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_benchmark_run_template_is_valid() -> None:
    document = _load_template()
    errors = validate_benchmark_run_document(document)
    assert errors == []


def test_publication_grade_requires_signed_provenance() -> None:
    document = copy.deepcopy(_load_template())
    document["spec"]["provenance"]["signing"]["required"] = False

    errors = validate_benchmark_run_document(document)

    assert any("spec.provenance.signing.required" in error for error in errors)


def test_comparison_requires_frozen_workload_controls() -> None:
    document = copy.deepcopy(_load_template())
    del document["spec"]["comparison"]["controls"]["fixed"]["sequenceLengthMix"]

    errors = validate_benchmark_run_document(document)

    assert any("spec.comparison.controls.fixed" in error for error in errors)


def test_inference_spec_requires_cost_metric() -> None:
    document = copy.deepcopy(_load_template())
    document["spec"]["metrics"]["inference"]["primary"] = [
        "ttft_ms",
        "tokens_per_second",
        "p99_latency_ms",
        "jitter_ms",
    ]

    errors = validate_benchmark_run_document(document)

    assert any("cost_per_token_usd or cost_per_request_usd" in error for error in errors)


def test_observability_requires_serving_join_keys() -> None:
    document = copy.deepcopy(_load_template())
    document["spec"]["observability"]["correlation"]["stableJoinKeys"].remove("trace_id")

    errors = validate_benchmark_run_document(document)

    assert any("spec.observability.correlation.stableJoinKeys" in error for error in errors)


def test_sinks_require_curated_lineage_contract() -> None:
    document = copy.deepcopy(_load_template())
    document["spec"]["sinks"]["curatedWarehouse"]["lineage"]["manifestDigestColumn"] = False

    errors = validate_benchmark_run_document(document)

    assert any("spec.sinks.curatedWarehouse.lineage.manifestDigestColumn" in error for error in errors)
