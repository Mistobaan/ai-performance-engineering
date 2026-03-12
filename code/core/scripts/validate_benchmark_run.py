#!/usr/bin/env python3
"""Validate declarative BenchmarkRun specs used by the repo methodology docs.

The goal is not to implement a full Kubernetes admission controller. This
validator enforces the repo's benchmarking methodology so publication-grade and
realism-grade runs carry the same minimum contract in local review and CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REQUIRED_FIXED_CONTROLS = {
    "model",
    "sequenceLengthMix",
    "precision",
    "batchingPolicy",
    "concurrencyModel",
}
REQUIRED_LAYERS = {"micro", "component", "end_to_end"}
REQUIRED_BOTTLENECKS = {
    "compute_bound",
    "comm_bound",
    "input_bound",
    "control_plane_bound",
}
REQUIRED_DECOMPOSITION = {"compute", "communication", "storage", "orchestration"}
REQUIRED_DISTRIBUTED_CHECKS = {
    "rdma_pathing",
    "gpu_nic_affinity",
    "pcie_link_health",
    "nvlink_health",
    "thermal_throttling",
}
REQUIRED_OBSERVABILITY_JOIN_KEYS = {
    "run_id",
    "benchmark_case_id",
    "scheduler_run_id",
    "job_uid",
    "pod_uid",
    "node_name",
    "gpu_uuid",
}
REQUIRED_INFERENCE_JOIN_KEYS = {"request_id", "trace_id"}
REQUIRED_TRAINING_JOIN_KEYS = {"rank_id"}
REQUIRED_INFRA_TELEMETRY = {
    "dcgm-exporter",
    "nvlink-exporter",
    "node-pci-exporter",
    "ping-exporter",
    "node-problem-detector",
    "hpc-verification",
}
REQUIRED_SCENARIO_PLAYBOOKS = {
    "tail_latency_regression",
    "low_gpu_utilization",
    "distributed_straggler",
    "scheduler_backpressure",
}
REQUIRED_PROVENANCE_CAPTURE = {
    "pinnedWorkloadSpec",
    "imageDigest",
    "driverCudaNcclRuntimeVersions",
    "hardwareTopology",
    "immutableRawArtifacts",
    "auditTrail",
}
ALLOWED_BENCHMARK_CLASSES = {"publication_grade", "realism_grade"}
ALLOWED_WORKLOAD_TYPES = {"training", "inference", "mixed"}
ALLOWED_VARIABLES = {
    "hardware_generation",
    "runtime_version",
    "scheduler_path",
    "control_plane_path",
    "driver_stack",
    "network_topology",
    "storage_stack",
}
REQUIRED_TRAINING_METRICS = {
    "time_to_train_hours",
    "mfu_pct",
    "scaling_efficiency_pct",
    "training_reliability_pct",
}
REQUIRED_INFERENCE_METRICS = {
    "ttft_ms",
    "tokens_per_second",
    "p99_latency_ms",
    "jitter_ms",
}
PUBLICATION_MODE_VALUES = {
    "dedicatedNodes",
    "stableBackgroundLoad",
    "fixedTopology",
    "topologyExclusiveScheduling",
}
REQUIRED_RAW_ARTIFACTS = {
    "logs",
    "traces",
    "profiler_reports",
    "manifests",
}
REQUIRED_WAREHOUSE_FACTS = {"benchmark_run_fact", "telemetry_slice_fact"}
REQUIRED_WAREHOUSE_DIMS = {
    "software_version_dim",
    "hardware_topology_dim",
    "cluster_region_dim",
    "workload_dim",
    "artifact_lineage_dim",
}


def _append_error(errors: List[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _expect_mapping(value: Any, path: str, errors: List[str]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    _append_error(errors, path, "expected mapping")
    return {}


def _expect_list(value: Any, path: str, errors: List[str]) -> List[Any]:
    if isinstance(value, list):
        return value
    _append_error(errors, path, "expected list")
    return []


def _validate_layers(spec: Dict[str, Any], errors: List[str]) -> Tuple[List[str], List[str]]:
    layers = _expect_list(spec.get("layers"), "spec.layers", errors)
    enabled_names: List[str] = []
    present_names: List[str] = []
    for index, layer in enumerate(layers):
        item = _expect_mapping(layer, f"spec.layers[{index}]", errors)
        name = item.get("name")
        if not isinstance(name, str) or not name:
            _append_error(errors, f"spec.layers[{index}].name", "expected non-empty string")
            continue
        present_names.append(name)
        enabled = item.get("enabled")
        if not isinstance(enabled, bool):
            _append_error(errors, f"spec.layers[{index}].enabled", "expected boolean")
            continue
        if enabled:
            enabled_names.append(name)
        suites = _expect_list(item.get("suites"), f"spec.layers[{index}].suites", errors)
        if enabled and not suites:
            _append_error(errors, f"spec.layers[{index}].suites", "enabled layer must declare at least one suite")
    if not enabled_names:
        _append_error(errors, "spec.layers", "at least one layer must be enabled")
    missing_required = REQUIRED_LAYERS - set(present_names)
    if missing_required:
        _append_error(
            errors,
            "spec.layers",
            f"template must model all methodology layers; missing {sorted(missing_required)}",
        )
    return present_names, enabled_names


def _validate_workload(spec: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    workload = _expect_mapping(spec.get("workload"), "spec.workload", errors)
    for field in ("model", "precision", "batchingPolicy", "concurrencyModel"):
        value = workload.get(field)
        if not isinstance(value, str) or not value:
            _append_error(errors, f"spec.workload.{field}", "expected non-empty string")
    sequence_mix = _expect_list(workload.get("sequenceLengthMix"), "spec.workload.sequenceLengthMix", errors)
    if not sequence_mix:
        _append_error(errors, "spec.workload.sequenceLengthMix", "expected at least one sequence mix entry")
    return workload


def _validate_comparison(spec: Dict[str, Any], errors: List[str]) -> None:
    comparison = _expect_mapping(spec.get("comparison"), "spec.comparison", errors)
    variable = comparison.get("variableUnderTest")
    if not isinstance(variable, str) or variable not in ALLOWED_VARIABLES:
        _append_error(
            errors,
            "spec.comparison.variableUnderTest",
            f"expected one of {sorted(ALLOWED_VARIABLES)}",
        )
    controls = _expect_mapping(comparison.get("controls"), "spec.comparison.controls", errors)
    fixed = _expect_mapping(controls.get("fixed"), "spec.comparison.controls.fixed", errors)
    missing = sorted(REQUIRED_FIXED_CONTROLS - set(fixed.keys()))
    if missing:
        _append_error(
            errors,
            "spec.comparison.controls.fixed",
            f"missing fixed workload controls {missing}",
        )
    if controls.get("compareOneVariableAtATime") is not True:
        _append_error(
            errors,
            "spec.comparison.controls.compareOneVariableAtATime",
            "must be true",
        )


def _validate_metrics(spec: Dict[str, Any], benchmark_type: str, errors: List[str]) -> None:
    metrics = _expect_mapping(spec.get("metrics"), "spec.metrics", errors)
    training = _expect_mapping(metrics.get("training"), "spec.metrics.training", errors)
    inference = _expect_mapping(metrics.get("inference"), "spec.metrics.inference", errors)

    training_primary = set(_expect_list(training.get("primary"), "spec.metrics.training.primary", errors))
    inference_primary = set(_expect_list(inference.get("primary"), "spec.metrics.inference.primary", errors))

    if benchmark_type in {"training", "mixed"}:
        if training.get("enabled") is not True:
            _append_error(errors, "spec.metrics.training.enabled", "must be true for training or mixed workloads")
        missing_training = REQUIRED_TRAINING_METRICS - training_primary
        if missing_training:
            _append_error(
                errors,
                "spec.metrics.training.primary",
                f"missing training metrics {sorted(missing_training)}",
            )
    if benchmark_type in {"inference", "mixed"}:
        if inference.get("enabled") is not True:
            _append_error(errors, "spec.metrics.inference.enabled", "must be true for inference or mixed workloads")
        missing_inference = REQUIRED_INFERENCE_METRICS - inference_primary
        if missing_inference:
            _append_error(
                errors,
                "spec.metrics.inference.primary",
                f"missing inference metrics {sorted(missing_inference)}",
            )
        cost_metrics = {"cost_per_token_usd", "cost_per_request_usd"} & inference_primary
        if not cost_metrics:
            _append_error(
                errors,
                "spec.metrics.inference.primary",
                "must include cost_per_token_usd or cost_per_request_usd",
            )


def _validate_trials(spec: Dict[str, Any], errors: List[str]) -> None:
    trials = _expect_mapping(spec.get("trials"), "spec.trials", errors)
    min_replicates = trials.get("minReplicates")
    if not isinstance(min_replicates, int) or min_replicates < 3:
        _append_error(errors, "spec.trials.minReplicates", "must be an integer >= 3")
    confidence = trials.get("confidenceLevel")
    if not isinstance(confidence, (int, float)) or not (0.8 <= float(confidence) < 1.0):
        _append_error(errors, "spec.trials.confidenceLevel", "must be between 0.8 and 1.0")
    outlier = _expect_mapping(trials.get("outlierPolicy"), "spec.trials.outlierPolicy", errors)
    for field in ("method", "action"):
        value = outlier.get(field)
        if not isinstance(value, str) or not value:
            _append_error(errors, f"spec.trials.outlierPolicy.{field}", "expected non-empty string")
    report = _expect_mapping(trials.get("report"), "spec.trials.report", errors)
    for field in ("distributions", "confidenceIntervals", "rankLevelOutliers"):
        if report.get(field) is not True:
            _append_error(errors, f"spec.trials.report.{field}", "must be true")


def _validate_bottleneck_analysis(spec: Dict[str, Any], errors: List[str]) -> None:
    analysis = _expect_mapping(spec.get("bottleneckAnalysis"), "spec.bottleneckAnalysis", errors)
    taxonomy = set(_expect_list(analysis.get("taxonomy"), "spec.bottleneckAnalysis.taxonomy", errors))
    missing_taxonomy = REQUIRED_BOTTLENECKS - taxonomy
    if missing_taxonomy:
        _append_error(
            errors,
            "spec.bottleneckAnalysis.taxonomy",
            f"missing bottleneck classes {sorted(missing_taxonomy)}",
        )
    decomposition = set(
        _expect_list(analysis.get("decomposeOverheads"), "spec.bottleneckAnalysis.decomposeOverheads", errors)
    )
    missing_decomposition = REQUIRED_DECOMPOSITION - decomposition
    if missing_decomposition:
        _append_error(
            errors,
            "spec.bottleneckAnalysis.decomposeOverheads",
            f"missing decomposition buckets {sorted(missing_decomposition)}",
        )
    instrumentation = _expect_mapping(
        analysis.get("instrumentation"),
        "spec.bottleneckAnalysis.instrumentation",
        errors,
    )
    for key in REQUIRED_BOTTLENECKS:
        probes = _expect_list(instrumentation.get(key), f"spec.bottleneckAnalysis.instrumentation.{key}", errors)
        if not probes:
            _append_error(
                errors,
                f"spec.bottleneckAnalysis.instrumentation.{key}",
                "must declare at least one probe",
            )


def _validate_distributed(spec: Dict[str, Any], errors: List[str]) -> None:
    distributed = _expect_mapping(spec.get("distributed"), "spec.distributed", errors)
    if distributed.get("requireRankLevelVisibility") is not True:
        _append_error(errors, "spec.distributed.requireRankLevelVisibility", "must be true")
    collectives = set(_expect_list(distributed.get("collectives"), "spec.distributed.collectives", errors))
    if not {"latency", "bandwidth"} <= collectives:
        _append_error(
            errors,
            "spec.distributed.collectives",
            "must include latency and bandwidth",
        )
    diagnosis = _expect_mapping(distributed.get("nodeDiagnosis"), "spec.distributed.nodeDiagnosis", errors)
    checks = set(_expect_list(diagnosis.get("validate"), "spec.distributed.nodeDiagnosis.validate", errors))
    missing_checks = REQUIRED_DISTRIBUTED_CHECKS - checks
    if missing_checks:
        _append_error(
            errors,
            "spec.distributed.nodeDiagnosis.validate",
            f"missing node-level checks {sorted(missing_checks)}",
        )
    remediation = set(
        _expect_list(diagnosis.get("remediation"), "spec.distributed.nodeDiagnosis.remediation", errors)
    )
    if not {"cordon_problematic_node", "isolate_bad_node"} <= remediation:
        _append_error(
            errors,
            "spec.distributed.nodeDiagnosis.remediation",
            "must include cordon_problematic_node and isolate_bad_node",
        )


def _validate_observability(spec: Dict[str, Any], workload_type: str, scheduler_path: str, errors: List[str]) -> None:
    observability = _expect_mapping(spec.get("observability"), "spec.observability", errors)
    correlation = _expect_mapping(observability.get("correlation"), "spec.observability.correlation", errors)
    join_keys = set(
        _expect_list(correlation.get("stableJoinKeys"), "spec.observability.correlation.stableJoinKeys", errors)
    )
    missing_join_keys = REQUIRED_OBSERVABILITY_JOIN_KEYS - join_keys
    if missing_join_keys:
        _append_error(
            errors,
            "spec.observability.correlation.stableJoinKeys",
            f"missing stable join keys {sorted(missing_join_keys)}",
        )
    if workload_type in {"training", "mixed"}:
        missing_training_keys = REQUIRED_TRAINING_JOIN_KEYS - join_keys
        if missing_training_keys:
            _append_error(
                errors,
                "spec.observability.correlation.stableJoinKeys",
                f"training or mixed runs must include {sorted(missing_training_keys)}",
            )
    if workload_type in {"inference", "mixed"}:
        missing_inference_keys = REQUIRED_INFERENCE_JOIN_KEYS - join_keys
        if missing_inference_keys:
            _append_error(
                errors,
                "spec.observability.correlation.stableJoinKeys",
                f"inference or mixed runs must include {sorted(missing_inference_keys)}",
            )

    lineage = _expect_mapping(
        correlation.get("publishedNumberLineage"),
        "spec.observability.correlation.publishedNumberLineage",
        errors,
    )
    for field in ("rawArtifactManifestDigest", "warehouseRowLineage", "querySpecCaptured"):
        if lineage.get(field) is not True:
            _append_error(errors, f"spec.observability.correlation.publishedNumberLineage.{field}", "must be true")

    telemetry = _expect_mapping(observability.get("telemetrySources"), "spec.observability.telemetrySources", errors)
    service = _expect_list(telemetry.get("service"), "spec.observability.telemetrySources.service", errors)
    if not service:
        _append_error(errors, "spec.observability.telemetrySources.service", "must declare at least one service signal")
    infra = set(
        _expect_list(
            telemetry.get("infrastructure"),
            "spec.observability.telemetrySources.infrastructure",
            errors,
        )
    )
    missing_infra = REQUIRED_INFRA_TELEMETRY - infra
    if missing_infra:
        _append_error(
            errors,
            "spec.observability.telemetrySources.infrastructure",
            f"missing infrastructure telemetry sources {sorted(missing_infra)}",
        )
    scheduler = set(
        _expect_list(telemetry.get("scheduler"), "spec.observability.telemetrySources.scheduler", errors)
    )
    if "kubernetes_events" not in scheduler:
        _append_error(
            errors,
            "spec.observability.telemetrySources.scheduler",
            "must include kubernetes_events",
        )
    for required in ("kueue", "slinky"):
        if required in scheduler_path and required not in scheduler:
            _append_error(
                errors,
                "spec.observability.telemetrySources.scheduler",
                f"scheduler path {scheduler_path!r} must include telemetry source {required!r}",
            )

    playbooks = set(_expect_list(observability.get("scenarioPlaybooks"), "spec.observability.scenarioPlaybooks", errors))
    missing_playbooks = REQUIRED_SCENARIO_PLAYBOOKS - playbooks
    if missing_playbooks:
        _append_error(
            errors,
            "spec.observability.scenarioPlaybooks",
            f"missing scenario playbooks {sorted(missing_playbooks)}",
        )


def _validate_provenance(spec: Dict[str, Any], benchmark_class: str, errors: List[str]) -> None:
    provenance = _expect_mapping(spec.get("provenance"), "spec.provenance", errors)
    capture = _expect_mapping(provenance.get("capture"), "spec.provenance.capture", errors)
    for field in REQUIRED_PROVENANCE_CAPTURE:
        if capture.get(field) is not True:
            _append_error(errors, f"spec.provenance.capture.{field}", "must be true")

    signing = _expect_mapping(provenance.get("signing"), "spec.provenance.signing", errors)
    required = signing.get("required")
    if benchmark_class == "publication_grade" and required is not True:
        _append_error(errors, "spec.provenance.signing.required", "must be true for publication-grade runs")
    for field in ("backend", "attestationFormat"):
        value = signing.get(field)
        if benchmark_class == "publication_grade" and (not isinstance(value, str) or not value):
            _append_error(errors, f"spec.provenance.signing.{field}", "expected non-empty string")


def _validate_execution_policy(spec: Dict[str, Any], benchmark_class: str, errors: List[str]) -> None:
    execution = _expect_mapping(spec.get("executionPolicy"), "spec.executionPolicy", errors)
    publication = _expect_mapping(execution.get("publicationGrade"), "spec.executionPolicy.publicationGrade", errors)
    realism = _expect_mapping(execution.get("realismGrade"), "spec.executionPolicy.realismGrade", errors)

    if benchmark_class == "publication_grade":
        for key in PUBLICATION_MODE_VALUES:
            if publication.get(key) is not True:
                _append_error(errors, f"spec.executionPolicy.publicationGrade.{key}", "must be true")
    if benchmark_class == "realism_grade":
        if realism.get("multiTenantScenarios") is not True:
            _append_error(
                errors,
                "spec.executionPolicy.realismGrade.multiTenantScenarios",
                "must be true for realism-grade runs",
            )
        if realism.get("captureClusterContext") is not True:
            _append_error(
                errors,
                "spec.executionPolicy.realismGrade.captureClusterContext",
                "must be true for realism-grade runs",
            )


def _validate_sinks(spec: Dict[str, Any], workload_type: str, errors: List[str]) -> None:
    sinks = _expect_mapping(spec.get("sinks"), "spec.sinks", errors)

    raw = _expect_mapping(sinks.get("rawArtifacts"), "spec.sinks.rawArtifacts", errors)
    for field in ("store", "pathTemplate", "retentionClass"):
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            _append_error(errors, f"spec.sinks.rawArtifacts.{field}", "expected non-empty string")
    raw_artifacts = set(_expect_list(raw.get("artifacts"), "spec.sinks.rawArtifacts.artifacts", errors))
    missing_raw = REQUIRED_RAW_ARTIFACTS - raw_artifacts
    if missing_raw:
        _append_error(
            errors,
            "spec.sinks.rawArtifacts.artifacts",
            f"missing raw artifact classes {sorted(missing_raw)}",
        )

    hot = _expect_mapping(sinks.get("hotMetrics"), "spec.sinks.hotMetrics", errors)
    hot_store = hot.get("store")
    if not isinstance(hot_store, str) or not hot_store:
        _append_error(errors, "spec.sinks.hotMetrics.store", "expected non-empty string")
    retention_days = hot.get("retentionDays")
    if not isinstance(retention_days, int) or retention_days <= 0:
        _append_error(errors, "spec.sinks.hotMetrics.retentionDays", "must be a positive integer")
    budget = _expect_mapping(hot.get("cardinalityBudget"), "spec.sinks.hotMetrics.cardinalityBudget", errors)
    max_active_series = budget.get("maxActiveSeries")
    if not isinstance(max_active_series, int) or max_active_series <= 0:
        _append_error(
            errors,
            "spec.sinks.hotMetrics.cardinalityBudget.maxActiveSeries",
            "must be a positive integer",
        )
    drop_dimensions = set(
        _expect_list(
            budget.get("dropDimensions"),
            "spec.sinks.hotMetrics.cardinalityBudget.dropDimensions",
            errors,
        )
    )
    if not {"request_id", "trace_id"} <= drop_dimensions:
        _append_error(
            errors,
            "spec.sinks.hotMetrics.cardinalityBudget.dropDimensions",
            "must include request_id and trace_id to keep hot-metric cardinality bounded",
        )

    warehouse = _expect_mapping(sinks.get("curatedWarehouse"), "spec.sinks.curatedWarehouse", errors)
    warehouse_store = warehouse.get("store")
    if not isinstance(warehouse_store, str) or not warehouse_store:
        _append_error(errors, "spec.sinks.curatedWarehouse.store", "expected non-empty string")
    layout = _expect_mapping(warehouse.get("layout"), "spec.sinks.curatedWarehouse.layout", errors)
    facts = set(_expect_list(layout.get("facts"), "spec.sinks.curatedWarehouse.layout.facts", errors))
    missing_facts = REQUIRED_WAREHOUSE_FACTS - facts
    if missing_facts:
        _append_error(
            errors,
            "spec.sinks.curatedWarehouse.layout.facts",
            f"missing warehouse facts {sorted(missing_facts)}",
        )
    if workload_type in {"inference", "mixed"} and "serving_outcome_fact" not in facts:
        _append_error(
            errors,
            "spec.sinks.curatedWarehouse.layout.facts",
            "inference or mixed runs must include serving_outcome_fact",
        )
    if workload_type in {"training", "mixed"} and "training_outcome_fact" not in facts:
        _append_error(
            errors,
            "spec.sinks.curatedWarehouse.layout.facts",
            "training or mixed runs must include training_outcome_fact",
        )
    dims = set(_expect_list(layout.get("dimensions"), "spec.sinks.curatedWarehouse.layout.dimensions", errors))
    missing_dims = REQUIRED_WAREHOUSE_DIMS - dims
    if missing_dims:
        _append_error(
            errors,
            "spec.sinks.curatedWarehouse.layout.dimensions",
            f"missing warehouse dimensions {sorted(missing_dims)}",
        )

    retention = _expect_mapping(warehouse.get("retention"), "spec.sinks.curatedWarehouse.retention", errors)
    hot_days = retention.get("hotDays")
    warm_days = retention.get("warmDays")
    cold_days = retention.get("coldDays")
    for field, value in (("hotDays", hot_days), ("warmDays", warm_days), ("coldDays", cold_days)):
        if not isinstance(value, int) or value <= 0:
            _append_error(errors, f"spec.sinks.curatedWarehouse.retention.{field}", "must be a positive integer")
    if all(isinstance(value, int) and value > 0 for value in (hot_days, warm_days, cold_days)):
        if not (hot_days <= warm_days <= cold_days):
            _append_error(
                errors,
                "spec.sinks.curatedWarehouse.retention",
                "must satisfy hotDays <= warmDays <= coldDays",
            )

    lineage = _expect_mapping(warehouse.get("lineage"), "spec.sinks.curatedWarehouse.lineage", errors)
    for field in ("publishedNumbersTraceableToRaw", "manifestDigestColumn", "workloadSpecDigestColumn"):
        if lineage.get(field) is not True:
            _append_error(errors, f"spec.sinks.curatedWarehouse.lineage.{field}", "must be true")


def _validate_automation(spec: Dict[str, Any], errors: List[str]) -> None:
    automation = _expect_mapping(spec.get("automation"), "spec.automation", errors)
    ci = _expect_mapping(automation.get("ci"), "spec.automation.ci", errors)
    for field in ("canary", "nightly", "preRelease"):
        if not isinstance(ci.get(field), bool):
            _append_error(errors, f"spec.automation.ci.{field}", "expected boolean")


def summarize_benchmark_run(document: Dict[str, Any]) -> Dict[str, Any]:
    spec = document["spec"]
    enabled_layers = [layer["name"] for layer in spec["layers"] if layer.get("enabled")]
    return {
        "name": document["metadata"]["name"],
        "benchmark_class": spec["intent"]["benchmarkClass"],
        "workload_type": spec["intent"]["workloadType"],
        "variable_under_test": spec["comparison"]["variableUnderTest"],
        "enabled_layers": enabled_layers,
        "scheduler_path": spec["intent"]["schedulerPath"],
        "hot_metrics_store": spec["sinks"]["hotMetrics"]["store"],
        "warehouse_store": spec["sinks"]["curatedWarehouse"]["store"],
        "ci_schedule": {
            "canary": spec["automation"]["ci"]["canary"],
            "nightly": spec["automation"]["ci"]["nightly"],
            "preRelease": spec["automation"]["ci"]["preRelease"],
        },
    }


def validate_benchmark_run_document(document: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(document, dict):
        return ["document: expected YAML mapping"]

    api_version = document.get("apiVersion")
    if not isinstance(api_version, str) or not api_version:
        _append_error(errors, "apiVersion", "expected non-empty string")
    kind = document.get("kind")
    if kind != "BenchmarkRun":
        _append_error(errors, "kind", "expected BenchmarkRun")

    metadata = _expect_mapping(document.get("metadata"), "metadata", errors)
    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        _append_error(errors, "metadata.name", "expected non-empty string")

    spec = _expect_mapping(document.get("spec"), "spec", errors)
    intent = _expect_mapping(spec.get("intent"), "spec.intent", errors)
    benchmark_class = intent.get("benchmarkClass")
    if benchmark_class not in ALLOWED_BENCHMARK_CLASSES:
        _append_error(
            errors,
            "spec.intent.benchmarkClass",
            f"expected one of {sorted(ALLOWED_BENCHMARK_CLASSES)}",
        )
        benchmark_class = "publication_grade"
    workload_type = intent.get("workloadType")
    if workload_type not in ALLOWED_WORKLOAD_TYPES:
        _append_error(
            errors,
            "spec.intent.workloadType",
            f"expected one of {sorted(ALLOWED_WORKLOAD_TYPES)}",
        )
        workload_type = "mixed"
    for field in ("schedulerPath", "cadence"):
        value = intent.get(field)
        if not isinstance(value, str) or not value:
            _append_error(errors, f"spec.intent.{field}", "expected non-empty string")

    _validate_layers(spec, errors)
    _validate_workload(spec, errors)
    _validate_comparison(spec, errors)
    _validate_metrics(spec, workload_type, errors)
    _validate_trials(spec, errors)
    _validate_bottleneck_analysis(spec, errors)
    _validate_distributed(spec, errors)
    _validate_observability(spec, workload_type, str(intent.get("schedulerPath", "")), errors)
    _validate_provenance(spec, benchmark_class, errors)
    _validate_execution_policy(spec, benchmark_class, errors)
    _validate_sinks(spec, workload_type, errors)
    _validate_automation(spec, errors)
    return errors


def validate_benchmark_run_file(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_benchmark_run_document(document)
    summary = summarize_benchmark_run(document) if not errors else {}
    return errors, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a declarative BenchmarkRun YAML spec.")
    parser.add_argument("--file", required=True, help="Path to the BenchmarkRun YAML file.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    path = Path(args.file)
    errors, summary = validate_benchmark_run_file(path)

    if args.json:
        payload = {
            "valid": not errors,
            "file": str(path),
            "errors": errors,
            "summary": summary,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if errors:
            print(f"BenchmarkRun invalid: {path}")
            for error in errors:
                print(f"- {error}")
        else:
            print(f"BenchmarkRun valid: {path}")
            print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
