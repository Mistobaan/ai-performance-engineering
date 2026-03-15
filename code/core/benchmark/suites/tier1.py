"""Tier-1 canonical benchmark suite definition and artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Tier1Target:
    key: str
    target: str
    category: str
    rationale: str
    profile: str = "minimal"


@dataclass(frozen=True)
class Tier1SuiteDefinition:
    name: str
    version: int
    description: str
    history_root: str
    default_profile: str
    default_output_format: str
    targets: List[Tier1Target]

    def target_strings(self) -> List[str]:
        return [target.target for target in self.targets]

    def by_target(self) -> Dict[str, Tier1Target]:
        return {target.target: target for target in self.targets}


def default_tier1_config_path(repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root or _repo_root())
    return root / "configs" / "benchmark_suites" / "tier1.yaml"


def _coerce_target(payload: Dict[str, Any]) -> Tier1Target:
    return Tier1Target(
        key=str(payload["key"]),
        target=str(payload["target"]),
        category=str(payload["category"]),
        rationale=str(payload.get("rationale", "")).strip(),
        profile=str(payload.get("profile", "minimal")).strip() or "minimal",
    )


def load_tier1_suite(config_path: Optional[Path] = None) -> Tier1SuiteDefinition:
    path = Path(config_path or default_tier1_config_path()).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Tier1SuiteDefinition(
        name=str(data.get("suite_name", "tier1")),
        version=int(data.get("version", 1)),
        description=str(data.get("description", "")).strip(),
        history_root=str(data.get("history_root", "artifacts/history/tier1")).strip(),
        default_profile=str(data.get("default_profile", "minimal")).strip() or "minimal",
        default_output_format=str(data.get("default_output_format", "both")).strip() or "both",
        targets=[_coerce_target(entry) for entry in data.get("targets", [])],
    )


def _load_json_object(
    path: Path,
    *,
    label: str,
    required: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = f"Failed to read {label} {path}: {exc}"
        if required:
            raise ValueError(message) from exc
        return None, message

    if not isinstance(payload, dict):
        message = f"Expected JSON object in {label} {path}, got {type(payload).__name__}"
        if required:
            raise ValueError(message)
        return None, message

    return payload, None


def _chapter_key_from_target(target: str) -> str:
    chapter = target.split(":", 1)[0].strip()
    return chapter.replace("/", "_").replace("-", "_")


def _example_from_target(target: str) -> Optional[str]:
    if ":" not in target:
        return None
    return target.split(":", 1)[1].strip() or None


def _find_best_optimization_name(benchmark: Dict[str, Any]) -> Optional[str]:
    best_speedup = float(benchmark.get("best_speedup", 0.0) or 0.0)
    best_name = benchmark.get("best_optimization")
    if best_name:
        return str(best_name)
    optimizations = benchmark.get("optimizations", []) or []
    if not optimizations:
        return None
    best_entry = max(
        optimizations,
        key=lambda entry: float(entry.get("speedup", 0.0) or 0.0),
    )
    if math.isclose(float(best_entry.get("speedup", 0.0) or 0.0), best_speedup, rel_tol=1e-6, abs_tol=1e-6):
        return str(best_entry.get("technique") or best_entry.get("file") or "")
    return str(best_entry.get("technique") or best_entry.get("file") or "")


def _artifact_refs(benchmark: Dict[str, Any]) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for key, value in benchmark.items():
        if not isinstance(value, str):
            continue
        if key.endswith(("_rep", "_trace", "_json")) and value:
            refs[key] = value
    return refs


def _geometric_mean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _single_target_suite(suite: Tier1SuiteDefinition, target: Tier1Target) -> Tier1SuiteDefinition:
    return Tier1SuiteDefinition(
        name=suite.name,
        version=suite.version,
        description=suite.description,
        history_root=suite.history_root,
        default_profile=suite.default_profile,
        default_output_format=suite.default_output_format,
        targets=[target],
    )


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "target"


def _confirm_speedup_regressions(
    *,
    comparison: Dict[str, Any],
    current_summary: Dict[str, Any],
    previous_summary: Optional[Dict[str, Any]],
    suite: Tier1SuiteDefinition,
    suite_run_dir: Path,
    bench_root: Optional[Path],
    execution_run_id: str,
    profile_type: str,
    output_format: str,
    suite_timeout: Optional[int],
    timeout_multiplier: float,
    validity_profile: str,
    allow_portable_expectations_update: bool,
    reproducible: bool,
    cold_start: bool,
    force_synchronize: bool,
    iterations: Optional[int],
    warmup: Optional[int],
    gpu_sm_clock_mhz: Optional[int],
    gpu_mem_clock_mhz: Optional[int],
    artifacts_dir: Optional[str],
    log_level: str,
    log_file: Optional[str],
    single_gpu: bool,
    accept_regressions: bool,
    update_expectations: bool,
    allow_mixed_provenance: bool,
    ncu_metric_set: str,
    ncu_replay_mode: Optional[str],
    pm_sampling_interval: Optional[int],
    nsys_timeout_seconds: Optional[int],
    ncu_timeout_seconds: Optional[int],
    launch_via: str,
    nproc_per_node: Optional[int],
    nnodes: Optional[str],
    rdzv_backend: Optional[str],
    rdzv_endpoint: Optional[str],
    torchrun_env: Optional[List[str]],
    target_extra_args: Optional[List[str]],
    verify_input: bool,
    verify_output: bool,
    llm_analysis: bool,
    force_llm: bool,
    llm_provider: Optional[str],
    apply_llm_patches: bool,
    rebenchmark_llm_patches: bool,
    patch_strategy: str,
    llm_patch_retries: int,
    use_llm_cache: bool,
    llm_explain: bool,
) -> Dict[str, Any]:
    from core.analysis.regressions import compare_suite_summaries
    from core.benchmark.bench_commands import _execute_benchmarks

    if previous_summary is None:
        comparison["rechecks"] = []
        comparison["suppressed_regressions"] = []
        return comparison

    regressions = list(comparison.get("regressions", []))
    confirmed: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    rechecks: List[Dict[str, Any]] = []
    target_map = suite.by_target()

    for regression in regressions:
        if regression.get("reason") != "speedup":
            confirmed.append(regression)
            continue
        target_name = str(regression.get("target") or "")
        target = target_map.get(target_name)
        if target is None:
            confirmed.append(regression)
            continue

        recheck_run_id = f"{execution_run_id}__recheck__{_sanitize_component(target.key)}"
        recheck_execution = _execute_benchmarks(
            targets=[target.target],
            bench_root=bench_root,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=suite_timeout,
            timeout_multiplier=timeout_multiplier,
            validity_profile=validity_profile,
            allow_portable_expectations_update=allow_portable_expectations_update,
            reproducible=reproducible,
            cold_start=cold_start,
            force_synchronize=force_synchronize,
            iterations=iterations,
            warmup=warmup,
            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
            artifacts_dir=artifacts_dir,
            run_id=recheck_run_id,
            log_level=log_level,
            log_file=log_file,
            single_gpu=single_gpu,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            allow_mixed_provenance=allow_mixed_provenance,
            ncu_metric_set=ncu_metric_set,
            ncu_replay_mode=ncu_replay_mode,
            pm_sampling_interval=pm_sampling_interval,
            nsys_timeout_seconds=nsys_timeout_seconds,
            ncu_timeout_seconds=ncu_timeout_seconds,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
            verify_input=verify_input,
            verify_output=verify_output,
            llm_analysis=llm_analysis,
            force_llm=force_llm,
            llm_provider=llm_provider,
            apply_llm_patches=apply_llm_patches,
            rebenchmark_llm_patches=rebenchmark_llm_patches,
            patch_strategy=patch_strategy,
            llm_patch_retries=llm_patch_retries,
            use_llm_cache=use_llm_cache,
            llm_explain=llm_explain,
            exit_on_failure=False,
        )

        recheck_summary = build_tier1_suite_summary(
            Path(recheck_execution["output_json"]),
            _single_target_suite(suite, target),
            run_id=recheck_execution["run_id"],
            manifest_path=Path(recheck_execution["manifest_path"]) if recheck_execution.get("manifest_path") else None,
            report_path=Path(recheck_execution["output_markdown"]) if recheck_execution.get("output_markdown") else None,
        )
        recheck_target = recheck_summary["targets"][0]
        recheck_comparison = compare_suite_summaries(recheck_summary, previous_summary)
        recheck_record = {
            "target": target_name,
            "recheck_run_id": recheck_execution["run_id"],
            "recheck_output_json": recheck_execution["output_json"],
            "recheck_summary": recheck_target,
            "confirmed_regression": any(
                row.get("target") == target_name and row.get("reason") == "speedup"
                for row in recheck_comparison.get("regressions", [])
            ),
        }
        rechecks.append(recheck_record)

        if recheck_record["confirmed_regression"]:
            confirmed.append(regression)
        else:
            suppressed.append(
                {
                    **regression,
                    "suppression_reason": "recheck_not_regressed",
                    "recheck_run_id": recheck_execution["run_id"],
                    "recheck_speedup": recheck_target.get("best_speedup"),
                    "recheck_optimized_time_ms": recheck_target.get("best_optimized_time_ms"),
                }
            )

    comparison["regressions"] = confirmed
    comparison["suppressed_regressions"] = suppressed
    comparison["rechecks"] = rechecks

    recheck_path = suite_run_dir / "regression_rechecks.json"
    recheck_path.parent.mkdir(parents=True, exist_ok=True)
    recheck_path.write_text(json.dumps(rechecks, indent=2), encoding="utf-8")
    comparison["regression_rechecks_path"] = str(recheck_path)
    return comparison


def build_tier1_suite_summary(
    result_json_path: Path,
    suite: Tier1SuiteDefinition,
    *,
    run_id: str,
    manifest_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload, _ = _load_json_object(
        Path(result_json_path),
        label="tier-1 benchmark result JSON",
        required=True,
    )
    assert payload is not None
    target_map = suite.by_target()

    chapter_results = {
        (chapter.get("chapter"), bench.get("example")): bench
        for chapter in payload.get("results", [])
        for bench in chapter.get("benchmarks", []) or []
    }

    targets: List[Dict[str, Any]] = []
    for target in suite.targets:
        chapter_key = _chapter_key_from_target(target.target)
        example = _example_from_target(target.target)
        bench = chapter_results.get((chapter_key, example))
        if bench is None:
            targets.append(
                {
                    "key": target.key,
                    "target": target.target,
                    "category": target.category,
                    "status": "missing",
                    "rationale": target.rationale,
                }
            )
            continue

        targets.append(
            {
                "key": target.key,
                "target": target.target,
                "category": target.category,
                "rationale": target.rationale,
                "status": bench.get("status", "unknown"),
                "baseline_time_ms": bench.get("baseline_time_ms"),
                "best_speedup": bench.get("best_speedup"),
                "best_optimized_time_ms": (
                    float(bench.get("baseline_time_ms", 0.0) or 0.0)
                    / float(bench.get("best_speedup", 0.0) or 1.0)
                    if float(bench.get("baseline_time_ms", 0.0) or 0.0) > 0.0
                    and float(bench.get("best_speedup", 0.0) or 0.0) > 0.0
                    else None
                ),
                "best_optimization": _find_best_optimization_name(bench),
                "optimization_goal": bench.get("optimization_goal"),
                "baseline_memory_mb": bench.get("baseline_memory_mb"),
                "best_memory_savings_pct": bench.get("best_memory_savings_pct"),
                "baseline_p75_ms": bench.get("baseline_p75_ms"),
                "baseline_file": bench.get("baseline_file"),
                "artifacts": _artifact_refs(bench),
            }
        )

    speedups = [float(target.get("best_speedup", 0.0) or 0.0) for target in targets if target.get("status") == "succeeded"]
    succeeded = sum(1 for target in targets if target.get("status") == "succeeded")
    failed = sum(1 for target in targets if str(target.get("status", "")).startswith("failed"))
    skipped = sum(1 for target in targets if str(target.get("status", "")).startswith("skipped"))
    missing = sum(1 for target in targets if target.get("status") == "missing")

    return {
        "suite_name": suite.name,
        "suite_version": suite.version,
        "description": suite.description,
        "run_id": run_id,
        "generated_at": payload.get("timestamp"),
        "source_result_json": str(Path(result_json_path)),
        "source_manifest_json": str(manifest_path) if manifest_path else None,
        "source_markdown_report": str(report_path) if report_path else None,
        "targets": targets,
        "summary": {
            "target_count": len(targets),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "missing": missing,
            "avg_speedup": (sum(speedups) / len(speedups)) if speedups else 0.0,
            "median_speedup": statistics.median(speedups) if speedups else 0.0,
            "geomean_speedup": _geometric_mean(speedups),
            "representative_speedup": _geometric_mean(speedups),
            "max_speedup": max(speedups) if speedups else 0.0,
        },
    }


def run_tier1_suite(
    *,
    config_path: Optional[Path] = None,
    history_root: Optional[Path] = None,
    bench_root: Optional[Path] = None,
    profile_type: Optional[str] = None,
    output_format: Optional[str] = None,
    suite_timeout: Optional[int] = 14400,
    timeout_multiplier: float = 3.0,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    reproducible: bool = False,
    cold_start: bool = False,
    force_synchronize: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    gpu_sm_clock_mhz: Optional[int] = None,
    gpu_mem_clock_mhz: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    single_gpu: bool = False,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    pm_sampling_interval: Optional[int] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    torchrun_env: Optional[List[str]] = None,
    target_extra_args: Optional[List[str]] = None,
    verify_input: bool = True,
    verify_output: bool = True,
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
) -> Dict[str, Any]:
    from core.analysis.history_index import load_history_index_with_warnings, update_history_index
    from core.analysis.regressions import compare_suite_summaries, render_regression_summary
    from core.analysis.trends import build_trend_snapshot
    from core.benchmark.bench_commands import _execute_benchmarks

    suite = load_tier1_suite(config_path)
    execution = _execute_benchmarks(
        targets=suite.target_strings(),
        bench_root=bench_root,
        output_format=output_format or suite.default_output_format,
        profile_type=profile_type or suite.default_profile,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        validity_profile=validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        reproducible=reproducible,
        cold_start=cold_start,
        force_synchronize=force_synchronize,
        iterations=iterations,
        warmup=warmup,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        log_level=log_level,
        log_file=log_file,
        single_gpu=single_gpu,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        pm_sampling_interval=pm_sampling_interval,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        torchrun_env=torchrun_env,
        target_extra_args=target_extra_args,
        verify_input=verify_input,
        verify_output=verify_output,
        llm_analysis=llm_analysis,
        force_llm=force_llm,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
        exit_on_failure=False,
    )

    history_root_path = Path(history_root or (_repo_root() / suite.history_root)).resolve()
    suite_run_dir = history_root_path / execution["run_id"]
    suite_run_dir.mkdir(parents=True, exist_ok=True)

    summary = build_tier1_suite_summary(
        Path(execution["output_json"]),
        suite,
        run_id=execution["run_id"],
        manifest_path=Path(execution["manifest_path"]) if execution.get("manifest_path") else None,
        report_path=Path(execution["output_markdown"]) if execution.get("output_markdown") else None,
    )
    summary_path = suite_run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    index_path = history_root_path / "index.json"
    history_warnings: List[str] = []
    previous_index, index_warnings = load_history_index_with_warnings(index_path)
    history_warnings.extend(index_warnings)
    previous_summary = None
    for entry in reversed(previous_index.get("runs", [])):
        previous_summary_path_raw = str(entry.get("summary_path") or "").strip()
        if not previous_summary_path_raw:
            continue
        previous_summary_path = Path(previous_summary_path_raw)
        if previous_summary_path.exists():
            previous_summary, previous_summary_warning = _load_json_object(
                previous_summary_path,
                label="previous tier-1 summary",
            )
            if previous_summary_warning:
                history_warnings.append(previous_summary_warning)
                continue
            break

    regression_summary_path = suite_run_dir / "regression_summary.md"
    comparison = compare_suite_summaries(summary, previous_summary)
    if history_warnings:
        comparison.setdefault("warnings", []).extend(history_warnings)
    comparison = _confirm_speedup_regressions(
        comparison=comparison,
        current_summary=summary,
        previous_summary=previous_summary,
        suite=suite,
        suite_run_dir=suite_run_dir,
        bench_root=bench_root,
        execution_run_id=execution["run_id"],
        profile_type=profile_type or suite.default_profile,
        output_format=output_format or suite.default_output_format,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        validity_profile=validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        reproducible=reproducible,
        cold_start=cold_start,
        force_synchronize=force_synchronize,
        iterations=iterations,
        warmup=warmup,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
        artifacts_dir=artifacts_dir,
        log_level=log_level,
        log_file=log_file,
        single_gpu=single_gpu,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        pm_sampling_interval=pm_sampling_interval,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        torchrun_env=torchrun_env,
        target_extra_args=target_extra_args,
        verify_input=verify_input,
        verify_output=verify_output,
        llm_analysis=llm_analysis,
        force_llm=force_llm,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
    )
    regression_summary_path.write_text(
        render_regression_summary(summary, previous_summary, comparison),
        encoding="utf-8",
    )
    regression_json_path = suite_run_dir / "regression_summary.json"
    regression_json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    updated_index = update_history_index(
        history_root=history_root_path,
        suite=suite,
        summary=summary,
        summary_path=summary_path,
        regression_summary_path=regression_summary_path,
        regression_json_path=regression_json_path,
    )
    run_warnings = list(history_warnings)
    run_warnings.extend(updated_index.get("warnings", []))

    trend_snapshot = build_trend_snapshot(updated_index)
    trend_snapshot_path = suite_run_dir / "trend_snapshot.json"
    trend_snapshot_path.write_text(json.dumps(trend_snapshot, indent=2), encoding="utf-8")

    updated_index = update_history_index(
        history_root=history_root_path,
        suite=suite,
        summary=summary,
        summary_path=summary_path,
        regression_summary_path=regression_summary_path,
        regression_json_path=regression_json_path,
        trend_snapshot_path=trend_snapshot_path,
    )

    return {
        "suite": suite,
        "execution": execution,
        "summary": summary,
        "summary_path": summary_path,
        "regression_summary_path": regression_summary_path,
        "regression_json_path": regression_json_path,
        "trend_snapshot_path": trend_snapshot_path,
        "index": updated_index,
        "history_root": history_root_path,
        "comparison": comparison,
        "warnings": list(dict.fromkeys(run_warnings)),
    }
