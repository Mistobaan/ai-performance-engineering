# Cluster Evaluation Harness

This folder has one contract that matters for new work:

```text
cluster/runs/<run_id>/
  manifest.json
  structured/
  raw/
  figures/
  reports/
```

If you are running benchmarks, consuming results from MCP/CLI/dashboard, or handing the output to another tool, use the run directory as the unit of record.

## Start Here
Use this for the common "evaluate this system" ask:

```bash
python -m cli.aisp cluster common-eval \
  --preset common-answer-fast \
  --run-id <run_id> \
  --hosts localhost \
  --labels localhost \
  --ssh-user "$(id -un)" \
  --primary-label localhost \
  --extra-arg --skip-render-localhost-report
```

Presets:
- `common-answer-fast`: the fastest useful answer bundle; NCCL, vLLM concurrency, vLLM request-rate, GEMM, GPU STREAM, fio, quick nvbandwidth, scorecard, coverage, MLPerf alignment.
- `core-system`: the standard full single-node/system bundle.
- `modern-llm`: the canonical full bundle with advanced LLM/distributed signals.
- `multinode-readiness`: checks contract and environment only; no workloads.

## Declarative BenchmarkRuns
When you need the benchmarking service contract instead of just the scripts:

- freeze the workload first with `templates/performance_intake.yaml` and `templates/benchmark_workload_spec.yaml`
- start from `templates/benchmark_run.yaml`
- validate it with `python -m core.scripts.validate_benchmark_run --file templates/benchmark_run.yaml`
- use `cluster/configs/benchmarkrun-crd.yaml` as the Kubernetes CRD shape
- read `cluster/docs/kubernetes_benchmark_service.md` for the intended SLINKY + Kueue operator mapping

That path is for declarative scheduling, CI automation, and publication-vs-realism policy. The existing `common-eval` presets remain the current execution backend.

## What To Open
- `cluster/runs/<run_id>/`: new run results.
- `cluster/published/current/`: the current published canonical package.
- `cluster/field-report.md`: published report package.
- `cluster/field-report-notes.md`: published evidence ledger.
- `cluster/docs/advanced-runbook.md`: operator reference for running, promoting, and validating canonical packages.
- `cluster/docs/field-report-template.md`: promotion template, not the entrypoint.

## What To Ignore Most Of The Time
- `cluster/published/current/structured/`: published structured artifacts for the current canonical package.
- `cluster/published/current/figures/`: published figures for the current canonical package.
- `cluster/published/current/raw/`: raw logs; useful for debugging, not for first-pass consumption.

That published package is for readers and stakeholder deliverables. It is not the forward runtime contract.
Archived historical baselines live under `cluster/archive/runs/<run_id>/`.

## Current Published State
- Published canonical localhost package: `2026-03-05_localhost_modern_profile_r24_full20b`
- Archived comparison baseline: `2026-03-04_localhost_modern_profile_r22_fastcanon` in `cluster/archive/runs/2026-03-04_localhost_modern_profile_r22_fastcanon/`

Interpretation:
- `r24` is the official published package.
- `r22` is preserved only for historical delta work and does not stay in the live publication surfaces.

## Promotion Rules
When a run should become the new published package:
1. Run `common-eval` or the raw suite and inspect `cluster/runs/<run_id>/`.
2. Promote the localhost package into `cluster/published/current/` explicitly:

```bash
python -m cli.aisp cluster promote-run \
  --run-id <run_id> \
  --label localhost \
  --allow-run-id <historical_baseline_if_any> \
  --cleanup
```

3. If this run also becomes the stakeholder report package, sync:
   - `cluster/field-report.md`
   - `cluster/field-report-notes.md`
   - `cluster/docs/field-report-template.md`
   - `cluster/docs/advanced-runbook.md`
4. Re-run `cluster/scripts/validate_field_report_requirements.sh`.

## Multi-Node
Before the first real multi-node run, use the readiness-only preset:

```bash
python -m cli.aisp cluster common-eval \
  --preset multinode-readiness \
  --run-id <run_id> \
  --hosts <h1,h2> \
  --labels <l1,l2> \
  --ssh-user <user> \
  --ssh-key <key> \
  --extra-arg --oob-if \
  --extra-arg <iface> \
  --extra-arg --socket-ifname \
  --extra-arg <iface>
```

That writes:
- `cluster/runs/<run_id>/structured/<run_id>_multinode_readiness.json`

Use `modern-llm` only after readiness is green.

## Optional Export
`python -m cli.aisp cluster build-canonical-package ...` is optional. Most users can ignore it. It only exists to export already-produced runs into a clean shareable bundle after collection/promotion are already done.
