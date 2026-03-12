'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Braces,
  Copy,
  Download,
  FileCode2,
  FileText,
  Layers3,
  Link2,
  RefreshCw,
  Waypoints,
  WandSparkles,
} from 'lucide-react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { getBenchmarkContracts } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { BenchmarkContractEntry, BenchmarkContractsSummary } from '@/types';

type GeneratorForm = {
  name: string;
  benchmarkClass: 'publication_grade' | 'realism_grade';
  workloadType: 'training' | 'inference' | 'mixed';
  schedulerPath: string;
  cadence: 'canary' | 'nightly' | 'pre_release';
  model: string;
  precision: string;
  batchingPolicy: string;
  concurrencyModel: string;
  comparisonVariable:
    | 'hardware_generation'
    | 'runtime_version'
    | 'scheduler_path'
    | 'control_plane_path'
    | 'driver_stack'
    | 'network_topology'
    | 'storage_stack';
};

const DEFAULT_FORM: GeneratorForm = {
  name: 'publication-inference-stack-b200',
  benchmarkClass: 'publication_grade',
  workloadType: 'inference',
  schedulerPath: 'slinky-kueue',
  cadence: 'pre_release',
  model: 'openai/gpt-oss-20b',
  precision: 'bf16',
  batchingPolicy: 'continuous',
  concurrencyModel: 'closed_loop',
  comparisonVariable: 'runtime_version',
};

function contractIcon(kind: BenchmarkContractEntry['kind']) {
  return kind === 'yaml' ? FileCode2 : FileText;
}

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'benchmark-run';
}

function yamlString(value: string) {
  return JSON.stringify(value);
}

function buildBenchmarkRunYaml(form: GeneratorForm) {
  const trainingEnabled = form.workloadType === 'training' || form.workloadType === 'mixed';
  const inferenceEnabled = form.workloadType === 'inference' || form.workloadType === 'mixed';
  const realismMultiTenant = form.benchmarkClass === 'realism_grade';

  return `apiVersion: benchmarking.aisp.dev/v1alpha1
kind: BenchmarkRun
metadata:
  name: ${slugify(form.name)}
  labels:
    aisp.dev/benchmark-class: ${form.benchmarkClass}
    aisp.dev/owner: performance-engineering
spec:
  intent:
    benchmarkClass: ${form.benchmarkClass}
    workloadType: ${form.workloadType}
    schedulerPath: ${yamlString(form.schedulerPath)}
    cadence: ${form.cadence}
  layers:
    - name: micro
      enabled: true
      objective: Isolate subsystem ceilings and regressions before rolling up to user-visible behavior.
      suites:
        - name: nccl-allreduce
          repoCommand: cluster/scripts/run_allreduce_stability.sh --run-id \${RUN_ID}
    - name: component
      enabled: true
      objective: Measure serving, data pipeline, and control-plane subsystems with stable workload specs.
      suites:
        - name: stack-under-test
          repoCommand: python -m cli.aisp cluster common-eval --preset common-answer-fast
    - name: end_to_end
      enabled: true
      objective: Validate realistic workflows after micro and component bottlenecks are understood.
      suites:
        - name: customer-workflow
          repoCommand: python -m cli.aisp cluster common-eval --preset modern-llm
  workload:
    model: ${yamlString(form.model)}
    sequenceLengthMix:
      - inputTokens: 512
        outputTokens: 128
        weight: 0.7
      - inputTokens: 1024
        outputTokens: 256
        weight: 0.3
    precision: ${yamlString(form.precision)}
    batchingPolicy: ${yamlString(form.batchingPolicy)}
    concurrencyModel: ${yamlString(form.concurrencyModel)}
    datasetRef: ${yamlString('eval_datasets/README.md')}
    fixedAcrossTrials:
      - model
      - sequenceLengthMix
      - precision
      - batchingPolicy
      - concurrencyModel
  comparison:
    variableUnderTest: ${form.comparisonVariable}
    baseline:
      artifactRef: ${yamlString('cluster/published/current')}
      description: ${yamlString('currently published canonical package')}
    candidate:
      artifactRef: ${yamlString('cluster/runs/${RUN_ID}')}
      description: ${yamlString('run under test')}
    controls:
      fixed:
        model: ${yamlString(form.model)}
        sequenceLengthMix: ${yamlString('[{512/128@0.7},{1024/256@0.3}]')}
        precision: ${yamlString(form.precision)}
        batchingPolicy: ${yamlString(form.batchingPolicy)}
        concurrencyModel: ${yamlString(form.concurrencyModel)}
      compareOneVariableAtATime: true
  metrics:
    training:
      enabled: ${trainingEnabled}
      primary:
        - time_to_train_hours
        - mfu_pct
        - scaling_efficiency_pct
        - training_reliability_pct
    inference:
      enabled: ${inferenceEnabled}
      primary:
        - ttft_ms
        - tokens_per_second
        - p99_latency_ms
        - jitter_ms
        - cost_per_request_usd
        - cost_per_token_usd
  trials:
    minReplicates: 5
    confidenceLevel: 0.95
    outlierPolicy:
      method: mad
      action: flag_and_keep
    report:
      distributions: true
      confidenceIntervals: true
      rankLevelOutliers: true
  bottleneckAnalysis:
    taxonomy:
      - compute_bound
      - comm_bound
      - input_bound
      - control_plane_bound
    decomposeOverheads:
      - compute
      - communication
      - storage
      - orchestration
    instrumentation:
      compute_bound:
        - model_server_metrics
        - gpu_counters
        - ncu
      comm_bound:
        - nccl_traces
        - rdma_probes
        - nvlink_exporter
      input_bound:
        - storage_probes
        - dataloader_metrics
        - network_probes
      control_plane_bound:
        - scheduler_timing
        - queue_depth
        - job_startup_trace
  distributed:
    requireRankLevelVisibility: true
    collectives:
      - latency
      - bandwidth
      - outliers
    nodeDiagnosis:
      validate:
        - rdma_pathing
        - gpu_nic_affinity
        - pcie_link_health
        - nvlink_health
        - thermal_throttling
      remediation:
        - isolate_bad_node
        - cordon_problematic_node
  observability:
    correlation:
      stableJoinKeys:
        - run_id
        - benchmark_case_id
        - scheduler_run_id
        - job_uid
        - pod_uid
        - node_name
        - gpu_uuid
        - rank_id
        - request_id
        - trace_id
      publishedNumberLineage:
        rawArtifactManifestDigest: true
        warehouseRowLineage: true
        querySpecCaptured: true
    telemetrySources:
      service:
        - model_server_metrics
        - request_traces
        - queue_time
      infrastructure:
        - dcgm-exporter
        - nvlink-exporter
        - node-pci-exporter
        - ping-exporter
        - node-problem-detector
        - hpc-verification
      scheduler:
        - kubernetes_events
        - kueue
        - slinky
    scenarioPlaybooks:
      - tail_latency_regression
      - low_gpu_utilization
      - distributed_straggler
      - scheduler_backpressure
  provenance:
    capture:
      pinnedWorkloadSpec: true
      imageDigest: true
      driverCudaNcclRuntimeVersions: true
      hardwareTopology: true
      immutableRawArtifacts: true
      auditTrail: true
    signing:
      required: true
      backend: sigstore
      attestationFormat: in_toto
  executionPolicy:
    publicationGrade:
      dedicatedNodes: true
      stableBackgroundLoad: true
      fixedTopology: true
      topologyExclusiveScheduling: true
    realismGrade:
      multiTenantScenarios: ${realismMultiTenant}
      captureClusterContext: true
  sinks:
    rawArtifacts:
      store: object_storage
      pathTemplate: ${yamlString('s3://benchmarks/raw/${RUN_ID}/')}
      retentionClass: cold_immutable
      artifacts:
        - logs
        - traces
        - profiler_reports
        - manifests
    hotMetrics:
      store: prometheus
      retentionDays: 30
      cardinalityBudget:
        maxActiveSeries: 2000000
        dropDimensions:
          - request_id
          - trace_id
    curatedWarehouse:
      store: parquet_duckdb
      layout:
        facts:
          - benchmark_run_fact
          - serving_outcome_fact
          - training_outcome_fact
          - telemetry_slice_fact
        dimensions:
          - software_version_dim
          - hardware_topology_dim
          - cluster_region_dim
          - workload_dim
          - artifact_lineage_dim
      retention:
        hotDays: 30
        warmDays: 180
        coldDays: 730
      lineage:
        publishedNumbersTraceableToRaw: true
        manifestDigestColumn: true
        workloadSpecDigestColumn: true
  automation:
    ci:
      canary: true
      nightly: true
      preRelease: true
`;
}

function ContractsSkeleton() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={`contracts-skel-${index}`} className="card p-5 animate-pulse">
          <div className="h-4 w-28 bg-white/10 rounded mb-4" />
          <div className="h-3 w-full bg-white/10 rounded mb-2" />
          <div className="h-3 w-5/6 bg-white/10 rounded mb-5" />
          <div className="space-y-2">
            <div className="h-7 bg-white/5 rounded-lg" />
            <div className="h-7 bg-white/5 rounded-lg" />
          </div>
        </div>
      ))}
    </div>
  );
}

function SurfaceCard({
  entry,
  onCopyPath,
  onCopyLink,
}: {
  entry: BenchmarkContractEntry;
  onCopyPath: (path: string) => void;
  onCopyLink: (anchor: string) => void;
}) {
  const Icon = contractIcon(entry.kind);
  const summary = entry.summary;
  const anchor = `contract-${entry.name}`;

  return (
    <div id={anchor} className="card scroll-mt-24">
      <div className="card-header items-start gap-3">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-white/5 border border-white/10">
            <Icon className="w-4 h-4 text-accent-primary" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-base font-semibold text-white">{entry.name}</h2>
              <span className={cn('badge', entry.exists ? 'badge-success' : 'badge-danger')}>
                {entry.exists ? entry.kind : 'missing'}
              </span>
            </div>
            <p className="text-sm text-white/60 mt-1">{entry.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onCopyPath(entry.path)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
          >
            <span className="inline-flex items-center gap-2">
              <Copy className="w-3.5 h-3.5" />
              Path
            </span>
          </button>
          <button
            onClick={() => onCopyLink(anchor)}
            className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
          >
            <span className="inline-flex items-center gap-2">
              <Link2 className="w-3.5 h-3.5" />
              Link
            </span>
          </button>
        </div>
      </div>
      <div className="card-body space-y-4">
        <div className="rounded-lg border border-white/10 bg-black/20 p-3">
          <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Path</div>
          <div className="font-mono text-xs text-white/75 break-all">{entry.path}</div>
        </div>

        {summary && (
          <div className="space-y-3">
            <div>
              <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Top-level keys</div>
              <div className="flex flex-wrap gap-2">
                {summary.top_level_keys.map((key) => (
                  <span key={`${entry.name}-top-${key}`} className="badge badge-info">
                    {key}
                  </span>
                ))}
              </div>
            </div>

            {summary.spec_keys && summary.spec_keys.length > 0 && (
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Spec keys</div>
                <div className="flex flex-wrap gap-2">
                  {summary.spec_keys.map((key) => (
                    <span key={`${entry.name}-spec-${key}`} className="badge badge-warning">
                      {key}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {(summary.enabled_layers || summary.has_observability || summary.has_sinks) && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Layers</div>
                  <div className="text-sm text-white/80">
                    {summary.enabled_layers?.length ? summary.enabled_layers.join(', ') : '—'}
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Observability</div>
                  <div className={cn('text-sm font-medium', summary.has_observability ? 'text-accent-success' : 'text-white/50')}>
                    {summary.has_observability ? 'present' : '—'}
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Sinks</div>
                  <div className={cn('text-sm font-medium', summary.has_sinks ? 'text-accent-success' : 'text-white/50')}>
                    {summary.has_sinks ? 'present' : '—'}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ContractsPage() {
  const { showToast } = useToast();
  const [data, setData] = useState<BenchmarkContractsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<GeneratorForm>(DEFAULT_FORM);

  const yamlPreview = useMemo(() => buildBenchmarkRunYaml(form), [form]);

  const loadContracts = useCallback(async (isRefresh = false) => {
    try {
      if (!isRefresh) setLoading(true);
      setError(null);
      const payload = await getBenchmarkContracts();
      setData(payload as BenchmarkContractsSummary);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load benchmark contracts.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadContracts();
  }, [loadContracts]);

  useEffect(() => {
    if (!data || typeof window === 'undefined' || !window.location.hash) return;
    const target = document.getElementById(window.location.hash.slice(1));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [data]);

  const copyText = useCallback(
    async (text: string, successMessage: string) => {
      try {
        await navigator.clipboard.writeText(text);
        showToast(successMessage, 'success');
      } catch (e) {
        showToast(e instanceof Error ? e.message : 'Failed to copy.', 'error');
      }
    },
    [showToast]
  );

  const copyDeepLink = useCallback(
    async (anchor: string) => {
      const url = `${window.location.origin}/contracts#${anchor}`;
      await copyText(url, 'Deep link copied to clipboard.');
    },
    [copyText]
  );

  const downloadYaml = useCallback(() => {
    const blob = new Blob([yamlPreview], { type: 'text/yaml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slugify(form.name)}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('BenchmarkRun YAML downloaded.', 'success');
  }, [form.name, showToast, yamlPreview]);

  const entries = useMemo(() => Object.values(data?.contracts || {}), [data?.contracts]);
  const docCount = entries.filter((entry) => entry.kind === 'doc').length;
  const yamlCount = entries.filter((entry) => entry.kind === 'yaml').length;

  const actions = (
    <button
      onClick={() => {
        setRefreshing(true);
        loadContracts(true);
      }}
      className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
      disabled={refreshing}
    >
      <RefreshCw className={cn('w-4 h-4', refreshing ? 'animate-spin' : '')} />
      Refresh
    </button>
  );

  return (
    <DashboardShell
      title="Benchmark Contracts"
      subtitle="Methodology, warehouse, BenchmarkRun, and interface surfaces in one place."
      actions={actions}
    >
      {loading && !data ? (
        <ContractsSkeleton />
      ) : error && !data ? (
        <div className="card">
          <div className="card-body py-16 text-center">
            <p className="text-lg text-white/80">Failed to load contract surfaces.</p>
            <p className="text-sm text-white/50 mt-2">{error}</p>
          </div>
        </div>
      ) : (
        <>
          <section className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            <div id="interfaces" className="card xl:col-span-2 scroll-mt-24">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Waypoints className="w-5 h-5 text-accent-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Interface Surfaces</h2>
                    <p className="text-xs text-white/50">One shared contract, three thin entrypoints.</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-success">{data?.available ? 'available' : 'offline'}</span>
                  <button
                    onClick={() => copyDeepLink('interfaces')}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Link2 className="w-3.5 h-3.5" />
                      Link
                    </span>
                  </button>
                </div>
              </div>
              <div className="card-body grid grid-cols-1 gap-3">
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">CLI</div>
                      <div className="font-mono text-sm text-white/85">{data?.interfaces.cli}</div>
                    </div>
                    <button
                      onClick={() => copyText(data?.interfaces.cli || '', 'CLI command copied.')}
                      className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Dashboard API</div>
                      <div className="font-mono text-sm text-white/85">{data?.interfaces.dashboard_api}</div>
                    </div>
                    <button
                      onClick={() => copyText(data?.interfaces.dashboard_api || '', 'Dashboard API route copied.')}
                      className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">MCP</div>
                      <div className="font-mono text-sm text-white/85">{data?.interfaces.mcp_tool}</div>
                    </div>
                    <button
                      onClick={() => copyText(data?.interfaces.mcp_tool || '', 'MCP tool name copied.')}
                      className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Layers3 className="w-5 h-5 text-accent-secondary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Surface Count</h2>
                    <p className="text-xs text-white/50">Current exposed contract files.</p>
                  </div>
                </div>
              </div>
              <div className="card-body space-y-4">
                <div>
                  <div className="text-3xl font-semibold text-white">{entries.length}</div>
                  <div className="text-sm text-white/50">total surfaces</div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xl font-semibold text-white">{docCount}</div>
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mt-1">Docs</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                    <div className="text-xl font-semibold text-white">{yamlCount}</div>
                    <div className="text-xs text-white/50 uppercase tracking-[0.2em] mt-1">YAML</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <Braces className="w-5 h-5 text-accent-warning" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Repo Root</h2>
                    <p className="text-xs text-white/50">Path used by the shared contract surface.</p>
                  </div>
                </div>
              </div>
              <div className="card-body">
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="font-mono text-xs text-white/75 break-all">{data?.repo_root}</div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div id="generator" className="card scroll-mt-24">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <WandSparkles className="w-5 h-5 text-accent-success" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">BenchmarkRun Generator</h2>
                    <p className="text-xs text-white/50">Small form, valid YAML, copy/download ready.</p>
                  </div>
                </div>
                <button
                  onClick={() => copyDeepLink('generator')}
                  className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                >
                  <span className="inline-flex items-center gap-2">
                    <Link2 className="w-3.5 h-3.5" />
                    Link
                  </span>
                </button>
              </div>
              <div className="card-body space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Run Name</div>
                    <input
                      value={form.name}
                      onChange={(e) => setForm((current) => ({ ...current, name: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Benchmark Class</div>
                    <select
                      value={form.benchmarkClass}
                      onChange={(e) => setForm((current) => ({ ...current, benchmarkClass: e.target.value as GeneratorForm['benchmarkClass'] }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      <option value="publication_grade">publication_grade</option>
                      <option value="realism_grade">realism_grade</option>
                    </select>
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Workload Type</div>
                    <select
                      value={form.workloadType}
                      onChange={(e) => setForm((current) => ({ ...current, workloadType: e.target.value as GeneratorForm['workloadType'] }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      <option value="inference">inference</option>
                      <option value="training">training</option>
                      <option value="mixed">mixed</option>
                    </select>
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Cadence</div>
                    <select
                      value={form.cadence}
                      onChange={(e) => setForm((current) => ({ ...current, cadence: e.target.value as GeneratorForm['cadence'] }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      <option value="canary">canary</option>
                      <option value="nightly">nightly</option>
                      <option value="pre_release">pre_release</option>
                    </select>
                  </label>
                  <label className="space-y-1 md:col-span-2">
                    <div className="text-xs uppercase text-white/40">Scheduler Path</div>
                    <input
                      value={form.schedulerPath}
                      onChange={(e) => setForm((current) => ({ ...current, schedulerPath: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1 md:col-span-2">
                    <div className="text-xs uppercase text-white/40">Model</div>
                    <input
                      value={form.model}
                      onChange={(e) => setForm((current) => ({ ...current, model: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Precision</div>
                    <input
                      value={form.precision}
                      onChange={(e) => setForm((current) => ({ ...current, precision: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Batching Policy</div>
                    <input
                      value={form.batchingPolicy}
                      onChange={(e) => setForm((current) => ({ ...current, batchingPolicy: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Concurrency Model</div>
                    <input
                      value={form.concurrencyModel}
                      onChange={(e) => setForm((current) => ({ ...current, concurrencyModel: e.target.value }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    />
                  </label>
                  <label className="space-y-1">
                    <div className="text-xs uppercase text-white/40">Variable Under Test</div>
                    <select
                      value={form.comparisonVariable}
                      onChange={(e) => setForm((current) => ({ ...current, comparisonVariable: e.target.value as GeneratorForm['comparisonVariable'] }))}
                      className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                    >
                      <option value="hardware_generation">hardware_generation</option>
                      <option value="runtime_version">runtime_version</option>
                      <option value="scheduler_path">scheduler_path</option>
                      <option value="control_plane_path">control_plane_path</option>
                      <option value="driver_stack">driver_stack</option>
                      <option value="network_topology">network_topology</option>
                      <option value="storage_stack">storage_stack</option>
                    </select>
                  </label>
                </div>

                <div className="rounded-lg border border-accent-info/20 bg-accent-info/10 px-4 py-3 text-sm text-white/75">
                  This is a dense starter that stays valid against the current repo contract. It is meant to be copied, then tightened against the actual workload and cluster before submission.
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                    <FileCode2 className="w-5 h-5 text-accent-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Generated YAML</h2>
                    <p className="text-xs text-white/50">Copy, download, or apply from here.</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => copyText(yamlPreview, 'BenchmarkRun YAML copied.')}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Copy className="w-3.5 h-3.5" />
                      Copy YAML
                    </span>
                  </button>
                  <button
                    onClick={downloadYaml}
                    className="px-3 py-2 rounded-lg border border-white/10 bg-white/5 text-xs text-white/80 hover:bg-white/10"
                  >
                    <span className="inline-flex items-center gap-2">
                      <Download className="w-3.5 h-3.5" />
                      Download
                    </span>
                  </button>
                </div>
              </div>
              <div className="card-body space-y-4">
                <div className="rounded-lg border border-white/10 bg-black/40 p-4">
                  <pre className="font-mono text-xs text-white/80 overflow-x-auto whitespace-pre-wrap">{yamlPreview}</pre>
                </div>
                <div className="rounded-lg border border-white/10 bg-black/20 p-4">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-white/40 mb-2">Suggested next command</div>
                  <div className="font-mono text-xs text-white/80 break-all">
                    kubectl apply -f {slugify(form.name)}.yaml
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {entries.map((entry) => (
              <SurfaceCard
                key={entry.name}
                entry={entry}
                onCopyPath={(path) => copyText(path, 'Contract path copied.')}
                onCopyLink={copyDeepLink}
              />
            ))}
          </section>
        </>
      )}
    </DashboardShell>
  );
}
