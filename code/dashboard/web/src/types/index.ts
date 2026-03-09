export interface Benchmark {
  name: string;
  chapter: string;
  type: string;
  status: 'succeeded' | 'failed' | 'skipped';
  baseline_time_ms: number;
  optimized_time_ms: number;
  speedup: number;
  raw_speedup?: number;
  speedup_capped?: boolean;
  optimization_goal?: 'speed' | 'memory';
  baseline_memory_mb?: number | null;
  optimized_memory_mb?: number | null;
  memory_savings_pct?: number | null;
  p75_ms?: number | null;
  error?: string;
}

export interface BenchmarkSummary {
  total: number;
  succeeded: number;
  failed: number;
  skipped: number;
  avg_speedup: number;
  max_speedup: number;
  min_speedup: number;
}

export interface BenchmarkPagination {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface BenchmarkFilters {
  search?: string | null;
  status?: string[];
  chapter?: string[];
  benchmark?: string | null;
  optimization_goal?: string | null;
  sort_field?: string;
  sort_dir?: string;
}

export interface BenchmarkOverview {
  timestamp?: string;
  summary: BenchmarkSummary;
  status_counts: {
    succeeded: number;
    failed: number;
    skipped: number;
  };
  top_speedups: Benchmark[];
  chapter_stats: Array<{
    chapter: string;
    count: number;
    succeeded: number;
    avg_speedup: number;
    max_speedup: number;
  }>;
}

export interface BenchmarkPage {
  timestamp?: string;
  summary: BenchmarkSummary;
  benchmarks: Benchmark[];
  pagination: BenchmarkPagination;
  filters: BenchmarkFilters;
}

export interface BenchmarkData {
  benchmarks: Benchmark[];
  summary: BenchmarkSummary;
  timestamp: string;
  speedup_cap?: number;
}

export interface GpuInfo {
  name: string;
  memory_total: number;      // MB
  memory_used: number;       // MB
  utilization: number;
  temperature: number;
  temperature_hbm?: number | null;
  power?: number;            // watts
  power_draw?: number;       // legacy key
  power_limit?: number;      // watts
  compute_capability?: string;
  driver_version?: string;
  cuda_version?: string;
  clock_graphics?: number;
  clock_memory?: number;
  fan_speed?: number | null;
  pstate?: string | null;
  live?: boolean;
}

export interface SoftwareInfo {
  python_version: string;
  pytorch_version: string;
  cuda_version: string;
  cudnn_version: string;
  triton_version?: string;
}

export interface LLMAnalysis {
  summary: string;
  key_findings: string[];
  recommendations: string[];
  bottlenecks: Array<{
    name: string;
    severity: 'high' | 'medium' | 'low';
    description: string;
    recommendation: string;
  }>;
}

export interface ProfilerData {
  kernels: Array<{
    name: string;
    duration_ms: number;
    memory_mb: number;
    occupancy: number;
  }>;
  memory_timeline: Array<{
    timestamp: number;
    allocated_mb: number;
    reserved_mb: number;
  }>;
  flame_data?: unknown;
}

export interface ProfilePair {
  chapter: string;
  name: string;
  path: string;
  run_id: string;
  type?: string;
  has_nsys: boolean;
  has_ncu: boolean;
  baseline_nsys: string[];
  optimized_nsys: string[];
  baseline_ncu: string[];
  optimized_ncu: string[];
}

export interface ProfilePairsResult {
  pairs: ProfilePair[];
  count: number;
}

export interface CompileAnalysisBenchmark {
  name: string;
  chapter: string;
  speedup: number;
  baseline_time_ms: number;
  optimized_time_ms: number;
}

export interface CompileAnalysis {
  speedup: number;
  compile_time_ms: number;
  graph_breaks: number;
  fusion_ratio: number;
  recommendations: string[];
  compile_benchmarks: CompileAnalysisBenchmark[];
  has_real_data: boolean;
  mode_comparison?: Record<string, unknown>;
  graph_breaks_list?: unknown[];
}

export interface NcuKernelRow {
  id: number;
  kernel_name: string;
  block_size: string;
  grid_size: string;
  stream: string;
  device: string;
  cc: string;
  time_avg_ms?: number | null;
  time_sum_ms?: number | null;
  time_pct?: number | null;
  occupancy_limit_reason?: string | null;
  metrics: Record<string, number>;
}

export interface NcuSummaryResult {
  success: boolean;
  report_path?: string;
  top_k?: number;
  sort_by?: string;
  kernel_count?: number;
  total_time_sum_ms?: number | null;
  kernels?: NcuKernelRow[];
  metrics_requested?: string[];
  command?: string[] | null;
  stderr?: string | null;
  returncode?: number | null;
  error?: string;
}

export interface ClockLockSnapshot {
  app_sm_mhz?: number | null;
  app_mem_mhz?: number | null;
  cur_sm_mhz?: number | null;
  cur_mem_mhz?: number | null;
  error?: string;
}

export interface ClockLockResultRow {
  device: number;
  physical_index: number;
  locked: boolean;
  error?: string;
  physical_index_error?: string;
  theoretical_tflops_fp16?: number;
  theoretical_gbps?: number;
  before?: ClockLockSnapshot;
  during?: ClockLockSnapshot;
  after?: ClockLockSnapshot;
}

export interface ClockLockCheckResult {
  success: boolean;
  gpu_count?: number;
  results?: ClockLockResultRow[];
  error?: string;
}

export interface CommandRunResult {
  command: string[];
  returncode: number | null;
  stdout: string;
  stderr: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterEvalSuiteResult {
  success: boolean;
  mode?: string;
  run_id?: string;
  run_dir?: string;
  structured_dir?: string;
  raw_dir?: string;
  figures_dir?: string;
  reports_dir?: string;
  primary_label?: string;
  meta_path?: string;
  manifest_path?: string | null;
  collect?: CommandRunResult;
  manifest?: CommandRunResult;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterCommonEvalResult extends ClusterEvalSuiteResult {
  preset?: string;
  preset_description?: string;
  artifact_roles?: string[];
  coverage_baseline_run_id?: string | null;
}

export interface ClusterCanonicalPackageResult {
  success: boolean;
  canonical_run_id?: string;
  comparison_run_ids?: string[];
  historical_run_ids?: string[];
  output_dir?: string;
  package_readme_path?: string;
  package_manifest_path?: string;
  cleanup_keep_run_ids_path?: string;
  historical_reference_path?: string;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface ClusterPromoteRunResult {
  success: boolean;
  run_id?: string;
  label?: string;
  run_dir?: string;
  published_root?: string;
  published_structured_dir?: string;
  published_raw_dir?: string;
  published_figures_dir?: string;
  published_reports_dir?: string;
  published_manifest_path?: string;
  published_localhost_report_path?: string;
  published_localhost_notes_path?: string;
  allow_run_ids?: string[];
  steps?: Record<string, unknown>;
  command?: string[];
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface FieldReportValidationResult {
  success: boolean;
  returncode?: number | null;
  stdout?: string;
  stderr?: string;
  duration_ms?: number;
  error?: string;
}

export interface BenchmarkRunSummary {
  date: string;
  timestamp: string;
  benchmark_count: number;
  avg_speedup: number;
  max_speedup: number;
  successful: number;
  failed: number;
  source: string;
}

export interface BenchmarkHistory {
  total_runs: number;
  latest?: string | null;
  runs: BenchmarkRunSummary[];
}

export interface BenchmarkTrendPoint {
  date: string;
  avg_speedup: number;
  max_speedup: number;
  benchmark_count: number;
}

export interface BenchmarkTrends {
  history: BenchmarkTrendPoint[];
  by_date: BenchmarkTrendPoint[];
  avg_speedup: number;
  run_count: number;
  best_ever?: {
    date?: string;
    speedup?: number;
  };
  improvements?: Array<{
    date: string;
    delta: number;
  }>;
  regressions?: Array<{
    date: string;
    delta: number;
  }>;
}

export interface Tier1RunSummary {
  run_id: string;
  generated_at?: string | null;
  suite_name?: string;
  suite_version?: number;
  target_count: number;
  succeeded: number;
  failed: number;
  skipped: number;
  missing: number;
  avg_speedup: number;
  median_speedup: number;
  geomean_speedup: number;
  representative_speedup: number;
  max_speedup: number;
  summary_path?: string | null;
  regression_summary_path?: string | null;
  regression_summary_json_path?: string | null;
  trend_snapshot_path?: string | null;
  source_result_json?: string | null;
  improvement_count?: number;
  regression_count?: number;
}

export interface Tier1TargetSummary {
  key: string;
  target: string;
  category: string;
  rationale?: string;
  status: string;
  baseline_time_ms?: number | null;
  best_speedup?: number | null;
  best_optimized_time_ms?: number | null;
  best_optimization?: string | null;
  optimization_goal?: string | null;
  baseline_memory_mb?: number | null;
  best_memory_savings_pct?: number | null;
  baseline_p75_ms?: number | null;
  baseline_file?: string | null;
  artifacts?: Record<string, string>;
}

export interface Tier1LatestRunDetails {
  run: Tier1RunSummary | null;
  summary: Record<string, unknown>;
  targets: Tier1TargetSummary[];
  regressions: Tier1Delta[];
  improvements: Tier1Delta[];
  new_targets: Tier1Delta[];
  missing_targets: Tier1Delta[];
}

export interface Tier1Delta {
  key?: string;
  target?: string;
  reason?: string;
  before?: string | number | null;
  after?: string | number | null;
  delta?: number | null;
  delta_pct?: number | null;
}

export interface Tier1History {
  suite_name: string;
  suite_version?: number;
  history_root?: string;
  total_runs: number;
  latest_run_id?: string | null;
  runs: Tier1RunSummary[];
  latest: Tier1LatestRunDetails;
}

export interface Tier1TrendPoint {
  run_id: string;
  generated_at: string;
  avg_speedup: number;
  median_speedup: number;
  geomean_speedup: number;
  representative_speedup: number;
  max_speedup: number;
  succeeded: number;
  failed: number;
  skipped: number;
  missing: number;
}

export interface Tier1Trends {
  suite_name: string;
  run_count: number;
  latest_run_id?: string | null;
  history: Tier1TrendPoint[];
  avg_speedup: number;
  avg_median_speedup?: number;
  avg_geomean_speedup?: number;
  representative_speedup?: number;
  best_speedup_seen?: number;
}

export interface Tier1TargetHistoryPoint {
  run_id: string;
  generated_at?: string | null;
  key?: string | null;
  target?: string | null;
  category?: string | null;
  status: string;
  baseline_time_ms?: number | null;
  best_optimized_time_ms?: number | null;
  best_speedup?: number | null;
  best_optimization?: string | null;
  baseline_memory_mb?: number | null;
  best_memory_savings_pct?: number | null;
  artifacts?: Record<string, string>;
}

export interface Tier1TargetHistory {
  suite_name: string;
  suite_version?: number;
  history_root?: string;
  selected_key?: string | null;
  selected_target?: string | null;
  category?: string | null;
  rationale?: string | null;
  run_count: number;
  best_speedup_seen?: number;
  latest?: Tier1TargetHistoryPoint | null;
  history: Tier1TargetHistoryPoint[];
}

export interface BenchmarkCompareDelta {
  key: string;
  chapter: string;
  name: string;
  baseline_speedup: number;
  candidate_speedup: number;
  delta: number;
  delta_pct?: number | null;
  baseline_status: string;
  candidate_status: string;
  status_changed: boolean;
  baseline_time_ms?: number | null;
  candidate_time_ms?: number | null;
  baseline_optimized_time_ms?: number | null;
  candidate_optimized_time_ms?: number | null;
}

export interface BenchmarkCompareRun {
  path: string;
  timestamp?: string | null;
  summary: BenchmarkSummary;
}

export interface BenchmarkCompareBenchmark {
  key: string;
  chapter: string;
  name: string;
  status: string;
  speedup: number;
  baseline_time_ms?: number | null;
  optimized_time_ms?: number | null;
  optimization_goal?: string | null;
}

export interface BenchmarkCompareResult {
  baseline: BenchmarkCompareRun;
  candidate: BenchmarkCompareRun;
  overlap: {
    common: number;
    added: number;
    removed: number;
    baseline_total: number;
    candidate_total: number;
  };
  deltas: BenchmarkCompareDelta[];
  regressions: BenchmarkCompareDelta[];
  improvements: BenchmarkCompareDelta[];
  added_benchmarks: BenchmarkCompareBenchmark[];
  removed_benchmarks: BenchmarkCompareBenchmark[];
  status_transitions: Record<string, number>;
}

export interface Tab {
  id: string;
  label: string;
  icon: string;
}
