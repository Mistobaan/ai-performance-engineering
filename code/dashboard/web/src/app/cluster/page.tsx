'use client';

import { useCallback, useMemo, useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { useToast } from '@/components/Toast';
import { runClusterEvalSuite, validateFieldReport } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { ClusterEvalSuiteResult, FieldReportValidationResult } from '@/types';
import { RefreshCw, FileCheck2, Server } from 'lucide-react';

function splitList(value: string): string[] | undefined {
  const items = value
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0);
  return items.length ? items : undefined;
}

function badge(ok: boolean | undefined) {
  if (ok) return 'badge badge-success';
  return 'badge badge-danger';
}

function codeBlock(text: string | undefined | null) {
  if (!text) return null;
  return (
    <pre className="text-xs text-white/80 bg-black/30 border border-white/10 rounded-lg p-4 overflow-x-auto">
      {text}
    </pre>
  );
}

export default function ClusterPage() {
  const { showToast } = useToast();

  const [mode, setMode] = useState<'smoke' | 'full'>('smoke');
  const [runId, setRunId] = useState<string>('');
  const [hosts, setHosts] = useState<string>('');
  const [labels, setLabels] = useState<string>('');
  const [sshUser, setSshUser] = useState<string>('');
  const [sshKey, setSshKey] = useState<string>('');
  const [socketIfname, setSocketIfname] = useState<string>('');
  const [ncclIbHca, setNcclIbHca] = useState<string>('');
  const [primaryLabel, setPrimaryLabel] = useState<string>('');
  const [extraArgs, setExtraArgs] = useState<string>('');
  const [timeoutSeconds, setTimeoutSeconds] = useState<string>('');

  const [runningSuite, setRunningSuite] = useState(false);
  const [suiteResult, setSuiteResult] = useState<ClusterEvalSuiteResult | null>(null);

  const [reportPath, setReportPath] = useState('cluster/field-report.md');
  const [notesPath, setNotesPath] = useState('cluster/field-report-notes.md');
  const [templatePath, setTemplatePath] = useState('cluster/docs/field-report-template.md');
  const [runbookPath, setRunbookPath] = useState('cluster/docs/advanced-runbook.md');
  const [canonicalRunId, setCanonicalRunId] = useState<string>('');
  const [allowRunId, setAllowRunId] = useState<string>('');
  const [runningValidator, setRunningValidator] = useState(false);
  const [validatorResult, setValidatorResult] = useState<FieldReportValidationResult | null>(null);

  const suiteMeta = useMemo(() => {
    if (!suiteResult) return null;
    const modeValue = suiteResult.mode || '—';
    const idValue = suiteResult.run_id || '—';
    return `${modeValue} • ${idValue}`;
  }, [suiteResult]);

  const runSuite = useCallback(async () => {
    try {
      setRunningSuite(true);
      setSuiteResult(null);
      const params: Record<string, unknown> = {
        mode,
      };
      if (runId.trim()) params.run_id = runId.trim();
      if (timeoutSeconds.trim()) params.timeout_seconds = Number(timeoutSeconds);

      const hostsList = splitList(hosts);
      const labelsList = splitList(labels);
      const extraArgsList = splitList(extraArgs);

      if (hostsList) params.hosts = hostsList;
      if (labelsList) params.labels = labelsList;
      if (sshUser.trim()) params.ssh_user = sshUser.trim();
      if (sshKey.trim()) params.ssh_key = sshKey.trim();
      if (socketIfname.trim()) params.socket_ifname = socketIfname.trim();
      if (ncclIbHca.trim()) params.nccl_ib_hca = ncclIbHca.trim();
      if (primaryLabel.trim()) params.primary_label = primaryLabel.trim();
      if (extraArgsList) params.extra_args = extraArgsList;

      const data = (await runClusterEvalSuite(params)) as ClusterEvalSuiteResult;
      setSuiteResult(data);
      if (data.success) {
        showToast('Cluster eval suite completed successfully.', 'success');
      } else {
        showToast(data.error || 'Cluster eval suite failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Cluster eval suite failed.', 'error');
      setSuiteResult(null);
    } finally {
      setRunningSuite(false);
    }
  }, [
    extraArgs,
    hosts,
    labels,
    mode,
    ncclIbHca,
    primaryLabel,
    runId,
    showToast,
    socketIfname,
    sshKey,
    sshUser,
    timeoutSeconds,
  ]);

  const runValidator = useCallback(async () => {
    try {
      setRunningValidator(true);
      setValidatorResult(null);
      const params: Record<string, unknown> = {};
      if (reportPath.trim()) params.report = reportPath.trim();
      if (notesPath.trim()) params.notes = notesPath.trim();
      if (templatePath.trim()) params.template = templatePath.trim();
      if (runbookPath.trim()) params.runbook = runbookPath.trim();
      if (canonicalRunId.trim()) params.canonical_run_id = canonicalRunId.trim();
      const allow = splitList(allowRunId);
      if (allow) params.allow_run_id = allow;

      const data = (await validateFieldReport(params)) as FieldReportValidationResult;
      setValidatorResult(data);
      if (data.success) {
        showToast('Field report validation passed.', 'success');
      } else {
        showToast(data.error || 'Field report validation failed.', 'warning');
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Field report validation failed.', 'error');
      setValidatorResult(null);
    } finally {
      setRunningValidator(false);
    }
  }, [
    allowRunId,
    canonicalRunId,
    notesPath,
    reportPath,
    runbookPath,
    showToast,
    templatePath,
  ]);

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Cluster evaluation workflows under cluster/ (field report, discovery, reproducibility)."
      actions={
        <button
          onClick={runSuite}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-lg text-sm text-white disabled:opacity-50"
          disabled={runningSuite}
        >
          <RefreshCw className={cn('w-4 h-4', runningSuite ? 'animate-spin' : '')} />
          Run Suite
        </button>
      }
    >
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <Server className="w-5 h-5 text-accent-info" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Eval Suite</h2>
                <p className="text-xs text-white/50">
                  Runs a safe local smoke bundle or the full multi-node suite.
                </p>
              </div>
            </div>
            {suiteResult && (
              <span className={badge(suiteResult.success)}>
                {suiteResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">Mode</div>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value as 'smoke' | 'full')}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                >
                  <option value="smoke">smoke</option>
                  <option value="full">full</option>
                </select>
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">RUN_ID</div>
                <input
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                  placeholder="(optional) default: YYYY-MM-DD"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">Hosts (full mode)</div>
                <input
                  value={hosts}
                  onChange={(e) => setHosts(e.target.value)}
                  placeholder="comma-separated, required for full mode"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">Labels (optional)</div>
                <input
                  value={labels}
                  onChange={(e) => setLabels(e.target.value)}
                  placeholder="comma-separated, same count as hosts"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">SSH User</div>
                <input
                  value={sshUser}
                  onChange={(e) => setSshUser(e.target.value)}
                  placeholder="(optional)"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">SSH Key</div>
                <input
                  value={sshKey}
                  onChange={(e) => setSshKey(e.target.value)}
                  placeholder="(optional) /path/to/key"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">socket_ifname</div>
                <input
                  value={socketIfname}
                  onChange={(e) => setSocketIfname(e.target.value)}
                  placeholder="(optional) e.g. eth0"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">NCCL_IB_HCA</div>
                <input
                  value={ncclIbHca}
                  onChange={(e) => setNcclIbHca(e.target.value)}
                  placeholder="(optional) e.g. mlx5_0"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">primary_label</div>
                <input
                  value={primaryLabel}
                  onChange={(e) => setPrimaryLabel(e.target.value)}
                  placeholder="(optional) for smoke/local steps"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">timeout_seconds</div>
                <input
                  value={timeoutSeconds}
                  onChange={(e) => setTimeoutSeconds(e.target.value)}
                  placeholder="(optional)"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1 lg:col-span-2">
                <div className="text-xs uppercase text-white/40">extra_args (optional)</div>
                <input
                  value={extraArgs}
                  onChange={(e) => setExtraArgs(e.target.value)}
                  placeholder="comma-separated args appended to run_cluster_eval_suite.sh"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
            </div>

            <button
              onClick={runSuite}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningSuite}
            >
              <RefreshCw className={cn('w-4 h-4', runningSuite ? 'animate-spin' : '')} />
              {runningSuite ? 'Running...' : 'Run Eval Suite'}
            </button>

            {suiteResult && (
              <div className="space-y-3">
                <div className="text-xs text-white/50">{suiteMeta}</div>
                {suiteResult.meta_path && (
                  <div className="text-xs text-white/70">
                    meta: <span className="font-mono text-white/90">{suiteResult.meta_path}</span>
                  </div>
                )}
                {suiteResult.manifest_path && (
                  <div className="text-xs text-white/70">
                    manifest: <span className="font-mono text-white/90">{suiteResult.manifest_path}</span>
                  </div>
                )}
                {suiteResult.error && (
                  <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                    {suiteResult.error}
                  </div>
                )}
                {codeBlock(suiteResult.stdout || suiteResult.collect?.stdout)}
                {codeBlock(suiteResult.stderr || suiteResult.collect?.stderr)}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/5 border border-white/10">
                <FileCheck2 className="w-5 h-5 text-accent-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Field Report Validator</h2>
                <p className="text-xs text-white/50">
                  Validates required sections plus artifact hygiene.
                </p>
              </div>
            </div>
            {validatorResult && (
              <span className={badge(validatorResult.success)}>
                {validatorResult.success ? 'OK' : 'FAILED'}
              </span>
            )}
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 gap-4">
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">report</div>
                <input
                  value={reportPath}
                  onChange={(e) => setReportPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">notes</div>
                <input
                  value={notesPath}
                  onChange={(e) => setNotesPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">template</div>
                <input
                  value={templatePath}
                  onChange={(e) => setTemplatePath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">runbook</div>
                <input
                  value={runbookPath}
                  onChange={(e) => setRunbookPath(e.target.value)}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">canonical_run_id (optional)</div>
                <input
                  value={canonicalRunId}
                  onChange={(e) => setCanonicalRunId(e.target.value)}
                  placeholder="RUN_ID expected by the report"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <div className="text-xs uppercase text-white/40">allow_run_id (optional)</div>
                <input
                  value={allowRunId}
                  onChange={(e) => setAllowRunId(e.target.value)}
                  placeholder="comma-separated extra run ids allowed for hygiene checks"
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none"
                />
              </label>
            </div>

            <button
              onClick={runValidator}
              className="btn btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={runningValidator}
            >
              <RefreshCw className={cn('w-4 h-4', runningValidator ? 'animate-spin' : '')} />
              {runningValidator ? 'Validating...' : 'Validate Field Report'}
            </button>

            {validatorResult?.error && (
              <div className="rounded-lg border border-accent-danger/30 bg-accent-danger/10 px-4 py-3 text-sm text-white/80">
                {validatorResult.error}
              </div>
            )}
            {validatorResult && (
              <div className="space-y-3">
                {codeBlock(validatorResult.stdout)}
                {codeBlock(validatorResult.stderr)}
              </div>
            )}
          </div>
        </div>
      </section>
    </DashboardShell>
  );
}
