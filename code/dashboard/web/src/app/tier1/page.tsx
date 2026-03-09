'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { DashboardShell } from '@/components/DashboardShell';
import { StatsCard } from '@/components/StatsCard';
import { getTier1History, getTier1TargetHistory, getTier1Trends } from '@/lib/api';
import type { Tier1Delta, Tier1History, Tier1TargetHistory, Tier1TargetSummary, Tier1Trends } from '@/types';
import { BarChart3, Clock3, Gauge, ShieldCheck } from 'lucide-react';

function Tier1Skeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={`tier1-skel-${index}`} className="card p-5 animate-pulse">
            <div className="h-3 w-24 bg-white/10 rounded mb-3" />
            <div className="h-8 w-32 bg-white/10 rounded" />
            <div className="h-3 w-28 bg-white/10 rounded mt-3" />
          </div>
        ))}
      </div>
      <div className="card p-6 animate-pulse">
        <div className="h-4 w-48 bg-white/10 rounded mb-4" />
        <div className="h-56 bg-white/5 rounded" />
      </div>
    </div>
  );
}

function formatSpeedup(value?: number | null) {
  return `${(value || 0).toFixed(2)}x`;
}

function formatTimestamp(value?: string | null) {
  if (!value) return '—';
  return value.replace('T', ' ').replace(/:\d\d(?:\.\d+)?$/, '');
}

function formatDelta(change: Tier1Delta) {
  if (typeof change.delta_pct === 'number') {
    return `${change.delta_pct >= 0 ? '+' : ''}${change.delta_pct.toFixed(2)}%`;
  }
  if (typeof change.delta === 'number') {
    return `${change.delta >= 0 ? '+' : ''}${change.delta.toFixed(2)}`;
  }
  if (change.before !== undefined || change.after !== undefined) {
    return `${String(change.before ?? '—')} -> ${String(change.after ?? '—')}`;
  }
  return change.reason || 'changed';
}

function targetStatusBadge(status: string) {
  if (status === 'succeeded') return 'badge badge-success';
  if (status === 'failed') return 'badge badge-danger';
  if (status === 'skipped') return 'badge badge-warning';
  return 'badge badge-info';
}

export default function Tier1Page() {
  const [history, setHistory] = useState<Tier1History | null>(null);
  const [trends, setTrends] = useState<Tier1Trends | null>(null);
  const [targetHistory, setTargetHistory] = useState<Tier1TargetHistory | null>(null);
  const [selectedTargetKey, setSelectedTargetKey] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [targetLoading, setTargetLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTier1 = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [historyData, trendsData] = await Promise.all([getTier1History(), getTier1Trends()]);
      setHistory(historyData as Tier1History);
      setTrends(trendsData as Tier1Trends);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load tier-1 history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTier1();
  }, [loadTier1]);

  const latestRun = history?.latest?.run ?? null;
  const latestSummary = (history?.latest?.summary ?? {}) as Record<string, number>;
  const latestTargets = useMemo(
    () =>
      [...(history?.latest?.targets ?? [])]
        .sort((a, b) => (b.best_speedup || 0) - (a.best_speedup || 0))
        .slice(0, 8),
    [history?.latest?.targets]
  );
  const chartData = trends?.history ?? [];
  const latestRegressions = (history?.latest?.regressions ?? []).slice(0, 6);
  const latestImprovements = (history?.latest?.improvements ?? []).slice(0, 6);
  const targetOptions = useMemo(
    () =>
      [...(history?.latest?.targets ?? [])].sort((a, b) =>
        String(a.target || a.key || '').localeCompare(String(b.target || b.key || ''))
      ),
    [history?.latest?.targets]
  );
  const targetChartData = targetHistory?.history ?? [];

  useEffect(() => {
    if (!selectedTargetKey && latestTargets.length > 0) {
      setSelectedTargetKey(latestTargets[0].key);
    }
  }, [latestTargets, selectedTargetKey]);

  useEffect(() => {
    if (!selectedTargetKey) {
      setTargetHistory(null);
      return;
    }
    let cancelled = false;
    const loadTarget = async () => {
      try {
        setTargetLoading(true);
        const payload = await getTier1TargetHistory({ key: selectedTargetKey });
        if (!cancelled) {
          setTargetHistory(payload as Tier1TargetHistory);
        }
      } catch (e) {
        if (!cancelled) {
          setTargetHistory(null);
          setError(e instanceof Error ? e.message : 'Failed to load target history');
        }
      } finally {
        if (!cancelled) {
          setTargetLoading(false);
        }
      }
    };
    loadTarget();
    return () => {
      cancelled = true;
    };
  }, [selectedTargetKey]);

  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Canonical tier-1 benchmark history, regressions, and latest snapshot."
      onRefresh={loadTier1}
    >
      {loading ? (
        <Tier1Skeleton />
      ) : error ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">{error}</div>
        </div>
      ) : !history || history.total_runs === 0 ? (
        <div className="card">
          <div className="card-body text-center py-16 text-white/70">
            No canonical tier-1 runs found yet.
          </div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            <StatsCard
              title="Canonical Runs"
              value={history.total_runs}
              subtitle={history.latest_run_id || '—'}
              icon={Clock3}
            />
            <StatsCard
              title="Representative"
              value={formatSpeedup(latestRun?.representative_speedup)}
              subtitle="Latest geomean speedup"
              icon={Gauge}
              variant="success"
            />
            <StatsCard
              title="Median"
              value={formatSpeedup(latestRun?.median_speedup)}
              subtitle="Latest median speedup"
              icon={BarChart3}
            />
            <StatsCard
              title="Best Seen"
              value={formatSpeedup(trends?.best_speedup_seen)}
              subtitle="Across canonical history"
              icon={ShieldCheck}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="card lg:col-span-2">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Canonical Trend</h2>
                <span className="badge badge-info">{chartData.length} runs</span>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="run_id" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }} />
                    <YAxis tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(16, 16, 24, 0.95)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number, name) => [`${value.toFixed(2)}x`, name]}
                    />
                    <Line type="monotone" dataKey="geomean_speedup" name="Geomean" stroke="#00f5d4" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="median_speedup" name="Median" stroke="#f72585" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="max_speedup" name="Max" stroke="#f9c74f" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Latest Snapshot</h2>
                <span className="badge badge-info">{latestRun?.run_id || '—'}</span>
              </div>
              <div className="card-body space-y-3 text-sm text-white/70">
                <div className="flex items-center justify-between">
                  <span>Generated</span>
                  <span className="text-white">{formatTimestamp(latestRun?.generated_at)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Targets</span>
                  <span className="text-white">
                    {latestRun?.succeeded ?? 0}/{latestRun?.target_count ?? 0} green
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Geomean</span>
                  <span className="text-accent-success">{formatSpeedup(latestRun?.geomean_speedup)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Median</span>
                  <span className="text-white">{formatSpeedup(latestRun?.median_speedup)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Average</span>
                  <span className="text-white">{formatSpeedup(latestRun?.avg_speedup)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Regressions</span>
                  <span className="text-accent-danger">{history.latest.regressions.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Improvements</span>
                  <span className="text-accent-success">{history.latest.improvements.length}</span>
                </div>
                <div className="pt-2 border-t border-white/10 text-xs text-white/50 space-y-1">
                  <div>Summary: {latestRun?.summary_path || '—'}</div>
                  <div>Regressions: {latestRun?.regression_summary_path || '—'}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Latest Target Winners</h2>
                <span className="badge badge-success">{latestTargets.length} shown</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5 text-xs uppercase text-white/50">
                      <th className="px-5 py-3 text-left">Target</th>
                      <th className="px-5 py-3 text-right">Category</th>
                      <th className="px-5 py-3 text-right">Speedup</th>
                      <th className="px-5 py-3 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {latestTargets.map((target: Tier1TargetSummary) => (
                      <tr key={target.key} className="border-b border-white/5 text-sm text-white/70">
                        <td className="px-5 py-3">
                          <div className="text-white">{target.key}</div>
                          <div className="text-xs text-white/40">{target.target}</div>
                        </td>
                        <td className="px-5 py-3 text-right">{target.category}</td>
                        <td className="px-5 py-3 text-right text-accent-success">
                          {formatSpeedup(target.best_speedup)}
                        </td>
                        <td className="px-5 py-3 text-right">
                          <span className={targetStatusBadge(target.status)}>{target.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Latest Summary</h2>
                <span className="badge badge-info">{history.suite_name}</span>
              </div>
              <div className="card-body grid grid-cols-2 gap-4 text-sm text-white/70">
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Representative</div>
                  <div className="text-white">{formatSpeedup(Number(latestSummary.representative_speedup || 0))}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Max</div>
                  <div className="text-white">{formatSpeedup(Number(latestSummary.max_speedup || 0))}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Succeeded</div>
                  <div className="text-white">{String(latestSummary.succeeded || 0)}</div>
                </div>
                <div>
                  <div className="text-xs uppercase text-white/40 mb-1">Missing</div>
                  <div className="text-white">{String(latestSummary.missing || 0)}</div>
                </div>
                <div className="col-span-2 pt-2 border-t border-white/10">
                  <div className="text-xs uppercase text-white/40 mb-1">History Root</div>
                  <div className="text-white/60 break-all">{history.history_root || '—'}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Latest Regressions</h2>
                <span className="badge badge-danger">{history.latest.regressions.length}</span>
              </div>
              <div className="card-body space-y-3">
                {latestRegressions.length === 0 ? (
                  <div className="text-sm text-white/50">No tracked regressions in the latest canonical run.</div>
                ) : (
                  latestRegressions.map((change, index) => (
                    <div key={`reg-${change.key || change.target || index}`} className="rounded-lg border border-white/10 p-3">
                      <div className="text-sm text-white">{change.target || change.key || 'unknown target'}</div>
                      <div className="mt-1 text-xs text-white/50">{change.reason || 'regression'}</div>
                      <div className="mt-2 text-sm text-accent-danger">{formatDelta(change)}</div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-white">Latest Improvements</h2>
                <span className="badge badge-success">{history.latest.improvements.length}</span>
              </div>
              <div className="card-body space-y-3">
                {latestImprovements.length === 0 ? (
                  <div className="text-sm text-white/50">No tracked improvements in the latest canonical run.</div>
                ) : (
                  latestImprovements.map((change, index) => (
                    <div key={`imp-${change.key || change.target || index}`} className="rounded-lg border border-white/10 p-3">
                      <div className="text-sm text-white">{change.target || change.key || 'unknown target'}</div>
                      <div className="mt-1 text-xs text-white/50">{change.reason || 'improvement'}</div>
                      <div className="mt-2 text-sm text-accent-success">{formatDelta(change)}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-white">Per-Target History</h2>
              <div className="flex items-center gap-3">
                <span className="badge badge-info">{targetHistory?.run_count ?? 0} runs</span>
                <select
                  className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white"
                  value={selectedTargetKey}
                  onChange={(event) => setSelectedTargetKey(event.target.value)}
                >
                  {targetOptions.map((target) => (
                    <option key={target.key} value={target.key}>
                      {target.target || target.key}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="card-body">
              {targetLoading ? (
                <div className="text-sm text-white/50">Loading target history…</div>
              ) : !targetHistory || targetChartData.length === 0 ? (
                <div className="text-sm text-white/50">No target history available yet.</div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2">
                    <div className="mb-3 text-sm text-white/60">
                      <div className="text-white">{targetHistory.selected_target}</div>
                      <div>{targetHistory.rationale || 'Canonical target history from tier-1 summaries.'}</div>
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={targetChartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="run_id" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }} />
                        <YAxis tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'rgba(16, 16, 24, 0.95)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                          }}
                          formatter={(value: number, name) =>
                            name === 'best_speedup'
                              ? [`${value.toFixed(2)}x`, 'Speedup']
                              : [`${value.toFixed(3)} ms`, name === 'baseline_time_ms' ? 'Baseline' : 'Optimized']
                          }
                        />
                        <Line type="monotone" dataKey="best_speedup" name="best_speedup" stroke="#00f5d4" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="baseline_time_ms" name="baseline_time_ms" stroke="#f72585" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="best_optimized_time_ms" name="best_optimized_time_ms" stroke="#f9c74f" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="space-y-3">
                    <div className="rounded-lg border border-white/10 p-4">
                      <div className="text-xs uppercase text-white/40 mb-1">Best Seen</div>
                      <div className="text-xl text-accent-success">{formatSpeedup(targetHistory.best_speedup_seen)}</div>
                    </div>
                    <div className="rounded-lg border border-white/10 p-4">
                      <div className="text-xs uppercase text-white/40 mb-1">Latest Status</div>
                      <div className="mt-1">
                        <span className={targetStatusBadge(targetHistory.latest?.status || 'unknown')}>
                          {targetHistory.latest?.status || 'unknown'}
                        </span>
                      </div>
                    </div>
                    <div className="rounded-lg border border-white/10 p-4">
                      <div className="text-xs uppercase text-white/40 mb-1">Latest Speedup</div>
                      <div className="text-white">{formatSpeedup(targetHistory.latest?.best_speedup)}</div>
                    </div>
                    <div className="rounded-lg border border-white/10 p-4">
                      <div className="text-xs uppercase text-white/40 mb-1">Latest Optimization</div>
                      <div className="text-white break-words">{targetHistory.latest?.best_optimization || '—'}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </DashboardShell>
  );
}
