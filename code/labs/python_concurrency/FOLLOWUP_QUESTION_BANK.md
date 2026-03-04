# Follow-Up Question Bank

Use this file as a rapid drilling bank for likely second-order questions.
Each answer is deliberately concise and designed to survive aggressive follow-up.

## Quick Summary

Answer algorithm:
1. State the controlling principle (bounded pressure, idempotency, determinism).
2. Name the mechanism used in code (queue, semaphore, timeout/retry, dedupe lock).
3. Add one metric to validate the claim (queue depth, p95, error rate, retries).

## 1) Runtime and Model Choice

Q: Why not always use `asyncio`?
A: `asyncio` is best for wait-heavy I/O concurrency; CPU-heavy pure-Python work usually needs processes for true parallelism.

Q: If external packages are disallowed and `concurrent.futures` is disallowed, can I use `threading`?
A: Yes. `threading` and `queue` are standard-library modules, so they satisfy stdlib-only constraints.

Q: Why not always use processes?
A: Processes add serialization and IPC overhead, and are unnecessary for simple I/O-bound workloads where `asyncio` is lighter.

Q: How do you defend your runtime choice quickly?
A: State workload classification, expected bottleneck, and one metric you will validate (p95 latency or throughput).

Q: Why does 50x `sleep(1)` finish near 1 second with threads?
A: The tasks are mostly waiting, so threads overlap wait time instead of running sequentially.

Q: What if workload shifts over time?
A: Instrument stage metrics and switch to hybrid architecture when one stage becomes CPU-bound.

## 2) Queueing, Backpressure, and Ordering

Q: What is the first guard against overload?
A: Bounded queues and bounded workers; this creates explicit backpressure.

Q: Why does unbounded fanout fail in practice?
A: It converts load spikes into memory growth and dependency overload.

Q: How do you preserve output order under concurrency?
A: Tag input index and write output into a pre-sized result array at that index.

Q: How do you run 32 jobs with concurrency capped at 4?
A: Enqueue all 32 jobs into `queue.Queue`, run exactly 4 workers, and let workers pull until sentinels arrive.

Q: Why enqueue one sentinel per worker?
A: Each worker needs its own stop token so every worker exits its loop cleanly without hanging.

Q: Is ordered output always required?
A: No; if not required, unordered streaming can reduce tail latency and memory pressure.

Q: What queue metric is most useful?
A: Queue depth trend over time; rising depth with flat completions indicates saturation.

Q: Do you need locks for indexed result writes?
A: Not when each worker owns a unique index and reads happen only after `join()`, but shared counters/maps still need locks.

## 3) Timeouts, Retries, and Failure Taxonomy

Q: What is a retryable error?
A: A transient failure likely to succeed later (e.g., temporary timeout/429/service unavailable).

Q: What is non-retryable?
A: Validation errors, deterministic schema violations, and permanent semantic failures.

Q: Why exponential backoff with jitter?
A: It reduces synchronized retry waves and protects unstable dependencies.

Q: How many retries is reasonable?
A: Keep retries bounded and policy-driven by operation criticality and downstream SLO.

Q: Why both per-attempt timeout and global timeout?
A: Per-attempt timeout bounds single-call stall; global timeout bounds whole batch lifetime.

## 4) Idempotency and Exactly-Once Effects

Q: What is a good `idempotency_key`?
A: Stable across retries for the same logical operation, unique across distinct operations, scoped by tenant/resource.

Q: Example key format?
A: `tenant_id:operation_type:client_request_id[:payload_fingerprint]`.

Q: What is a bad key?
A: Anything that changes on retry, such as newly generated timestamp-only IDs.

Q: Where should dedupe state live?
A: Durable shared store with atomic check-and-commit semantics near the side-effect boundary.

Q: How do you handle same key with different payload?
A: Treat as conflict and reject explicitly; do not apply side effect.

Q: How long should key TTL be?
A: At least max replay window (retry horizon + queue delay + recovery interval).

Q: Can idempotency guarantee exactly-once processing?
A: It provides exactly-once effect at side-effect boundaries over at-least-once delivery.

## 5) Terminal Status and Recovery

Q: Why is `started` not terminal?
A: `started` is transitional; terminal states must be final and immutable outcomes.

Q: How do you detect missing terminal states?
A: Compare `queued/started/completed`, scan for missing/duplicate indexes, and fail summary/export on mismatch.

Q: What if process crashes mid-batch?
A: Reconcile unresolved IDs (`started_ids - terminal_ids`) and replay under idempotency protection.

Q: Should unknown outcomes be dropped?
A: No. Emit deterministic terminal placeholder (`cancelled` with recovery context) and reconcile.

Q: What is the minimal terminal invariant set?
A: One terminal record per input index, no missing records, allowed-status membership only.

## 6) Cancellation and Shutdown

Q: What happens on global stop signal?
A: Stop intake, drain/cancel in-flight by policy, terminalize backlog explicitly, then shutdown workers.

Q: Why sentinel shutdown?
A: Sentinels provide deterministic worker exit without hanging on `queue.get()`.

Q: Why one sentinel per worker?
A: Each worker needs an explicit stop signal to avoid orphan waits.

Q: What is graceful vs hard cancellation?
A: Graceful lets controlled cleanup/terminalization happen; hard cancellation risks partial state and lost outcomes.

## 7) Observability and Tuning

Q: Which metrics are mandatory?
A: Status counts, retries, queue depth, in-flight count, p50/p95/p99 latency.

Q: What indicates host bottleneck vs dependency bottleneck?
A: High queue depth + low dependency utilization suggests host scheduling/admission bottleneck.

Q: How do you tune worker count?
A: Increase incrementally and stop when p95/p99 or error rate worsens.

Q: Why can average latency mislead?
A: Averages hide tail failures; p99 may be broken while mean looks healthy.

Q: What logs should every terminal record include?
A: item ID, status, attempts, latency, error type/message, and key routing fields.

## 8) Docker and Kubernetes

Q: Deployment or Job for workers?
A: `Deployment` for continuous consumers; `Job` for finite work units.

Q: How should probes be configured?
A: `startupProbe` for boot, `readinessProbe` for traffic gating, cautious `livenessProbe` for true stuck conditions.

Q: Why does `CrashLoopBackOff` happen frequently?
A: Startup dependency failures, bad config/env, or premature liveness checks.

Q: Requests vs limits?
A: Requests drive scheduling guarantees; limits constrain runtime and can induce throttling.

Q: How do you shutdown safely on Kubernetes?
A: Handle `SIGTERM`, stop intake, drain terminal states, respect `terminationGracePeriodSeconds`.

## 9) Slurm on Kubernetes

Q: What is the biggest integration risk?
A: Double scheduling ambiguity between Slurm policy and Kubernetes placement.

Q: How to avoid double scheduling?
A: Define one authoritative allocator and align the other layer’s behavior to that contract.

Q: What symptoms indicate integration mismatch?
A: Slurm says allocated while pods remain pending due to taints/labels/resources.

Q: What metrics span both layers?
A: Slurm queue wait, pod pending reasons, GPU utilization, and job completion latency.

## 10) GPU Control-Plane Follow-Ups

Q: What failures are retryable on GPU workloads?
A: Transient launch/runtime/service failures; deterministic OOM should require resource-shaping before retry.

Q: How to handle OOM properly?
A: Reduce batch/sequence size, adjust placement, or route to higher-memory pool before retry.

Q: Why per-GPU slot limits?
A: Prevent over-admission that causes memory thrash and unstable tail latency.

Q: Which GPU metrics matter most for control plane?
A: Utilization, memory pressure, queue wait, launch failure taxonomy, and p95 completion time.

Q: How do you discuss GPU concurrency without kernel coding?
A: Focus on admission, scheduling, retries, terminalization, and observability around GPU-backed tasks.

## 11) Rapid "If Pushed" Responses

Q: "How do you know this is correct?"
A: "Terminal-state invariants + stress tests + replay safety checks prove correctness under concurrency and shutdown paths."

Q: "How do you know this scales?"
A: "Backpressure keeps resource usage bounded; scaling decisions are driven by queue-depth and tail-latency trends."

Q: "What breaks first in production?"
A: "Usually uncontrolled fanout, weak retry taxonomy, or missing shutdown contracts."

Q: "What would you add first for production hardening?"
A: "Durable progress ledger, idempotent replay controls, and SLO-focused telemetry."
