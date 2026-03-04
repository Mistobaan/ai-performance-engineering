# Advanced Scenarios

This file extends the core scripts with additional scenario patterns that
commonly appear in concurrency/system-design coding tasks.

## Quick Summary

Use each scenario as a repeatable algorithm:
1. Identify the pressure point (memory, latency tail, retries, shutdown).
2. Apply bounded controls (queues, workers, rate limits, lock ordering).
3. Define explicit terminal outcomes and replay strategy.
4. Prove behavior with stage metrics and invariant checks.

## Scenario 1: Massive Stream Processing

Goal:
- Process millions of items without full in-memory materialization.

Design:
1. Stream/chunk input.
2. Use bounded queues (`maxsize`) for explicit backpressure.
3. Keep fixed workers and bounded retries.
4. Checkpoint progress offsets for restart.

Key metrics:
- queue depth trend,
- throughput over time,
- memory stability,
- timeout/failure rates.

## Scenario 2: Strict Tail-Latency Target

Goal:
- Keep p95/p99 under threshold while maximizing throughput.

Design:
1. Instrument per-stage latency.
2. Tune concurrency incrementally.
3. Apply load shedding when saturation starts.
4. Prefer stable tails over peak average throughput.

Key metrics:
- p95/p99 latency,
- queue growth slope,
- error and timeout spikes.

## Scenario 3: Graceful Cancellation

Goal:
- Handle global stop signal with deterministic final states.

Design:
1. Stop new intake immediately.
2. Drain/cancel in-flight by explicit policy.
3. Mark queued leftovers as `cancelled`.
4. Ensure no missing terminal states.

Terminal-state invariant:
- input count == output count == terminal-record count.

## Scenario 4: Idempotent At-Least-Once Processing

Goal:
- Handle duplicates safely under retrying delivery semantics.

Design:
1. Require `idempotency_key` on input.
2. Track `in_progress` and `completed` keys.
3. Retry transient errors with backoff.
4. Route exhausted retryable failures to poison handling.

Terminal statuses:
- `success`, `deduped`, `poison`, `failed`.

## Scenario 5: Hybrid CPU + I/O Pipeline

Goal:
- Mix async network/disk stages with CPU-heavy parsing/transforms.

Design:
1. Stage A async fetch.
2. Stage B process-pool parse.
3. Stage C async write.
4. Bounded A->B and B->C handoff queues.

Key metrics:
- stage A/B/C latency,
- stage-specific failure counters,
- end-to-end p95.

## Scenario 6: Rate-Limited Downstream Service

Goal:
- Respect start-rate limits while preserving throughput.

Design:
1. Use rate limiter for request starts per second.
2. Use semaphore for concurrent in-flight limit.
3. Honor retry-after/timeout semantics.

Common pitfall:
- Using only semaphore can still violate starts-per-second limits.
  A semaphore caps in-flight concurrency, but it does not limit how quickly new requests start after completions.
  If requests are short, you can still exceed downstream RPS policies, so pair semaphore control with a real rate limiter.

## Scenario 7: Priority Classes

Goal:
- Serve high-priority tasks faster under load.

Design:
1. Separate priority queues.
2. Weighted scheduling or reserved worker slots.
3. Per-class p95 monitoring.

Terminal status extension:
- Include `priority` in terminal records for observability.

## Scenario 8: Batch Windows and Micro-Batching

Goal:
- Improve throughput by batching work while protecting latency.

Design:
1. Buffer up to batch size or max wait window.
2. Flush on whichever trigger occurs first.
3. Track batch size distribution and queue wait time.

Tradeoff:
- Larger batches improve throughput but can hurt tail latency.

## Scenario 9: Poison Queue Triage

Goal:
- Prevent retry storms and preserve debuggability.

Design:
1. Route exhausted retryable failures to poison storage.
2. Record full context (attempts, last error, key fields).
3. Provide replay tooling for poison entries.

Terminal status extension:
- Track `poison_reason` and `last_error_type`.

## Scenario 10: Exactly-Once Illusion at Boundaries

Goal:
- Provide exactly-once effect over at-least-once delivery.

Design:
1. At-least-once delivery + idempotent side effects.
2. Dedupe ledger keyed by idempotency key.
3. Atomic commit/mark-completed sequence.

## Scenario 11: Multi-Lock Coordination Safety

Goal:
- Prevent deadlocks in code paths that require multiple locks.

Design:
1. Define a global lock ordering contract.
2. Acquire locks only in that order.
3. Add timeout guards during lock acquisition in high-risk paths.
4. Alert on repeated lock-timeout events.

## Scenario 12: Event-Loop Health Under Mixed Workloads

Goal:
- Keep async event loop responsive when CPU or blocking calls are present.

Design:
1. Keep coroutine code non-blocking.
2. Offload blocking work to executor threads/processes.
3. Measure event-loop lag and callback delay.
4. Bound offloaded work queue to avoid hidden backlogs.

## Scenario 13: GPU Control-Plane Scheduling (No Kernel Coding)

Goal:
- Manage GPU-backed jobs safely under bursty demand.

Design:
1. Bounded admission queue for pending GPU jobs.
2. Per-GPU slot control (one queue/semaphore per device or pool).
3. Terminalize outcomes (`success`, `oom`, `timed_out`, `failed`, `cancelled`).
4. Async health polling to inform admission and throttling.
5. Deterministic shutdown drain so unresolved jobs are never dropped silently.

Key metrics:
- queue depth and queue wait,
- per-GPU utilization and memory pressure,
- launch failures by class (OOM/timeout/other),
- completion latency p95/p99 by GPU pool.

## Architecture Checklist for Any Scenario

1. Define terminal statuses first.
2. Bound all pressure points (queue size, workers, in-flight, start rate).
3. Classify failures as retryable vs non-retryable.
4. Add timeout policy per stage.
5. Capture per-item terminal records with stable schema.
6. Assert invariants in tests (ordering, completeness, status membership).
7. Tune with measured tails (p95/p99), not only averages.
