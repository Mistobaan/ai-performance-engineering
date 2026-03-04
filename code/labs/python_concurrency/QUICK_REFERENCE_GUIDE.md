# Concurrency Quick Reference

## Quick Summary

Core flow:
1. Pick runtime model from workload shape.
2. Bound fanout and in-flight work.
3. Add timeout/retry/cancellation/idempotency controls.
4. Emit deterministic terminal records and validate invariants.

## 0) One-File Reference Run

- Script: `labs/python_concurrency/all_in_one_pipeline.py`
- Sample input: `labs/python_concurrency/sample_all_in_one_items.json`

```bash
python labs/python_concurrency/all_in_one_pipeline.py \
  --input labs/python_concurrency/sample_all_in_one_items.json \
  --stage-a-workers 3 \
  --stage-b-workers 2 \
  --stage-c-workers 2 \
  --queue-size 4 \
  --rps 12 \
  --fetch-inflight 3 \
  --write-inflight 2 \
  --fetch-timeout-ms 200 \
  --write-timeout-ms 200 \
  --cpu-timeout-ms 1200 \
  --fetch-retries 1 \
  --write-retries 1 \
  --cpu-rounds 8000 \
  --seed 7
```

## 1) Runtime Model Selection

- I/O-bound:
  - Prefer `asyncio` (or threads) because tasks spend most time waiting.
- CPU-bound pure Python:
  - Prefer `ProcessPoolExecutor` because one interpreter lock limits thread parallelism.
- Mixed pipeline:
  - Use stage-specific models: async for I/O, process pool for CPU.

## 2) Safe Default Architecture

1. Producer -> bounded queue (`Queue(maxsize=N)`).
2. Fixed worker pool (`K` workers).
3. Optional semaphore for external dependency limits.
4. Timeout + bounded retries.
5. Structured terminal result per item.
6. Ordered rendering by input index.

## 2b) Stdlib-Only Thread Pool Pattern (Common Constraint Prompt)

When constraints disallow external packages and `concurrent.futures`:
1. Use `threading` + `queue.Queue`.
2. Set fixed worker count for concurrency cap (`max_workers`).
3. Enqueue all jobs, then enqueue one sentinel (`None`) per worker.
4. `join()` queue and workers before rendering final ordered results.

Why one sentinel per worker:
- each worker needs a distinct stop token to exit its loop cleanly.

## 3) Terminal Status Rules

A terminal status is final and immutable.
Every item must end in exactly one terminal status.

Recommended terminal set:
- `success`
- `failed`
- `timed_out`
- `cancelled`
- `deduped` (for idempotency flows)
- `poison` (retryable failures exhausted)

Lifecycle note:
- `started` is a progress marker, not a terminal state; track it in counters/logs.

## 4) Reliability Patterns

- Backpressure:
  - Bounded queue + bounded workers prevents uncontrolled memory growth.
- Failure isolation:
  - One item failure must not crash the batch.
- Retry safety:
  - Use idempotency keys before retrying side-effecting calls.
- Cancellation correctness:
  - Stop intake first, then drain or cancel in-flight work by policy.

## 5) Observability Baseline

Collect at minimum:
- `success`, `failed`, `timed_out`, `cancelled`, `deduped`, `poison`
- retry count
- queue depth/backlog
- in-flight count
- p50/p95/p99 latency
- stage-level latency in multi-stage pipelines

## 6) Tuning Loop

1. Measure baseline.
2. Increase workers gradually.
3. Monitor p95/p99 and failure rates.
4. Stop when tail latency or errors degrade.
5. Re-check after each policy change.

## 7) Snippets Worth Memorizing

### Ordered terminal output
```python
results = [None] * len(items)
for idx, item in enumerate(items):
    queue.put_nowait((idx, item))
# worker writes: results[idx] = terminal_result
```

### Timeout + retry + backoff
```python
for attempt in range(retries + 1):
    try:
        value = await asyncio.wait_for(do_work(item), timeout=0.8)
        break
    except Exception:
        if attempt < retries:
            await asyncio.sleep(0.05 * (2 ** attempt))
```

### Bounded in-flight external calls
```python
sem = asyncio.Semaphore(max_workers)
async with sem:
    await call_external()
```

### CPU-heavy stage offload
```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as pool:
    list(pool.map(cpu_heavy_fn, data))
```

## 8) Coroutines Quick Recall

Definition:
- Coroutine = `async def` unit that can pause at `await` and resume later.

Execution rule:
- `async def fn(...)` call returns coroutine object.
- It runs only when:
  - `await fn(...)`, or
  - scheduled with `asyncio.create_task(fn(...))`.

Best fit:
- I/O-heavy concurrency (many waits, many in-flight tasks).

Not best fit alone:
- CPU-heavy pure-Python loops.

One-line memory aid:
- "Coroutines are for waiting; processes are for heavy compute."

## 9) Fast Architecture Narrative

- Classify workload type first.
- Bound concurrency and queue depth.
- Add explicit terminal states and retry policy.
- Preserve deterministic output ordering.
- Instrument latency and failure modes.
- Tune with measured data.

## 10) Common Failure Classes (Broken -> Fixed)

1. Unbounded fanout -> queue + fixed worker pool.
2. Shared-state race -> lock-protected critical section.
3. Deadlock via inconsistent lock order -> global lock ordering rule.
4. Blocking call inside coroutine -> offload via executor/to_thread.
5. Immediate retries (retry storm) -> exponential backoff + jitter.
6. Ignored cancellation -> catch `CancelledError`, clean up, re-raise.

## 11) Free-Threaded Python (No-GIL) Quick Facts

As of March 4, 2026:

- CPython 3.13 introduced free-threaded mode as an experimental opt-in build.
- CPython 3.14 made free-threaded mode officially supported but optional.
- Default CPython builds still use the GIL.

Use this answer pattern:

1. Runtime support first: "Is this deployment actually free-threaded?"
2. Dependency support second: "Are our C extensions and wheels thread-safe here?"
3. Model choice third:
   - I/O-heavy: `asyncio` remains strong.
   - CPU-heavy: threads can improve with free-threaded builds, but processes remain the portable safe default across mixed environments.

## 12) Missing Terminal Status Runbook

Detection:
1. Validate `len(results) == len(inputs)`.
2. Scan for unresolved slots (`None`/placeholder statuses).
3. Check counter parity (`started == completed` at close).

Correction:
1. No-crash timeout path: mark unresolved queued work as `cancelled`.
2. Crash recovery path: replay `started_ids - terminal_ids` from checkpoint.
3. Guard replay with idempotency keys so recovery does not duplicate side effects.

One-line memory aid:
- "No item is allowed to disappear; unresolved always becomes explicit, then reconciled."

## 13) NVIDIA GPU Control-Plane Extension

Use the same control-plane patterns:
1. `Queue(maxsize=N)` for GPU job admission.
2. Per-GPU slot caps (queue/semaphore) for safe concurrency.
3. Explicit terminal statuses (`success`, `oom`, `timed_out`, `failed`, `cancelled`).
4. Async health polling + deterministic shutdown drain.

Reference:
- `labs/python_concurrency/NVIDIA_GPU_EXTENSION.md`

## 14) Docker/K8s/Slurm-on-K8s Recall

1. Primitive choice:
   - `Deployment` for continuous workers.
   - `Job` for finite batch work.
2. Shutdown contract:
   - stop intake on `SIGTERM`, drain in-flight, emit terminal states.
3. GPU scheduling:
   - request `nvidia.com/gpu`, enforce labels/taints/affinity policy.
4. Slurm-on-K8s:
   - avoid double-scheduling; define one source of truth for allocation.

## 15) Follow-Up Drill Shortcut

Use:
- `labs/python_concurrency/FOLLOWUP_QUESTION_BANK.md`
- `labs/python_concurrency/TOP30_HARDEST_FOLLOWUPS.md`

Fast routine:
1. 10 prompts per session.
2. 20-30 second answer each.
3. Include one metric and one failure mode in every answer.
