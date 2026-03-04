# Python Concurrency Playbook (Labs)

This lab is a full preparation pack for practical Python concurrency work.
It is designed for fast implementation under pressure: clear runtime selection,
bounded concurrency, retry/cancellation correctness, deterministic output, and
measurable performance behavior.

## Quick Summary

Flow:
1. Bound pressure: queues, worker limits, semaphore/rate limits.
2. Classify workload: async for I/O, processes for pure-Python CPU.
3. Handle failures: timeout, retry, dedupe, poison, cancellation.
4. Preserve correctness: one terminal status per item, ordered output.
5. Validate with metrics: counters + p95/p99 and invariant tests.

## 30-Second Opening Script

Use this opening structure before going into details:

1. "I will classify workload stages first: async for I/O and process pool for pure-Python CPU."
2. "I will bound pressure at every layer: bounded queues, fixed workers, semaphore, and rate limiter."
3. "I will make failures explicit: timeout/retry with jitter, dedupe ownership, poison routing, and cancellation."
4. "I will guarantee closure: one terminal status per input index and deterministic ordered output."
5. "I will validate behavior with counters plus p95/p99 and strict invariants."

### First Blocks To Type (in order)

1. Statuses + dataclasses: `all_in_one_pipeline.py` lines `37-117`
2. Rate limiter + backoff helpers: lines `122-200`
3. Stage simulation functions: lines `205-285`
4. Producer + workers (A/B/C): lines `290-735`
5. Orchestration + global-timeout drain + invariants: lines `830-1047`
6. Summary + CLI + entrypoint: lines `1053-1195`

### Minimal Skeleton (for blank editor starts)

```python
# 1) statuses + result schema
# 2) rate limiter + backoff helper
# 3) stage funcs: fetch (async), cpu_transform (sync), write (async)
# 4) workers: stage_a -> stage_b(process pool) -> stage_c
# 5) orchestrator: bounded queues + sentinels + global timeout drain
# 6) summarize + invariant checks + main()
```

## Constraint-Prompt Quick Recipe (Stdlib Only)

Use this pattern when constraints are:
- no external packages/frameworks,
- no `concurrent.futures`.
- `slow_work()` represents some slow work

Allowed:
- `threading` and `queue` (both standard library).

Algorithm:
1. Queue all jobs (`num_jobs`, e.g. 1000).
2. Start fixed workers (`max_workers`, e.g. 10).
3. Each worker pops jobs and writes to `results[job_id]`.
4. Push one sentinel (`None`) per worker for clean exit.
5. `join()` workers and render ordered results.

Reference snippet:

```python
import queue
import threading

num_jobs = 1000
max_workers = 10
work_queue: queue.Queue[int | None] = queue.Queue()
results: list[str | None] = [None] * num_jobs

def worker() -> None:
    while True:
        job_id = work_q.get()
        try:
            if job_id is None:
                return
            results[job_id] = slow_work(job_id)
        finally:
            work_q.task_done()

threads = [threading.Thread(target=worker) for _ in range(max_workers)]
for t in threads:
    t.start()

for job_id in range(num_jobs):
    work_queue.put(job_id)

for _ in range(max_workers):
    work_queue.put(None)

work_queue.join()
for t in threads:
    t.join()
```

Why no lock in this exact snippet:
- each worker writes to a unique index (`results[job_id]`),
- final reads happen only after all `join()` calls.

When to add a lock:
- shared counters/dicts with read-modify-write,
- any chance of two workers writing same slot/key.

## Table Of Contents

1. [30-Second Opening Script](#30-second-opening-script)
2. [Constraint-Prompt Quick Recipe (Stdlib Only)](#constraint-prompt-quick-recipe-stdlib-only)
3. [Start Here](#start-here)
4. [All-In-One Annotated Reference](#all-in-one-annotated-reference)
5. [Complete End-To-End Walkthrough](#complete-end-to-end-walkthrough)
6. [Coroutines Quick Recall](#coroutines-quick-recall)
7. [Free-Threaded Python (No-GIL) Status](#free-threaded-python-no-gil-status)
8. [Terminal Status Model](#terminal-status-model)
9. [Core Patterns And Examples](#core-patterns-and-examples)
10. [Scenario Q&A (High Probability)](#scenario-qa-high-probability)
11. [Platform Q&A: Docker, Kubernetes, Slurm on Kubernetes](#platform-qa-docker-kubernetes-slurm-on-kubernetes)
12. [Advanced Scenarios](#advanced-scenarios)
13. [NVIDIA GPU Extension Track](#nvidia-gpu-extension-track)
14. [Follow-Up Drill Bank](#follow-up-drill-bank)
15. [Reference Guides](#reference-guides)
16. [Validation And Stress Tests](#validation-and-stress-tests)
17. [Coverage Matrix](#coverage-matrix)

## Start Here

If you only have a short prep window, run in this order:

1. `all_in_one_pipeline.py`
   - Single-file reference that combines queue backpressure, retries, dedupe/poison, global timeout, and hybrid async+process stages.
2. `taskrun_round1_asyncio.py`
   - Learn bounded workers + retries + ordered output.
3. `taskrun_round2_controls.py`
   - Add global timeout, cancellation, and rate limiting.
4. `taskrun_round3_idempotency.py`
   - Add dedupe and poison routing for retry-safe side effects.
5. `hybrid_three_stage_pipeline.py`
   - Combine async I/O + process-pool CPU + async I/O.
6. `failure_patterns.py`
   - Review broken vs fixed patterns for common production bugs.

## All-In-One Annotated Reference

Primary file:
- `labs/python_concurrency/all_in_one_pipeline.py`

Sample input:
- `labs/python_concurrency/sample_all_in_one_items.json`

Run command:

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

What this single file demonstrates in one place:
- bounded producer + bounded handoff queues,
- fixed worker pools per stage,
- semaphore + rate limiter together,
- per-attempt timeout + retry + jittered backoff,
- idempotency ownership (`completed_keys` + `in_progress_keys`),
- dedupe and poison terminal outcomes,
- async I/O stage A -> process-pool CPU stage B -> async I/O stage C,
- global deadline cancellation and queue draining,
- strict terminal-state reconciliation and invariant checks.

### Section-By-Section Map (with line ranges)

| Lines in `all_in_one_pipeline.py` | Section | Why it matters |
| --- | --- | --- |
| `33-117` | Status/constants + dataclasses | Defines stable terminal schema and lifecycle counters. |
| `119-200` | Rate limiter + helpers | Shows why starts/sec control is separate from in-flight control. |
| `202-285` | Stage simulation functions | Deterministic and probabilistic failure modes for realistic drills. |
| `287-735` | Producer + stage workers | Core concurrency implementation with retries, dedupe, and terminalization. |
| `738-1047` | Orchestration + cancellation + invariants | Sentinel ordering, timeout handling, queue drain, and final reconciliation. |
| `1050-1195` | Summary + CLI + entrypoint | Operational metrics, argument controls, and executable single-file flow. |

### Line-By-Line Checkpoints To Talk Through

1. `37-44`: terminal statuses declared explicitly.
2. `48-57`: allowed terminal set used for invariant enforcement.
3. `93-116`: shared mutable state contract (`completed_keys`, `in_progress_keys`, counters).
4. `122-143`: rate limiter controls request starts per second.
5. `153-157`: jittered exponential backoff helper.
6. `160-190`: single constructor for uniform terminal records.
7. `193-199`: atomic key release/commit helper.
8. `205-227`: stage-A failure simulation controls.
9. `230-250`: CPU stage function intentionally pure Python for process-pool relevance.
10. `253-284`: stage-C failure simulation controls.
11. `290-308`: bounded producer + one sentinel per stage-A worker.
12. `355-381`: dedupe ownership decision point (`completed` / `in_progress`).
13. `398-413`: stage-A rate-limit + semaphore + timeout call path.
14. `414-442`: stage-A retry classification (`timed_out`, `poison`, permanent failures).
15. `444-456`: successful stage-A handoff to stage B.
16. `559-568`: process-pool CPU execution with timeout wrapper.
17. `671-677`: stage-C bounded async write with timeout.
18. `683-711`: stage-C retry + poison classification logic.
19. `953-969`: sentinel-based shutdown ordering across stages.
20. `979-1009`: global-timeout cancellation and draining of leftover queue items.
21. `1014-1045`: final reconciliation and hard invariant checks.
22. `1053-1092`: summary metrics (`queued`, `started`, `completed`, retries, latencies).

### Single-File Live Coding Sequence

Use this build order if coding from a blank editor:

1. Define terminal statuses and result dataclass.
2. Add shared state for dedupe and counters.
3. Add rate limiter and backoff helper.
4. Implement stage simulations (`simulated_fetch`, `cpu_transform`, `simulated_write`).
5. Implement producer with bounded queue and sentinels.
6. Implement stage-A worker with dedupe + retry + timeout + rate limit + semaphore.
7. Implement stage-B worker with process-pool and timeout.
8. Implement stage-C worker with retry + poison routing.
9. Implement orchestration and sentinel shutdown ordering.
10. Add global-timeout cancellation and queue-drain terminalization.
11. Add final invariant checks and summary.
12. Wire CLI args and entrypoint.

## Complete End-To-End Walkthrough

Use this as a full narrative you can talk through from first principles.

### Problem statement

Process a large list of items concurrently with these constraints:
- bounded worker concurrency,
- per-item timeout,
- retry policy,
- deterministic output ordering,
- partial success under failures,
- clear summary metrics and terminal states.

### Runtime classification

- If work is mostly waiting (network/disk): use `asyncio` workers.
- If a stage is CPU-heavy pure Python: offload to `ProcessPoolExecutor`.
- If mixed: use stage-specific models in one pipeline.

### Coroutines Quick Recall

Use this section when asked what coroutines are and when to use them.

### What a coroutine is

- A coroutine is an `async def` function execution unit.
- Calling an `async def` function does not run it immediately; it creates a coroutine object.
- The coroutine runs only when:
  - `await`ed directly, or
  - scheduled as a task (for example with `asyncio.create_task`).

### Mental model

Think of a coroutine as "pausable work" managed by the event loop:
- it runs until an `await` point,
- yields control while waiting,
- resumes later when awaited operation is ready.

### When to use coroutines

- High-concurrency I/O workloads:
  - HTTP calls,
  - database/network requests,
  - disk/network waits,
  - many independent latency-bound tasks.
- Pipelines where most stages are wait-heavy and need many concurrent in-flight operations.

### When not to use coroutines alone

- Heavy pure-Python CPU loops.
- Blocking calls inside coroutine bodies (for example `time.sleep`).

For CPU-heavy work:
- use `ProcessPoolExecutor`, or
- use a free-threaded runtime only when runtime + dependencies are validated.

### "Are coroutines standard now?"

Yes. `async`/`await` coroutines have been mainstream in Python since 3.5 and are the standard model for structured async I/O in modern Python services and SDKs.

### 20-second answer

"A coroutine is an `async def` unit that the event loop can pause and resume. It is ideal for I/O-bound concurrency because tasks yield while waiting instead of blocking threads. For CPU-bound pure Python, use processes (or validated free-threaded runtime paths), not coroutine-only designs."

### Architecture

1. Input list is enqueued as `(index, item)`.
2. Fixed workers pull from queue.
3. Each item runs with timeout + bounded retry loop.
4. Results are written into `results[index]` to preserve output order.
5. Shutdown uses explicit sentinels (`None`) for deterministic worker exit.
6. Summary reports success/failure/timeout/cancellation and latency percentiles.

### Worked end-to-end example (single run, full trace)

Sample input (`sample_round2_items.json` pattern):

```json
[
  {"id": "u0", "delay_ms": 50, "fail_rate": 0.0},
  {"id": "u1", "delay_ms": 900, "fail_rate": 0.0},
  {"id": "u2", "delay_ms": 50, "fail_rate": 1.0},
  {"id": "u3", "delay_ms": 50, "fail_rate": 0.0}
]
```

Run:

```bash
python labs/python_concurrency/taskrun_round2_controls.py \
  --input labs/python_concurrency/sample_round2_items.json \
  --max-workers 2 \
  --rps 6 \
  --timeout-ms 200 \
  --retries 1 \
  --global-timeout-ms 1200
```

How each item should finish:

1. `u0` usually becomes `success`.
2. `u1` usually becomes `timed_out` (delay is higher than per-attempt timeout).
3. `u2` usually becomes `failed` (simulated non-timeout failure).
4. `u3` usually becomes `success` unless cancelled by global deadline pressure.

Why this is a complete example:

- bounded worker pool (`--max-workers`)
- rate control (`--rps`)
- per-attempt timeout (`--timeout-ms`)
- retry policy (`--retries`)
- global stop condition (`--global-timeout-ms`)
- ordered terminal output by input index
- summary counters that must match record-level outcomes

### Complete end-to-end code snippet (single-file pattern)

```python
#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, asdict

SUCCESS = "success"
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"


@dataclass
class ItemResult:
    idx: int
    item_id: str
    status: str
    attempts: int
    latency_ms: float
    error: str | None
    output: str | None


async def io_step(item: dict) -> str:
    await asyncio.sleep(float(item.get("delay_ms", 50)) / 1000.0)
    if random.random() < float(item.get("fail_rate", 0.0)):
        raise RuntimeError("simulated failure")
    return f"ok:{item['id']}"


async def run_one(item: dict, timeout_ms: int) -> str:
    return await asyncio.wait_for(io_step(item), timeout=float(timeout_ms) / 1000.0)


async def worker(
    queue: asyncio.Queue[tuple[int, dict] | None],
    results: list[ItemResult | None],
    sem: asyncio.Semaphore,
    retries: int,
    timeout_ms: int,
    stop_event: asyncio.Event,
) -> None:
    while True:
        work = await queue.get()
        if work is None:
            queue.task_done()
            return

        idx, item = work
        started = time.perf_counter()
        attempts = 0
        status = FAILED
        error: str | None = None
        output: str | None = None

        if stop_event.is_set():
            results[idx] = ItemResult(idx, str(item["id"]), CANCELLED, 0, 0.0, "global_timeout_before_start", None)
            queue.task_done()
            continue

        for attempt in range(retries + 1):
            if stop_event.is_set():
                status = CANCELLED
                error = "global_timeout_during_execution"
                break
            attempts = attempt + 1
            try:
                async with sem:
                    output = await run_one(item, timeout_ms=timeout_ms)
                status = SUCCESS
                error = None
                break
            except asyncio.TimeoutError:
                status = TIMED_OUT
                error = f"timeout_after_{timeout_ms}ms"
            except Exception as exc:  # noqa: BLE001
                status = FAILED
                error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.05 * (2**attempt))

        latency_ms = (time.perf_counter() - started) * 1000.0
        results[idx] = ItemResult(idx, str(item["id"]), status, attempts, latency_ms, error, output)
        queue.task_done()


async def run_pipeline(
    items: list[dict],
    max_workers: int,
    timeout_ms: int,
    retries: int,
    global_timeout_ms: int,
) -> list[ItemResult]:
    queue: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue()
    for idx, item in enumerate(items):
        queue.put_nowait((idx, item))

    results: list[ItemResult | None] = [None] * len(items)
    sem = asyncio.Semaphore(max(1, max_workers))
    stop_event = asyncio.Event()
    workers = [
        asyncio.create_task(worker(queue, results, sem, retries, timeout_ms, stop_event))
        for _ in range(max(1, max_workers))
    ]

    try:
        await asyncio.wait_for(queue.join(), timeout=float(global_timeout_ms) / 1000.0)
    except asyncio.TimeoutError:
        stop_event.set()
        while True:
            try:
                pending = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if pending is None:
                queue.task_done()
                continue
            idx, item = pending
            if results[idx] is None:
                results[idx] = ItemResult(
                    idx,
                    str(item["id"]),
                    CANCELLED,
                    0,
                    0.0,
                    "global_timeout_before_start",
                    None,
                )
            queue.task_done()
        await queue.join()

    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    finalized: list[ItemResult] = []
    for idx, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing terminal result at index {idx}")
        finalized.append(result)
    return finalized


def summarize(results: list[ItemResult]) -> dict:
    return {
        "total": len(results),
        "success": sum(1 for r in results if r.status == SUCCESS),
        "failed": sum(1 for r in results if r.status == FAILED),
        "timed_out": sum(1 for r in results if r.status == TIMED_OUT),
        "cancelled": sum(1 for r in results if r.status == CANCELLED),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=700)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--global-timeout-ms", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    items = json.loads(open(args.input, encoding="utf-8").read())
    results = asyncio.run(
        run_pipeline(
            items=items,
            max_workers=args.max_workers,
            timeout_ms=args.timeout_ms,
            retries=args.retries,
            global_timeout_ms=args.global_timeout_ms,
        )
    )
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))
    summary = summarize(results)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["failed"] + summary["timed_out"] + summary["cancelled"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### Concrete command sequence

```bash
python labs/python_concurrency/taskrun_round2_controls.py \
  --input labs/python_concurrency/sample_round2_items.json \
  --max-workers 4 \
  --rps 6 \
  --timeout-ms 700 \
  --retries 1 \
  --global-timeout-ms 2400
```

### What to observe in output

- Per-item JSON lines with terminal statuses.
- Final summary JSON with totals and latency stats.
- Non-zero exit code if any item is non-success.
- Each item index appears exactly once with a terminal status.

### Fast talk track (design reasoning)

1. Classify workload: I/O-heavy, so use `asyncio` with fixed workers.
2. Bound pressure: queue + worker count + optional rate limiter.
3. Protect reliability: per-attempt timeout, bounded retries, explicit failure modes.
4. Preserve determinism: write results by input index.
5. Guarantee closure: every item must end in one terminal status.
6. Tune with evidence: watch p95/p99 and status counters while changing worker count.

### Why this design is robust

- bounded fanout (no unbounded gather),
- failure isolation per item,
- deterministic render order,
- explicit terminal outcomes,
- measurable, tunable behavior.

## Free-Threaded Python (No-GIL) Status

Use this section for current-version questions about GIL-free Python.

As of **March 4, 2026**:
- CPython **3.13** introduced free-threaded mode as an experimental opt-in build (`--disable-gil`).
- CPython **3.14** moved free-threaded mode to "officially supported but optional" (phase II in PEP 779).
- Default CPython remains GIL-enabled in standard installs.

| Question | Practical answer |
| --- | --- |
| Which versions have no-GIL support? | 3.13+ has free-threaded builds; 3.14 marks them officially supported but optional. |
| Is no-GIL the default? | No. Default builds are still GIL-enabled. |
| Does no-GIL remove need for process pools? | Not universally. Process pools remain robust when runtime/dependency compatibility is uncertain. |
| What is the biggest caveat? | Dependency ecosystem readiness (especially C extensions) must be validated on the exact runtime. |
| Can runtime behavior still toggle GIL in free-threaded builds? | Yes, runtime controls like `PYTHON_GIL` and `-X gil` exist for compatibility modes. |

Free-threaded deep Q&A:
- `labs/python_concurrency/SCENARIO_QA.md`

## Terminal Status Model

A terminal status is the final immutable outcome for one item.
Every input item must end in exactly one terminal status.

### Statuses used here

- Generic:
  - `success`, `failed`, `timed_out`, `cancelled`
- Reliability-specific:
  - `deduped`, `poison`
- Hybrid stage-specific:
  - `failed_stage_a`, `failed_stage_b`, `failed_stage_c`

### Why `started` is not listed as a terminal status

`started` is a lifecycle/progress marker, not a final outcome. An item can move from
`started` to `success`/`failed`/`timed_out`/`cancelled`, so `started` belongs in
telemetry counters and event logs (`queued`, `started`, `completed`) rather than the
terminal status vocabulary.

### Tiny status state machine

```text
Lifecycle / progress
  queued --> started --> completed

Terminal outcomes (subset reached at completed)
  started --> success
  started --> failed
  started --> timed_out
  started --> cancelled
  started --> deduped
  started --> poison
  started --> failed_stage_a | failed_stage_b | failed_stage_c
```

### Required invariants

1. `len(results) == len(inputs)`
2. exactly one terminal record per input index
3. no unresolved placeholders
4. status in allowed terminal set
5. counters in summary match record-level statuses

### If a terminal status is missing: identify and correct

Identification:

1. Run completeness checks before summary/export:
   - missing index scan (`results[idx] is None`),
   - duplicate index scan,
   - status-set membership scan.
2. Compare `queued/started/completed` counters. Any `started > completed` gap means unresolved work.
3. On restart after crash, reconcile durable logs:
   - `started_ids - terminal_ids = unresolved_ids`.

Correction:

1. In-process (no crash): convert unresolved queued work to explicit `cancelled` on global timeout and only then emit summary.
2. After process crash: replay unresolved IDs from checkpoint/input source, guarded by idempotency keys so retries are safe.
3. If side effects may have partially happened, reconcile with downstream state first, then finalize each unresolved item as `success`, `deduped`, `failed`, or `cancelled`.

Design rule:
- Never silently drop unresolved items. If state is unknown, emit a deterministic terminal status plus error context (for example `cancelled` + `error="crash_recovery_pending"`), then reconcile.

Detailed reference:
- `labs/python_concurrency/TERMINAL_STATUS_REFERENCE.md`

## Core Patterns And Examples

### 1) All-in-one combined pattern (single file)
- Script: `all_in_one_pipeline.py`
- End-to-end reference that combines bounded queues, dedupe, poison, retries, rate limiting, process-pool CPU stage, and cancellation.

### 2) Bounded worker pool and ordered output
- Script: `taskrun_round1_asyncio.py`
- Prevents unbounded fanout and keeps memory/concurrency controlled.

### 3) Global deadline + cancellation + rate limiting
- Script: `taskrun_round2_controls.py`
- Adds system-level controls beyond per-item retries.

### 4) Idempotency + dedupe + poison routing
- Script: `taskrun_round3_idempotency.py`
- Makes retries safe for side-effecting operations.

### 5) Hybrid async/process pipeline
- Script: `hybrid_three_stage_pipeline.py`
- Async stage A (fetch) -> process-pool stage B (CPU parse) -> async stage C (write).
- Bounded A->B and B->C queues provide backpressure.

### 6) Broken vs fixed production failures
- Script: `failure_patterns.py`
- Includes direct examples for:
  - unbounded fanout,
  - race conditions,
  - deadlock/lock-order,
  - event-loop blocking,
  - retry storm vs jitter,
  - cancellation handling.

## Scenario Q&A (High Probability)

| Question | Strong concise answer |
| --- | --- |
| Why not `asyncio.gather(*all_tasks)` for huge input? | Unbounded gather can explode memory and dependency pressure; queue + workers gives bounded concurrency. |
| How do you keep output deterministic under concurrency? | Keep input index and write result into that index. |
| Why per-attempt timeout and not only global timeout? | Per-attempt timeouts prevent a single call from monopolizing worker slots. |
| How do you make retries safe with side effects? | Use idempotency keys with lock-protected dedupe state. |
| How do you avoid deadlocks with multiple locks? | Enforce one global lock acquisition order. |
| How do you avoid freezing the event loop? | Never run blocking calls directly in coroutines; offload blocking work via executor/to_thread. |
| Why use jitter in backoff? | It spreads retries over time and avoids synchronized retry spikes. |
| What does cancellation correctness mean? | All items end in explicit terminal statuses; no silent drops on shutdown. |
| What is backpressure? | Producers slow down because queue/workers are saturated, preventing memory blowups. |
| Why choose processes for CPU-heavy pure Python? | One interpreter lock limits thread-level bytecode parallelism; processes provide true CPU parallelism. |
| How do you debug stuck workers? | Track queue depth, in-flight count, per-item timeout logs, and worker state transitions. |
| What should be tuned first? | Worker count and rate limits, then retry policy, while watching tail latency and error rate. |

More:
- `labs/python_concurrency/SCENARIO_QA.md`

## Platform Q&A: Docker, Kubernetes, Slurm on Kubernetes

| Question | Strong concise answer |
| --- | --- |
| How should container images be built for reliable concurrency workers? | Use slim immutable images, pin dependency versions, run as non-root, and keep startup deterministic. Smaller images reduce cold-start and pull time variance. |
| What Kubernetes primitive fits queue workers? | Use `Job` for finite batch runs and `Deployment` for continuous queue consumers. Pick based on lifecycle semantics, not convenience. |
| How do you prevent overload in Kubernetes workers? | Combine application-level bounded queues with pod-level resource requests/limits and autoscaling policies tied to queue depth. |
| What probes should be used for worker pods? | `startupProbe` for slow boot, `readinessProbe` for traffic eligibility, and `livenessProbe` only for true deadlock detection. Misused liveness can create restart storms. |
| How do you handle graceful shutdown in Kubernetes? | Use `SIGTERM` handling, stop intake first, drain/terminalize in-flight work, and set `terminationGracePeriodSeconds` long enough to finish deterministic shutdown. |
| Why is idempotency critical with Kubernetes restarts? | Pods can restart or be rescheduled at any time; idempotency prevents duplicate side effects during replay/retry after restarts. |
| How do GPU requests work in Kubernetes? | Request extended resources like `nvidia.com/gpu` and ensure the NVIDIA device plugin/runtime is installed. Scheduler placement must match node labels/taints and quota policy. |
| What does Slurm on Kubernetes add? | Slurm adds HPC-style queueing/allocation policy while Kubernetes provides container orchestration and cluster primitives. The combined stack needs clear ownership of scheduling decisions. |
| Slurm vs Kubernetes scheduler: who should decide placement? | Avoid double-scheduling ambiguity: define a single source of truth for resource allocation and keep the other layer policy-aware but non-conflicting. |
| What are common Slurm-on-Kubernetes failure modes? | Pending pods from insufficient GPU resources, misaligned node labels/taints, and mismatched job lifecycle between Slurm state and pod state. |
| How do you observe bottlenecks in this stack? | Correlate queue depth, job wait time, pod pending reasons, GPU utilization/memory pressure, and tail latency. One metric alone is rarely enough. |
| What is the fast architecture summary for this platform? | Bounded admission, explicit terminal states, idempotent replay, graceful shutdown, and evidence-driven autoscaling/placement. |

## Advanced Scenarios

| Scenario | Primary design move | Typical failure mode if skipped |
| --- | --- | --- |
| Massive stream processing | Stream input + bounded queue | OOM from full materialization |
| Tail-latency constrained throughput | Tune with p95/p99 guardrails | Good average latency but unstable tails |
| Graceful cancellation under deadlines | Stop intake, then drain/cancel deterministically | Missing/ambiguous item outcomes |
| Idempotent at-least-once processing | Dedupe ledger with key ownership | Duplicate side effects |
| Hybrid CPU + I/O pipelines | Async I/O stages + process-pool CPU stage | Event-loop stalls or CPU bottleneck |
| Rate-limited downstream services | Rate limiter + semaphore together | 429 storms and overload |
| Priority classes | Per-priority queues/scheduling | High-priority starvation |
| Batch-window micro-batching | Size-or-time flush policy | Throughput or tail-latency collapse |
| Poison queue triage and replay | Route exhausted retries to poison storage | Retry storms and poor debuggability |
| Exactly-once effect at boundaries | At-least-once + idempotent commit boundary | Non-repeatable outcomes |
| Multi-lock coordination safety | Global lock ordering rule | Deadlock |
| Event-loop health in mixed workloads | Offload blocking calls, monitor loop lag | Hidden serial execution in async code |

Details:
- `labs/python_concurrency/ADVANCED_SCENARIOS.md`

## NVIDIA GPU Extension Track

If you want to extend these same patterns into GPU-cloud control planes, use:
- `labs/python_concurrency/NVIDIA_GPU_EXTENSION.md`

What it covers:
- bounded GPU admission queues and slot scheduling,
- timeout/retry terminalization for GPU jobs,
- async GPU health polling and drain-safe shutdown,
- practical code examples focused on orchestration (not kernel programming).

## Follow-Up Drill Bank

Use this for rapid-fire second-order probes and concise responses:
- `labs/python_concurrency/FOLLOWUP_QUESTION_BANK.md`
- `labs/python_concurrency/TOP30_HARDEST_FOLLOWUPS.md`

Fast routing while answering follow-ups:
1. Runtime model choice -> `SCENARIO_QA.md` (Runtime Selection, Coroutines, Free-Threaded sections).
2. Reliability/idempotency questions -> `SCENARIO_QA.md` (Idempotency and Side Effects) + `taskrun_round3_idempotency.py`.
3. End-to-end implementation walk-through -> `all_in_one_pipeline.py` + the line map above.
4. Platform questions (Docker/K8s/Slurm-on-K8s) -> `SCENARIO_QA.md` and the platform table in this README.
5. Short answer drills -> `TOP30_HARDEST_FOLLOWUPS.md`.

Recommended drill loop:
1. Pick one section (for example Idempotency or Kubernetes).
2. Answer each prompt in 20-30 seconds.
3. Add one concrete metric or failure mode to each answer.
4. Repeat until answers are consistent and concise.

## Reference Guides

- `labs/python_concurrency/QUICK_REFERENCE_GUIDE.md`
  - concise patterns, snippets, and tuning sequence.
- `labs/python_concurrency/SCENARIO_QA.md`
  - concise Q&A for tradeoffs and edge cases.
- `labs/python_concurrency/FOLLOWUP_QUESTION_BANK.md`
  - likely follow-up probes with compact defensible answers.
- `labs/python_concurrency/TOP30_HARDEST_FOLLOWUPS.md`
  - one-line answers for the highest-frequency hard follow-up probes.
- `labs/python_concurrency/ADVANCED_SCENARIOS.md`
  - deeper operational scenarios.
- `labs/python_concurrency/TERMINAL_STATUS_REFERENCE.md`
  - terminal-status taxonomy and invariants.

## Validation And Stress Tests

### Functional invariants

```bash
pytest -q tests/test_python_concurrency_all_in_one.py
pytest -q tests/test_python_concurrency_terminal_states.py
pytest -q tests/test_python_concurrency_failure_patterns.py
```

### Stress invariants

```bash
pytest -q tests/test_python_concurrency_stress.py
```

Stress suite covers:
- high-volume bounded queue saturation,
- repeated cancellation races,
- dedupe + poison + global-deadline cross-invariants,
- terminal-state completeness under load.

### Manual runs

```bash
python labs/python_concurrency/all_in_one_pipeline.py \
  --input labs/python_concurrency/sample_all_in_one_items.json \
  --stage-a-workers 3 --stage-b-workers 2 --stage-c-workers 2 \
  --queue-size 4 --rps 12 --fetch-inflight 3 --write-inflight 2 \
  --fetch-timeout-ms 200 --write-timeout-ms 200 --cpu-timeout-ms 1200 \
  --fetch-retries 1 --write-retries 1 --cpu-rounds 8000 --seed 7

python labs/python_concurrency/taskrun_round1_asyncio.py \
  --input labs/python_concurrency/sample_round1_items.json \
  --max-workers 4 --timeout-ms 600 --retries 1

python labs/python_concurrency/taskrun_round2_controls.py \
  --input labs/python_concurrency/sample_round2_items.json \
  --max-workers 4 --rps 6 --timeout-ms 700 --retries 1 --global-timeout-ms 2400

python labs/python_concurrency/taskrun_round3_idempotency.py \
  --input labs/python_concurrency/sample_round3_items.json \
  --max-workers 4 --timeout-ms 500 --retries 2 --global-timeout-ms 1800 --seed 42

python labs/python_concurrency/hybrid_three_stage_pipeline.py \
  --input labs/python_concurrency/sample_hybrid_items.json \
  --stage-a-workers 3 --stage-b-workers 3 --stage-c-workers 3 --queue-size 4 --cpu-rounds 30000

python labs/python_concurrency/failure_patterns.py \
  --items 40 --workers 6 --delay-ms 10 --retries 4 --seed 42
```

## Coverage Matrix

| Concept | Example | Test validation |
| --- | --- | --- |
| Full all-in-one reference (queues + retries + dedupe + poison + hybrid stages + cancellation) | `all_in_one_pipeline.py` | `tests/test_python_concurrency_all_in_one.py` |
| Bounded worker pool + ordering | `taskrun_round1_asyncio.py` | `test_round1_terminal_state_invariants_ordering_and_retries` |
| Timeout + retry | `taskrun_round1_asyncio.py`, `taskrun_round2_controls.py` | round1/round2 tests |
| Global cancellation/deadline | `taskrun_round2_controls.py` | `test_round2_global_timeout_preserves_order_and_marks_cancellations` |
| Idempotency + dedupe + poison | `taskrun_round3_idempotency.py` | `test_round3_dedupe_and_poison_terminal_states` |
| Dedupe + poison + deadline cross-invariants | `taskrun_round3_idempotency.py` | `test_round3_cross_invariants_dedupe_poison_and_global_deadline` |
| Hybrid 3-stage pipeline | `hybrid_three_stage_pipeline.py` | `test_hybrid_pipeline_terminal_states_ordering_and_stage_failures` |
| Unbounded fanout bug/fix | `failure_patterns.py` | `test_unbounded_and_bounded_fanout_produce_same_outputs_for_small_input` |
| Shared-state race bug/fix | `failure_patterns.py` | `test_race_condition_broken_loses_updates_but_fixed_is_exact` |
| Deadlock bug/fix | `failure_patterns.py` | `test_deadlock_broken_is_detected_and_fixed_completes` |
| Event-loop blocking bug/fix | `failure_patterns.py` | `test_hidden_blocking_fixed_is_faster_than_broken` |
| Retry storm jitter behavior | `failure_patterns.py` | `test_retry_schedules_with_and_without_jitter` |
| Cancellation bug/fix | `failure_patterns.py` | `test_cancellation_handling_broken_vs_fixed` |
| Stress saturation/cancellation races | `taskrun_round2_controls.py`, `hybrid_three_stage_pipeline.py` | `tests/test_python_concurrency_stress.py` |
| GPU control-plane extension patterns | `NVIDIA_GPU_EXTENSION.md` | design reference (no execution required) |
