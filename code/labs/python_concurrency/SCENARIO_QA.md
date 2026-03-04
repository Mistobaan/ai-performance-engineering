# Scenario Q&A

## Quick Summary

Fast answer template:
1. Classify workload (I/O vs CPU vs mixed).
2. Name the control primitive (queue, semaphore, rate limiter, process pool).
3. State the correctness guarantee (terminal state, dedupe, ordering).
4. State the metric used to verify it.

## Runtime Selection

Q: How do you choose between `asyncio`, threads, and processes?
A: Classify workload first: I/O-bound uses `asyncio` or threads; CPU-bound pure Python uses processes; mixed flows use stage-specific models. The follow-up is usually about evidence, so mention that you validate the choice with p95 latency, throughput, and error-rate measurements.

Q: If I'm not allowed to use external libraries (e.g. `concurrent.futures`"), what should I use?
A: Use the `threading` library.  `threading` is part of Python's standard library, not an external package, and it does not use `concurrent.futures`. The follow-up is to state explicitly that you are staying within stdlib-only constraints.

Q: If you must implement this quickly in a blank single-file editor, what coding order is safest?
A: Build in this order: statuses/result schema, shared state, retry/timeout helpers, stage functions, workers, orchestration, then summary/invariant checks. The follow-up is reliability: this order keeps terminal-state correctness visible from the first runnable version.

Q: Why not run everything with threads?
A: Threads are usually effective for I/O waits, but pure-Python CPU loops are constrained by a single interpreter lock in default CPython builds. The practical follow-up is overhead: too many threads can increase context switching and tail latency.

Q: Why does a 50x `time.sleep(1)` workload still finish near 1 second with threads?
A: Because those tasks are wait-heavy, threads overlap waiting time instead of running sequentially. The follow-up is that this is concurrency benefit from I/O/wait overlap, not CPU parallel speedup.

Q: How do you return ordered results from concurrent workers?
A: Pre-size a result list and write each item to `results[<job_id>]`, then render after all joins. The follow-up is that order of completion can vary while final output order stays deterministic.

Q: Why might you not need locks in the simple indexed-result pattern?
A: If each thread writes to a unique index and final reads happen only after `join()`, there is no conflicting shared write. The follow-up caveat is that duplicate indexes or shared mutable counters would require synchronization.

Q: When do you definitely need a lock?
A: When multiple threads can touch the same mutable state, especially read-modify-write operations like counters or dict updates with invariants. The follow-up is that `Lock` protects critical sections, not whole functions.

Q: When can threads still help CPU-heavy tasks?
A: When heavy work runs inside native extensions that release the interpreter lock, thread pools can scale well. The follow-up is compatibility: always verify extension behavior and benchmark it on your actual runtime.

## Coroutines (async/await) Q&A

Q: What is a coroutine in Python?
A: A coroutine is an `async def` execution unit that can pause at `await` points and later resume under event-loop scheduling. A useful follow-up is that this model enables one thread to multiplex many waiting operations efficiently.

Q: Does calling an `async def` function execute it immediately?
A: No. Calling it returns a coroutine object, and it runs only when awaited or scheduled as a task. The common follow-up pitfall is "coroutine was never awaited" warnings, which indicate lifecycle mistakes.

Q: Why are coroutines useful?
A: They enable high-concurrency I/O with low overhead by letting one thread multiplex many waiting tasks. The follow-up is where they shine: API fanout, async DB calls, and queue workers with many in-flight waits.

Q: When should coroutines be preferred?
A: Prefer them for workloads dominated by waiting, such as network calls, async storage, and latency-bound pipelines. The follow-up is capacity control: combine coroutines with bounded queues/semaphores, not unbounded fanout.

Q: When should coroutines not be the only model?
A: Coroutines alone are not ideal for heavy pure-Python CPU work because event-loop fairness does not create true CPU parallelism. The follow-up is what to use instead: process pools (or validated free-threaded runtime paths).

Q: Are coroutines "standard" in modern Python?
A: Yes. `async`/`await` coroutines are first-class and widely used in production frameworks and SDKs. The follow-up is interoperability: many modern client libraries now expose native async APIs.

Q: What common coroutine mistake causes performance collapse?
A: Putting blocking calls inside coroutines (for example `time.sleep`) freezes the event loop and serializes progress. The follow-up fix is to use non-blocking async APIs or offload blocking work with executors/`asyncio.to_thread`.

## Free-Threaded CPython (No-GIL) Q&A

Q: Which Python versions include free-threaded (no-GIL) support?
A: CPython 3.13 introduced a free-threaded build as an experimental option (`--disable-gil`), and CPython 3.14 moved it to officially supported but still optional (phase II in PEP 779). As of March 4, 2026, this is still not the default interpreter mode.

Q: Is no-GIL the default in mainstream Python installs?
A: No. The free-threaded build is opt-in and default mainstream builds still use the GIL. The follow-up is deployment hygiene: confirm runtime mode explicitly instead of assuming.

Q: Can free-threaded Python still run with the GIL enabled?
A: Yes, free-threaded builds can expose runtime controls such as `PYTHON_GIL` and `-X gil` for compatibility behavior. The follow-up is reproducibility: record runtime flags in your run metadata.

Q: How do you explain model choice if no-GIL exists?
A: Use workload-first logic: free-threaded + thread-safe dependencies can make thread pools viable for CPU-heavy paths. The follow-up is portability: process pools remain a robust default across mixed environments.

Q: Do C extensions matter in free-threaded Python?
A: Yes, extension ecosystem readiness is often the practical constraint. The follow-up question is risk: if extension safety is uncertain, keep conservative concurrency defaults and test thoroughly.

Q: What is the concise "state of no-GIL" answer?
A: "3.13 introduced free-threaded Python experimentally; 3.14 made it officially supported but optional; default CPython remains GIL-enabled." The follow-up is design impact: runtime and dependency compatibility still drive architecture decisions.

Q: What should be measured before switching CPU strategy?
A: Compare throughput, p95/p99 latency, memory footprint, and correctness under your actual dependency set on both runtimes. The follow-up is rollout safety: do an A/B canary before changing production defaults.

## Throughput, Backpressure, and Queueing

Q: Why avoid unbounded `asyncio.gather(...)` on very large inputs?
A: Unbounded fanout can inflate memory and overload dependencies; queue + fixed workers gives explicit pressure control. The follow-up is stability: bounded systems degrade predictably instead of failing abruptly.

Q: How do you scale to 1000 jobs while capping concurrency at 10?
A: Use a bounded worker-pool pattern: enqueue all 1000 jobs into a `queue.Queue` (e.g. `worker_queue`), run exactly 10 worker threads, and let each worker pull until shutdown. The follow-up is that concurrency cap comes from worker count, not input size.

Q: How do I ensure that each worker shuts down upon completion?
A: Use `for _ in range(max_workers): work_queue.put(None)` to enqueue one sentinel per worker to signal a clean shutdown.

Q: How do you prevent OOM with huge streams?
A: Stream/chunk input, keep bounded queues, and cap worker concurrency. The follow-up is observability: track queue depth and resident memory to catch saturation early.

Q: What is backpressure in this context?
A: It is a control mechanism where producers slow down when consumer-side queues/workers are saturated. The follow-up is outcome: backpressure protects both memory and downstream error budgets.

Q: How do you debug deadlocks in async code?
A: Use timeout guards around multi-lock sections, enforce global lock ordering, and instrument lock-wait timings. The follow-up is prevention: simplify lock topology and keep critical sections short.

Q: How do you avoid event-loop freezes?
A: Do not call blocking functions directly inside coroutines; offload blocking work to executors or `asyncio.to_thread`. The follow-up is verification: monitor event-loop lag and callback delay.

Q: How do you preserve deterministic output order while processing concurrently?
A: Attach input index and write terminal result into a pre-sized result array at that index. The follow-up is correctness proof: execution order can vary while render order remains stable.

## Timeouts, Retries, and Failure Isolation

Q: What is a safe retry strategy?
A: Retry only retryable failures, cap attempts, use exponential backoff (plus jitter if needed), and preserve terminal status per item. The follow-up is policy: classify retryable vs non-retryable exceptions explicitly.

Q: Why is jitter important for retries?
A: Without jitter, many clients retry in synchronized waves; jitter spreads retries over time and reduces overload spikes. The follow-up is practical tuning: jitter range should scale with backoff window.

Q: Why should timeout be per attempt, not only global?
A: Per-attempt timeouts prevent individual calls from hanging and consuming worker slots indefinitely. The follow-up is composition: combine per-attempt timeout with a global deadline.

Q: How do you stop one bad item from crashing the whole batch?
A: Catch exceptions per item, emit terminal result, and continue processing other items. The follow-up is output design: include error type and attempts so failures are debuggable.

Q: What do you do after retry exhaustion?
A: Mark terminal failure (`timed_out`/`failed`) or route retryable exhausted items to a `poison` queue/list. The follow-up is operations: provide replay tooling with idempotency safety.

## Idempotency and Side Effects

Q: Why does idempotency matter for retries?
A: Retries can repeat execution, so idempotency keys prevent duplicate side effects. The follow-up is boundary design: dedupe must exist at the side-effect boundary, not only in caller memory.

Q: What is a good `idempotency_key`?
A: A good key is stable for the same logical operation across retries, unique across distinct operations, and scoped to tenant/resource boundaries. A common pattern is `tenant_id:operation_type:client_request_id` with a deterministic payload fingerprint when needed.

Q: What is a bad `idempotency_key`?
A: Keys derived from volatile fields like timestamps generated per retry are bad because they defeat dedupe. Random keys are acceptable only if the client reuses the same key on retries of the same operation.

Q: Should payload hash be part of the key?
A: Often yes, but only over canonicalized fields that define operation identity, not mutable metadata. The follow-up is conflict handling: if same key arrives with non-equivalent payload, reject with explicit conflict error.

Q: How long should idempotency keys be retained?
A: Retain for at least the maximum replay window (client retries + queue delays + recovery window), then expire via TTL. The follow-up is safety: too short TTL reintroduces duplicates; too long TTL increases storage cost.

Q: Where should idempotency state be stored?
A: Store in durable, low-latency shared storage reachable by all workers handling the operation. The follow-up is consistency: dedupe state must be atomic with side-effect commit boundaries.

Q: What is the difference between `deduped` and `success`?
A: `success` means side effect was applied now; `deduped` means side effect was already applied for that key and intentionally skipped. The follow-up is metrics: track both separately to distinguish duplicate traffic from real throughput.

Q: How do you avoid dedupe races with concurrent duplicates?
A: Use lock-protected ownership state (`in_progress` + `completed`) so only one worker can apply a given key. The follow-up is recovery: persist dedupe state when crash safety is required.

## Cancellation and Deadlines

Q: How do you handle a global timeout mid-run?
A: Stop dispatching new work, drain backlog into explicit `cancelled` statuses, and close workers via sentinels. The follow-up is predictability: this guarantees no item disappears silently.

Q: What are cancellation invariants?
A: Every item ends with one terminal status; no silent drops; no lingering non-terminal placeholders. The follow-up is verification: enforce these invariants in stress tests and shutdown-path tests.

## Hybrid Pipelines

Q: Why split stages across different models?
A: Stage characteristics differ; async models fit I/O stages, process pools fit CPU-heavy pure-Python transforms. The follow-up is tuning: optimize each stage independently and then rebalance end-to-end.

Q: How should stage handoff be designed?
A: Use bounded queues between stages and sentinel-based stage shutdown ordering. Bounded queues create explicit backpressure so fast producers cannot overwhelm slower downstream stages and cause memory blowups; sentinels (for example `None`, usually one per worker) provide an unambiguous "no more work" signal so each stage drains and exits cleanly.

Q: What should be measured in multi-stage pipelines?
A: Per-stage latency plus end-to-end latency and per-stage failure counters. The follow-up is bottleneck isolation: queue-depth differentials quickly show which stage is limiting throughput.

## Rate Limiting and External Constraints

Q: How is semaphore different from rate limiter?
A: Semaphore controls in-flight concurrency; rate limiter controls starts per unit time. The follow-up is combined control: use both when dependencies enforce concurrency and requests-per-second limits.

Q: When do you need both?
A: Use both when downstream has capacity limits (concurrency) and policy limits (request rate). The follow-up is failure mode: semaphore-only systems can still violate request-rate budgets.

## Terminal Status Design

Q: What is a terminal status?
A: A terminal status is a final, immutable state for an item after all retry/cancellation logic is complete. The follow-up is contract clarity: downstream consumers should never infer state from missing records.

Q: Which statuses should generally exist?
A: Usually `success`, `failed`, `timed_out`, `cancelled`, plus `deduped`/`poison` where side-effect retries exist. The follow-up is schema stability: keep status vocabulary explicit and versioned.

Q: Why is `started` not a terminal status?
A: `started` is transitional state, not final outcome; it should appear in counters/events, not final per-item records. The follow-up is integrity checks: use `started` vs `completed` parity to detect unresolved work before publishing results.

Q: How do you validate terminal-state correctness?
A: Assert one terminal record per input item, no missing entries, and status-set membership in tests. The follow-up is runtime guardrails: fail summary/export if any invariant is violated.

Q: How do you detect missing terminal statuses in a live run?
A: Compare queue/counter invariants (`queued`, `started`, `completed`) and scan for unresolved result slots (`None`/placeholder). A non-zero `started - completed` gap indicates unresolved items that must be reconciled before final summary.

Q: What do you do if the process crashes mid-batch?
A: Reconstruct unresolved items from durable progress logs/checkpoints (`started_ids - terminal_ids`) and replay them. Use idempotency keys so replay is safe even if side effects were partially applied before the crash.

Q: Should unresolved items be silently ignored?
A: No. Every item must end in explicit terminal state, even in recovery mode. If true outcome is unknown at first pass, mark deterministically (for example `cancelled` with recovery context) and reconcile later.

Q: What follow-up question usually comes next?
A: "How do you avoid double-apply during replay?" Use idempotency/dedupe state at the side-effect boundary and make replay idempotent by design. The follow-up is evidence: show replay tests and duplicate-injection tests.

## Performance Tuning

Q: Which metrics do you collect first?
A: Status counts, retry count, queue depth, in-flight count, and p50/p95/p99 latency. The follow-up is prioritization: tail metrics and error modes usually matter more than average latency.

Q: How do you tune worker count safely?
A: Increase incrementally and stop when p95/p99 or failure rates regress. The follow-up is steady-state testing: run long enough to include warmup and saturation behavior.

Q: Why focus on p95/p99 and not only average latency?
A: Tail latency dominates user-visible reliability and overload behavior. The follow-up is SLO alignment: average latency can look good while p99 is failing badly.

## NVIDIA GPU Control-Plane Q&A

Q: How does this concurrency model map to NVIDIA GPU cloud workloads?
A: The same control-plane primitives apply: bounded queues for admission, semaphores for per-GPU slot limits, and explicit terminal states for job lifecycle. The follow-up is resource awareness: include GPU memory and utilization checks in admission decisions.

Q: Do I need CUDA kernel coding to discuss GPU concurrency architecture?
A: No. Control-plane design is about job scheduling, backpressure, retries, deadlines, and observability around GPU-backed tasks. The follow-up is examples: job launch orchestration and health polling are enough for most discussions.

Q: What are common GPU-specific failure modes to mention?
A: OOM, launch timeouts, host-side queue buildup, and noisy-neighbor contention are common modes. The follow-up is mitigation: admission control, retry classification, and priority or quota policies.

Q: What extra metrics should be tracked for GPU-backed pipelines?
A: Queue depth and latency should be paired with GPU utilization, memory pressure, and per-device error counts. The follow-up is diagnosis: low utilization with high queue depth usually indicates host or scheduling bottlenecks.

Q: How do retries differ for GPU workloads?
A: Retry only transient failures; avoid immediate retries on deterministic OOM without changing batch size or placement. The follow-up is policy: include adaptive backoff and resource-shaping before replay.

## Docker, Kubernetes, and Slurm-on-Kubernetes Q&A

Q: Which Kubernetes primitive should run concurrency workers?
A: Use `Deployment` for long-running consumers and `Job` for finite batches. The follow-up is lifecycle control: match primitive semantics to workload completion behavior.

Q: How do you keep worker pods from thrashing on startup?
A: Use `startupProbe` for slow initialization and gate traffic with `readinessProbe`. The follow-up is restart safety: avoid aggressive liveness probes that trigger unnecessary churn.

Q: What is the shutdown contract for queue workers in Kubernetes?
A: On `SIGTERM`, stop intake first, then drain/terminalize in-flight items before exit. The follow-up is platform config: set `terminationGracePeriodSeconds` to cover worst-case drain time.

Q: Why are idempotency keys still required in container orchestration?
A: Pods can restart or be rescheduled unexpectedly, creating replay paths. The follow-up is correctness: idempotency keeps retries/replays from duplicating side effects.

Q: How do GPU requests work in Kubernetes?
A: Request extended resources like `nvidia.com/gpu`, and ensure the NVIDIA device plugin/runtime are configured cluster-wide. The follow-up is placement: align node labels/taints/affinity with GPU job policy.

Q: What is Slurm-on-Kubernetes in practical terms?
A: Slurm provides HPC queueing/allocation policy while Kubernetes handles container orchestration and cluster lifecycle. The follow-up is ownership: make one layer the clear authority for allocation decisions to avoid policy conflicts.

Q: What does \"double scheduling\" mean in Slurm-on-Kubernetes?
A: It means Slurm and Kubernetes both try to make independent placement decisions, creating mismatch and pending-state confusion. The follow-up mitigation is explicit integration rules and unified resource accounting.

Q: What are high-signal debug points in this stack?
A: Correlate queue depth, Slurm queue wait, Kubernetes pod pending reasons, and GPU utilization/memory pressure. The follow-up is diagnosis discipline: use cross-layer timelines, not single-metric guesses.

Q: What causes `CrashLoopBackOff` in worker pods most often?
A: Startup dependency failures, bad env/config, or liveness probes that trigger too early are common causes. The follow-up is action order: inspect container logs/events first, then probe configuration and startup timing.

Q: Requests vs limits: what matters most for scheduling?
A: Requests drive placement guarantees; limits cap runtime usage and can trigger throttling/eviction behavior when misconfigured. The follow-up is stability: set realistic requests to avoid noisy-neighbor instability.

Q: How do you avoid image pull bottlenecks during scale-out?
A: Use smaller immutable images, regional/local registries, and pre-pull strategies on node pools. The follow-up is observability: monitor image pull latency separately from app startup latency.

## Compact Architecture Pattern

1. Bounded queue for pending items.
2. Fixed worker pool.
3. Semaphore and/or rate limiter for external constraints.
4. Per-item timeout + retry + explicit terminal status.
5. Ordered rendering by input index.
6. Metrics for throughput and tail-latency tuning.

## Primary References (Free-Threaded Status)

- `https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython`
- `https://docs.python.org/3.14/whatsnew/3.14.html#free-threaded-python-is-officially-supported`
- `https://docs.python.org/3.14/howto/free-threading-python.html`
- `https://peps.python.org/pep-0703/`
- `https://peps.python.org/pep-0779/`
