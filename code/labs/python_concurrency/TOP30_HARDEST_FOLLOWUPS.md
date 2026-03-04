# Top 30 Hardest Follow-Ups (One-Line Answers)

## Quick Summary

Use these one-liners with this structure:
1. State the design rule.
2. Name the concrete mechanism.
3. Mention one failure mode it prevents.

1. Q: How do you choose `asyncio` vs processes quickly?
   A: Classify bottleneck first: wait-heavy I/O -> `asyncio`; heavy pure-Python CPU -> processes.

2. Q: Why not just `asyncio.gather` everything?
   A: Unbounded fanout turns input spikes into memory pressure and downstream overload.

3. Q: How do you prove ordered output under concurrency?
   A: Preserve input index and write result to that index regardless of completion order.

4. Q: What prevents one slow call from stalling the system?
   A: Per-attempt timeout plus bounded retries and bounded worker slots.

5. Q: What is the minimum retry policy you trust?
   A: Retry only retryable errors with capped exponential backoff and jitter.

6. Q: Why is jitter required, not optional?
   A: It desynchronizes retries and prevents thundering-herd replays.

7. Q: What is a good `idempotency_key`?
   A: Stable per logical operation, unique across operations, and scoped by tenant/resource.

8. Q: What makes an idempotency key bad?
   A: Any key that changes on retry (for example timestamp-only or regenerated random IDs).

9. Q: Should payload hash be included in idempotency?
   A: Include a canonical payload fingerprint when payload identity must be enforced.

10. Q: What if same key arrives with different payload?
    A: Treat as explicit conflict and reject; never silently apply ambiguous side effects.

11. Q: How long should idempotency records live?
    A: At least max replay window (retry horizon + queue delay + recovery interval).

12. Q: How do you avoid duplicate side effects during crash recovery?
    A: Replay unresolved items behind the same idempotency boundary used in normal processing.

13. Q: Why is `started` not a terminal status?
    A: `started` is transitional; terminal states must be final immutable outcomes.

14. Q: How do you detect missing terminal states live?
    A: Enforce `queued/started/completed` parity and fail export on missing result slots.

15. Q: What do you do when process crashes mid-batch?
    A: Reconcile `started_ids - terminal_ids` and replay safely via idempotent boundaries.

16. Q: How do you shut down without losing work?
    A: Stop intake, drain or terminalize backlog, and exit workers via sentinels.

17. Q: Why one sentinel per worker?
    A: Each worker needs an explicit termination token to avoid hanging on `queue.get()`.

18. Q: How do you avoid deadlock in async locks?
    A: Enforce global lock ordering and keep critical sections short with timeout guards.

19. Q: How do you avoid event-loop freezes?
    A: Keep coroutine paths non-blocking and offload blocking calls to executor/to_thread.

20. Q: What metrics are non-negotiable?
    A: Status counts, retries, queue depth, in-flight, and p50/p95/p99 latency.

21. Q: How do you tune worker count safely?
    A: Increase incrementally and stop when p95/p99 or error rates regress.

22. Q: Why can average latency mislead?
    A: Mean can improve while p99 degrades, masking overload and user-visible failures.

23. Q: Deployment or Job for Kubernetes workers?
    A: `Deployment` for continuous consumers, `Job` for finite completion semantics.

24. Q: What is the Kubernetes shutdown contract?
    A: On `SIGTERM`, stop intake first, then drain/terminalize before exit.

25. Q: Why do workers crash-loop in clusters?
    A: Startup dependency failures, bad config, or overly aggressive liveness probes.

26. Q: Requests vs limits in Kubernetes?
    A: Requests drive placement guarantees; limits constrain runtime and can throttle.

27. Q: How do GPU requests work on Kubernetes?
    A: Request `nvidia.com/gpu` and align labels/taints/affinity with placement policy.

28. Q: What is Slurm-on-Kubernetes risk number one?
    A: Double scheduling ambiguity if allocation authority is not clearly defined.

29. Q: What failure classes are common in GPU control planes?
    A: OOM, timeout, launch failure, and host-side queue saturation.

30. Q: How do you discuss GPU architecture without kernel coding?
    A: Focus on admission control, scheduling, retries, terminalization, and telemetry.
