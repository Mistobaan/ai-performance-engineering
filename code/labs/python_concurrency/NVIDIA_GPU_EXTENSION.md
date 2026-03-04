# NVIDIA GPU Extension Track (Control-Plane Focus)

This guide extends the Python concurrency material into GPU-backed service and batch systems.
It focuses on orchestration and reliability patterns, not CUDA kernel implementation.

## Quick Summary

Control-plane algorithm:
1. Admit GPU jobs through bounded queues.
2. Limit per-device concurrency with semaphores/slots.
3. Apply timeout/retry policy with explicit failure classes.
4. Persist terminal outcomes so replay and recovery are deterministic.

## Why This Extension Matters

The same concurrency primitives from this lab map directly to GPU control planes:
- bounded queues -> admission control for GPU jobs,
- semaphores -> per-GPU slot limits,
- terminal statuses -> deterministic job lifecycle outcomes,
- retries/timeouts -> robust failure handling under device and dependency faults.

## Concept Mapping

| Concurrency concept | GPU control-plane equivalent |
| --- | --- |
| `Queue(maxsize=N)` | Pending GPU job queue with backpressure |
| worker pool | Launch workers that submit jobs to GPUs |
| semaphore | Per-GPU or per-pool concurrency cap |
| timeout | Max job runtime / launch timeout |
| retry + jitter | Controlled replay for transient failures |
| terminal status | `success`, `oom`, `timed_out`, `failed`, `cancelled`, `deduped` |

## Example 1: Bounded GPU Slot Scheduler

```python
import asyncio
from dataclasses import dataclass

@dataclass
class GpuJob:
    job_id: str
    payload: dict
    timeout_s: float = 30.0

async def launch_on_gpu(job: GpuJob, gpu_id: int) -> dict:
    # placeholder for actual launch RPC/subprocess call
    await asyncio.sleep(0.2)
    return {"job_id": job.job_id, "gpu": gpu_id, "status": "success"}

async def run_gpu_jobs(jobs: list[GpuJob], gpu_ids: list[int], slots_per_gpu: int = 1):
    # Queue of available GPU slots; bounded by physical scheduling policy.
    gpu_slots: asyncio.Queue[int] = asyncio.Queue()
    for gpu in gpu_ids:
        for _ in range(slots_per_gpu):
            gpu_slots.put_nowait(gpu)

    results = [None] * len(jobs)

    async def run_one(idx: int, job: GpuJob):
        gpu = await gpu_slots.get()
        try:
            result = await asyncio.wait_for(launch_on_gpu(job, gpu), timeout=job.timeout_s)
            results[idx] = result
        except asyncio.TimeoutError:
            results[idx] = {"job_id": job.job_id, "gpu": gpu, "status": "timed_out"}
        except Exception as exc:  # noqa: BLE001
            results[idx] = {"job_id": job.job_id, "gpu": gpu, "status": "failed", "error": str(exc)}
        finally:
            gpu_slots.put_nowait(gpu)

    await asyncio.gather(*(run_one(i, job) for i, job in enumerate(jobs)))
    return results
```

Key points:
- availability is explicit (`gpu_slots`),
- capacity is bounded,
- timeout/failure are terminalized per job,
- slot release happens in `finally`, preventing dead slot leaks.

## Example 2: Async GPU Health Polling (Control Plane)

```python
import asyncio

async def poll_gpu_metrics(interval_s: float = 2.0):
    while True:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode == 0:
            lines = out.decode().strip().splitlines()
            # parse lines and emit structured metrics
            print({"gpu_rows": lines})
        else:
            print({"status": "poll_failed", "error": err.decode().strip()})

        await asyncio.sleep(interval_s)
```

Key points:
- non-blocking polling integrates into event loop cleanly,
- failures are captured as explicit events,
- metrics can drive admission or auto-throttle decisions.

## Example 3: Cancellation-Safe Global Drain

```python
import asyncio

async def drain_queue_with_cancellation(job_queue: asyncio.Queue, terminal_results: dict, stop_event: asyncio.Event):
    # Stop producers first, then mark unresolved queued work deterministically.
    stop_event.set()
    while True:
        try:
            job_id = job_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if job_id not in terminal_results:
            terminal_results[job_id] = {"status": "cancelled", "error": "global_shutdown"}
        job_queue.task_done()
```

Key points:
- no silent drops during shutdown,
- unresolved items become explicit terminal records,
- recovery/replay can operate from deterministic state.

## GPU-Focused Q&A Prompts

1. How do you prevent GPU OOM cascades under bursty load?
2. How do you decide retryable vs non-retryable GPU failures?
3. How do you handle per-GPU heterogeneity in scheduling policy?
4. What metrics prove bottleneck is host scheduling vs GPU compute?
5. How do you drain safely during deploy/shutdown without losing jobs?

## Practical Answer Pattern

1. Correctness first: terminal states for every job, no silent drops.
2. Pressure control: bounded queue + per-GPU slot limits.
3. Resilience: timeout/retry policy with jitter and explicit failure classes.
4. Observability: queue depth + p95 latency + GPU utilization/memory + error taxonomy.
5. Recovery: replay unresolved work with idempotency safeguards.
