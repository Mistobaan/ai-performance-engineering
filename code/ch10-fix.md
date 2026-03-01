# Chapter 10 Fixes

## Narrative / Prose

### BEFORE
```text
The key advantage of using the CUDA Pipeline API’s producer and consumer calls
(e.g., pipe.producer_acquire(), pipe.producer_commit(), pipe.consumer_wait(),
and pipe.consumer_release()) is that they synchronize only the specific warps or
stages that actually need to hand off data. This is in contrast to forcing every thread in
a block to wait.
A block-wide barrier would stall every warp—even those that are not involved with
the producer-consumer pipeline. All execution in that block must pause until every
thread reaches the barrier, as shown in Figure 10-7.
By comparison, the Pipeline API maintains per-stage state internally. When a pro-
ducer warp finishes its asynchronous copy and calls pipe.producer_commit, only the
warps that call pipe.consumer_wait will block until the data is ready. Other warps in
the block can continue running any work that does not depend on that stage. In prac-
tice, the CUDA Pipeline API reduces idle time and decreases stalled warps because it
eliminates the need to pause the entire block with a barrier. With pipelines you coor-
dinate producer and consumer handoffs at a finer granularity than a hand-coded
async-copy sequence (e.g., PTX cp.async + __syncthreads()).
```

### AFTER
```text
For a block-scoped pipeline, the producer and consumer calls
(pipe.producer_acquire(), pipe.producer_commit(), pipe.consumer_wait(), and
pipe.consumer_release()) are collective CTA operations. Every thread in the block
must execute these calls in the same order.
A block-wide barrier would stall every warp—even those that are not involved with
the producer-consumer pipeline. All execution in that block must pause until every
thread reaches the barrier, as shown in Figure 10-7.
By comparison, the Pipeline API maintains per-stage state internally while still requir-
ing collective participation at each acquire/commit/wait/release point. Warp special-
ization remains valid because only role-specific warps perform role-specific work be-
tween those collective calls (for example, loader memcpy_async work, compute, and
store), with explicit handoff synchronization points. In practice, this reduces idle time
and stalled warps compared with coarse hand-written barrier schemes while preserving
correctness and avoiding branch-divergent pipeline deadlocks.
```

## Code Pattern (Warp-Specialized Pipeline Collectives)

### BEFORE
```cpp
// UNSAFE pattern for a block-scoped pipeline:
// collectives are executed only by role-specific branches.
if (warp_id == 0) {
  pipe.producer_acquire();
  // load
  pipe.producer_commit();
}
if (warp_id == 1) {
  pipe.consumer_wait();
  // compute
  pipe.consumer_release();
}
if (warp_id == 2) {
  pipe.consumer_wait();
  // store
  pipe.consumer_release();
}
```

### AFTER
```cpp
// SAFE pattern for a block-scoped pipeline:
// collectives are CTA-uniform; role work is branch-specific.
pipe.producer_acquire();
// cooperative load
pipe.producer_commit();

pipe.consumer_wait();
cta.sync();

if (warp_id == 1) {
  // compute
}
cta.sync();

if (warp_id == 2) {
  // store
}
cta.sync();

pipe.consumer_release();
cta.sync();
```
