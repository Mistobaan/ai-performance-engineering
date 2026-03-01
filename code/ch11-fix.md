# Chapter 11 Fixes

## Narrative / Prose

### BEFORE
```text
One subtle but important legality rule is that for a block-scoped pipeline, all CTA
threads participate in producer_acquire/commit and consumer_wait/release. You can
still keep warp specialization by letting only selected warps issue memcpy_async work.
For memory visibility and cross-role handoff, combine these collectives with explicit
cta.sync() points where ownership changes.
```

### AFTER
```text
One subtle but important legality rule is that for a block-scoped pipeline, all CTA
threads participate in producer_acquire/commit and consumer_wait/release. You can
still keep warp specialization by letting only selected warps issue memcpy_async work.
For memory visibility and cross-role handoff, combine these collectives with explicit
cta.sync() points where ownership changes. Do not place block-scoped pipeline
collectives only inside warp-role branches; that mismatched collective sequence can
deadlock.
```

## Code Pattern (Two Pipelines and Cluster Pipeline)

### BEFORE
```cpp
// UNSAFE pattern for block-scoped pipelines in warp-specialized code.
if (warp_id == loader_warp) {
  pipe_lc.producer_acquire();
  // load
  pipe_lc.producer_commit();
}
if (warp_id == compute_warp) {
  pipe_lc.consumer_wait();
  // compute
  pipe_lc.consumer_release();
}
if (warp_id == storer_warp) {
  pipe_cs.consumer_wait();
  // store
  pipe_cs.consumer_release();
}
```

### AFTER
```cpp
// SAFE pattern used by chapter-11 implementations:
// collectives are CTA-uniform; work is role-specific between sync points.
pipe_lc.consumer_wait();
cta.sync();

if (warp_id == compute_warp) {
  // compute
}
cta.sync();
pipe_lc.consumer_release();

pipe_cs.producer_acquire();
pipe_cs.producer_commit();
pipe_cs.consumer_wait();
cta.sync();

if (warp_id == storer_warp) {
  // store
}
cta.sync();
pipe_cs.consumer_release();
```
