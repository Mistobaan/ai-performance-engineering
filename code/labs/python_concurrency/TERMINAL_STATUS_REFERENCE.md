# Terminal Status Reference

## Quick Summary

Terminal-state algorithm:
1. Every input must end in exactly one final status.
2. No unresolved placeholders at output time.
3. Lifecycle counters (`queued/started/completed`) must reconcile.
4. Unknown or missing states trigger recovery/replay, never silent drop.

## Definition

A terminal status is the final immutable state of one item after all logic
(timeouts, retries, dedupe, cancellation) has completed.

A correct concurrency pipeline must guarantee:
1. exactly one terminal status per input item,
2. no item left unresolved,
3. status transitions are explicit and testable.

## Why Terminal Status Matters

1. Correctness:
   - Prevents silent data loss and duplicate side effects.
2. Observability:
   - Enables accurate success/failure accounting.
3. Recovery:
   - Supports replay policies and checkpointing.
4. Testing:
   - Allows deterministic invariant checks.

## Status Taxonomy in This Lab

### Common statuses
- `success`: completed all required stages.
- `failed`: terminal non-retryable error.
- `timed_out`: retryable operation exceeded timeout and exhausted retries.
- `cancelled`: stopped by global shutdown/deadline policy.

### Reliability-specific statuses
- `deduped`: duplicate idempotency key detected; side effect intentionally skipped.
- `poison`: retryable failure exhausted; routed to poison handling.

### Stage-specific statuses (hybrid pipeline)
- `failed_stage_a`
- `failed_stage_b`
- `failed_stage_c`

## Valid Transition Patterns

### Timeout/retry flow
`pending -> running -> timed_out -> retrying -> running -> success|timed_out`

### Cancellation flow
`pending -> cancelled`
`running -> cancelled` (policy-dependent)

### Idempotent flow
`pending -> deduped` (if prior completion exists)
`pending -> running -> success -> completed_key`

### Poison flow
`pending -> running -> transient_error -> retrying -> ... -> poison`

## Anti-Patterns

1. Hidden non-terminal placeholders (`None`, missing records).
2. Multiple terminal statuses for same input key.
3. Retrying non-idempotent side effects without dedupe keys.
4. Using exception logs as status instead of explicit status fields.

## Validation Checklist

1. `len(results) == len(inputs)`
2. status of each result in allowed terminal set
3. index mapping preserved for deterministic output
4. retries counter consistent with policy
5. dedupe/poison counts align with observed item outcomes
