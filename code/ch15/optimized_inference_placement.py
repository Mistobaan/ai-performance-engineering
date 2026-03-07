"""Optimized inference placement policy honoring NVLink-local TP/EP."""

from __future__ import annotations

from typing import Optional

from ch15.baseline_inference_placement import (  # noqa: E402
    _PlacementBenchmark,
    PlacementConfig,
)


class OptimizedInferencePlacementBenchmark(_PlacementBenchmark):
    """Heuristic-aligned placement: TP intra-node for prefill, TP=1 for decode, sticky sessions."""

    def __init__(self) -> None:
        cfg = PlacementConfig(
            prefill_tp_size=8,
            prefill_span_nodes=False,  # keep TP inside the NVLink island
            decode_tp_size=1,  # collapse TP for decode to kill all-reduce
            decode_span_nodes=False,
            decode_microbatch=4,
            remote_expert_fraction=0.05,  # expert pinning favors local shards
            router_sticky_decode=True,
            kv_transfer_policy="local_only",  # never walk KV across nodes mid-session
            notes="Prefill TP within node, decode TP=1, MoE local-first, KV stickiness.",
        )
        super().__init__(cfg, prefix="placement_optimized")


    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for inference_placement."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 10.0),
            tpot_ms=getattr(self, '_tpot_ms', 1.0),
            total_tokens=getattr(self, '_total_tokens', 100),
            total_requests=getattr(self, '_total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark():
    return OptimizedInferencePlacementBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
