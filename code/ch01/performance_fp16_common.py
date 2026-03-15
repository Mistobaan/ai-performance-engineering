"""Local workload tuning for the Chapter 1 precision-only benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceFP16Workload:
    """Compute-heavy shape that makes precision-only gains measurable."""

    batch_size: int = 32
    num_microbatches: int = 16
    hidden_dim: int = 16384

    @property
    def samples_per_iteration(self) -> float:
        return float(self.batch_size * self.num_microbatches)


PERFORMANCE_FP16_WORKLOAD = PerformanceFP16Workload()

