"""Optimized training-speed benchmark using CUDA graph replay."""

from __future__ import annotations

import copy
from typing import Optional

import torch

from ch13.baseline_training_speed import BaselineTrainingSpeedBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTrainingSpeedBenchmark(BaselineTrainingSpeedBenchmark):
    """Replay the fixed-shape training step from a captured CUDA graph."""

    def __init__(self) -> None:
        super().__init__()
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        super().setup()
        if any(v is None for v in (self.model, self.input_ids, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        saved_model_state = copy.deepcopy(self.model.state_dict())
        saved_opt_state = copy.deepcopy(self.optimizer.state_dict())

        for _ in range(3):
            self._train_step(self.input_ids, self.targets)
        self._synchronize()
        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)

        self.static_input = self.input_ids.clone()
        self.static_target = self.targets.clone()
        self.output_buffer = torch.empty((1, 1, 8), device=self.device, dtype=torch.float32)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()

        def _captured_step() -> None:
            assert self.output_buffer is not None
            logits = self._train_step(self.static_input, self.static_target)
            self.output_buffer.copy_(logits[:1, :1, :8].detach().float())

        with torch.cuda.stream(self.capture_stream):
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                _captured_step()
        self.capture_stream.synchronize()

        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)
        self.output = None

    def benchmark_fn(self) -> None:
        if any(v is None for v in (self.graph, self.capture_stream, self.input_ids, self.targets, self.static_input, self.static_target, self.output_buffer)):
            raise RuntimeError("CUDA graph not initialized")

        with self._nvtx_range("optimized_training_speed"):
            with torch.cuda.stream(self.capture_stream):
                self.static_input.copy_(self.input_ids)
                self.static_target.copy_(self.targets)
                self.graph.replay()
            self.capture_stream.synchronize()
            self.output = self.output_buffer.detach().clone()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def teardown(self) -> None:
        self.graph = None
        self.capture_stream = None
        self.static_input = None
        self.static_target = None
        self.output_buffer = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "graph_replay": 1.0,
            "autocast_bf16": 1.0,
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedTrainingSpeedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
