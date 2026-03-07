"""Fixture module for wrapper-following contract tests."""

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark


class SharedBench(VerificationPayloadMixin, BaseBenchmark):
    def setup(self):
        pass

    def benchmark_fn(self):
        import torch

        self._x = torch.randn(8, 8, device=self.device)

    def validate_result(self):
        return None

    def teardown(self):
        pass

    def capture_verification_payload(self):
        self._set_verification_payload(
            inputs={"x": self._x},
            output=self._x,
            batch_size=1,
            parameter_count=0,
        )
