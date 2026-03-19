from __future__ import annotations

from pathlib import Path

import torch

from core.benchmark.verification import InputSignature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark
from core.scripts.ci.check_verification_compliance import check_file_compliance
from core.scripts.validate_benchmark_pairs import discover_benchmark_pairs, get_input_signature_safe


class _DummyPayloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, device=self.device)
        self.y = None

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self.y = self.x * 2

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.y,
            batch_size=int(self.x.shape[0]),
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )


def test_get_input_signature_safe_executes_payload_path() -> None:
    bench = _DummyPayloadBenchmark()
    sig, err = get_input_signature_safe(bench)
    assert err is None
    assert isinstance(sig, InputSignature)


def test_discover_benchmark_pairs_matches_optimized_variants(tmp_path: Path) -> None:
    ch_dir = tmp_path / "ch01"
    ch_dir.mkdir(parents=True)
    (ch_dir / "baseline_foo.py").write_text("# baseline\n")
    (ch_dir / "optimized_foo.py").write_text("# optimized\n")
    (ch_dir / "optimized_foo_bar.py").write_text("# optimized variant\n")

    pairs = discover_benchmark_pairs(tmp_path, chapter="ch01")
    assert "ch01:foo" in pairs
    assert "ch01:foo_bar" in pairs
    assert pairs["ch01:foo_bar"]["baseline"] == ch_dir / "baseline_foo.py"
    assert pairs["ch01:foo_bar"]["optimized"] == ch_dir / "optimized_foo_bar.py"


def test_ci_compliance_checker_accepts_payload_mixin_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class Dummy(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message and issue.severity == "error" for issue in issues)


def test_ci_compliance_checker_accepts_local_benchmark_base_inheritance(tmp_path: Path) -> None:
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
        "class ChildBenchmark(ParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)


def test_ci_compliance_checker_accepts_imported_benchmark_base_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base_module.py").write_text(
        "from core.benchmark.verification_mixin import VerificationPayloadMixin\n"
        "from core.harness.benchmark_harness import BaseBenchmark\n"
        "class ImportedParentBenchmark(VerificationPayloadMixin, BaseBenchmark):\n"
        "    allow_cpu = True\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    file_path = tmp_path / "baseline_dummy.py"
    file_path.write_text(
        "from base_module import ImportedParentBenchmark\n"
        "class ChildBenchmark(ImportedParentBenchmark):\n"
        "    def benchmark_fn(self):\n"
        "        return None\n"
    )
    issues = check_file_compliance(file_path)
    assert not any("get_input_signature" in issue.message for issue in issues)
