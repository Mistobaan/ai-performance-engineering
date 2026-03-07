from __future__ import annotations

import ast
import textwrap

from core.benchmark.contract import BenchmarkContract
from core.harness.validity_checks import (
    check_benchmark_fn_antipatterns,
    check_benchmark_fn_sync_calls,
)


def _parse_class(source: str) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(source))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def test_contract_warns_on_sync_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                self._synchronize()
                torch.cuda.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("_synchronize()" in warning for warning in warnings)
    assert any("torch.cuda.synchronize()" in warning for warning in warnings)


def test_contract_warns_on_stream_and_event_synchronize_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                stream.synchronize()
                self._poll_event.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("stream/event synchronize()" in warning for warning in warnings)


def test_contract_warns_on_local_event_variable_synchronize_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                end_event = torch.cuda.Event(enable_timing=True)
                alias = end_event
                alias.synchronize()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("stream/event synchronize()" in warning for warning in warnings)


def test_runtime_sync_check_detects_hot_path_synchronization() -> None:
    class SyncBench:
        def _synchronize(self) -> None:
            pass

        def benchmark_fn(self) -> None:
            self._synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("_synchronize()" in finding for finding in findings)


def test_runtime_sync_check_detects_stream_or_event_synchronization() -> None:
    class SyncBench:
        def benchmark_fn(self) -> None:
            stream.synchronize()
            self._poll_event.synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("stream/event synchronize()" in finding for finding in findings)


def test_runtime_sync_check_detects_local_event_variable_synchronization() -> None:
    class SyncBench:
        def benchmark_fn(self) -> None:
            end_event = torch.cuda.Event(enable_timing=True)
            alias = end_event
            alias.synchronize()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("stream/event synchronize()" in finding for finding in findings)


def test_contract_warns_on_sync_inside_same_class_helper_called_by_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class SyncBench:
            def setup(self):
                pass

            def _helper(self):
                torch.cuda.synchronize()

            def benchmark_fn(self):
                self._helper()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("torch.cuda.synchronize()" in warning for warning in warnings)


def test_runtime_sync_check_detects_same_class_helper_synchronization() -> None:
    class SyncBench:
        def _helper(self) -> None:
            torch.cuda.synchronize()

        def benchmark_fn(self) -> None:
            self._helper()

    ok, findings = check_benchmark_fn_sync_calls(SyncBench().benchmark_fn)

    assert not ok
    assert any("torch.cuda.synchronize()" in finding for finding in findings)


def test_runtime_sync_check_ignores_clean_benchmark_fn() -> None:
    class CleanBench:
        def benchmark_fn(self) -> None:
            x = 1 + 1
            assert x == 2

    ok, findings = check_benchmark_fn_sync_calls(CleanBench().benchmark_fn)

    assert ok
    assert findings == []


def test_runtime_sync_check_respects_allowlist() -> None:
    class AllowedBench:
        def benchmark_fn(self) -> None:
            torch.cuda.synchronize()

    ok, findings = check_benchmark_fn_sync_calls(
        AllowedBench().benchmark_fn,
        allowed_codes=("sync",),
    )

    assert ok
    assert findings == []


def test_contract_warns_on_random_input_regeneration_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                torch.randn(8, 8, device=self.device)

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_contract_warns_on_host_transfer_inside_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def benchmark_fn(self):
                x = y.cpu()
                z = y.to("cpu")
                return x, z

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any(".cpu()" in warning for warning in warnings)
    assert any(".to('cpu')" in warning for warning in warnings)


def test_runtime_antipattern_check_detects_hot_path_allocations() -> None:
    class AntiPatternBench:
        def benchmark_fn(self) -> None:
            torch.randn(8, 8)

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any("regenerates random inputs" in finding for finding in findings)


def test_contract_warns_on_antipattern_inside_same_class_helper_called_by_benchmark_fn() -> None:
    class_node = _parse_class(
        """
        class AntiPatternBench:
            def setup(self):
                pass

            def _helper(self):
                torch.randn(8, 8, device=self.device)

            def benchmark_fn(self):
                self._helper()

            def teardown(self):
                pass

            def get_input_signature(self):
                return {}

            def validate_result(self):
                return None

            def get_verify_output(self):
                return None

            def get_output_tolerance(self):
                return (1e-5, 1e-5)
        """
    )

    errors, warnings = BenchmarkContract.validate_benchmark_class_ast(class_node)

    assert not errors
    assert any("regenerates random inputs" in warning for warning in warnings)


def test_runtime_antipattern_check_detects_same_class_helper_antipattern() -> None:
    class AntiPatternBench:
        def _helper(self) -> None:
            torch.randn(8, 8)

        def benchmark_fn(self) -> None:
            self._helper()

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any("regenerates random inputs" in finding for finding in findings)


def test_runtime_antipattern_check_detects_host_transfer() -> None:
    class AntiPatternBench:
        def benchmark_fn(self) -> None:
            value.cpu()
            value.to("cpu")

    ok, findings = check_benchmark_fn_antipatterns(AntiPatternBench().benchmark_fn)

    assert not ok
    assert any(".cpu()" in finding for finding in findings)
    assert any(".to('cpu')" in finding for finding in findings)


def test_runtime_antipattern_check_respects_allowlist() -> None:
    class AllowedBench:
        def benchmark_fn(self) -> None:
            value.cpu()

    ok, findings = check_benchmark_fn_antipatterns(
        AllowedBench().benchmark_fn,
        allowed_codes=("host_transfer",),
    )

    assert ok
    assert findings == []
