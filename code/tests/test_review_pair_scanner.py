from __future__ import annotations

from pathlib import Path

from core.verification import review_baseline_optimized_pairs as pair_review
from core.verification.review_baseline_optimized_pairs import CodeReviewer, dedupe_issues


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


_BENCHMARK_PREFIX = """
from __future__ import annotations

import argparse
import torch
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata
"""


def test_compare_pair_does_not_flag_valid_graph_replay_as_work_reduction(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_graph.py",
        _BENCHMARK_PREFIX
        + """
class BaselineGraphBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def setup(self):
        self.tensor = torch.ones(self.batch_size, device=self.device)

    def _step(self):
        self.tensor.add_(1)

    def benchmark_fn(self):
        for _ in range(self.num_loops):
            self._step()


def get_benchmark():
    return BaselineGraphBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_graph.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedGraphBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def setup(self):
        self.tensor = torch.ones(self.batch_size, device=self.device)
        torch.cuda.synchronize()

    def _replay(self):
        self.tensor.add_(1)

    def benchmark_fn(self):
        self._replay()


def get_benchmark():
    return OptimizedGraphBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "work_reduction" for issue in issues)


def test_compare_pair_ignores_setup_only_synchronizations(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_sync.py",
        _BENCHMARK_PREFIX
        + """
class BaselineSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineSyncBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_sync.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedSyncBenchmark(BaseBenchmark):
    def setup(self):
        torch.cuda.synchronize()

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedSyncBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "sync_mismatch" for issue in issues)


def test_check_file_ignores_dummy_rows_and_mock_decode_help_text(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    source = _write(
        tmp_path,
        "optimized_decode_graphs.py",
        _BENCHMARK_PREFIX
        + """
class DecodeGraphBenchmark(BaseBenchmark):
    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden", help="Hidden size for the mock decode op.")

    def benchmark_fn(self):
        # without paying full attention cost on dummy rows.
        return None


def get_benchmark():
    return DecodeGraphBenchmark()
""",
    )

    issues = reviewer.check_file(source, "optimized")

    assert not any(issue["type"] == "suspicious_pattern" for issue in issues)


def test_dedupe_issues_collapses_alias_duplicates() -> None:
    duplicated = [
        {"type": "work_reduction", "file": "a vs b", "message": "same", "severity": "medium"},
        {"type": "work_reduction", "file": "a vs b", "message": "same", "severity": "medium"},
    ]

    assert dedupe_issues(duplicated) == [duplicated[0]]


def test_compare_pair_flags_true_workload_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_workload.py",
        _BENCHMARK_PREFIX
        + """
class BaselineWorkloadBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineWorkloadBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_workload.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedWorkloadBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.num_loops = 256
        self._workload = WorkloadMetadata(
            tokens_per_iteration=float(self.batch_size * self.num_loops),
        )

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedWorkloadBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "work_reduction" for issue in issues)


def test_compare_pair_flags_hot_path_helper_synchronize(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_helper_sync.py",
        _BENCHMARK_PREFIX
        + """
class BaselineHelperSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self._step()

    def _step(self):
        return None


def get_benchmark():
    return BaselineHelperSyncBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_helper_sync.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedHelperSyncBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self._step()

    def _step(self):
        torch.cuda.synchronize()


def get_benchmark():
    return OptimizedHelperSyncBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "sync_mismatch" for issue in issues)


def test_compare_pair_flags_seed_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_seed.py",
        _BENCHMARK_PREFIX
        + """
class BaselineSeedBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineSeedBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_seed.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedSeedBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedSeedBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "seed_mismatch" for issue in issues)


def test_compare_pair_flags_config_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_config.py",
        _BENCHMARK_PREFIX
        + """
from core.harness.benchmark_harness import BenchmarkConfig

class BaselineConfigBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        return None

    def get_config(self):
        return BenchmarkConfig(iterations=12, warmup=5)


def get_benchmark():
    return BaselineConfigBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_config.py",
        _BENCHMARK_PREFIX
        + """
from core.harness.benchmark_harness import BenchmarkConfig

class OptimizedConfigBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        return None

    def get_config(self):
        return BenchmarkConfig(iterations=3, warmup=5, enable_profiling=False)


def get_benchmark():
    return OptimizedConfigBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "config_mismatch" for issue in issues)


def test_compare_pair_flags_precision_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_precision.py",
        _BENCHMARK_PREFIX
        + """
class BaselinePrecisionBenchmark(BaseBenchmark):
    def setup(self):
        self.x = torch.randn(4, device=self.device, dtype=torch.float32)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselinePrecisionBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_precision.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedPrecisionBenchmark(BaseBenchmark):
    def setup(self):
        self.x = torch.randn(4, device=self.device, dtype=torch.float16)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedPrecisionBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "precision_mismatch" for issue in issues)


def test_compare_pair_flags_hot_path_extra_work(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_hot_path.py",
        _BENCHMARK_PREFIX
        + """
class BaselineHotPathBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self.output = self.model(self.x)


def get_benchmark():
    return BaselineHotPathBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_hot_path.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedHotPathBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self.output = self.model(self.x.to(dtype=torch.float16))


def get_benchmark():
    return OptimizedHotPathBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "hot_path_extra_work" for issue in issues)


def test_compare_pair_ignores_hot_path_work_reduction(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_hot_path_reduction.py",
        _BENCHMARK_PREFIX
        + """
class BaselineHotPathReductionBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self.output = self.model(self.x.clone())


def get_benchmark():
    return BaselineHotPathReductionBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_hot_path_reduction.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedHotPathReductionBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self.output = self.model(self.x)


def get_benchmark():
    return OptimizedHotPathReductionBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "hot_path_extra_work" for issue in issues)


def test_compare_pair_skips_hot_path_diff_for_cuda_graph_pairs(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_graph_hot_path.py",
        _BENCHMARK_PREFIX
        + """
class BaselineGraphHotPathBenchmark(BaseBenchmark):
    def benchmark_fn(self):
        self.output = self.model(self.x)


def get_benchmark():
    return BaselineGraphHotPathBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_graph_hot_path.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedGraphHotPathBenchmark(BaseBenchmark):
    def setup(self):
        self.graph = torch.cuda.CUDAGraph()

    def benchmark_fn(self):
        self.graph.replay()
        self.output = self.output_buffer.detach().clone()


def get_benchmark():
    return OptimizedGraphHotPathBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "hot_path_extra_work" for issue in issues)


def test_compare_pair_flags_algorithmic_pair_mismatch(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_algorithmic.py",
        _BENCHMARK_PREFIX
        + """
class BaselineAlgorithmicBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.route_mode = "uniform"

    def benchmark_fn(self):
        for pos in range(seq_len):
            pass


def get_benchmark():
    return BaselineAlgorithmicBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_algorithmic.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedAlgorithmicBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.route_mode = "topology_aware"

    def benchmark_fn(self):
        for pos in range(0, seq_len, self.block_size):
            pass


def get_benchmark():
    return OptimizedAlgorithmicBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert any(issue["type"] == "algorithmic_pair_mismatch" for issue in issues)


def test_compare_pair_dedupes_duplicate_seed_calls(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_seeded.py",
        _BENCHMARK_PREFIX
        + """
class BaselineSeededBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineSeededBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_seeded.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedSeededBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.manual_seed_all(42)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedSeededBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "seed_mismatch" for issue in issues)


def test_compare_pair_resolves_seed_info_from_imported_local_base(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    _write(
        tmp_path,
        "benchmark_base.py",
        _BENCHMARK_PREFIX
        + """
class SeededBaseBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def benchmark_fn(self):
        return None
""",
    )
    baseline = _write(
        tmp_path,
        "baseline_imported_seed.py",
        _BENCHMARK_PREFIX
        + """
class BaselineImportedSeedBenchmark(BaseBenchmark):
    def setup(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselineImportedSeedBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_imported_seed.py",
        _BENCHMARK_PREFIX
        + """
from benchmark_base import SeededBaseBenchmark


class OptimizedImportedSeedBenchmark(SeededBaseBenchmark):
    pass


def get_benchmark():
    return OptimizedImportedSeedBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "seed_mismatch" for issue in issues)


def test_compare_pair_skips_precision_mismatch_for_intentional_precision_targets(tmp_path: Path) -> None:
    reviewer = CodeReviewer()
    baseline = _write(
        tmp_path,
        "baseline_precisionfp8.py",
        _BENCHMARK_PREFIX
        + """
class BaselinePrecisionBenchmark(BaseBenchmark):
    def setup(self):
        self.x = torch.randn(8, device=self.device, dtype=torch.float32)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return BaselinePrecisionBenchmark()
""",
    )
    optimized = _write(
        tmp_path,
        "optimized_precisionfp8.py",
        _BENCHMARK_PREFIX
        + """
class OptimizedPrecisionBenchmark(BaseBenchmark):
    def setup(self):
        self.x = torch.randn(8, device=self.device, dtype=torch.float16)

    def benchmark_fn(self):
        return None


def get_benchmark():
    return OptimizedPrecisionBenchmark()
""",
    )

    issues = reviewer.compare_pair(baseline, [optimized])

    assert not any(issue["type"] == "precision_mismatch" for issue in issues)


def test_review_main_scopes_to_requested_chapters(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def _fake_discover(repo_root: Path, chapter: str):
        calls.append(chapter)
        if chapter == "ch12":
            return []
        raise AssertionError(f"unexpected chapter {chapter}")

    monkeypatch.setattr(pair_review, "discover_benchmark_pairs", _fake_discover)

    rc = pair_review.main(["--chapter", "ch12"])

    captured = capsys.readouterr()
    assert rc == 1
    assert calls == ["ch12"]
    assert "No benchmark pairs found for the requested scope." in captured.out
