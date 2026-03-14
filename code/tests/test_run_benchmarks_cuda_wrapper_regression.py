from pathlib import Path

from core.harness import run_benchmarks


class _DummyExpectationsStore:
    def __init__(self, chapter_dir: Path, *_args, **_kwargs) -> None:
        self.path = chapter_dir / "expectations_test.json"

    def save(self) -> None:
        return None


def test_test_chapter_impl_uses_cuda_wrapper_detector_without_nameerror(tmp_path, monkeypatch):
    chapter_dir = tmp_path / "ch03"
    chapter_dir.mkdir()
    baseline_path = chapter_dir / "baseline_numa_unaware.py"
    baseline_path.write_text(
        "from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark\n"
        "class DemoCudaWrapper(CudaBinaryBenchmark):\n"
        "    cuda_binary_path = 'demo'\n",
        encoding="utf-8",
    )

    detector_calls: list[Path] = []
    original_detector = run_benchmarks.is_cuda_binary_benchmark_file

    def _tracked_detector(path: Path) -> bool:
        detector_calls.append(path)
        return original_detector(path)

    monkeypatch.setattr(run_benchmarks, "is_cuda_binary_benchmark_file", _tracked_detector)
    monkeypatch.setattr(run_benchmarks, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(run_benchmarks, "detect_expectation_key", lambda: "test")
    monkeypatch.setattr(run_benchmarks, "detect_execution_environment", lambda: {"kind": "test"})
    monkeypatch.setattr(run_benchmarks, "get_git_info", lambda: {"commit": "deadbeef"})
    monkeypatch.setattr(run_benchmarks, "clean_build_directories", lambda _chapter_dir: None)
    monkeypatch.setattr(run_benchmarks, "reset_cuda_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "emit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_benchmarks, "start_progress_watchdog", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(run_benchmarks, "ExpectationsStore", _DummyExpectationsStore)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_benchmarks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        run_benchmarks,
        "_discover_chapter_benchmark_pairs",
        lambda *args, **kwargs: ([(baseline_path, [], "numa_unaware")], [], None, 0, 0),
    )

    result = run_benchmarks._test_chapter_impl(chapter_dir, enable_profiling=False)

    assert detector_calls == [baseline_path]
    assert result["status"] == "completed"
    assert result["summary"]["informational"] == 1
    assert result["benchmarks"] == []
