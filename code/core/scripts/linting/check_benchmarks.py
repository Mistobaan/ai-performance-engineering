#!/usr/bin/env python3
"""Linter to check that all benchmarks follow the benchmark contract.

Usage:
    python core/scripts/linting/check_benchmarks.py                    # Check all benchmarks
    python core/scripts/linting/check_benchmarks.py ch01/               # Check specific chapter
    python core/scripts/linting/check_benchmarks.py --fix              # Auto-fix issues (if possible)
"""

import argparse
import warnings
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from core.benchmark.contract import check_benchmark_file
from core.discovery import discover_benchmark_entrypoints, is_benchmark_entrypoint_file


def _display_path(file_path: Path) -> str:
    try:
        return str(file_path.relative_to(repo_root))
    except ValueError:
        return str(file_path)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Check benchmarks follow the contract")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to check (default: all chapters)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues (not implemented yet)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return nonzero if warnings are found",
    )
    parser.add_argument(
        "--run-setup",
        action="store_true",
        help="Actually import and instantiate benchmarks during validation (WARNING: executes module code and constructors, requires CUDA for BaseBenchmark, not suitable for pre-commit hooks). By default, uses AST parsing for side-effect free validation.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only fail on benchmark_fn() hot-path synchronization warnings. Structural contract debt is ignored except for parse/file errors.",
    )
    parser.add_argument(
        "--include-unpaired",
        action="store_true",
        help="Include benchmark entrypoints that expose get_benchmark()/registration decorators even when they are not named baseline_*/optimized_*.",
    )
    args = parser.parse_args()
    
    if args.fix:
        print("Auto-fix is not yet implemented")
        sys.exit(1)
    
    # Find benchmark files
    if args.paths:
        benchmark_files = []
        for path_str in args.paths:
            path = Path(path_str).resolve()
            if path.is_file() and path.suffix == ".py":
                benchmark_files.append(path)
            elif path.is_dir():
                for file in sorted(path.rglob("*.py")):
                    if "__pycache__" in file.parts or file.name.startswith("test_"):
                        continue
                    if args.include_unpaired:
                        if is_benchmark_entrypoint_file(file):
                            benchmark_files.append(file)
                        continue
                    if file.name.startswith(("baseline_", "optimized_")):
                        benchmark_files.append(file)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            benchmark_files = discover_benchmark_entrypoints(
                repo_root,
                include_unpaired=args.include_unpaired,
            )

    benchmark_files = _dedupe_paths(benchmark_files)
    
    if not benchmark_files:
        print("No benchmark files found")
        return 0
    
    print(f"Checking {len(benchmark_files)} benchmark files...")
    print()
    
    total_errors = 0
    total_warnings = 0
    failed_files = []
    
    for file_path in sorted(benchmark_files):
        is_valid, errors, file_warnings = check_benchmark_file(
            file_path,
            run_setup=args.run_setup,
        )
        if args.sync_only:
            relevant_errors = [
                error for error in errors
                if error.startswith("Syntax error:")
                or error.startswith("Failed to parse file:")
                or error.startswith("File does not exist:")
                or error.startswith("Not a Python file:")
            ]
            relevant_warnings = [
                warning for warning in file_warnings
                if "benchmark_fn() contains" in warning and "synchronize" in warning
            ]
            errors = relevant_errors
            file_warnings = relevant_warnings
            is_valid = len(errors) == 0
        
        if errors or file_warnings:
            print(f"❌ {_display_path(file_path)}")
            if errors:
                total_errors += len(errors)
                for error in errors:
                    print(f"   ERROR: {error}")
            if file_warnings:
                total_warnings += len(file_warnings)
                if args.verbose or args.sync_only:
                    for warning in file_warnings:
                        print(f"   WARNING: {warning}")
            failed_files.append(file_path)
        else:
            if args.verbose:
                print(f"✓ {_display_path(file_path)}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files checked: {len(benchmark_files)}")
    print(f"Files with errors: {len(failed_files)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    
    if failed_files:
        print()
        print("Failed files:")
        for file_path in failed_files:
            print(f"  - {_display_path(file_path)}")
    
    return 1 if total_errors > 0 or (args.fail_on_warnings and total_warnings > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
