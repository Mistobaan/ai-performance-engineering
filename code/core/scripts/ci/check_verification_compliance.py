#!/usr/bin/env python3
"""CI check script for benchmark verification compliance.

This script is designed to be run in CI pipelines or as a pre-commit hook.
It fails if:
- New benchmark files are missing required verification methods
- Baseline/optimized pairs have mismatched signatures
- Skip flags are used without proper justification

Usage:
    # Check all changed files (for CI)
    python -m core.scripts.ci.check_verification_compliance

    # Execute changed pairs and validate signatures (requires CUDA-capable environment)
    python -m core.scripts.ci.check_verification_compliance --validate-pairs
    
    # Check specific files (for pre-commit)
    python -m core.scripts.ci.check_verification_compliance --files ch07/baseline_attention.py
    
    # Check against a base branch
    python -m core.scripts.ci.check_verification_compliance --base-branch main
    
Exit codes:
    0: All checks passed
    1: Verification compliance issues found
    2: Script error
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComplianceIssue:
    """A single compliance issue."""
    file_path: str
    severity: str  # "error" or "warning"
    message: str
    line_number: Optional[int] = None


@dataclass
class ComplianceReport:
    """Complete compliance check report."""
    files_checked: int = 0
    errors: int = 0
    warnings: int = 0
    issues: List[ComplianceIssue] = field(default_factory=list)
    
    def add_error(self, file_path: str, message: str, line_number: Optional[int] = None) -> None:
        self.issues.append(ComplianceIssue(file_path, "error", message, line_number))
        self.errors += 1
    
    def add_warning(self, file_path: str, message: str, line_number: Optional[int] = None) -> None:
        self.issues.append(ComplianceIssue(file_path, "warning", message, line_number))
        self.warnings += 1


# =============================================================================
# File Analysis
# =============================================================================

class BenchmarkMethodChecker(ast.NodeVisitor):
    """AST visitor to check for required verification methods."""
    
    def __init__(self):
        self.class_info: dict[str, dict[str, object]] = {}
        self.benchmark_class: Optional[str] = None
        self.benchmark_class_line: Optional[int] = None
        self.bases: Set[str] = set()
        self.methods: Set[str] = set()
        self.attributes: Set[str] = set()
        self.skip_flags: List[tuple[str, int]] = []  # (flag_name, line_number)
        self._in_benchmark_class = False
    
    def _dotted_name(self, node: ast.AST) -> Optional[str]:
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            return None
        return ".".join(reversed(parts))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = {
            name
            for base in node.bases
            if (name := self._dotted_name(base)) is not None
        }
        methods = {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef)
        }
        self.class_info[node.name] = {
            "bases": bases,
            "methods": methods,
        }

        has_benchmark_fn = any(
            isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
            for item in node.body
        )
        
        if has_benchmark_fn:
            self.benchmark_class = node.name
            self.benchmark_class_line = node.lineno
            self._in_benchmark_class = True
            self.bases = set(bases)
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self.methods.add(item.name)
                
                # Check for skip flags
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            self.attributes.add(target.id)
                            if target.id in ("skip_output_check", "skip_input_check", "skip_verification"):
                                # Check if value is True
                                if isinstance(item.value, ast.Constant) and item.value.value:
                                    self.skip_flags.append((target.id, item.lineno))
            
            self._in_benchmark_class = False
        
        self.generic_visit(node)

    def class_or_bases_define_method(self, class_name: str, method_name: str) -> bool:
        visited: Set[str] = set()

        def _walk(name: str) -> bool:
            if name in visited:
                return False
            visited.add(name)
            info = self.class_info.get(name)
            if info is None:
                return False
            methods = info.get("methods", set())
            if isinstance(methods, set) and method_name in methods:
                return True
            bases = info.get("bases", set())
            if not isinstance(bases, set):
                return False
            for base_name in bases:
                if _walk(base_name):
                    return True
            return False

        return _walk(class_name)


def check_file_compliance(file_path: Path) -> List[ComplianceIssue]:
    """Check a single benchmark file for compliance issues."""
    issues: List[ComplianceIssue] = []
    
    # Only check baseline_*.py and optimized_*.py files
    if not (file_path.name.startswith("baseline_") or file_path.name.startswith("optimized_")):
        return issues
    
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        issues.append(ComplianceIssue(
            str(file_path), "error", f"Syntax error: {e}", e.lineno
        ))
        return issues
    except Exception as e:
        issues.append(ComplianceIssue(
            str(file_path), "error", f"Parse error: {e}"
        ))
        return issues
    
    checker = BenchmarkMethodChecker()
    checker.visit(tree)
    
    if not checker.benchmark_class:
        # Not a benchmark file
        return issues
    
    # Benchmarks typically use VerificationPayloadMixin, which provides
    # get_input_signature()/get_verify_output()/get_output_tolerance() via inheritance.
    # This checker is file-local and cannot always see base classes imported from elsewhere.
    uses_payload_mixin = any(base.endswith("VerificationPayloadMixin") for base in checker.bases)
    inherits_benchmark_locally = any(
        base != "BaseBenchmark" and checker.class_or_bases_define_method(base, "get_input_signature")
        for base in checker.bases
    )
    inherits_from_other_benchmark_base = any(
        base.endswith("Benchmark") and base != "BaseBenchmark"
        for base in checker.bases
    )
    inherits_verification_support_runtime = _class_supports_verification_via_runtime_load(
        file_path,
        checker.benchmark_class,
    )

    if (
        "get_input_signature" not in checker.methods
        and not uses_payload_mixin
        and not inherits_benchmark_locally
        and not inherits_from_other_benchmark_base
        and not inherits_verification_support_runtime
    ):
        issues.append(
            ComplianceIssue(
                str(file_path),
                "warning",
                f"Class {checker.benchmark_class} does not define get_input_signature() in this file "
                "(OK if inherited from a base class; ensure the class chain provides it).",
                checker.benchmark_class_line,
            )
        )
    
    # Check for skip flags without justification
    for flag_name, line_no in checker.skip_flags:
        # Check if there's a justification attribute
        justification_attr = f"{flag_name}_reason"
        if justification_attr not in checker.attributes:
            issues.append(ComplianceIssue(
                str(file_path),
                "warning",
                f"Skip flag '{flag_name}' used without justification (add {justification_attr} attribute)",
                line_no,
            ))
    
    return issues


def _class_supports_verification_via_runtime_load(file_path: Path, class_name: Optional[str]) -> bool:
    """Best-effort import path to resolve inherited verification helpers.

    This is only used to suppress file-local false positives when a benchmark
    inherits verification support through an imported benchmark base class.
    Any import failure falls back to the static warning path.
    """
    if not class_name:
        return False

    module_name = f"_verification_compliance_{file_path.stem}_{abs(hash(file_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    cleanup_names = [module_name]
    added_sys_path = False
    try:
        sys.modules[module_name] = module
        parent = str(file_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
            added_sys_path = True
        spec.loader.exec_module(module)
        cls = getattr(module, class_name, None)
        if cls is None:
            return False
        method = getattr(cls, "get_input_signature", None)
        return callable(method)
    except Exception:
        return False
    finally:
        for name in cleanup_names:
            sys.modules.pop(name, None)
        if added_sys_path:
            try:
                sys.path.remove(str(file_path.parent))
            except ValueError:
                pass


# =============================================================================
# Pair Validation
# =============================================================================

def find_paired_file(file_path: Path) -> Optional[Path]:
    """Find the corresponding baseline/optimized file for a benchmark."""
    if file_path.name.startswith("baseline_"):
        paired_name = file_path.name.replace("baseline_", "optimized_")
        paired_path = file_path.parent / paired_name
        return paired_path if paired_path.exists() else None
    elif file_path.name.startswith("optimized_"):
        paired_name = file_path.name.replace("optimized_", "baseline_")
        paired_path = file_path.parent / paired_name
        return paired_path if paired_path.exists() else None
    return None


def check_pair_signature_compliance(
    changed_benchmark_files: List[Path],
    *,
    repo_root: Path,
) -> List[ComplianceIssue]:
    """Validate baseline/optimized signature equivalence for changed pairs.

    This executes real benchmark code paths to capture verification payloads and
    extract validated InputSignatures (no dry-run/mocks).
    """
    from core.discovery import discover_benchmarks
    from core.scripts.validate_benchmark_pairs import validate_pair

    changed_abs = {p.resolve() for p in changed_benchmark_files}
    bench_dirs = {p.parent for p in changed_abs}

    issues: List[ComplianceIssue] = []
    seen: Set[tuple[Path, Path]] = set()

    for bench_dir in sorted(bench_dirs):
        discovered = discover_benchmarks(bench_dir, validate=False, warn_missing=False)
        if not discovered:
            continue

        try:
            chapter = str(bench_dir.relative_to(repo_root))
        except ValueError:
            chapter = str(bench_dir)

        for baseline_path, optimized_paths, _example_name in discovered:
            baseline_abs = baseline_path.resolve()
            optimized_abs = [p.resolve() for p in optimized_paths]

            if baseline_abs not in changed_abs and not any(p in changed_abs for p in optimized_abs):
                continue

            for opt_path in optimized_paths:
                pair = (baseline_abs, opt_path.resolve())
                if pair in seen:
                    continue
                seen.add(pair)

                example_name = opt_path.stem.replace("optimized_", "", 1)
                result = validate_pair(chapter, example_name, baseline_path, opt_path)
                if result.valid:
                    continue

                pair_label = f"{baseline_path} vs {opt_path}"
                if result.error:
                    issues.append(
                        ComplianceIssue(
                            pair_label,
                            "error",
                            f"Signature validation failed: {result.error}",
                        )
                    )
                    continue

                mismatch_keys = sorted({m.key for m in result.mismatches})
                issues.append(
                    ComplianceIssue(
                        pair_label,
                        "error",
                        f"Signature mismatch keys: {mismatch_keys}",
                    )
                )

    return issues


# =============================================================================
# Git Integration
# =============================================================================

def get_changed_files(base_branch: Optional[str] = None) -> List[Path]:
    """Get list of changed files (staged or compared to base branch)."""
    try:
        if base_branch:
            # Compare against base branch
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],
                capture_output=True,
                text=True,
                check=True,
            )
        
        files = [Path(f.strip()) for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.suffix == ".py"]
    except subprocess.CalledProcessError:
        return []


# =============================================================================
# Main
# =============================================================================

def check_compliance(
    files: Optional[List[Path]] = None,
    base_branch: Optional[str] = None,
    root_dir: Optional[Path] = None,
    validate_pairs: bool = False,
) -> ComplianceReport:
    """Run compliance checks on files."""
    report = ComplianceReport()
    
    if files is None:
        files = get_changed_files(base_branch)
    
    if not files:
        print("No files to check")
        return report
    
    # Filter to benchmark files only
    benchmark_files = [
        f for f in files 
        if f.name.startswith("baseline_") or f.name.startswith("optimized_")
    ]
    
    if not benchmark_files:
        print("No benchmark files to check")
        return report
    
    print(f"Checking {len(benchmark_files)} benchmark file(s)")
    
    # Resolve paths
    if root_dir:
        benchmark_files = [root_dir / f if not f.is_absolute() else f for f in benchmark_files]
    
    for file_path in benchmark_files:
        if not file_path.exists():
            print(f"  Skipping {file_path} (not found)")
            continue
        
        report.files_checked += 1
        issues = check_file_compliance(file_path)
        
        for issue in issues:
            if issue.severity == "error":
                report.add_error(issue.file_path, issue.message, issue.line_number)
            else:
                report.add_warning(issue.file_path, issue.message, issue.line_number)

    if validate_pairs and benchmark_files:
        pair_issues = check_pair_signature_compliance(benchmark_files, repo_root=root_dir or Path("."))
        for issue in pair_issues:
            if issue.severity == "error":
                report.add_error(issue.file_path, issue.message, issue.line_number)
            else:
                report.add_warning(issue.file_path, issue.message, issue.line_number)
    
    return report


def print_report(report: ComplianceReport) -> None:
    """Print compliance report."""
    print()
    print("=" * 60)
    print("VERIFICATION COMPLIANCE CHECK")
    print("=" * 60)
    print(f"Files checked: {report.files_checked}")
    print(f"Errors:        {report.errors}")
    print(f"Warnings:      {report.warnings}")
    
    if report.issues:
        print()
        print("ISSUES:")
        for issue in report.issues:
            prefix = "ERROR" if issue.severity == "error" else "WARNING"
            location = f":{issue.line_number}" if issue.line_number else ""
            print(f"  [{prefix}] {issue.file_path}{location}: {issue.message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check benchmark files for verification compliance"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="*",
        type=Path,
        help="Specific files to check (default: staged files)"
    )
    parser.add_argument(
        "--base-branch", "-b",
        help="Compare against a base branch (e.g., main)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=".",
        help="Root directory (default: current directory)"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--validate-pairs",
        action="store_true",
        help="Execute changed baseline/optimized pairs and validate input signatures match",
    )
    
    args = parser.parse_args()
    
    try:
        report = check_compliance(
            files=args.files,
            base_branch=args.base_branch,
            root_dir=args.root.resolve(),
            validate_pairs=args.validate_pairs,
        )
        
        print_report(report)
        
        if report.errors > 0:
            print()
            print("❌ Compliance check FAILED")
            print("   Run 'python -m core.scripts.migrate_verification_methods' to add missing methods")
            return 1
        
        if args.fail_on_warning and report.warnings > 0:
            print()
            print("❌ Compliance check FAILED (warnings treated as errors)")
            return 1
        
        if report.files_checked > 0:
            print()
            print("✓ Compliance check PASSED")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())



