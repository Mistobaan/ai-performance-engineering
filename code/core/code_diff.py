"""
Shared code diff helpers for baseline vs optimized benchmark files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional


def _read_code_file(path: Path, *, label: str, warnings: List[str]) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"Failed to read {label} code from {path}: {exc}")
        return None


def find_code_pair(chapter_dir: Path, name: str) -> Dict[str, Any]:
    """Locate baseline/optimized code snippets for a benchmark name."""
    baseline_code = None
    optimized_code = None
    baseline_path: Optional[str] = None
    optimized_path: Optional[str] = None
    warnings: List[str] = []

    baseline_patterns = [
        f"baseline_{name}.py",
        f"{name}_baseline.py",
        "baseline.py",
        f"{name}_slow.py",
    ]
    optimized_patterns = [
        f"optimized_{name}.py",
        f"{name}_optimized.py",
        "optimized.py",
        f"{name}_fast.py",
    ]

    # Search direct patterns
    for pattern in baseline_patterns:
        for f in chapter_dir.rglob(pattern):
            baseline_code = _read_code_file(f, label="baseline", warnings=warnings)
            if baseline_code is not None:
                baseline_path = str(f)
                break
        if baseline_code:
            break

    for pattern in optimized_patterns:
        for f in chapter_dir.rglob(pattern):
            optimized_code = _read_code_file(f, label="optimized", warnings=warnings)
            if optimized_code is not None:
                optimized_path = str(f)
                break
        if optimized_code:
            break

    # Fallback: scan lab-like subdirs
    if not baseline_code or not optimized_code:
        lab_dirs = list(chapter_dir.glob("lab*")) + list(chapter_dir.glob(f"*{name}*"))
        for lab_dir in lab_dirs:
            if lab_dir.is_dir():
                for f in lab_dir.glob("*.py"):
                    content = _read_code_file(f, label="candidate", warnings=warnings)
                    if content is None:
                        continue
                    fname = f.name.lower()
                    if ("baseline" in fname or "slow" in fname) and not baseline_code:
                        baseline_code = content
                        baseline_path = str(f)
                    elif ("optimized" in fname or "fast" in fname) and not optimized_code:
                        optimized_code = content
                        optimized_path = str(f)
                if baseline_code and optimized_code:
                    break

    return {
        "baseline_code": baseline_code,
        "optimized_code": optimized_code,
        "baseline_path": baseline_path,
        "optimized_path": optimized_path,
        "warnings": warnings,
    }


def summarize_diff(baseline: str, optimized: str) -> Dict[str, Any]:
    """Produce a lightweight diff summary between two code strings."""
    baseline_lines = baseline.splitlines()
    optimized_lines = optimized.splitlines()
    changes = []
    for i, (bl, ol) in enumerate(zip(baseline_lines, optimized_lines)):
        if bl != ol:
            changes.append({"line": i + 1, "baseline": bl[:100], "optimized": ol[:100]})

    return {
        "baseline_lines": len(baseline_lines),
        "optimized_lines": len(optimized_lines),
        "changes_count": len(changes),
        "key_changes": changes[:10],
    }
