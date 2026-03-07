"""Helpers for launching repo Python entrypoints without local sys.path hacks."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def _dedupe_entries(entries: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for entry in entries:
        text = str(entry).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def build_repo_python_env(
    repo_root: Path,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    extra_pythonpath: Optional[Sequence[str | Path]] = None,
) -> Dict[str, str]:
    """Return an env dict with ``repo_root`` prepended to ``PYTHONPATH``."""
    env = dict(os.environ if base_env is None else base_env)
    entries: List[str] = [str(Path(repo_root).resolve())]
    for entry in extra_pythonpath or ():
        entries.append(str(Path(entry).resolve()))
    existing = env.get("PYTHONPATH", "")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part.strip())
    env["PYTHONPATH"] = os.pathsep.join(_dedupe_entries(entries))
    return env


def build_python_entry_command(
    *,
    module_name: Optional[str] = None,
    script_path: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build ``python`` command for either a module or a script path."""
    if bool(module_name) == bool(script_path):
        raise ValueError("Specify exactly one of module_name or script_path.")
    python = python_executable or sys.executable
    if module_name:
        return [python, "-m", module_name, *(argv or [])]
    return [python, str(Path(script_path).resolve()), *(argv or [])]


def build_torchrun_entry_command(
    torchrun_executable: str,
    *,
    module_name: Optional[str] = None,
    script_path: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    nproc_per_node: int,
    nnodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
) -> List[str]:
    """Build ``torchrun`` command for either a module or a script path."""
    if bool(module_name) == bool(script_path):
        raise ValueError("Specify exactly one of module_name or script_path.")
    cmd = [
        torchrun_executable,
        "--nproc_per_node",
        str(int(nproc_per_node)),
    ]
    if nnodes is not None:
        cmd.extend(["--nnodes", str(int(nnodes))])
    if rdzv_backend is not None:
        cmd.extend(["--rdzv_backend", str(rdzv_backend)])
    if rdzv_endpoint is not None:
        cmd.extend(["--rdzv_endpoint", str(rdzv_endpoint)])
    if module_name:
        cmd.extend(["-m", module_name])
    else:
        cmd.append(str(Path(script_path).resolve()))
    cmd.extend(argv or [])
    return cmd
