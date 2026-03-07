"""Best-effort vLLM worker shutdown helpers for benchmark subprocesses."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _collect_descendant_pids(root_pid: int) -> List[int]:
    parent_to_children: dict[int, Set[int]] = {}
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        stat_text = _safe_read_text(proc_dir / "stat")
        if not stat_text:
            continue
        # /proc/<pid>/stat format: pid (comm) state ppid ...
        rparen = stat_text.rfind(")")
        if rparen < 0:
            continue
        tail = stat_text[rparen + 1 :].strip().split()
        if len(tail) < 2:
            continue
        try:
            pid = int(proc_dir.name)
            ppid = int(tail[1])
        except ValueError:
            continue
        parent_to_children.setdefault(ppid, set()).add(pid)

    descendants: List[int] = []
    stack: List[int] = [root_pid]
    seen: Set[int] = {root_pid}
    while stack:
        parent = stack.pop()
        for child in parent_to_children.get(parent, set()):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            stack.append(child)
    descendants.sort()
    return descendants


def _signal_pids(pids: Iterable[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _wait_for_exit(pids: Iterable[int], timeout_seconds: float) -> Set[int]:
    deadline = time.time() + max(timeout_seconds, 0.0)
    pending = set(int(pid) for pid in pids)
    while pending and time.time() < deadline:
        dead: Set[int] = set()
        for pid in pending:
            if not Path(f"/proc/{pid}").exists():
                dead.add(pid)
        pending -= dead
        if pending:
            time.sleep(0.05)
    return pending


def _call_if_present(obj: object, method_name: str) -> None:
    method = getattr(obj, method_name, None)
    if callable(method):
        method()


def shutdown_vllm_runtime(
    llm_obj: object,
    *,
    logger: Optional[Callable[[str, object], None]] = None,
    grace_seconds: float = 8.0,
) -> None:
    """Stop vLLM engine internals and terminate leftover child processes.

    This is intentionally aggressive because chapter benchmarks run inside
    dedicated subprocesses (`use_subprocess=True`), so terminating descendants
    cannot impact unrelated benchmark state in the parent harness process.
    """

    if llm_obj is None:
        return

    root_pid = os.getpid()
    engine = getattr(llm_obj, "llm_engine", None)
    for target in (engine, llm_obj):
        if target is None:
            continue
        for method_name in (
            "shutdown",
            "close",
            "stop",
            "shutdown_background_loop",
            "shutdown_worker_execution_loop",
        ):
            try:
                _call_if_present(target, method_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if logger is not None:
                    logger("vLLM cleanup warning (%s): %s", method_name, exc)

    # Some vLLM paths initialize torch.distributed/NCCL state; force teardown to
    # avoid lingering process-group resources and worker retention.
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                if logger is not None:
                    logger("vLLM cleanup warning (destroy_process_group): %s", exc)
    except Exception:
        pass

    # Recompute descendants after in-runtime shutdown attempts and clean any
    # workers that remain alive.
    after_children = set(_collect_descendant_pids(root_pid))
    lingering_children = sorted(after_children)
    if not lingering_children:
        return

    _signal_pids(lingering_children, signal.SIGTERM)
    remaining = _wait_for_exit(lingering_children, timeout_seconds=max(grace_seconds, 0.0))
    if remaining:
        _signal_pids(sorted(remaining), signal.SIGKILL)
        remaining = _wait_for_exit(sorted(remaining), timeout_seconds=2.0)

    if remaining and logger is not None:
        logger(
            "vLLM cleanup warning: descendant PID(s) still alive after SIGKILL: %s",
            sorted(remaining),
        )
