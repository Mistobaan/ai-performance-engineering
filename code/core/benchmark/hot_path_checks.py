"""Source-level checks for benchmark hot-path anti-patterns."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RANDOM_INPUT_CALLS = {
    ("torch", "rand"),
    ("torch", "rand_like"),
    ("torch", "randint"),
    ("torch", "randn"),
    ("torch", "randn_like"),
}

_COMPILE_CALLS = {
    ("torch", "compile"),
    ("compile_callable",),
}

_PROFILER_CALLS = {
    ("torch", "profiler", "profile"),
    ("torch", "cuda", "profiler", "start"),
    ("torch", "cuda", "profiler", "stop"),
}

_SUBPROCESS_OR_NETWORK_CALLS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "request"),
    ("subprocess", "Popen"),
    ("subprocess", "check_output"),
    ("subprocess", "run"),
}

_PATH_IO_METHODS = {
    "open",
    "read_bytes",
    "read_text",
    "write_bytes",
    "write_text",
}

_HOST_TRANSFER_METHODS = {
    "cpu": "benchmark_fn() transfers tensors to CPU via .cpu() "
    "(line {line}); keep host transfers out of the timed hot path",
    "numpy": "benchmark_fn() converts tensors to NumPy via .numpy() "
    "(line {line}); keep host conversions out of the timed hot path",
    "tolist": "benchmark_fn() materializes tensors as Python lists via .tolist() "
    "(line {line}); keep host conversions out of the timed hot path",
}

_Finding = Tuple[str, str]


def _function_node_map(
    class_node: ast.ClassDef,
) -> Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _self_method_call_name(node: ast.Call) -> Optional[str]:
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
        return None
    return node.func.attr


def _reachable_same_class_functions(
    class_node: ast.ClassDef,
    *,
    root_name: str = "benchmark_fn",
) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    methods = _function_node_map(class_node)
    root = methods.get(root_name)
    if root is None:
        return []

    visited: set[str] = set()
    ordered: List[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit(method_name: str) -> None:
        if method_name in visited:
            return
        method_node = methods.get(method_name)
        if method_node is None:
            return
        visited.add(method_name)
        ordered.append(method_node)
        for child in ast.walk(method_node):
            if not isinstance(child, ast.Call):
                continue
            callee = _self_method_call_name(child)
            if callee is None or callee == method_name:
                continue
            visit(callee)

    visit(root_name)
    return ordered


def _dedupe_messages(messages: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduped.append(message)
    return deduped


def _attribute_chain(node: ast.AST) -> Tuple[str, ...]:
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _call_target_chain(node: ast.AST) -> Tuple[str, ...]:
    if isinstance(node, ast.Call):
        return _call_target_chain(node.func)
    return _attribute_chain(node)


def _has_stream_or_event_hint(parts: Tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in parts)
    return any("stream" in part or "event" in part for part in lowered)


def _assigned_name_targets(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_assigned_name_targets(elt))
        return names
    return []


def _is_stream_or_event_factory(node: ast.AST) -> bool:
    target = _call_target_chain(node)
    if _has_stream_or_event_hint(target):
        return True
    if not target:
        return False
    tail = target[-1].lower()
    return tail in {"event", "stream", "current_stream", "default_stream"}


def _benchmark_fn_sync_like_names(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    known: set[str] = set()
    pending: List[Tuple[List[str], ast.AST]] = []

    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            pending.append(
                (
                    [name for target in node.targets for name in _assigned_name_targets(target)],
                    node.value,
                )
            )
        elif isinstance(node, ast.AnnAssign):
            pending.append((_assigned_name_targets(node.target), node.value))

    changed = True
    while changed:
        changed = False
        for targets, value in pending:
            if not targets or value is None:
                continue
            is_sync_like = False
            if _is_stream_or_event_factory(value):
                is_sync_like = True
            elif isinstance(value, ast.Name) and value.id in known:
                is_sync_like = True
            elif _has_stream_or_event_hint(_attribute_chain(value)):
                is_sync_like = True
            if not is_sync_like:
                continue
            for name in targets:
                if name not in known:
                    known.add(name)
                    changed = True
    return known


def _is_cpu_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() == "cpu"
    if isinstance(node, ast.Call):
        target = _attribute_chain(node.func)
        if target == ("torch", "device") and node.args:
            return _is_cpu_target(node.args[0])
    return False


def _normalize_allowed_codes(allowed_codes: Optional[Iterable[str]]) -> set[str]:
    return {str(code).strip().lower() for code in (allowed_codes or ()) if str(code).strip()}


def _filter_findings(findings: List[_Finding], *, allowed_codes: Optional[Iterable[str]] = None) -> List[str]:
    allowed = _normalize_allowed_codes(allowed_codes)
    return [message for code, message in findings if code not in allowed]


def benchmark_fn_sync_warnings(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return source-level warnings for explicit synchronization in benchmark_fn()."""
    findings: List[_Finding] = []
    sync_like_names = _benchmark_fn_sync_like_names(function_node)
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_synchronize":
            findings.append((
                "sync",
                "benchmark_fn() contains _synchronize() "
                f"(line {node.lineno}); this inflates harness timings and blocks CUDA graph capture",
            ))
            continue
        if _attribute_chain(node.func) == ("torch", "cuda", "synchronize"):
            findings.append((
                "sync",
                "benchmark_fn() contains torch.cuda.synchronize() "
                f"(line {node.lineno}); this inflates harness timings and blocks CUDA graph capture",
            ))
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize":
            owner_chain = _call_target_chain(node.func.value)
            owner_name = owner_chain[0] if owner_chain else ""
            if owner_chain and (
                _has_stream_or_event_hint(owner_chain) or owner_name in sync_like_names
            ):
                findings.append((
                    "sync",
                    "benchmark_fn() contains stream/event synchronize() "
                    f"(line {node.lineno}); move this post-timing or use stream dependencies instead",
                ))
    return _filter_findings(findings, allowed_codes=allowed_codes)


def benchmark_fn_sync_warnings_for_class(
    class_node: ast.ClassDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    messages: List[str] = []
    for function_node in _reachable_same_class_functions(class_node):
        messages.extend(
            benchmark_fn_sync_warnings(
                function_node,
                allowed_codes=allowed_codes,
            )
        )
    return _dedupe_messages(messages)


def benchmark_fn_antipattern_warnings(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return source-level warnings for common hot-path anti-patterns."""
    findings: List[_Finding] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue

        target = _attribute_chain(node.func)
        if target in _RANDOM_INPUT_CALLS:
            findings.append((
                "random_input_regeneration",
                "benchmark_fn() regenerates random inputs via "
                f"{'.'.join(target)}() (line {node.lineno}); allocate inputs in setup() "
                "and reuse/mutate buffers during timing",
            ))
            continue
        if target in _COMPILE_CALLS:
            findings.append((
                "compile",
                "benchmark_fn() triggers compilation via "
                f"{'.'.join(target)}() (line {node.lineno}); compile kernels/functions in setup()",
            ))
            continue
        if target in _PROFILER_CALLS:
            findings.append((
                "profiling",
                "benchmark_fn() starts/stops profiling via "
                f"{'.'.join(target)}() (line {node.lineno}); profiling setup must stay out of the timed hot path"
                ,
            ))
            continue
        if target in _SUBPROCESS_OR_NETWORK_CALLS:
            findings.append((
                "io",
                "benchmark_fn() performs subprocess/network I/O via "
                f"{'.'.join(target)}() (line {node.lineno}); external I/O invalidates timing measurements"
                ,
            ))
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "open":
            findings.append((
                "io",
                "benchmark_fn() performs file I/O via open() "
                f"(line {node.lineno}); file reads/writes invalidate timing measurements",
            ))
            continue

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in _HOST_TRANSFER_METHODS:
                findings.append(("host_transfer", _HOST_TRANSFER_METHODS[attr].format(line=node.lineno)))
                continue
            if attr in _PATH_IO_METHODS:
                findings.append((
                    "io",
                    "benchmark_fn() performs file I/O via "
                    f".{attr}() (line {node.lineno}); file reads/writes invalidate timing measurements",
                ))
                continue
            if attr == "to":
                if node.args and _is_cpu_target(node.args[0]):
                    findings.append((
                        "host_transfer",
                        "benchmark_fn() transfers tensors to CPU via .to('cpu') "
                        f"(line {node.lineno}); keep host transfers out of the timed hot path",
                    ))
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "device" and keyword.value is not None and _is_cpu_target(
                        keyword.value
                    ):
                        findings.append((
                            "host_transfer",
                            "benchmark_fn() transfers tensors to CPU via .to(device='cpu') "
                            f"(line {node.lineno}); keep host transfers out of the timed hot path",
                        ))
                        break
    return _filter_findings(findings, allowed_codes=allowed_codes)


def benchmark_fn_antipattern_warnings_for_class(
    class_node: ast.ClassDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    messages: List[str] = []
    for function_node in _reachable_same_class_functions(class_node):
        messages.extend(
            benchmark_fn_antipattern_warnings(
                function_node,
                allowed_codes=allowed_codes,
            )
        )
    return _dedupe_messages(messages)


def _parse_benchmark_fn(benchmark_fn: Any) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        source = textwrap.dedent(inspect.getsource(benchmark_fn))
    except (OSError, TypeError):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )


def _parse_benchmark_class(
    benchmark_fn: Any,
) -> Optional[ast.ClassDef]:
    owner = getattr(benchmark_fn, "__self__", None)
    if owner is None:
        return None
    cls = owner if inspect.isclass(owner) else owner.__class__
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    ]
    if not candidates:
        return None
    return candidates[0]


def check_benchmark_fn_sync_calls(
    benchmark_fn: Any,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Detect explicit CUDA synchronization calls in benchmark_fn source."""
    class_node = _parse_benchmark_class(benchmark_fn)
    if class_node is not None:
        findings = benchmark_fn_sync_warnings_for_class(
            class_node,
            allowed_codes=allowed_codes,
        )
        return len(findings) == 0, findings
    function_node = _parse_benchmark_fn(benchmark_fn)
    if function_node is None:
        return True, []
    findings = benchmark_fn_sync_warnings(function_node, allowed_codes=allowed_codes)
    return len(findings) == 0, findings


def check_benchmark_fn_antipatterns(
    benchmark_fn: Any,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Detect common performance anti-patterns in benchmark_fn source."""
    class_node = _parse_benchmark_class(benchmark_fn)
    if class_node is not None:
        findings = benchmark_fn_antipattern_warnings_for_class(
            class_node,
            allowed_codes=allowed_codes,
        )
        return len(findings) == 0, findings
    function_node = _parse_benchmark_fn(benchmark_fn)
    if function_node is None:
        return True, []
    findings = benchmark_fn_antipattern_warnings(function_node, allowed_codes=allowed_codes)
    return len(findings) == 0, findings
