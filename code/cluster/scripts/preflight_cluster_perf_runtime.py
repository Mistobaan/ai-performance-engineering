#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _extract_last_json_blob(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("no JSON payload found in command stdout")


def _probe_stack_via_python(cmd_prefix: list[str]) -> Dict[str, Any]:
    snippet = r"""
import json
import importlib.metadata as im
import torch

try:
    import torch.cuda.nccl as nccl
    nccl_v = nccl.version()
    if isinstance(nccl_v, (tuple, list)):
        nccl_version = ".".join(str(x) for x in nccl_v)
    else:
        nccl_version = str(nccl_v)
except Exception:
    nccl_version = None

try:
    import deep_gemm  # noqa: F401
    deep_gemm_installed = True
    deep_gemm_version = str(im.version("deep_gemm"))
except Exception:
    deep_gemm_installed = False
    deep_gemm_version = None

payload = {
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda),
    "cudnn_version": str(torch.backends.cudnn.version()),
    "nccl_version": nccl_version,
    "deep_gemm_installed": deep_gemm_installed,
    "deep_gemm_version": deep_gemm_version,
}
print(json.dumps(payload, sort_keys=True))
""".strip()
    cmd = [*cmd_prefix, "python", "-c", snippet]
    proc = _run(cmd)
    if proc.returncode != 0:
        quoted = " ".join(shlex.quote(x) for x in cmd)
        raise RuntimeError(
            "stack probe failed:\n"
            f"cmd={quoted}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return _extract_last_json_blob(proc.stdout)


def _probe_math_policy_via_python(cmd_prefix: list[str], allow_tf32: bool, float32_matmul_precision: str) -> Dict[str, Any]:
    snippet = f"""
import json
import torch

allow_tf32 = {str(bool(allow_tf32))}
precision = {float32_matmul_precision!r}

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision(precision)
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = allow_tf32

payload = {{
    "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", allow_tf32)),
    "float32_matmul_precision": precision,
}}
print(json.dumps(payload, sort_keys=True))
""".strip()
    cmd = [*cmd_prefix, "python", "-c", snippet]
    proc = _run(cmd)
    if proc.returncode != 0:
        quoted = " ".join(shlex.quote(x) for x in cmd)
        raise RuntimeError(
            "math policy probe failed:\n"
            f"cmd={quoted}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return _extract_last_json_blob(proc.stdout)


def _probe_host_stack(host_python: str) -> Dict[str, Any]:
    cmd_prefix = [host_python]
    snippet = r"""
import json
import importlib.metadata as im
import torch

try:
    import torch.cuda.nccl as nccl
    nccl_v = nccl.version()
    if isinstance(nccl_v, (tuple, list)):
        nccl_version = ".".join(str(x) for x in nccl_v)
    else:
        nccl_version = str(nccl_v)
except Exception:
    nccl_version = None

try:
    import deep_gemm  # noqa: F401
    deep_gemm_installed = True
    deep_gemm_version = str(im.version("deep_gemm"))
except Exception:
    deep_gemm_installed = False
    deep_gemm_version = None

payload = {
    "torch_version": str(torch.__version__),
    "cuda_version": str(torch.version.cuda),
    "cudnn_version": str(torch.backends.cudnn.version()),
    "nccl_version": nccl_version,
    "deep_gemm_installed": deep_gemm_installed,
    "deep_gemm_version": deep_gemm_version,
}
print(json.dumps(payload, sort_keys=True))
""".strip()
    proc = _run([*cmd_prefix, "-c", snippet])
    if proc.returncode != 0:
        quoted = " ".join(shlex.quote(x) for x in [*cmd_prefix, "-c", snippet])
        raise RuntimeError(
            "host stack probe failed:\n"
            f"cmd={quoted}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return _extract_last_json_blob(proc.stdout)


def _probe_host_math_policy(host_python: str, allow_tf32: bool, float32_matmul_precision: str) -> Dict[str, Any]:
    snippet = f"""
import json
import torch

allow_tf32 = {str(bool(allow_tf32))}
precision = {float32_matmul_precision!r}

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision(precision)
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = allow_tf32

payload = {{
    "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", allow_tf32)),
    "float32_matmul_precision": precision,
}}
print(json.dumps(payload, sort_keys=True))
""".strip()
    proc = _run([host_python, "-c", snippet])
    if proc.returncode != 0:
        quoted = " ".join(shlex.quote(x) for x in [host_python, "-c", snippet])
        raise RuntimeError(
            "host math policy probe failed:\n"
            f"cmd={quoted}\n"
            f"stdout={proc.stdout}\n"
            f"stderr={proc.stderr}"
        )
    return _extract_last_json_blob(proc.stdout)


def _probe_container_stack(image: str) -> Dict[str, Any]:
    cmd_prefix = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        image,
    ]
    return _probe_stack_via_python(cmd_prefix)


def _probe_container_math_policy(image: str, allow_tf32: bool, float32_matmul_precision: str) -> Dict[str, Any]:
    cmd_prefix = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        image,
    ]
    return _probe_math_policy_via_python(cmd_prefix, allow_tf32=allow_tf32, float32_matmul_precision=float32_matmul_precision)


def _probe_container_image_info(image: str) -> Dict[str, Any]:
    proc = _run(["docker", "image", "inspect", image])
    if proc.returncode != 0:
        return {"inspect_error": proc.stderr.strip() or proc.stdout.strip()}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"inspect_error": "unable to parse docker image inspect output"}
    if not payload:
        return {"inspect_error": "docker image inspect returned empty payload"}
    entry = payload[0]
    return {
        "id": str(entry.get("Id") or ""),
        "repo_digests": list(entry.get("RepoDigests") or []),
    }


def _digest_from_ref(ref: str) -> Optional[str]:
    if "@sha256:" not in ref:
        return None
    return ref.split("@", 1)[1].strip()


def _is_sha256_image_id(ref: str) -> bool:
    candidate = ref.strip()
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", candidate))


def _is_pinned_image_ref(ref: str) -> bool:
    candidate = ref.strip()
    return "@sha256:" in candidate or _is_sha256_image_id(candidate)


def _peermem_loaded() -> bool:
    if Path("/sys/module/nvidia_peermem").exists():
        return True
    proc = _run(["bash", "-lc", "lsmod | awk '{print $1}'"])
    if proc.returncode != 0:
        return False
    mods = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return "nvidia_peermem" in mods


def _try_load_peermem() -> Dict[str, Any]:
    loaded_before = _peermem_loaded()
    attempted = False
    errors: list[str] = []
    if loaded_before:
        return {
            "loaded_before": True,
            "attempted_modprobe": False,
            "loaded_after": True,
            "modprobe_errors": [],
        }

    attempted = True
    for module in ("nvidia-peermem", "nvidia_peermem"):
        proc = _run(["sudo", "-n", "modprobe", module])
        if proc.returncode == 0 and _peermem_loaded():
            return {
                "loaded_before": loaded_before,
                "attempted_modprobe": attempted,
                "loaded_after": True,
                "modprobe_errors": errors,
            }
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"returncode={proc.returncode}"
        errors.append(f"{module}: {detail}")

    return {
        "loaded_before": loaded_before,
        "attempted_modprobe": attempted,
        "loaded_after": _peermem_loaded(),
        "modprobe_errors": errors,
    }


def _load_profiles(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - explicit failure path
        raise RuntimeError(f"failed to read profiles JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid profiles JSON root in {path}")
    return data


def _compare_stack(expected: Dict[str, Any], actual: Dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for key, exp in expected.items():
        got = actual.get(key)
        if isinstance(exp, bool):
            if bool(got) != exp:
                errs.append(f"stack mismatch for {key}: expected={exp} got={got!r}")
        else:
            if str(got) != str(exp):
                errs.append(f"stack mismatch for {key}: expected={exp!r} got={got!r}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict stack+host preflight for cluster perf FP4/grouped-GEMM runs.")
    ap.add_argument("--runtime", choices=["host", "container"], required=True)
    ap.add_argument("--stack-profile", required=True, help="Profile key from configs/cluster_perf_stack_profiles.json")
    ap.add_argument("--image", default="", help="Container image ref (required for runtime=container)")
    ap.add_argument("--host-python", default="", help="Host Python interpreter for runtime=host stack probe")
    ap.add_argument("--profiles-json", default="", help="Path to stack profile JSON (default: repo config)")
    ap.add_argument("--out-json", default="", help="Optional output JSON path")
    ap.add_argument(
        "--require-peermem",
        action="store_true",
        help="Require nvidia_peermem to be loaded. Use only for GPUDirect RDMA paths.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    profiles_path = Path(args.profiles_json) if args.profiles_json else root_dir / "configs" / "cluster_perf_stack_profiles.json"
    profiles_data = _load_profiles(profiles_path)
    profiles = profiles_data.get("profiles") or {}
    entry = profiles.get(args.stack_profile)
    if entry is None:
        print(f"ERROR: unknown stack profile: {args.stack_profile}", file=sys.stderr)
        return 2

    allowed = set(entry.get("allowed_runtimes") or [])
    errors: list[str] = []
    if args.runtime not in allowed:
        errors.append(f"profile {args.stack_profile!r} does not allow runtime {args.runtime!r}")

    expected_image = str(entry.get("image_ref") or "")
    image_info: Dict[str, Any] = {}
    if args.runtime == "container":
        if not args.image:
            errors.append("runtime=container requires --image")
        else:
            if not _is_pinned_image_ref(args.image):
                errors.append(
                    "container image must be immutable (expected repo@sha256:... or local sha256:<image_id>)"
                )
            expected_digest = _digest_from_ref(expected_image) if expected_image else None
            actual_digest = _digest_from_ref(args.image)
            if expected_digest and actual_digest and expected_digest != actual_digest:
                errors.append(
                    "container image digest drift: "
                    f"expected={expected_digest} got={actual_digest}"
                )
            image_info = _probe_container_image_info(args.image)
            repo_digests = set(image_info.get("repo_digests") or [])
            if expected_image and repo_digests and expected_image not in repo_digests:
                if expected_digest and not any(d.endswith(expected_digest) for d in repo_digests):
                    errors.append(
                        "local image repo digest drift: "
                        f"expected digest {expected_digest} not found in {sorted(repo_digests)}"
                    )

    peermem_state = _try_load_peermem() if args.require_peermem else {
        "loaded_before": _peermem_loaded(),
        "attempted_modprobe": False,
        "loaded_after": _peermem_loaded(),
        "modprobe_errors": [],
        "required": False,
    }
    peermem_loaded = bool(peermem_state.get("loaded_after"))
    if args.require_peermem and not peermem_loaded:
        detail = "; ".join(peermem_state.get("modprobe_errors") or [])
        if detail:
            errors.append(
                "missing host prereq: nvidia_peermem kernel module is not loaded "
                f"(modprobe attempts: {detail})"
            )
        else:
            errors.append("missing host prereq: nvidia_peermem kernel module is not loaded")

    detected_stack: Dict[str, Any] = {}
    probe_error = ""
    try:
        if args.runtime == "host":
            host_python = args.host_python or str(root_dir / "env" / "venv" / "bin" / "python")
            detected_stack = _probe_host_stack(host_python)
        else:
            detected_stack = _probe_container_stack(args.image)
    except Exception as exc:
        probe_error = str(exc)
        errors.append(f"stack probe failed: {exc}")

    expected_stack = dict(entry.get("expected_stack") or {})
    if detected_stack and expected_stack:
        errors.extend(_compare_stack(expected_stack, detected_stack))

    requested_math_policy = dict(entry.get("math_policy") or {})
    requested_allow_tf32 = bool(requested_math_policy.get("allow_tf32", False))
    requested_float32_matmul_precision = str(requested_math_policy.get("float32_matmul_precision") or "high")
    detected_math_policy: Dict[str, Any] = {}
    math_policy_probe_error = ""
    try:
        if args.runtime == "host":
            host_python = args.host_python or str(root_dir / "env" / "venv" / "bin" / "python")
            detected_math_policy = _probe_host_math_policy(
                host_python=host_python,
                allow_tf32=requested_allow_tf32,
                float32_matmul_precision=requested_float32_matmul_precision,
            )
        else:
            detected_math_policy = _probe_container_math_policy(
                image=args.image,
                allow_tf32=requested_allow_tf32,
                float32_matmul_precision=requested_float32_matmul_precision,
            )
    except Exception as exc:
        math_policy_probe_error = str(exc)
        errors.append(f"math policy probe failed: {exc}")

    if detected_math_policy:
        if bool(detected_math_policy.get("allow_tf32")) != requested_allow_tf32:
            errors.append(
                "math policy mismatch for allow_tf32: "
                f"expected={requested_allow_tf32} got={detected_math_policy.get('allow_tf32')!r}"
            )
        if bool(detected_math_policy.get("cudnn_allow_tf32")) != requested_allow_tf32:
            errors.append(
                "math policy mismatch for cudnn_allow_tf32: "
                f"expected={requested_allow_tf32} got={detected_math_policy.get('cudnn_allow_tf32')!r}"
            )
        if str(detected_math_policy.get("float32_matmul_precision")) != requested_float32_matmul_precision:
            errors.append(
                "math policy mismatch for float32_matmul_precision: "
                f"expected={requested_float32_matmul_precision!r} got={detected_math_policy.get('float32_matmul_precision')!r}"
            )

    status = "ok" if not errors else "failed"
    out_payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "runtime": args.runtime,
        "stack_profile": args.stack_profile,
        "description": str(entry.get("description") or ""),
        "profiles_json": str(profiles_path),
        "requested_image": args.image if args.runtime == "container" else "",
        "expected_image": expected_image,
        "image_info": image_info,
        "peermem_loaded": peermem_loaded,
        "peermem": peermem_state,
        "require_peermem": args.require_peermem,
        "expected_stack": expected_stack,
        "detected_stack": detected_stack,
        "requested_math_policy": {
            "allow_tf32": requested_allow_tf32,
            "float32_matmul_precision": requested_float32_matmul_precision,
        },
        "detected_math_policy": detected_math_policy,
        "math_policy_probe_error": math_policy_probe_error,
        "math_policy": dict(entry.get("math_policy") or {}),
        "probe_error": probe_error,
        "errors": errors,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"PRECHECK stack_profile={args.stack_profile} runtime={args.runtime} status={status}")
    if detected_stack:
        print(f"PRECHECK detected_stack={json.dumps(detected_stack, sort_keys=True)}")
    if detected_math_policy:
        print(f"PRECHECK detected_math_policy={json.dumps(detected_math_policy, sort_keys=True)}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
