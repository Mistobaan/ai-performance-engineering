"""Shared helpers for the TRT-LLM Phi-3.5-MoE lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import importlib.machinery
import importlib.util
import os
import platform
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "phi-3.5-moe" / "original"
MODEL_PATH_ENV = "AISP_PHI35_MOE_MODEL_PATH"
ENGINE_PATH_ENV = "AISP_PHI35_MOE_ENGINE_PATH"
PROMPT_TEXT = "Explain GPU kernel fusion in one sentence."


def ensure_trtllm_assets(
    model_path: Path,
    *,
    engine_path: Optional[Path] = None,
    require_engine: bool = False,
) -> None:
    """Fail fast with explicit remediation when required TRT-LLM assets are absent."""
    missing = []
    if not model_path.exists():
        missing.append(f"model_path={model_path}")
    if require_engine and (engine_path is None or not engine_path.exists()):
        if engine_path is None:
            missing.append("engine_path=<unset>")
        else:
            missing.append(f"engine_path={engine_path}")
    if not missing:
        return
    missing_text = ", ".join(missing)
    raise RuntimeError(
        "SKIPPED: TRT-LLM Phi-3.5-MoE assets unavailable "
        f"({missing_text}). "
        "Remediation: provide a local Phi-3.5-MoE checkout via --model-path "
        f"(or ${MODEL_PATH_ENV}) and a built TensorRT-LLM engine via --engine-path "
        f"(or ${ENGINE_PATH_ENV})."
    )


def load_trtllm_runtime():
    """Load TensorRT-LLM runtime without importing the full package __init__."""
    os.environ.setdefault("OMPI_MCA_coll_ucc_enable", "0")
    if platform.system() == "Linux":
        try:
            from ctypes import cdll

            v_major, v_minor, *_ = sys.version_info
            cdll.LoadLibrary(f"libpython{v_major}.{v_minor}.so.1.0")
            cdll.LoadLibrary(f"libpython{v_major}.{v_minor}.so")
        except Exception:
            pass

    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("TensorRT-LLM is not installed")
    root = Path(spec.submodule_search_locations[0])

    pkg_name = "tensorrt_llm"
    if pkg_name not in sys.modules:
        pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(pkg_name, None))
        pkg.__path__ = [str(root)]
        sys.modules[pkg_name] = pkg

    common_path = root / "_common.py"
    spec_common = importlib.util.spec_from_file_location("tensorrt_llm._common", common_path)
    if spec_common is None or spec_common.loader is None:
        raise RuntimeError("TensorRT-LLM _common module not found")
    common_mod = importlib.util.module_from_spec(spec_common)
    spec_common.loader.exec_module(common_mod)  # type: ignore[union-attr]
    common_mod._init()

    runtime_path = root / "runtime" / "__init__.py"
    spec_runtime = importlib.util.spec_from_file_location("tensorrt_llm.runtime", runtime_path)
    if spec_runtime is None or spec_runtime.loader is None:
        raise RuntimeError("TensorRT-LLM runtime module not found")
    runtime_mod = importlib.util.module_from_spec(spec_runtime)
    spec_runtime.loader.exec_module(runtime_mod)  # type: ignore[union-attr]
    return runtime_mod


def parse_trtllm_args() -> argparse.Namespace:
    default_model_path = os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH))
    default_engine_path = os.environ.get(ENGINE_PATH_ENV)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=str, default=default_model_path)
    parser.add_argument("--engine-path", type=str, default=default_engine_path)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-slice", type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


def build_prompt_tokens(tokenizer, *, prompt_len: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer.encode(PROMPT_TEXT, add_special_tokens=True)
    if len(encoded) > prompt_len:
        encoded = encoded[:prompt_len]
    else:
        encoded = encoded + [tokenizer.pad_token_id] * (prompt_len - len(encoded))
    input_ids = torch.tensor([encoded] * batch_size, dtype=torch.long)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attention_mask


def slice_logits(logits: torch.Tensor, vocab_slice: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("Expected logits of shape [batch, vocab]")
    return logits[:, :vocab_slice]
