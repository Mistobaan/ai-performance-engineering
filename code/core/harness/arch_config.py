#!/usr/bin/env python3
"""Architecture helpers for Blackwell and Grace-Blackwell GPUs."""

from typing import Any, Dict, List, Optional
import os
import subprocess
import shutil
import warnings
from pathlib import Path
from importlib import metadata as importlib_metadata
from contextlib import nullcontext

from core.utils.compile_utils import enable_tf32
from core.utils.warning_filters import suppress_known_cuda_capability_warnings
from core.benchmark.triton_compat import (
    ENABLE_TRITON_PATCH as _TRITON_PATCH_ENABLED,
    ensure_triton_compat,
)

with suppress_known_cuda_capability_warnings():
    import torch

try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel
    from torch.nn.attention import SDPBackend as _SDPBackend
    _NEW_SDPA_API_AVAILABLE = True
except ImportError:
    _sdpa_kernel = None  # type: ignore[assignment]
    _SDPBackend = None  # type: ignore[assignment]
    _NEW_SDPA_API_AVAILABLE = False

def _default_sdpa_backends() -> List[Any]:
    if _SDPBackend is None:
        return []
    order: List[Any] = []
    # Prefer TE fused attention on Blackwell/GB200 where available, then Flash, then other fused paths.
    for name in ("TRANSFORMER_ENGINE", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN"):
        if hasattr(_SDPBackend, name):
            order.append(getattr(_SDPBackend, name))
    return order


_PREFERRED_SDPA_BACKENDS: List[Any] = _default_sdpa_backends()


def prefer_sdpa_backends(order: Optional[List[Any]] = None):
    """
    Return a context manager that routes scaled_dot_product_attention to preferred backends.
    
    Uses the new torch.nn.attention.sdpa_kernel() API. Never falls back to the
    deprecated torch.backends.cuda.sdp_kernel() API.

    Example:
        with prefer_sdpa_backends():
            F.scaled_dot_product_attention(...)
    """
    if not _NEW_SDPA_API_AVAILABLE:
        # Return no-op context manager - do NOT use deprecated API
        return nullcontext()
    
    if order is None:
        order = _PREFERRED_SDPA_BACKENDS
    if _sdpa_kernel is None or not order:
        return nullcontext()
    return _sdpa_kernel(order)


def prefer_flash_sdpa():
    """Alias retained for backwards compatibility with earlier chapter drafts."""
    return prefer_sdpa_backends()

BLACKWELL_CC = "10.0"
BLACKWELL_ULTRA_CC = "10.3"
GRACE_BLACKWELL_MAJOR = 12

def _parse_version_tuple(version: str) -> tuple:
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    return tuple(parts)

class ArchitectureConfig:
    """Provide configuration details for NVIDIA Blackwell GPUs."""

    def __init__(self) -> None:
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()
        self.cutlass_version = None

    def _detect_architecture(self) -> str:
        with suppress_known_cuda_capability_warnings():
            if not torch.cuda.is_available():
                return "cpu"
            props = torch.cuda.get_device_properties(0)
        major, minor = props.major, props.minor
        compute_capability = f"{major}.{minor}"

        if major == GRACE_BLACKWELL_MAJOR:
            return "grace_blackwell"

        if major >= 10:
            if major == 10 and minor >= 3:
                return "blackwell_ultra"
            return "blackwell"

        return "other"

    def _get_architecture_config(self) -> Dict[str, Any]:
        configs = {
            "blackwell": {
                "name": "Blackwell B200/B300",
                "compute_capability": BLACKWELL_CC,
                "sm_version": "sm_100",
                "memory_bandwidth": "up to ~8 TB/s",
                "tensor_cores": "5th Gen",
                "features": ["HBM3e", "TMA", "NVLink-C2C", "Stream-ordered Memory"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimisations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "TMA-aware kernels",
                    "HBM3e-aware allocation",
                    "Stream-ordered memory APIs",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.5 Blackwell optimisations",
                    "HBM3e access patterns",
                    "TMA intrinsic support",
                    "Stream-ordered memory",
                    "Blackwell-tuned kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2025.x",
                    "Nsight Compute 2025.x",
                    "HTA",
                    "PyTorch Profiler",
                    "perf"
                ],
            },
            "blackwell_ultra": {
                "name": "Blackwell Ultra B300",
                "compute_capability": BLACKWELL_ULTRA_CC,
                "sm_version": "sm_103",
                "memory_bandwidth": "up to ~9 TB/s",
                "tensor_cores": "5th Gen",
                "features": ["HBM3e (288 GB)", "TMA", "NVLink-C2C", "Stream-ordered Memory"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimisations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "TMA-aware kernels",
                    "HBM3e-aware allocation",
                    "Stream-ordered memory APIs",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.5 Blackwell optimisations",
                    "HBM3e access patterns",
                    "TMA intrinsic support",
                    "Stream-ordered memory",
                    "Blackwell Ultra tuned kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2025.x",
                    "Nsight Compute 2025.x",
                    "HTA",
                    "PyTorch Profiler",
                    "perf"
                ],
            },
            "grace_blackwell": {
                "name": "Grace-Blackwell GB10",
                "compute_capability": "12.1",
                "sm_version": "sm_121",
                "memory_bandwidth": "Grace LPDDR5X up to ~500 GB/s (single Grace)",
                "tensor_cores": "5th Gen (Blackwell-class)",
                "features": ["Grace-Blackwell coherence fabric", "TMA", "HBM3e", "NVLink-C2C"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e optimizations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "TMA-aware kernels",
                    "HBM3e-aware allocation",
                    "Stream-ordered memory APIs",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.5 GB10 optimizations",
                    "HBM3e access patterns",
                    "TMA intrinsic support",
                    "Stream-ordered memory",
                    "GB10-tuned kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2025.x",
                    "Nsight Compute 2025.x",
                    "HTA",
                    "PyTorch Profiler",
                    "perf"
                ],
            },
        }

        generic = {
            "name": "Generic CUDA GPU",
            "compute_capability": "Unknown",
            "sm_version": "sm_unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": [],
            "cuda_features": [],
            "pytorch_optimizations": [],
            "triton_features": [],
            "profiling_tools": [],
        }

        return configs.get(self.arch, generic)

    def get_sm_version(self) -> str:
        return self.config["sm_version"]

    def get_architecture_name(self) -> str:
        return self.config["name"]

    def get_features(self) -> list:
        return self.config["features"]

    def get_cuda_features(self) -> list:
        return self.config["cuda_features"]

    def get_pytorch_optimizations(self) -> list:
        return self.config["pytorch_optimizations"]

    def get_triton_features(self) -> list:
        return self.config["triton_features"]

    def get_profiling_tools(self) -> list:
        return self.config["profiling_tools"]

    def _sanitize_arch_value(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        sanitized = value
        replacements = {
            "sm_121a": "sm_120",
            "sm121a": "sm120",
            "121a": "120",
            "12.1a": "12.0",
        }
        for needle, repl in replacements.items():
            sanitized = sanitized.replace(needle, repl)
        return sanitized

    def _set_arch_env(self, key: str, fallback: str) -> None:
        current = os.environ.get(key)
        if current:
            sanitized = self._sanitize_arch_value(current)
            if sanitized and sanitized != current:
                os.environ[key] = sanitized
        else:
            os.environ[key] = fallback

    def configure_pytorch_optimizations(self) -> None:
        with suppress_known_cuda_capability_warnings():
            if not torch.cuda.is_available():
                return

        if self.arch in ("blackwell", "blackwell_ultra"):
            arch_list = "10.3" if self.arch == "blackwell_ultra" else "10.0"
            cmake_arch = "103" if self.arch == "blackwell_ultra" else "100"
            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
            if "CMAKE_CUDA_ARCHITECTURES" not in os.environ:
                os.environ["CMAKE_CUDA_ARCHITECTURES"] = cmake_arch
        elif self.arch == "grace_blackwell":
            # CUDA 13.0's ptxas refuses tcgen05/tensormap opcodes for sm_121/121a, so clamp to sm_120.
            self._set_arch_env("TORCH_CUDA_ARCH_LIST", "12.0")
            self._set_arch_env("CMAKE_CUDA_ARCHITECTURES", "120")
            self._set_arch_env("CUDAARCHS", "120")
        
        # PyTorch Inductor configuration
        inductor = getattr(torch, "_inductor", None)
        if inductor and hasattr(inductor, "config"):
            cfg = inductor.config
            # Enable PyTorch 2.10 features
            if hasattr(cfg, "triton"):
                triton_cfg = cfg.triton
                if hasattr(triton_cfg, "unique_kernel_names"):
                    triton_cfg.unique_kernel_names = True
                # Avoid automatic cudagraph wrapping to prevent RNG capture issues in setup code.
                if hasattr(triton_cfg, "cudagraph_trees"):
                    triton_cfg.cudagraph_trees = False
                if hasattr(triton_cfg, "cudagraphs"):
                    triton_cfg.cudagraphs = False
            
            # Enable max-autotune GEMM backends (PyTorch 2.10)
            # CUTLASS provides optimized GEMM kernels for NVIDIA GPUs
            if hasattr(cfg, "max_autotune_gemm_backends"):
                cfg.max_autotune_gemm_backends = "CUTLASS,TRITON,ATEN"
            
            # Enable CUTLASS for all operations
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_enabled_ops"):
                cfg.cuda.cutlass_enabled_ops = "all"
            
            # Enable aggressive Triton optimization for Blackwell
            if hasattr(cfg, "aggressive_fusion"):
                cfg.aggressive_fusion = True
        
        # Triton 3.5 configuration for Blackwell and Grace-Blackwell
        if self.arch in ("blackwell", "blackwell_ultra", "grace_blackwell"):
            try:
                import triton
                # Configure Triton 3.5 for appropriate architecture
                if hasattr(triton.runtime, "driver"):
                    if self.arch == "blackwell":
                        triton.runtime.driver.set_active_device_capability(10, 0)
                    elif self.arch == "blackwell_ultra":
                        triton.runtime.driver.set_active_device_capability(10, 3)
                    elif self.arch == "grace_blackwell":
                        # CUDA 13.0 PTXAS lacks SM 12.1 support for tensormap ops; target SM 12.0 instead.
                        triton.runtime.driver.set_active_device_capability(12, 0)
            except (ImportError, AttributeError):
                pass
            
            # Note: TMA is enabled automatically via compute capability configuration above
            # and TMA API usage in kernels. No environment variables needed.
            
            # Configure CUTLASS for torch.compile backend
            # Fix the cutlass_dir path to point to nvidia-cutlass-dsl installation
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_dir"):
                try:
                    import cutlass
                    # Get the nvidia_cutlass_dsl root directory
                    cutlass_module_path = os.path.dirname(cutlass.__file__)
                    nvidia_cutlass_root = os.path.dirname(os.path.dirname(cutlass_module_path))
                    cfg.cuda.cutlass_dir = nvidia_cutlass_root
                    try:
                        cutlass_pkg_version = importlib_metadata.version("nvidia-cutlass-dsl")
                        self.cutlass_version = cutlass_pkg_version
                        if _parse_version_tuple(cutlass_pkg_version) < (4, 2, 0):
                            warnings.warn(
                                "nvidia-cutlass-dsl < 4.2 detected; upgrade recommended for full Blackwell support.",
                                RuntimeWarning,
                            )
                    except importlib_metadata.PackageNotFoundError:
                        warnings.warn(
                            "nvidia-cutlass-dsl package not found; CUTLASS kernels may be skipped.",
                            RuntimeWarning,
                        )
                except ImportError:
                    # If cutlass not installed, unset cutlass_dir
                    # PyTorch will skip CUTLASS backend
                    pass

            if "TRITON_PTXAS_PATH" not in os.environ:
                try:
                    triton_root = Path(triton.__file__).resolve().parent
                    bundled_ptxas = triton_root / "backends" / "nvidia" / "bin" / "ptxas"
                    system_ptxas = shutil.which("ptxas")
                    version_ok = False
                    if bundled_ptxas.exists():
                        try:
                            result = subprocess.run(
                                [str(bundled_ptxas), "--version"],
                                capture_output=True,
                                text=True,
                                timeout=2,
                                check=False,
                            )
                        except (subprocess.SubprocessError, OSError):
                            result = None
                        if result and "release 13." in result.stdout:
                            version_ok = True
                    if not version_ok and system_ptxas:
                        os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                        if VERBOSE_EXPERIMENTAL_FEATURES:
                            print(f"PASSED: Triton: using system ptxas at {system_ptxas} for SM 12.1 support")
                except Exception as ex:
                    if VERBOSE_EXPERIMENTAL_FEATURES:
                        print(f"WARNING: Triton ptxas selection failed: {ex}")
        
        # Standard CUDA configurations
        os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
        # Use new PYTORCH_ALLOC_CONF (preferred), fallback to legacy PYTORCH_CUDA_ALLOC_CONF
        alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
        legacy_alloc = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if alloc_conf is None:
            alloc_conf = legacy_alloc or "max_split_size_mb:256,expandable_segments:True"
            if VERBOSE_EXPERIMENTAL_FEATURES and legacy_alloc:
                print("Migrated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF (new API)")
        os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
        
        # Configure TF32 using the shared helper to avoid legacy/new API mixing.
        enable_tf32()

    def print_info(self) -> None:
        cfg = self.config
        print(f"Architecture: {cfg['name']}")
        print(f"Compute Capability: {cfg['compute_capability']}")
        print(f"SM Version: {cfg['sm_version']}")
        print(f"Memory Bandwidth: {cfg['memory_bandwidth']}")
        print(f"Tensor Cores: {cfg['tensor_cores']}")
        if cfg['features']:
            print(f"Features: {', '.join(cfg['features'])}")
        if cfg['cuda_features']:
            print(f"CUDA Features: {', '.join(cfg['cuda_features'])}")
        if cfg['pytorch_optimizations']:
            print(f"PyTorch Optimisations: {', '.join(cfg['pytorch_optimizations'])}")
        if cfg['triton_features']:
            print(f"Triton Features: {', '.join(cfg['triton_features'])}")
        if cfg['profiling_tools']:
            print(f"Profiling Tools: {', '.join(cfg['profiling_tools'])}")

_OPTIMIZATIONS_APPLIED = False

# Feature flags (can be disabled via environment variables)
VERBOSE_EXPERIMENTAL_FEATURES = os.environ.get("VERBOSE_EXPERIMENTAL_FEATURES", "0") == "1"
ENABLE_TRITON_PATCH = _TRITON_PATCH_ENABLED


def configure_optimizations() -> None:
    global _OPTIMIZATIONS_APPLIED
    if _OPTIMIZATIONS_APPLIED:
        return
    ArchitectureConfig().configure_pytorch_optimizations()
    ensure_triton_compat()
    _OPTIMIZATIONS_APPLIED = True
    
    # Optionally pre-warm CUDA extensions in background
    # Enable via: export PREWARM_CUDA_EXTENSIONS=1
    if os.environ.get("PREWARM_CUDA_EXTENSIONS", "0") == "1":
        try:
            from core.utils.extension_prewarm import prewarm_extensions
            prewarm_extensions(background=True)
        except ImportError:
            pass  # Prewarm module not available


arch_config = ArchitectureConfig()
configure_optimizations()
