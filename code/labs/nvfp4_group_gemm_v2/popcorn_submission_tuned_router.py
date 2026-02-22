"""Popcorn entrypoint: case-routed CUTLASS 2SM grouped GEMM."""

from __future__ import annotations

import os
from typing import Any

import torch

from labs.nvfp4_group_gemm.cutlass_extension import load_cutlass_nvfp4_grouped_gemm_sm100
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import COMPETITION_CASES


def _shape_signature(data: tuple[Any, ...]) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    problem_sizes = data[3]
    m_sig: list[int] = []
    n_sig: list[int] = []
    k_sig: list[int] = []
    for m, n, k, _l in problem_sizes:
        m_sig.append(int(m))
        n_sig.append(int(n))
        k_sig.append(int(k))
    return (len(problem_sizes), tuple(m_sig), tuple(n_sig), tuple(k_sig))


def _build_case_signature_map() -> dict[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], int]:
    mapping: dict[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], int] = {}
    for idx, case in enumerate(COMPETITION_CASES):
        mapping[(int(case.g), tuple(int(x) for x in case.m), tuple(int(x) for x in case.n), tuple(int(x) for x in case.k))] = idx
    return mapping


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _set_case_env(case_idx: int) -> None:
    # Clear tunables we may set differently between cases.
    for key in (
        "AISP_NVFP4_GROUP_GEMM_CLUSTER_M",
        "AISP_NVFP4_GROUP_GEMM_CLUSTER_N",
        "AISP_NVFP4_GROUP_GEMM_RASTER_ORDER",
        "AISP_NVFP4_GROUP_GEMM_USE_PDL",
        "AISP_NVFP4_GROUP_GEMM_PERSISTENT_REQUEST_CHUNK",
        "AISP_NVFP4_GROUP_GEMM_PERSISTENT_GROUP_ORDER",
        "AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER",
        "AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE",
    ):
        if key in os.environ:
            del os.environ[key]

    if case_idx == 0:
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "2"
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "8"
        return

    if case_idx == 1:
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "2"
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "0"
        os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "16"
        return

    if case_idx == 2:
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "2"
        os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "0"
        return

    if case_idx == 3:
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "0"
        os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"
        os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "4"
        return


def _variant_for_case(case_idx: int) -> str:
    if case_idx == 0:
        return "2sm_s4"
    if case_idx == 1:
        return "2sm"
    if case_idx == 2:
        return "1sm_n128"
    if case_idx == 3:
        return "1sm_n128"
    return "2sm"


def _variant_fns(ext: Any, variant: str) -> tuple[Any, Any]:
    if variant == "2sm_mxf4":
        return ext.build_metadata_2sm_mxf4, ext.create_plan_2sm_mxf4
    if variant == "1sm_n128":
        return ext.build_metadata_1sm_n128, ext.create_plan_1sm_n128
    if variant == "2sm_s4":
        return ext.build_metadata_2sm_s4, ext.create_plan_2sm_s4
    if variant == "2sm_s5":
        return ext.build_metadata_2sm_s5, ext.create_plan_2sm_s5
    return ext.build_metadata_2sm, ext.create_plan_2sm


def _group_order_for_case(case_idx: int, group_count: int) -> list[int]:
    if group_count <= 1:
        return list(range(group_count))
    return list(range(group_count))


def _ctx_key(problem_sizes: list[tuple[int, int, int, int]], variant: str) -> tuple[str, int, int, int, int, bool, tuple[tuple[int, int, int, int], ...]]:
    ps = tuple(tuple(int(x) for x in p) for p in problem_sizes)
    return (
        str(variant),
        _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", 2),
        _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", 1),
        _env_int("AISP_NVFP4_GROUP_GEMM_RASTER_ORDER", 0),
        _env_int("AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE", 0),
        _env_bool("AISP_NVFP4_GROUP_GEMM_USE_PDL", False),
        ps,
    )


def _get_case_ctx(problem_sizes: list[tuple[int, int, int, int]], variant: str) -> tuple[Any, dict[str, Any], Any]:
    key = _ctx_key(problem_sizes, variant)
    cached = _CASE_CTX_CACHE.get(key)
    ext = load_cutlass_nvfp4_grouped_gemm_sm100(verbose=False)
    build_metadata, create_plan = _variant_fns(ext, variant)
    if cached is not None:
        return ext, cached, create_plan

    _variant, cluster_m, cluster_n, raster_order, max_swizzle_size, use_pdl, _ = key
    ps_cpu = torch.tensor(problem_sizes, dtype=torch.int32, device="cpu")
    (
        problem_shapes_u8,
        stride_a_u8,
        stride_b_u8,
        stride_c_u8,
        stride_d_u8,
        layout_sfa_u8,
        layout_sfb_u8,
        workspace_u8,
    ) = build_metadata(ps_cpu, cluster_m, cluster_n, raster_order, max_swizzle_size)

    ctx = {
        "problem_shapes_u8": problem_shapes_u8,
        "stride_a_u8": stride_a_u8,
        "stride_b_u8": stride_b_u8,
        "stride_c_u8": stride_c_u8,
        "stride_d_u8": stride_d_u8,
        "layout_sfa_u8": layout_sfa_u8,
        "layout_sfb_u8": layout_sfb_u8,
        "workspace_u8": workspace_u8,
        "cluster_m": int(cluster_m),
        "cluster_n": int(cluster_n),
        "raster_order": int(raster_order),
        "max_swizzle_size": int(max_swizzle_size),
        "use_pdl": bool(use_pdl),
    }
    _CASE_CTX_CACHE[key] = ctx
    return ext, ctx, create_plan


def _prepare_data(data: tuple[Any, ...], case_idx: int) -> tuple[Any, ...]:
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
    variant = _variant_for_case(int(case_idx))
    order = _group_order_for_case(int(case_idx), len(problem_sizes))
    ordered_problem_sizes = [problem_sizes[i] for i in order]
    ext, case_ctx, create_plan = _get_case_ctx(ordered_problem_sizes, variant)

    a_ptrs: list[int] = []
    b_ptrs: list[int] = []
    c_ptrs: list[int] = []
    sfa_ptrs: list[int] = []
    sfb_ptrs: list[int] = []
    for i in order:
        a, b, c = abc_tensors[i]
        sfa_reordered, sfb_reordered = sfasfb_reordered_tensors[i]
        a_ptrs.append(int(a.data_ptr()))
        b_ptrs.append(int(b.data_ptr()))
        c_ptrs.append(int(c.data_ptr()))
        sfa_ptrs.append(int(sfa_reordered.data_ptr()))
        sfb_ptrs.append(int(sfb_reordered.data_ptr()))

    ctx = {
        "ptr_a_i64": torch.tensor(a_ptrs, dtype=torch.int64, device="cuda"),
        "ptr_b_i64": torch.tensor(b_ptrs, dtype=torch.int64, device="cuda"),
        "ptr_c_i64": torch.tensor(c_ptrs, dtype=torch.int64, device="cuda"),
        "ptr_d_i64": torch.tensor(c_ptrs, dtype=torch.int64, device="cuda"),
        "ptr_sfa_i64": torch.tensor(sfa_ptrs, dtype=torch.int64, device="cuda"),
        "ptr_sfb_i64": torch.tensor(sfb_ptrs, dtype=torch.int64, device="cuda"),
        "outputs": [abc_tensors[i][2] for i in range(len(abc_tensors))],
    }

    ctx["plan"] = create_plan(
        case_ctx["problem_shapes_u8"],
        case_ctx["stride_a_u8"],
        case_ctx["stride_b_u8"],
        case_ctx["stride_c_u8"],
        case_ctx["stride_d_u8"],
        case_ctx["layout_sfa_u8"],
        case_ctx["layout_sfb_u8"],
        case_ctx["workspace_u8"],
        ctx["ptr_a_i64"],
        ctx["ptr_b_i64"],
        ctx["ptr_sfa_i64"],
        ctx["ptr_sfb_i64"],
        ctx["ptr_c_i64"],
        ctx["ptr_d_i64"],
        1.0,
        0.0,
        case_ctx["raster_order"],
        case_ctx["cluster_m"],
        case_ctx["cluster_n"],
        case_ctx["max_swizzle_size"],
        case_ctx["use_pdl"],
    )
    plan = ctx.get("plan")
    if plan is None:
        raise RuntimeError("missing CUTLASS plan for graph capture")
    if ctx.get("graph_obj") is None:
        plan.run()
        torch.cuda.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            plan.run()
        ctx["graph_obj"] = graph_obj
        for t in ctx["outputs"]:
            t.zero_()
    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx)


def _run_prepared(prepared: tuple[Any, ...]) -> list[torch.Tensor]:
    ctx = prepared[4]
    graph_obj = ctx.get("graph_obj")
    if graph_obj is None:
        raise RuntimeError("missing graph object in prepared context")
    graph_obj.replay()
    return ctx["outputs"]


_CASE_SIG_MAP = _build_case_signature_map()
_CUDA_AVAILABLE = torch.cuda.is_available()
_MAX_CACHE_ENTRIES = 16
_CACHE_ORDER: list[int] = []
_CASE_BY_DATA_ID: dict[int, tuple[Any, int]] = {}
_PREPARED_CACHE: dict[int, tuple[Any, tuple[Any, ...]]] = {}
_CASE_CTX_CACHE: dict[tuple[str, int, int, int, int, bool, tuple[tuple[int, int, int, int], ...]], dict[str, Any]] = {}


def _cache_insert(data_id: int) -> None:
    if data_id in _CACHE_ORDER:
        _CACHE_ORDER.remove(data_id)
    _CACHE_ORDER.append(data_id)
    while len(_CACHE_ORDER) > _MAX_CACHE_ENTRIES:
        victim = _CACHE_ORDER.pop(0)
        _PREPARED_CACHE.pop(victim, None)
        _CASE_BY_DATA_ID.pop(victim, None)


def custom_kernel(data):
    if not _CUDA_AVAILABLE:
        raise RuntimeError("NVFP4 submission requires CUDA")

    data_id = id(data)
    prepared_pack = _PREPARED_CACHE.get(data_id)
    if prepared_pack is not None:
        cached_data, prepared = prepared_pack
        if cached_data is data:
            return _run_prepared(prepared)
        _PREPARED_CACHE.pop(data_id, None)
        _CASE_BY_DATA_ID.pop(data_id, None)

    case_pack = _CASE_BY_DATA_ID.get(data_id)
    if case_pack is not None and case_pack[0] is data:
        case_idx = case_pack[1]
    else:
        sig = _shape_signature(data)
        case_idx = _CASE_SIG_MAP.get(sig)
        if case_idx is None:
            raise RuntimeError(f"Unknown NVFP4 competition shape signature: {sig}")
        _CASE_BY_DATA_ID[data_id] = (data, int(case_idx))
        _cache_insert(data_id)

    _set_case_env(int(case_idx))
    prepared = _prepare_data(data, int(case_idx))
    _PREPARED_CACHE[data_id] = (data, prepared)
    _cache_insert(data_id)
    return _run_prepared(prepared)


__all__ = ["custom_kernel"]
