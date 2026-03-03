"""Input generation for NVFP4 grouped GEMM benchmarks.

Ported from GPU MODE reference-kernels, but rewritten to avoid mutating global RNG
state (the harness enforces seed immutability in verify mode).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Scaling factor vector size
_SF_VEC_SIZE = 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _create_fp4_tensors(l: int, mn: int, k: int, *, gen: torch.Generator) -> torch.Tensor:
    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type.
    ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    # Keep sign bit + two LSBs for each nibble.
    ref_i8 = ref_i8 & 0b1011_1011
    return ref_i8.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)


def create_reordered_scale_factor_tensor(
    l: int,
    mn: int,
    k: int,
    ref_f8_tensor: torch.Tensor,
    *,
    gen: torch.Generator,
) -> torch.Tensor:
    """Create reordered scale factor tensor per the cuBLAS block scaling layout."""
    sf_k = _ceil_div(k, _SF_VEC_SIZE)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # batch size
        _ceil_div(mn, atom_m[0] * atom_m[1]),
        _ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device="cuda", generator=gen)
    # Keep the reordered scale dtype consistent with the reference scale dtype so data layout
    # and byte-encoding stay aligned with the chosen FP8 scale format.
    reordered_f8_tensor = rand_int_tensor.to(dtype=ref_f8_tensor.dtype).permute(*mma_permute_order)

    if ref_f8_tensor.device.type == "cpu":
        ref_f8_tensor = ref_f8_tensor.cuda()

    i_idx = torch.arange(mn, device="cuda")
    j_idx = torch.arange(sf_k, device="cuda")
    b_idx = torch.arange(l, device="cuda")
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k

    reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[i_grid, j_grid, b_grid]
    return reordered_f8_tensor


def generate_input(
    *,
    m: Tuple[int, ...],
    n: Tuple[int, ...],
    k: Tuple[int, ...],
    g: int,
    seed: int,
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor]],
    List[Tuple[int, int, int, int]],
]:
    """Generate input tensors for NVFP4 block-scaled group GEMM."""
    l = 1

    # Separate generators per device to avoid mutating global RNG state.
    gen_cuda = torch.Generator(device="cuda")
    gen_cpu = torch.Generator(device="cpu")
    gen_cuda.manual_seed(int(seed))
    gen_cpu.manual_seed(int(seed))

    abc_tensors: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sfasfb_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    sfasfb_reordered_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
    problem_sizes: List[Tuple[int, int, int, int]] = []

    for group_idx in range(int(g)):
        mi = int(m[group_idx])
        ni = int(n[group_idx])
        ki = int(k[group_idx])

        a_ref = _create_fp4_tensors(l, mi, ki, gen=gen_cuda)
        b_ref = _create_fp4_tensors(l, ni, ki, gen=gen_cuda)

        c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda", generator=gen_cuda).permute(1, 2, 0)

        sf_k = _ceil_div(ki, _SF_VEC_SIZE)
        sfa_ref_cpu = (
            torch.randint(1, 3, (l, mi, sf_k), dtype=torch.int8, device="cpu", generator=gen_cpu)
            .to(dtype=torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        sfb_ref_cpu = (
            torch.randint(1, 3, (l, ni, sf_k), dtype=torch.int8, device="cpu", generator=gen_cpu)
            .to(dtype=torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )

        sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu, gen=gen_cuda)
        sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu, gen=gen_cuda)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
        problem_sizes.append((mi, ni, ki, l))

    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)


__all__ = ["generate_input"]
