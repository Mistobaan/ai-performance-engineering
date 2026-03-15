"""Shared launch-policy helpers for the Chapter 10 persistent TMA example."""

from __future__ import annotations


def persistent_tile_count(m: int, n: int, block_m: int, block_n: int) -> int:
    """Return the number of output tiles in the launch grid."""
    grid_m = (m + block_m - 1) // block_m
    grid_n = (n + block_n - 1) // block_n
    return grid_m * grid_n


def persistent_program_count(
    m: int,
    n: int,
    block_m: int,
    block_n: int,
    num_sms: int,
) -> int:
    """Launch at most one persistent program per SM."""
    return min(num_sms, persistent_tile_count(m, n, block_m, block_n))
