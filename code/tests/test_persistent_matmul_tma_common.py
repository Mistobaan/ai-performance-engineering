from ch10.persistent_matmul_tma_common import (
    persistent_program_count,
    persistent_tile_count,
)


def test_persistent_tile_count_matches_grid_shape():
    assert persistent_tile_count(4096, 4096, 128, 128) == 1024
    assert persistent_tile_count(256, 512, 128, 128) == 8


def test_persistent_program_count_caps_at_one_program_per_sm():
    assert persistent_program_count(4096, 4096, 128, 128, num_sms=148) == 148
    assert persistent_program_count(256, 256, 128, 128, num_sms=148) == 4
