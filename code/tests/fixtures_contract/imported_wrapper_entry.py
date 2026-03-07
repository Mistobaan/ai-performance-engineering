"""Wrapper fixture module for imported benchmark class contract tests."""

from tests.fixtures_contract.imported_wrapper_common import SharedBench


def get_benchmark():
    bench = SharedBench()
    return bench
