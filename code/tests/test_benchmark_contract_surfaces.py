from __future__ import annotations

from core.api.registry import get_routes
from core.benchmark.contracts_surface import get_benchmark_contracts_summary
from core.tools.tools_commands import TOOLS


def test_benchmark_contracts_summary_exposes_expected_surfaces() -> None:
    summary = get_benchmark_contracts_summary()

    assert summary["available"] is True
    assert summary["interfaces"]["cli"] == "python -m cli.aisp tools benchmark-contracts"
    assert summary["interfaces"]["dashboard_api"] == "/api/benchmark/contracts"
    assert summary["interfaces"]["mcp_tool"] == "benchmark_contracts"

    contracts = summary["contracts"]
    assert contracts["warehouse"]["exists"] is True
    assert contracts["benchmark_run"]["summary"]["has_observability"] is True
    assert contracts["benchmark_run"]["summary"]["has_sinks"] is True


def test_benchmark_contracts_are_exposed_in_cli_and_dashboard_registry() -> None:
    assert "benchmark-contracts" in TOOLS
    assert any(route.name == "benchmark.contracts" for route in get_routes())
