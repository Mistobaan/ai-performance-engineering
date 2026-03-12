#!/usr/bin/env python3
"""Print the benchmark methodology and warehouse contract surfaces as JSON."""

from __future__ import annotations

import json

from core.benchmark.contracts_surface import get_benchmark_contracts_summary


def main() -> int:
    print(json.dumps(get_benchmark_contracts_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
