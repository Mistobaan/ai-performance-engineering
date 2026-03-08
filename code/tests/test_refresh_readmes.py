from __future__ import annotations

from pathlib import Path

import json

from core.scripts.refresh_readmes import (
    ENTRIES,
    REPO_ROOT,
    _format_markdown,
    _render_current_representative_deltas_body,
)


PRIORITY_EVIDENCE_DOCS = (
    "ch10",
    "ch14",
    "ch18",
    "labs/block_scaling",
    "labs/flashattention4",
    "labs/persistent_decode",
    "labs/real_world_models",
)


def _output_path(slug: str) -> Path:
    if slug.endswith(".md"):
        return REPO_ROOT / slug
    return REPO_ROOT / slug / "README.md"


def test_root_readme_preserves_evidence_first_sections() -> None:
    markdown = _format_markdown(ENTRIES["README.md"])

    assert "## Tier-1 Canonical Suite" in markdown
    assert "## Current Representative Deltas" in markdown
    assert "## Profiler Evidence" in markdown
    assert markdown.index("## Tier-1 Canonical Suite") < markdown.index("## Learning Goals")


def test_ch10_and_priority_labs_render_custom_evidence_sections() -> None:
    ch10_markdown = _format_markdown(ENTRIES["ch10"])
    ch14_markdown = _format_markdown(ENTRIES["ch14"])
    ch18_markdown = _format_markdown(ENTRIES["ch18"])
    block_scaling_markdown = _format_markdown(ENTRIES["labs/block_scaling"])
    models_markdown = _format_markdown(ENTRIES["labs/real_world_models"])

    assert "## Problem" in ch10_markdown
    assert "## Baseline Path" in ch10_markdown
    assert "## Optimized Path" in ch10_markdown
    assert "## Measured Delta" in ch10_markdown
    assert "## Repro Commands" in ch10_markdown
    assert ch10_markdown.index("## Problem") < ch10_markdown.index("## Learning Goals")

    for markdown in (ch14_markdown, ch18_markdown):
        assert "## Problem" in markdown
        assert "## Baseline Path" in markdown
        assert "## Optimized Path" in markdown
        assert "## Measured Delta" in markdown
        assert "## Profiler Evidence" in markdown
        assert "## Repro Commands" in markdown
        assert markdown.index("## Problem") < markdown.index("## Learning Goals")

    assert "## Running the Lab" in block_scaling_markdown
    assert "## Recommended Knobs" in block_scaling_markdown
    assert "## Harness vs Microbenchmark" in block_scaling_markdown

    assert "## Problem" in models_markdown
    assert "## Profiler Evidence" in models_markdown
    assert "## Repro Commands" in models_markdown


def test_priority_readmes_match_generated_content() -> None:
    slugs = ("README.md",) + PRIORITY_EVIDENCE_DOCS

    for slug in slugs:
        expected = _format_markdown(ENTRIES[slug]).rstrip() + "\n"
        actual = _output_path(slug).read_text(encoding="utf-8")
        assert actual == expected, f"{slug} is out of sync with core/scripts/refresh_readmes.py"


def test_current_representative_deltas_prefer_tier1_history_when_available(tmp_path: Path) -> None:
    history_root = tmp_path / "artifacts" / "history" / "tier1" / "20260308_070000_manual"
    history_root.mkdir(parents=True)
    summary_path = history_root / "summary.json"
    summary_payload = {
        "run_id": "20260308_070000_manual",
        "targets": [
            {
                "target": "labs/block_scaling:block_scaling",
                "status": "succeeded",
                "baseline_time_ms": 0.2,
                "best_speedup": 2.0,
            },
            {
                "target": "ch04:gradient_fusion",
                "status": "succeeded",
                "baseline_time_ms": 4.0,
                "best_speedup": 8.0,
            },
        ],
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    index_path = tmp_path / "artifacts" / "history" / "tier1" / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "runs": [
                    {
                        "run_id": "20260308_070000_manual",
                        "summary_path": str(summary_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    body = _render_current_representative_deltas_body(tmp_path)

    assert "latest canonical tier-1 history summary" in body
    assert "`labs/block_scaling:block_scaling`" in body
    assert "`0.200 ms`" in body
    assert "`0.100 ms`" in body
    assert "artifacts/history/tier1/20260308_070000_manual/summary.json" in body
