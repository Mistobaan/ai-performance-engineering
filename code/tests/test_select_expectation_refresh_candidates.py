import json
from pathlib import Path

from core.analysis.select_expectation_refresh_candidates import (
    select_candidates,
    write_candidate_outputs,
)


def test_select_candidates_filters_by_recommendation_and_materiality():
    rows = [
        {"target": "ch01:gemm", "refresh_recommendation": "update_now", "material_mismatch": True},
        {"target": "ch01:gemm_small", "refresh_recommendation": "update_now", "material_mismatch": False},
        {"target": "ch02:foo", "refresh_recommendation": "rerun_once_more", "material_mismatch": True},
    ]

    selected = select_candidates(rows, recommendations=["update_now"], include_non_material=False)

    assert [row["target"] for row in selected] == ["ch01:gemm"]


def test_write_candidate_outputs_materializes_target_list(tmp_path: Path):
    ledger_json = tmp_path / "ledger.json"
    ledger_json.write_text(
        json.dumps(
            [
                {
                    "target": "ch01:gemm",
                    "chapter": "ch01",
                    "example": "gemm",
                    "refresh_recommendation": "update_now",
                    "material_mismatch": True,
                },
                {
                    "target": "ch02:foo",
                    "chapter": "ch02",
                    "example": "foo",
                    "refresh_recommendation": "hold_noisy",
                    "material_mismatch": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    outputs = write_candidate_outputs(
        ledger_json=ledger_json,
        output_dir=tmp_path / "out",
        recommendations=["update_now"],
        include_non_material=False,
    )

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["selected_count"] == 1
    assert summary["targets"] == ["ch01:gemm"]
    assert outputs["txt"].read_text(encoding="utf-8") == "ch01:gemm\n"
