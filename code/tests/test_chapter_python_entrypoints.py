from __future__ import annotations

from pathlib import Path


def test_chapter_entrypoints_import_path_when_using_path_dunder_file() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in sorted(root.glob("ch*/*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "Path(__file__)" not in text:
            continue
        if "from pathlib import Path" in text or "import pathlib" in text:
            continue
        offenders.append(str(path.relative_to(root)))

    assert not offenders, (
        "Chapter Python entrypoints use Path(__file__) without importing Path: "
        + ", ".join(offenders)
    )
