from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_classic_baseline import BASELINE_EXAMPLES, generate_baseline


def test_classic_baseline_script_generates_five_examples_without_pdf(tmp_path: Path) -> None:
    records = generate_baseline(tmp_path, compile_pdf=False)

    assert len(records) == len(BASELINE_EXAMPLES) == 5
    assert (tmp_path / "baseline_metrics.json").exists()
    assert (tmp_path / "README.md").exists()

    written_metrics = json.loads((tmp_path / "baseline_metrics.json").read_text(encoding="utf-8"))
    assert [item["slug"] for item in written_metrics] == [item["slug"] for item in records]

    for record in records:
        assert record["mode"] == "classic"
        assert record["effective_mode"] == "classic"
        assert record["pdf_status"] == "skipped"
        assert record["pdf"] is None
        assert record["tex_bytes"] > 0
        assert record["processing_seconds"] >= 0.0
        assert (tmp_path / f"{record['slug']}.png").exists()
        assert (tmp_path / f"{record['slug']}.tex").exists()
