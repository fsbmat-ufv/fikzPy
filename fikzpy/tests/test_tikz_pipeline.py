from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fikzpy.core.diagnostics import sha256_file
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.tikz_pipeline import build_tikz_from_image


def _line_art_image() -> np.ndarray:
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    cv2.circle(image, (50, 40), 24, (0, 0, 0), 2)
    cv2.line(image, (25, 40), (75, 40), (0, 0, 0), 2)
    return image


def test_classic_mode_uses_contour_tikz_pipeline() -> None:
    result = build_tikz_from_image(
        _line_art_image(),
        ProcessingSettings(vectorization_mode="classic"),
        TikzOptions(use_bezier=False),
    )

    assert result.effective_mode == "classic"
    assert result.vector_objects == ()
    assert "% FIKZPY VECTOR MODE" not in result.tikz_code


def test_vector_mode_uses_vector_object_pipeline() -> None:
    result = build_tikz_from_image(
        _line_art_image(),
        ProcessingSettings(vectorization_mode="vector"),
        TikzOptions(use_bezier=True),
    )

    assert result.effective_mode == "vector"
    assert result.vector_objects
    assert result.vector_stats.total > 0
    assert "% FIKZPY VECTOR MODE" not in result.tikz_code
    assert ".. controls" in result.tikz_code


def test_fidelity_mode_preserves_more_vector_detail_than_vector_mode() -> None:
    image = _line_art_image()
    vector = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="vector"),
        TikzOptions(use_bezier=True),
    )
    fidelity = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="fidelity"),
        TikzOptions(use_bezier=True),
    )

    assert fidelity.effective_mode == "fidelity"
    assert fidelity.vector_stats.total >= vector.vector_stats.total
    assert len(fidelity.tikz_code) >= len(vector.tikz_code)


def test_visual_mode_uses_filled_svg_trace_pipeline() -> None:
    result = build_tikz_from_image(
        _line_art_image(),
        ProcessingSettings(vectorization_mode="visual"),
        TikzOptions(use_bezier=True),
    )

    assert result.effective_mode == "visual"
    assert result.vector_objects == ()
    assert result.visual_stats.paths > 0
    assert "\\path[fill=black" in result.tikz_code
    assert "cycle" in result.tikz_code


def test_classic_and_vector_tex_outputs_are_different(tmp_path: Path) -> None:
    image = _line_art_image()
    classic = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="classic"),
        TikzOptions(use_bezier=False),
    )
    vector = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="vector"),
        TikzOptions(use_bezier=True),
    )
    classic_path = tmp_path / "fikzpy_classic_001.tex"
    vector_path = tmp_path / "fikzpy_vector_001.tex"
    classic_path.write_text(wrap_standalone_document(classic.tikz_code), encoding="utf-8")
    vector_path.write_text(wrap_standalone_document(vector.tikz_code), encoding="utf-8")

    assert sha256_file(classic_path) != sha256_file(vector_path)


def test_vector_pipeline_failure_is_not_silently_classic(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_vector_pipeline(*args, **kwargs):
        raise RuntimeError("vector pipeline failed")

    monkeypatch.setattr("fikzpy.core.tikz_pipeline.fit_contours_to_vector_objects", broken_vector_pipeline)

    with pytest.raises(RuntimeError, match="vector pipeline failed"):
        build_tikz_from_image(
            _line_art_image(),
            ProcessingSettings(vectorization_mode="vector"),
            TikzOptions(use_bezier=True),
        )
