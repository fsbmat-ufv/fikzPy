"""Visual regression tests for the dot-to-dot mode on a real complex image."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from fikzpy.core.dot_to_dot import DotToDotConfig, run_dot_to_dot
from fikzpy.core.image_processor import (
    ProcessingSettings,
    process_image,
    visual_settings_from_processing,
)
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.tikz_pipeline import build_tikz_from_image
from fikzpy.core.visual_pipeline import contours_to_svg

_IMAGE_PATH = Path(__file__).resolve().parent / "25.jpg"

pytestmark = pytest.mark.skipif(not _IMAGE_PATH.exists(), reason="tests/25.jpg not present")


@pytest.fixture(scope="module")
def dino_svg() -> tuple[str, tuple[int, ...]]:
    """Trace tests/25.jpg once and reuse the intermediate SVG copy."""
    image = cv2.imread(str(_IMAGE_PATH))
    settings = ProcessingSettings(vectorization_mode="dot_to_dot")
    result = process_image(image, settings)
    svg = contours_to_svg(
        result.contours,
        image.shape,
        options=TikzOptions(),
        settings=visual_settings_from_processing(settings),
    )
    return svg, image.shape


def _labels(code: str) -> list[str]:
    return re.findall(r"\\node\[dotToDotLabel[^\]]*\] at \([^)]*\) \{([^}]*)\};", code)


def _label_positions(code: str) -> list[tuple[float, float]]:
    matches = re.findall(r"\\node\[dotToDotLabel[^\]]*\] at \(([-\d.]+),([-\d.]+)\)", code)
    return [(float(x), float(y)) for x, y in matches]


# --- 10 points -----------------------------------------------------------------------


def test_real_image_10_points(dino_svg) -> None:
    svg, shape = dino_svg
    result = run_dot_to_dot(svg, shape, DotToDotConfig(total_points=10))
    code = result.tikz_code

    assert code.count("\\fill[dotToDotPoint]") == 10
    assert code.count("\\node[dotToDotLabel") == 10
    assert _labels(code) == [str(number) for number in range(1, 11)]
    # Single silhouette, single sequence, no original lines, no letters.
    assert len(result.sequences) == 1
    assert "\\draw" not in code
    assert not re.search(r"\{[A-Z]+\d+\}", code)
    # No duplicate points.
    positions = [
        (round(dot.position.x, 3), round(dot.position.y, 3))
        for dot in result.sequences[0].points
    ]
    assert len(set(positions)) == 10
    # The dotted silhouette spans most of the figure (page coverage).
    xs = [dot.position.x for dot in result.sequences[0].points]
    ys = [dot.position.y for dot in result.sequences[0].points]
    assert (max(xs) - min(xs)) > 0.5 * shape[1]
    assert (max(ys) - min(ys)) > 0.5 * shape[0]


# --- 50 points -----------------------------------------------------------------------


def test_real_image_50_points(dino_svg) -> None:
    svg, shape = dino_svg
    result = run_dot_to_dot(svg, shape, DotToDotConfig(total_points=50))
    code = result.tikz_code

    assert code.count("\\fill[dotToDotPoint]") == 50
    assert _labels(code) == [str(number) for number in range(1, 51)]
    assert "\\draw" not in code
    assert not re.search(r"\{[A-Z]+\d+\}", code)

    # Recognizable silhouette: points hug the extracted outline.
    for dot in result.sequences[0].points:
        _length, _point, distance = result.silhouette.closest_length(dot.position)
        assert distance < 3.0

    # Numbers predominantly outside the silhouette.
    polygon = np.array(
        [[p.x, p.y] for p in result.silhouette.polyline()], np.float32
    ).reshape(-1, 1, 2)
    scale = 10.0 / shape[1]
    outside = 0
    for x, y in _label_positions(code):
        # Convert the TikZ label position back to SVG space for the test.
        svg_x = x / scale
        svg_y = shape[0] - y / scale
        if cv2.pointPolygonTest(polygon, (svg_x, svg_y), False) < 0:
            outside += 1
    assert outside >= 35  # at least 70% outside

    # No severe overlap: label anchor points keep a minimum separation.
    placed = _label_positions(code)
    min_gap = min(
        float(np.hypot(ax - bx, ay - by))
        for i, (ax, ay) in enumerate(placed)
        for bx, by in placed[i + 1 :]
    )
    assert min_gap > 0.12  # TikZ units (~1.2 mm at the default scale)


def test_real_image_details_keep_lines_but_not_outline(dino_svg) -> None:
    svg, shape = dino_svg
    config = DotToDotConfig(total_points=50, preserve_internal_details=True)
    result = run_dot_to_dot(svg, shape, config)
    # Details are drawn, but the outer silhouette is never restored.
    assert len(result.remaining_subpaths) > 0
    outline_w = result.silhouette.bounding_box()[2] - result.silhouette.bounding_box()[0]
    for detail in result.remaining_subpaths:
        box = detail.bounding_box()
        assert (box[2] - box[0]) < 0.8 * outline_w
    # Same point budget as without details.
    assert result.point_count == 50


# --- Manual selection on the real trace ----------------------------------------------


def test_real_image_manual_selection_flow(dino_svg) -> None:
    from fikzpy.core.dot_to_dot.segment_selection import InteractiveSelectionController
    from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths

    svg, shape = dino_svg
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(svg))
    subpaths = [sp for p in controller.paths for sp in p.subpaths]
    main = max(subpaths, key=lambda sp: sp.total_length)

    controller.set_point_count(6)
    p1 = main.point_at_length(main.total_length * 0.1)
    p2 = main.point_at_length(main.total_length * 0.3)
    assert controller.click(p1.x, p1.y) is not None
    assert controller.click(p2.x, p2.y) is not None
    pending = controller.pending_segment()
    assert pending is not None

    # Inversion flips the chosen arc.
    controller.invert()
    assert controller.pending_segment().reverse is True
    controller.invert()

    controller.confirm()
    # Second arc continues the numbering.
    p3 = main.point_at_length(main.total_length * 0.5)
    p4 = main.point_at_length(main.total_length * 0.7)
    controller.set_point_count(4)
    controller.click(p3.x, p3.y)
    controller.click(p4.x, p4.y)
    controller.confirm()

    assert controller.model.numbering_ranges() == [(1, 6), (7, 10)]

    config = DotToDotConfig(
        generation_mode="selected_segments",
        selected_segments=controller.model.segments,
    )
    result = run_dot_to_dot(svg, shape, config)
    assert result.point_count == 10
    assert _labels(result.tikz_code) == [str(number) for number in range(1, 11)]
    # The selected arcs were genuinely removed: the affected subpath is now
    # split into open runs instead of the original closed loop.
    assert any(not subpath.closed for subpath in result.remaining_subpaths)


# --- Non-regression of the other four modes -------------------------------------------


def _shape_image() -> np.ndarray:
    image = np.full((120, 120, 3), 255, np.uint8)
    cv2.rectangle(image, (25, 25), (95, 95), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 50), (70, 70), (255, 255, 255), -1)
    return image


@pytest.mark.parametrize(
    ("mode", "strategy"),
    [("classic", "auto"), ("classic", "filled"), ("visual", "auto"), ("contours", "auto")],
)
def test_dot_to_dot_config_never_affects_other_modes(mode: str, strategy: str) -> None:
    image = _shape_image()
    options = TikzOptions(classic_strategy=strategy)
    plain = build_tikz_from_image(image, ProcessingSettings(vectorization_mode=mode), options)
    with_config = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode=mode),
        options,
        dot_to_dot_config=DotToDotConfig(total_points=10),
    )
    assert plain.tikz_code == with_config.tikz_code
    assert "dotToDot" not in plain.tikz_code


# --- Real compilation ------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_real_pdf_compilation_10_points(dino_svg, tmp_path: Path) -> None:
    svg, shape = dino_svg
    result = run_dot_to_dot(svg, shape, DotToDotConfig(total_points=10))
    tex_path = tmp_path / "dots10.tex"
    tex_path.write_text(wrap_standalone_document(result.tikz_code), encoding="utf-8")
    completed = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-2000:]
    pdf_path = tex_path.with_suffix(".pdf")
    assert pdf_path.exists() and pdf_path.stat().st_size > 1000
