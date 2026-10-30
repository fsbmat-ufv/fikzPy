"""Tests for full-figure dot-to-dot generation (single silhouette output)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from fikzpy.core.dot_to_dot import DotToDotConfig, DotToDotError, run_dot_to_dot
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.tikz_pipeline import build_tikz_from_image


def _svg(paths: str, *, width: float = 200.0, height: float = 200.0) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f"{paths}</svg>"
    )


_MAIN_WITH_DETAIL = _svg(
    '<path d="M 20 20 L 180 20 L 180 180 L 20 180 Z '
    'M 80 80 L 120 80 L 120 120 L 80 120 Z"/>'
)

_TWO_BLOBS = _svg(
    '<path d="M 10 10 L 90 10 L 90 90 L 10 90 Z"/>'
    '<path d="M 120 120 L 160 120 L 160 160 L 120 160 Z"/>'
)


def _tikz_labels(code: str) -> list[str]:
    return re.findall(r"\\node\[dotToDotLabel[^\]]*\] at \([^)]*\) \{([^}]*)\};", code)


# --- Exact counts and pure numeric labels -------------------------------------------


@pytest.mark.parametrize("total", [10, 50])
def test_exact_point_and_node_counts(total: int) -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=total))
    code = result.tikz_code
    assert code.count("\\fill[dotToDotPoint]") == total
    assert code.count("\\node[dotToDotLabel") == total
    assert result.point_count == total


@pytest.mark.parametrize("total", [10, 50])
def test_labels_are_exactly_one_to_n(total: int) -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=total))
    labels = _tikz_labels(result.tikz_code)
    assert labels == [str(number) for number in range(1, total + 1)]


def test_no_alphanumeric_sequences() -> None:
    result = run_dot_to_dot(_TWO_BLOBS, (200, 200), DotToDotConfig(total_points=30))
    labels = _tikz_labels(result.tikz_code)
    assert all(re.fullmatch(r"\d+", label) for label in labels)
    assert not re.search(r"[A-Z]+\d+", result.tikz_code)


def test_single_sequence_even_with_disconnected_blobs() -> None:
    # Two disconnected squares: full figure still yields ONE numbered sequence
    # over the main region only.
    result = run_dot_to_dot(_TWO_BLOBS, (200, 200), DotToDotConfig(total_points=12))
    assert len(result.sequences) == 1
    assert result.point_count == 12


# --- Original drawing removed by default --------------------------------------------


def test_no_original_paths_by_default() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=10))
    assert result.remaining_subpaths == ()
    assert "\\draw" not in result.tikz_code


def test_internal_details_only_when_requested() -> None:
    config = DotToDotConfig(total_points=10, preserve_internal_details=True)
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), config)
    # The inner square detail is kept, rendered as Visual-style filled ink...
    assert len(result.remaining_subpaths) >= 1
    assert result.tikz_code.count("\\draw[dotToDotInk]") == 1
    # ...and the point count is unchanged.
    assert result.point_count == 10
    labels = _tikz_labels(result.tikz_code)
    assert labels == [str(number) for number in range(1, 11)]


def test_details_never_restore_the_replaced_outline() -> None:
    config = DotToDotConfig(total_points=10, preserve_internal_details=True)
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), config)
    outline_min_x, _, outline_max_x, _ = result.silhouette.bounding_box()
    for detail in result.remaining_subpaths:
        min_x, _, max_x, _ = detail.bounding_box()
        # Every kept detail is strictly smaller than the outer silhouette.
        assert (max_x - min_x) < 0.8 * (outline_max_x - outline_min_x)


# --- Start point, direction and distribution ----------------------------------------


def test_first_point_is_topmost() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8))
    points = result.sequences[0].points
    min_y = min(dot.position.y for dot in points)
    assert points[0].position.y == pytest.approx(min_y, abs=3.0)


def test_start_point_hint_moves_point_one() -> None:
    auto = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8))
    hinted = run_dot_to_dot(
        _MAIN_WITH_DETAIL,
        (200, 200),
        DotToDotConfig(total_points=8, start_point_hint=(180.0, 180.0)),
    )
    auto_first = auto.sequences[0].points[0].position
    hinted_first = hinted.sequences[0].points[0].position
    # The hinted start is near the bottom-right corner, not the top.
    assert hinted_first.y > auto_first.y
    assert hinted_first.x == pytest.approx(180.0, abs=6.0)


def test_direction_defaults_to_clockwise_and_can_invert() -> None:
    forward = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8))
    reverse = run_dot_to_dot(
        _MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8, direction="reverse")
    )
    assert forward.silhouette.signed_area() > 0.0  # clockwise on screen
    fwd = [(p.position.x, p.position.y) for p in forward.sequences[0].points]
    rev = [(p.position.x, p.position.y) for p in reverse.sequences[0].points]
    assert fwd[0] == pytest.approx(rev[0], abs=1e-6)
    assert fwd[1] == pytest.approx(rev[-1], abs=1e-6)


def test_no_duplicate_points_on_closed_silhouette() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=16))
    positions = [
        (round(dot.position.x, 4), round(dot.position.y, 4))
        for dot in result.sequences[0].points
    ]
    assert len(set(positions)) == len(positions)


def test_spacing_is_roughly_uniform() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=16))
    points = [dot.position for dot in result.sequences[0].points]
    spacings = [points[i].distance_to(points[(i + 1) % len(points)]) for i in range(len(points))]
    expected = result.silhouette.total_length / 16.0
    for spacing in spacings:
        assert spacing == pytest.approx(expected, rel=0.35)


def test_excessive_point_count_warns_but_generates() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=400))
    assert result.point_count == 400
    assert any("quantidade" in warning.lower() or "pontos" in warning.lower() for warning in result.warnings)


def test_selected_segments_mode_raises_without_segments() -> None:
    config = DotToDotConfig(generation_mode="selected_segments", total_points=8)
    with pytest.raises(DotToDotError):
        run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), config)


# --- Isolation from the other four modes --------------------------------------------


def _shape_image() -> np.ndarray:
    import cv2

    image = np.full((120, 120, 3), 255, np.uint8)
    cv2.rectangle(image, (25, 25), (95, 95), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 50), (70, 70), (255, 255, 255), -1)
    return image


@pytest.mark.parametrize("mode", ["classic", "visual", "contours"])
def test_other_modes_produce_no_dot_to_dot_artifacts(mode: str) -> None:
    image = _shape_image()
    result = build_tikz_from_image(image, ProcessingSettings(vectorization_mode=mode), TikzOptions())
    assert "dotToDotPoint" not in result.tikz_code
    assert result.effective_mode == mode


def test_visual_output_is_deterministic_and_unaffected_by_dot_to_dot() -> None:
    image = _shape_image()
    options = TikzOptions()
    first = build_tikz_from_image(image, ProcessingSettings(vectorization_mode="visual"), options)
    # Running dot-to-dot in between must not change the Visual result.
    build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="dot_to_dot"),
        options,
        dot_to_dot_config=DotToDotConfig(total_points=10),
    )
    second = build_tikz_from_image(image, ProcessingSettings(vectorization_mode="visual"), options)
    assert first.tikz_code == second.tikz_code


def test_dot_to_dot_mode_emits_native_points_via_pipeline() -> None:
    image = _shape_image()
    result = build_tikz_from_image(
        image,
        ProcessingSettings(vectorization_mode="dot_to_dot"),
        TikzOptions(),
        dot_to_dot_config=DotToDotConfig(total_points=10),
    )
    assert result.effective_mode == "dot_to_dot"
    assert result.dot_to_dot_result is not None
    assert result.dot_to_dot_result.point_count == 10
    assert "\\fill[dotToDotPoint]" in result.tikz_code
