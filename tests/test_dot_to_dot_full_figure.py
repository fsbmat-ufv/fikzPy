"""Stage 2 tests: full-figure dot-to-dot generation and mode isolation."""

from __future__ import annotations

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

_TWO_CONTOURS = _svg(
    '<path d="M 10 10 L 90 10 L 90 90 L 10 90 Z"/>'
    '<path d="M 120 120 L 160 120 L 160 160 L 120 160 Z"/>'
)


# --- Main contour and counts --------------------------------------------------------


def test_main_contour_gets_exact_point_count() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=12))
    assert len(result.sequences) == 1
    assert len(result.sequences[0].points) == 12
    assert result.point_count == 12


def test_total_points_are_exactly_requested_with_internal_contours() -> None:
    config = DotToDotConfig(total_points=30, include_internal_contours=True, min_points_per_path=3)
    result = run_dot_to_dot(_TWO_CONTOURS, (200, 200), config)
    assert result.point_count == 30
    assert sum(len(sequence.points) for sequence in result.sequences) == 30


def test_internal_contours_distribute_proportionally_to_length() -> None:
    config = DotToDotConfig(total_points=30, include_internal_contours=True, min_points_per_path=3)
    result = run_dot_to_dot(_TWO_CONTOURS, (200, 200), config)
    assert len(result.sequences) == 2
    # The longer (larger) contour must receive more points than the shorter one.
    counts = [len(sequence.points) for sequence in result.sequences]
    assert counts[0] > counts[1]
    assert all(count >= 3 for count in counts)


# --- Internal detail preservation ---------------------------------------------------


def test_internal_details_are_preserved_by_default() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8))
    # The inner square detail is kept in the remaining drawing.
    assert len(result.remaining_subpaths) == 1


def test_internal_details_can_be_dropped() -> None:
    config = DotToDotConfig(total_points=8, preserve_internal_details=False)
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), config)
    assert result.remaining_subpaths == ()


# --- Removed original outline -------------------------------------------------------


def test_original_main_outline_is_replaced_not_drawn() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8))
    main = result.dotted_refs[0]
    # The dotted contour is not part of the remaining drawing.
    dotted_ids = {(ref.path_id, ref.subpath_id) for ref in result.dotted_refs}
    assert (main.path_id, main.subpath_id) not in {
        (path.path_id, subpath_id)
        for path in result.paths
        for subpath_id, subpath in enumerate(path.subpaths)
        if subpath in result.remaining_subpaths
    }
    # Exactly one remaining draw command (the internal detail), plus the points.
    draw_count = result.tikz_code.count("\\draw[draw=")
    assert draw_count == len(result.remaining_subpaths) == 1
    assert "\\fill[dotToDotPoint]" in result.tikz_code


# --- Valid TikZ / numbering ---------------------------------------------------------


def test_generated_tikz_is_structurally_valid() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=6))
    code = result.tikz_code
    assert code.count("\\begin{tikzpicture}") == 1
    assert code.count("\\end{tikzpicture}") == 1
    assert code.count("\\begin{scope}") == code.count("\\end{scope}") == 1
    assert "dotToDotPoint/.style" in code
    assert "dotToDotLabel/.style" in code
    assert code.count("\\fill[dotToDotPoint]") == 6
    assert code.count("\\node[dotToDotLabel") == 6


def test_numbering_uses_start_number_for_main_sequence() -> None:
    result = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=5, start_number=10))
    labels = [dot.label for dot in result.sequences[0].points]
    assert labels == ["10", "11", "12", "13", "14"]


def test_secondary_sequences_use_letter_prefixes() -> None:
    config = DotToDotConfig(total_points=20, include_internal_contours=True, min_points_per_path=3)
    result = run_dot_to_dot(_TWO_CONTOURS, (200, 200), config)
    assert result.sequences[0].prefix == ""
    assert result.sequences[1].prefix == "A"
    assert result.sequences[1].points[0].label == "A1"
    assert result.sequences[1].points[-1].label.startswith("A")


def test_reverse_direction_changes_numbering_order() -> None:
    forward = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8, direction="forward"))
    reverse = run_dot_to_dot(_MAIN_WITH_DETAIL, (200, 200), DotToDotConfig(total_points=8, direction="reverse"))
    fwd_positions = [(p.position.x, p.position.y) for p in forward.sequences[0].points]
    rev_positions = [(p.position.x, p.position.y) for p in reverse.sequences[0].points]
    # Same start point, opposite traversal order.
    assert fwd_positions[0] == pytest.approx(rev_positions[0])
    assert fwd_positions[1] != pytest.approx(rev_positions[1])
    assert fwd_positions[1] == pytest.approx(rev_positions[-1])


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
