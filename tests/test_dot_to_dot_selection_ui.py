"""Tests for the main-preview selection layer (coordinates and overlay)."""

from __future__ import annotations

import numpy as np
import pytest

from fikzpy.core.dot_to_dot.segment_selection import InteractiveSelectionController
from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths
from fikzpy.gui.dot_to_dot_selection import render_selection_overlay, widget_to_image_coords


_SQUARE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<path d="M 10 10 L 90 10 L 90 90 L 10 90 Z"/></svg>'
)


# --- widget -> image coordinate mapping ----------------------------------------------


def test_identity_mapping_when_pixmap_fills_label() -> None:
    mapped = widget_to_image_coords(50.0, 40.0, (200, 100), (200, 100), (200, 100))
    assert mapped == pytest.approx((50.0, 40.0))


def test_mapping_scales_back_to_original_resolution() -> None:
    # 400x200 image displayed at 200x100 (half size), no letterbox.
    mapped = widget_to_image_coords(50.0, 40.0, (200, 100), (200, 100), (400, 200))
    assert mapped == pytest.approx((100.0, 80.0))


def test_mapping_compensates_horizontal_letterbox() -> None:
    # Square pixmap 100x100 centered in a 300x100 label: 100 px margins.
    mapped = widget_to_image_coords(150.0, 50.0, (300, 100), (100, 100), (200, 200))
    assert mapped == pytest.approx((100.0, 100.0))


def test_mapping_compensates_vertical_letterbox() -> None:
    # Pixmap 100x100 centered in a 100x300 label: 100 px top margin.
    mapped = widget_to_image_coords(50.0, 150.0, (100, 300), (100, 100), (200, 200))
    assert mapped == pytest.approx((100.0, 100.0))


def test_click_outside_pixmap_returns_none() -> None:
    assert widget_to_image_coords(10.0, 50.0, (300, 100), (100, 100), (200, 200)) is None
    assert widget_to_image_coords(290.0, 50.0, (300, 100), (100, 100), (200, 200)) is None


def test_mapping_with_zoom_like_upscale() -> None:
    # Pixmap larger than the image (zoom): 100x100 image shown at 400x400.
    mapped = widget_to_image_coords(200.0, 300.0, (400, 400), (400, 400), (100, 100))
    assert mapped == pytest.approx((50.0, 75.0))


def test_degenerate_sizes_return_none() -> None:
    assert widget_to_image_coords(0.0, 0.0, (100, 100), (0, 0), (100, 100)) is None


# --- overlay rendering ----------------------------------------------------------------


def _controller_with_square() -> InteractiveSelectionController:
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(_SQUARE_SVG))
    return controller


def test_overlay_draws_paths_over_faded_base() -> None:
    base = np.full((100, 100, 3), 255, np.uint8)
    controller = _controller_with_square()
    overlay = render_selection_overlay(base, controller)
    assert overlay.shape == base.shape
    # The traced square must be visible (non-white pixels along the top edge).
    assert (overlay[10, 10:90] < 250).any()


def test_overlay_marks_pending_selection_and_markers() -> None:
    base = np.full((100, 100, 3), 255, np.uint8)
    controller = _controller_with_square()
    controller.set_point_count(4)
    plain = render_selection_overlay(base, controller)
    controller.click(50.0, 10.0)
    with_start = render_selection_overlay(base, controller)
    controller.click(90.0, 50.0)
    with_arc = render_selection_overlay(base, controller)
    # Each interaction stage adds visible pixels.
    assert (plain != with_start).any()
    assert (with_start != with_arc).any()
    # The pending arc uses a blue-ish overlay color (BGR: high blue channel).
    blue_mask = (with_arc[:, :, 0].astype(int) - with_arc[:, :, 2].astype(int)) > 80
    assert blue_mask.any()


def test_overlay_confirmed_segments_use_distinct_color() -> None:
    base = np.full((100, 100, 3), 255, np.uint8)
    controller = _controller_with_square()
    controller.set_point_count(4)
    controller.click(50.0, 10.0)
    controller.click(90.0, 50.0)
    controller.confirm()
    overlay = render_selection_overlay(base, controller)
    # Confirmed arcs are orange-ish (BGR: high red channel vs blue).
    orange_mask = (overlay[:, :, 2].astype(int) - overlay[:, :, 0].astype(int)) > 80
    assert orange_mask.any()


def test_overlay_colors_never_reach_tikz_output() -> None:
    from fikzpy.core.dot_to_dot import DotToDotConfig, run_dot_to_dot

    controller = _controller_with_square()
    controller.set_point_count(4)
    controller.click(50.0, 10.0)
    controller.click(90.0, 50.0)
    controller.confirm()
    config = DotToDotConfig(
        generation_mode="selected_segments",
        selected_segments=controller.model.segments,
    )
    result = run_dot_to_dot(_SQUARE_SVG, (100, 100), config)
    assert "orange" not in result.tikz_code
    assert "blue" not in result.tikz_code
    assert "green" not in result.tikz_code
    assert "red" not in result.tikz_code
