"""Unit tests for the independent dot-to-dot geometric core (Stage 1)."""

from __future__ import annotations

from math import hypot

import pytest

from fikzpy.core.dot_to_dot.models import (
    CubicSegment,
    LineSegment,
    QuadraticSegment,
    SubPath,
    Vec2,
)
from fikzpy.core.dot_to_dot.path_sampler import sample_positions, sample_subpath
from fikzpy.core.dot_to_dot.silhouette import extract_silhouette
from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths
from fikzpy.core.dot_to_dot.tikz_exporter import CoordinateTransform


def _svg(path_d: str, *, width: float = 100.0, height: float = 100.0, extra: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<path d="{path_d}"/>{extra}</svg>'
    )


def _spacings(points) -> list[float]:
    return [points[i].position.distance_to(points[i + 1].position) for i in range(len(points) - 1)]


# --- Segment / arc-length primitives ------------------------------------------------


def test_straight_line_arc_length_matches_distance() -> None:
    segment = LineSegment(Vec2(0.0, 0.0), Vec2(30.0, 40.0))
    subpath = SubPath([segment], closed=False)
    assert subpath.total_length == pytest.approx(50.0, rel=1e-6)
    midpoint = subpath.point_at_length(25.0)
    assert (midpoint.x, midpoint.y) == pytest.approx((15.0, 20.0), abs=1e-6)


def test_cubic_bezier_arc_length_positive_and_monotonic() -> None:
    segment = CubicSegment(Vec2(0.0, 0.0), Vec2(0.0, 40.0), Vec2(40.0, 40.0), Vec2(40.0, 0.0))
    subpath = SubPath([segment], closed=False)
    length = subpath.total_length
    assert length > 40.0  # longer than the straight chord
    quarter = subpath.point_at_length(length * 0.25)
    half = subpath.point_at_length(length * 0.5)
    assert quarter.x < half.x  # progresses along the curve


def test_quadratic_split_preserves_endpoints() -> None:
    segment = QuadraticSegment(Vec2(0.0, 0.0), Vec2(10.0, 20.0), Vec2(20.0, 0.0))
    left, right = segment.split(0.5)
    assert (left.start.x, left.start.y) == pytest.approx((0.0, 0.0))
    assert (right.end.x, right.end.y) == pytest.approx((20.0, 0.0))
    assert (left.end.x, left.end.y) == pytest.approx((right.start.x, right.start.y))


def test_cubic_subsegment_points_lie_on_original_curve() -> None:
    segment = CubicSegment(Vec2(0.0, 0.0), Vec2(10.0, 30.0), Vec2(30.0, 30.0), Vec2(40.0, 0.0))
    sub = segment.subsegment(0.25, 0.75)
    # Endpoints of the sub-arc equal the original curve at t=0.25 and t=0.75.
    assert (sub.start.x, sub.start.y) == pytest.approx((segment.point(0.25).x, segment.point(0.25).y), abs=1e-6)
    assert (sub.end.x, sub.end.y) == pytest.approx((segment.point(0.75).x, segment.point(0.75).y), abs=1e-6)


# --- SVG loading --------------------------------------------------------------------


def test_load_open_path_is_not_closed() -> None:
    paths = load_svg_paths(_svg("M 0 0 L 10 0 L 10 10"))
    assert len(paths) == 1
    assert len(paths[0].subpaths) == 1
    assert paths[0].subpaths[0].closed is False


def test_load_closed_path_is_closed() -> None:
    paths = load_svg_paths(_svg("M 0 0 L 10 0 L 10 10 L 0 10 Z"))
    subpath = paths[0].subpaths[0]
    assert subpath.closed is True
    assert subpath.total_length == pytest.approx(40.0, rel=1e-6)


def test_load_multiple_subpaths() -> None:
    paths = load_svg_paths(_svg("M 0 0 L 10 0 L 10 10 Z M 20 20 L 30 20 L 30 30 Z"))
    assert len(paths) == 1
    assert len(paths[0].subpaths) == 2
    assert all(subpath.closed for subpath in paths[0].subpaths)


def test_load_cubic_command() -> None:
    paths = load_svg_paths(_svg("M 0 0 C 0 40 40 40 40 0"))
    segments = paths[0].subpaths[0].segments
    assert len(segments) == 1
    assert isinstance(segments[0], CubicSegment)


def test_transform_translate_is_applied() -> None:
    plain = load_svg_paths(_svg("M 0 0 L 10 0 L 10 10 Z"))
    translated = load_svg_paths(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<g transform="translate(5,7)"><path d="M 0 0 L 10 0 L 10 10 Z"/></g></svg>'
    )
    plain_start = plain[0].subpaths[0].start
    translated_start = translated[0].subpaths[0].start
    assert (translated_start.x, translated_start.y) == pytest.approx((plain_start.x + 5.0, plain_start.y + 7.0))
    # Arc length is translation-invariant.
    assert translated[0].subpaths[0].total_length == pytest.approx(plain[0].subpaths[0].total_length)


def test_transform_scale_scales_length() -> None:
    scaled = load_svg_paths(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M 0 0 L 10 0 L 10 10 L 0 10 Z" transform="scale(2)"/></svg>'
    )
    # 10x10 square perimeter (40) scaled by 2 -> 80.
    assert scaled[0].subpaths[0].total_length == pytest.approx(80.0, rel=1e-6)


# --- Sampling -----------------------------------------------------------------------


def test_sample_positions_closed_has_no_duplicate_endpoint() -> None:
    positions = sample_positions(40.0, 4, closed=True)
    assert positions == pytest.approx([0.0, 10.0, 20.0, 30.0])
    # s = L (== 40) is never emitted for a closed path.
    assert 40.0 not in positions


def test_sample_positions_open_includes_both_endpoints() -> None:
    positions = sample_positions(30.0, 4, closed=False)
    assert positions == pytest.approx([0.0, 10.0, 20.0, 30.0])


def test_sample_closed_square_exact_count_and_no_duplicate() -> None:
    paths = load_svg_paths(_svg("M 0 0 L 40 0 L 40 40 L 0 40 Z"))
    subpath = paths[0].subpaths[0]
    points = sample_subpath(subpath, 8)
    assert len(points) == 8
    positions = [(round(p.position.x, 6), round(p.position.y, 6)) for p in points]
    assert len(set(positions)) == 8  # first != last, no closure duplicate


def test_sample_uniform_spacing_on_closed_circle_like_path() -> None:
    # A closed square sampled densely should have near-uniform arc spacing.
    paths = load_svg_paths(_svg("M 0 0 L 40 0 L 40 40 L 0 40 Z"))
    subpath = paths[0].subpaths[0]
    points = sample_subpath(subpath, 16)
    spacings = _spacings(points)
    expected = subpath.total_length / 16.0
    for spacing in spacings:
        assert spacing == pytest.approx(expected, abs=expected * 0.15)


def test_sample_uniform_spacing_on_cubic() -> None:
    paths = load_svg_paths(_svg("M 0 0 C 0 60 60 60 60 0"))
    subpath = paths[0].subpaths[0]
    points = sample_subpath(subpath, 12)
    spacings = _spacings(points)
    expected = subpath.total_length / 11.0
    for spacing in spacings:
        assert spacing == pytest.approx(expected, abs=expected * 0.2)


# --- Coordinate conversion ----------------------------------------------------------


def test_coordinate_transform_flips_y_and_scales() -> None:
    transform = CoordinateTransform.from_image(image_width=100.0, image_height=80.0, width_units=10.0)
    # scale = 10/100 = 0.1; y flips about the image height.
    assert transform.to_tikz(Vec2(0.0, 0.0)) == pytest.approx((0.0, 8.0))
    assert transform.to_tikz(Vec2(100.0, 80.0)) == pytest.approx((10.0, 0.0))
    assert transform.to_tikz(Vec2(50.0, 40.0)) == pytest.approx((5.0, 4.0))


# --- Silhouette extraction ----------------------------------------------------------


def test_silhouette_excludes_full_canvas_frame() -> None:
    # Big frame rectangle covering the whole canvas plus a smaller shape.
    svg = _svg(
        "M 0 0 L 100 0 L 100 100 L 0 100 Z",
        extra='<path d="M 30 30 L 60 30 L 60 60 L 30 60 Z"/>',
    )
    result = extract_silhouette(load_svg_paths(svg), 100.0, 100.0)
    min_x, min_y, max_x, max_y = result.outline.bounding_box()
    # The silhouette follows the inner square, not the page frame.
    assert (max_x - min_x) < 60.0 and (max_y - min_y) < 60.0


def test_silhouette_merges_overlapping_subpaths_into_one_outline() -> None:
    # Two overlapping squares must yield one merged external outline.
    svg = _svg(
        "M 10 10 L 50 10 L 50 50 L 10 50 Z",
        extra='<path d="M 40 40 L 80 40 L 80 80 L 40 80 Z"/>',
    )
    result = extract_silhouette(load_svg_paths(svg), 100.0, 100.0)
    min_x, min_y, max_x, max_y = result.outline.bounding_box()
    assert max_x - min_x == pytest.approx(70.0, abs=4.0)
    assert max_y - min_y == pytest.approx(70.0, abs=4.0)
    assert result.outline.closed is True


def test_silhouette_ignores_white_cutouts() -> None:
    # A ring: outer square plus inner cutout; only the external contour counts.
    svg = _svg("M 10 10 L 90 10 L 90 90 L 10 90 Z M 30 30 L 70 30 L 70 70 L 30 70 Z")
    result = extract_silhouette(load_svg_paths(svg), 100.0, 100.0)
    min_x, min_y, max_x, max_y = result.outline.bounding_box()
    assert max_x - min_x == pytest.approx(80.0, abs=4.0)
    # Interior details are classified separately, never merged into the outline.
    assert result.outline.area() == pytest.approx(6400.0, rel=0.1)


def test_silhouette_is_clockwise_by_default() -> None:
    svg = _svg("M 20 20 L 80 20 L 80 80 L 20 80 Z")
    result = extract_silhouette(load_svg_paths(svg), 100.0, 100.0)
    # Positive shoelace area in image coordinates = visually clockwise.
    assert result.outline.signed_area() > 0.0


def test_silhouette_internal_details_exclude_boundary_ribbons() -> None:
    # Outer ring (thin boundary ribbon, ~1% stroke) plus an inner detail square.
    svg = _svg(
        "M 10 10 L 90 10 L 90 90 L 10 90 Z M 11 11 L 89 11 L 89 89 L 11 89 Z",
        extra='<path d="M 45 45 L 55 45 L 55 55 L 45 55 Z"/>',
    )
    result = extract_silhouette(load_svg_paths(svg), 100.0, 100.0, collect_internal_details=True)
    boxes = [detail.bounding_box() for group in result.internal_details for detail in group]
    # The inner detail square is kept; the outline ribbons are excluded.
    assert any((box[2] - box[0]) < 20.0 for box in boxes)
    assert all((box[2] - box[0]) < 60.0 for box in boxes)
