from __future__ import annotations

import numpy as np

from fikzpy.core.bezier_fit import can_use_bezier, catmull_rom_to_bezier, evaluate_cubic_bezier
from fikzpy.core.bezier_fit import fit_cubic_beziers
from fikzpy.core.vector_objects import BezierCurve, Line, Point, Polyline


def test_catmull_rom_to_bezier_returns_segments() -> None:
    points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)

    segments = catmull_rom_to_bezier(points)

    assert len(segments) == 3
    assert np.allclose(segments[0].start, points[0])
    assert np.allclose(segments[-1].end, points[-1])


def test_closed_bezier_has_one_segment_per_point() -> None:
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    segments = catmull_rom_to_bezier(points, closed=True)

    assert len(segments) == len(points)
    assert np.allclose(segments[-1].end, points[0])


def test_can_use_bezier_requires_enough_points() -> None:
    assert not can_use_bezier(np.array([[0, 0], [1, 1], [2, 0]], dtype=float))
    assert can_use_bezier(np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float))


def test_fit_cubic_beziers_recognizes_straight_line() -> None:
    points = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])

    fitted = fit_cubic_beziers(points, error_tolerance=0.05)

    assert len(fitted) == 1
    assert isinstance(fitted[0], Line)


def test_fit_cubic_beziers_approximates_arc_with_few_curves() -> None:
    angles = np.linspace(0, np.pi / 2, 24)
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    fitted = fit_cubic_beziers(points, error_tolerance=0.01, straightness_tolerance=0.001)

    assert 1 <= _count_type(fitted, BezierCurve) <= 3
    assert _count_type(fitted, Polyline) == 0


def test_fit_cubic_beziers_s_curve_splits_when_needed() -> None:
    xs = np.linspace(0, 2 * np.pi, 48)
    points = np.column_stack([xs, np.sin(xs)])

    fitted = fit_cubic_beziers(points, error_tolerance=0.02, straightness_tolerance=0.001)

    assert _count_type(fitted, BezierCurve) > 1


def test_fit_cubic_beziers_higher_tolerance_reduces_curve_count() -> None:
    xs = np.linspace(0, 2 * np.pi, 64)
    points = np.column_stack([xs, np.sin(xs)])

    strict = fit_cubic_beziers(points, error_tolerance=0.01, straightness_tolerance=0.001)
    loose = fit_cubic_beziers(points, error_tolerance=0.2, straightness_tolerance=0.001)

    assert _count_geometric(strict) >= _count_geometric(loose)


def test_fit_cubic_beziers_strict_tolerance_improves_fidelity() -> None:
    angles = np.linspace(0, np.pi / 2, 32)
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    strict = fit_cubic_beziers(points, error_tolerance=0.005, straightness_tolerance=0.001)
    loose = fit_cubic_beziers(points, error_tolerance=0.2, straightness_tolerance=0.001)

    assert _max_sample_error(strict, points) <= _max_sample_error(loose, points) + 1e-9


def test_fit_cubic_beziers_tiny_curve_does_not_generate_degenerate_bezier() -> None:
    points = np.array([[0, 0], [0.01, 0.002], [0.02, 0.0]], dtype=float)

    fitted = fit_cubic_beziers(points, error_tolerance=0.001, min_bezier_length=0.1)

    assert _count_type(fitted, BezierCurve) == 0


def test_fit_cubic_beziers_preserves_first_and_last_points() -> None:
    xs = np.linspace(0, 1, 12)
    points = np.column_stack([xs, xs**2])

    fitted = fit_cubic_beziers(points, error_tolerance=0.01, straightness_tolerance=0.001)

    assert _start_point(fitted).distance_to(Point(float(points[0, 0]), float(points[0, 1]))) < 1e-9
    assert _end_point(fitted).distance_to(Point(float(points[-1, 0]), float(points[-1, 1]))) < 1e-9


def test_fit_cubic_beziers_closed_contour_returns_closed_sequence() -> None:
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    fitted = fit_cubic_beziers(points, error_tolerance=0.05, closed=True, straightness_tolerance=0.001)

    assert _start_point(fitted).distance_to(_end_point(fitted)) < 1e-9


def _count_type(items, item_type: type) -> int:
    return sum(isinstance(item, item_type) for item in items)


def _count_geometric(items) -> int:
    return len(items)


def _start_point(items) -> Point:
    first = items[0]
    if isinstance(first, Line | BezierCurve):
        return first.start
    if isinstance(first, Polyline):
        return first.points[0]
    raise AssertionError(f"Unsupported item: {first!r}")


def _end_point(items) -> Point:
    last = items[-1]
    if isinstance(last, Line | BezierCurve):
        return last.end
    if isinstance(last, Polyline):
        return last.points[-1]
    raise AssertionError(f"Unsupported item: {last!r}")


def _max_sample_error(items, points: np.ndarray) -> float:
    samples: list[np.ndarray] = []
    for item in items:
        if isinstance(item, BezierCurve):
            samples.append(evaluate_cubic_bezier(item, np.linspace(0, 1, 25)))
        elif isinstance(item, Line):
            start = np.array(item.start.as_tuple())
            end = np.array(item.end.as_tuple())
            t = np.linspace(0, 1, 25)[:, None]
            samples.append(start * (1 - t) + end * t)
        elif isinstance(item, Polyline):
            samples.append(np.array([point.as_tuple() for point in item.points], dtype=float))
    sample_array = np.vstack(samples)
    distances = [float(np.linalg.norm(sample_array - point, axis=1).min()) for point in points]
    return max(distances)
