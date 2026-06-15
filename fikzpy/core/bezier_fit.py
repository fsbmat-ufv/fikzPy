"""Small utilities for turning polylines into cubic Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fikzpy.core.vector_objects import BezierCurve, Line, Point, Polyline, VectorPrimitive


@dataclass(frozen=True)
class CubicBezier:
    """A cubic Bezier segment."""

    start: np.ndarray
    control1: np.ndarray
    control2: np.ndarray
    end: np.ndarray


def _point_at(points: np.ndarray, index: int, closed: bool) -> np.ndarray:
    if closed:
        return points[index % len(points)]
    return points[min(max(index, 0), len(points) - 1)]


def catmull_rom_to_bezier(
    points: np.ndarray,
    *,
    closed: bool = False,
    tension: float = 1.0,
) -> list[CubicBezier]:
    """Approximate a polyline with smooth cubic Bezier segments.

    The method is intentionally small and predictable: it converts each
    Catmull-Rom span into one cubic Bezier span. Douglas-Peucker
    simplification should be applied before this step.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Bezier input points must have shape (n, 2).")
    if len(pts) < 2:
        return []

    segment_count = len(pts) if closed and len(pts) > 2 else len(pts) - 1
    factor = float(tension) / 6.0
    segments: list[CubicBezier] = []

    for index in range(segment_count):
        p0 = _point_at(pts, index - 1, closed)
        p1 = _point_at(pts, index, closed)
        p2 = _point_at(pts, index + 1, closed)
        p3 = _point_at(pts, index + 2, closed)

        control1 = p1 + (p2 - p0) * factor
        control2 = p2 - (p3 - p1) * factor
        segments.append(CubicBezier(start=p1, control1=control1, control2=control2, end=p2))

    return segments


def can_use_bezier(points: np.ndarray, *, min_points: int = 4) -> bool:
    """Return True when a contour has enough points for smooth conversion."""
    return len(np.asarray(points)) >= min_points


def fit_cubic_beziers(
    points: Sequence[Point] | np.ndarray,
    error_tolerance: float,
    *,
    closed: bool = False,
    min_bezier_length: float = 0.08,
    min_points_for_bezier: int = 6,
    straightness_tolerance: float = 0.03,
    control_point_epsilon: float = 1e-4,
    simplify_tolerance: float = 0.0,
    max_depth: int = 16,
) -> tuple[VectorPrimitive, ...]:
    """Fit lines and cubic Bezier curves to an ordered point sequence.

    This is a compact recursive cubic fitting pass inspired by Schneider's
    Graphics Gems approach. It fits one cubic to many points, measures the
    maximum deviation, and recursively splits only when the error is too large.
    """
    pts = _as_points_array(points)
    if len(pts) < 2:
        return ()

    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    pts = _remove_repeated_points(pts)
    if simplify_tolerance > 0 and len(pts) > 2:
        pts = _douglas_peucker(pts, simplify_tolerance, closed=False)

    if len(pts) < 2:
        return ()
    if len(pts) == 2:
        return (_line_from_array(pts[0], pts[-1]),)

    fitted = _fit_recursive(
        pts,
        error_tolerance=max(float(error_tolerance), 0.0),
        min_bezier_length=max(float(min_bezier_length), 0.0),
        min_points_for_bezier=max(4, int(min_points_for_bezier)),
        straightness_tolerance=max(float(straightness_tolerance), 0.0),
        control_point_epsilon=max(float(control_point_epsilon), 0.0),
        max_depth=max(0, int(max_depth)),
    )
    return tuple(fitted)


def simplify_points_for_bezier_fit(
    points: Sequence[Point] | np.ndarray,
    tolerance: float,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Conservatively simplify points before fitting Bezier curves."""
    pts = _as_points_array(points)
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    pts = _remove_repeated_points(pts)
    if tolerance <= 0 or len(pts) <= 2:
        return pts
    return _douglas_peucker(pts, float(tolerance), closed=False)


def evaluate_cubic_bezier(curve: BezierCurve, parameters: np.ndarray) -> np.ndarray:
    """Evaluate a Bezier curve at one or more parameter values."""
    u = np.asarray(parameters, dtype=np.float64)
    p0 = _point_to_array(curve.start)
    p1 = _point_to_array(curve.control1)
    p2 = _point_to_array(curve.control2)
    p3 = _point_to_array(curve.end)
    return _bezier_values((p0, p1, p2, p3), u)


def _fit_recursive(
    points: np.ndarray,
    *,
    error_tolerance: float,
    min_bezier_length: float,
    min_points_for_bezier: int,
    straightness_tolerance: float,
    control_point_epsilon: float,
    max_depth: int,
) -> list[VectorPrimitive]:
    if len(points) < min_points_for_bezier or _path_length(points) < min_bezier_length:
        return [_fallback_small_path(points, straightness_tolerance)]

    line_error = _max_line_distance(points)
    if line_error <= straightness_tolerance:
        return [_line_from_array(points[0], points[-1])]

    tangent_start = _unit(points[1] - points[0])
    tangent_end = _unit(points[-2] - points[-1])
    curve = _generate_cubic(points, tangent_start, tangent_end)
    max_error, split_index = _max_bezier_error(points, curve)

    if max_error <= error_tolerance and _is_valid_curve(curve, control_point_epsilon, min_bezier_length):
        return [_bezier_from_arrays(*curve)]

    if max_depth == 0 or split_index <= 1 or split_index >= len(points) - 2:
        return [_fallback_small_path(points, straightness_tolerance)]

    center_tangent = _unit(points[split_index + 1] - points[split_index - 1])
    if np.linalg.norm(center_tangent) == 0:
        center_tangent = tangent_start

    left = _fit_recursive(
        points[: split_index + 1],
        error_tolerance=error_tolerance,
        min_bezier_length=min_bezier_length,
        min_points_for_bezier=min_points_for_bezier,
        straightness_tolerance=straightness_tolerance,
        control_point_epsilon=control_point_epsilon,
        max_depth=max_depth - 1,
    )
    right = _fit_recursive(
        points[split_index:],
        error_tolerance=error_tolerance,
        min_bezier_length=min_bezier_length,
        min_points_for_bezier=min_points_for_bezier,
        straightness_tolerance=straightness_tolerance,
        control_point_epsilon=control_point_epsilon,
        max_depth=max_depth - 1,
    )
    return left + right


def _generate_cubic(points: np.ndarray, tangent_start: np.ndarray, tangent_end: np.ndarray) -> tuple[np.ndarray, ...]:
    parameters = _chord_length_parameters(points)
    p0 = points[0]
    p3 = points[-1]
    c_matrix = np.zeros((2, 2), dtype=np.float64)
    x_vector = np.zeros(2, dtype=np.float64)

    for point, u in zip(points, parameters):
        b0, b1, b2, b3 = _bernstein(float(u))
        a1 = tangent_start * b1
        a2 = tangent_end * b2
        target = point - ((b0 + b1) * p0 + (b2 + b3) * p3)
        c_matrix[0, 0] += np.dot(a1, a1)
        c_matrix[0, 1] += np.dot(a1, a2)
        c_matrix[1, 0] += np.dot(a1, a2)
        c_matrix[1, 1] += np.dot(a2, a2)
        x_vector[0] += np.dot(a1, target)
        x_vector[1] += np.dot(a2, target)

    segment_length = float(np.linalg.norm(p3 - p0))
    fallback_alpha = segment_length / 3.0
    try:
        alpha_start, alpha_end = np.linalg.solve(c_matrix, x_vector)
    except np.linalg.LinAlgError:
        alpha_start = alpha_end = fallback_alpha

    if alpha_start <= 1e-6 or alpha_end <= 1e-6:
        alpha_start = alpha_end = fallback_alpha

    p1 = p0 + tangent_start * alpha_start
    p2 = p3 + tangent_end * alpha_end
    return p0, p1, p2, p3


def _max_bezier_error(points: np.ndarray, curve: tuple[np.ndarray, ...]) -> tuple[float, int]:
    parameters = _chord_length_parameters(points)
    values = _bezier_values(curve, parameters)
    errors = np.linalg.norm(values - points, axis=1)
    if len(errors) <= 2:
        return 0.0, 1
    split_index = int(np.argmax(errors[1:-1]) + 1)
    return float(errors[split_index]), split_index


def _bezier_values(curve: tuple[np.ndarray, ...], parameters: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = curve
    u = np.asarray(parameters, dtype=np.float64)[:, None]
    return ((1 - u) ** 3) * p0 + 3 * u * ((1 - u) ** 2) * p1 + 3 * (u**2) * (1 - u) * p2 + (u**3) * p3


def _chord_length_parameters(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(distances.sum())
    if total == 0.0:
        return np.linspace(0.0, 1.0, len(points))
    return np.concatenate([[0.0], np.cumsum(distances) / total])


def _fallback_small_path(points: np.ndarray, straightness_tolerance: float) -> VectorPrimitive:
    if len(points) <= 2 or _max_line_distance(points) <= straightness_tolerance:
        return _line_from_array(points[0], points[-1])
    return Polyline(tuple(_point_from_array(point) for point in points), closed=False)


def _is_valid_curve(
    curve: tuple[np.ndarray, ...],
    control_point_epsilon: float,
    min_bezier_length: float,
) -> bool:
    p0, p1, p2, p3 = curve
    if np.linalg.norm(p3 - p0) < min_bezier_length:
        return False
    control_span = min(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p3))
    return bool(control_span > control_point_epsilon)


def _max_line_distance(points: np.ndarray) -> float:
    start = points[0]
    end = points[-1]
    baseline = end - start
    length = float(np.linalg.norm(baseline))
    if length == 0.0:
        return float(np.linalg.norm(points - start, axis=1).max(initial=0.0))
    offsets = points - start
    cross = baseline[0] * offsets[:, 1] - baseline[1] * offsets[:, 0]
    return float(np.abs(cross).max(initial=0.0) / length)


def _douglas_peucker(points: np.ndarray, tolerance: float, *, closed: bool) -> np.ndarray:
    if len(points) <= 2:
        return points
    distances = _line_distances(points, points[0], points[-1])
    index = int(np.argmax(distances))
    max_distance = float(distances[index])
    if max_distance <= tolerance:
        return np.vstack([points[0], points[-1]])
    left = _douglas_peucker(points[: index + 1], tolerance, closed=False)
    right = _douglas_peucker(points[index:], tolerance, closed=False)
    return np.vstack([left[:-1], right])


def _line_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    baseline = end - start
    length = float(np.linalg.norm(baseline))
    if length == 0.0:
        return np.linalg.norm(points - start, axis=1)
    offsets = points - start
    cross = baseline[0] * offsets[:, 1] - baseline[1] * offsets[:, 0]
    return np.abs(cross) / length


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _remove_repeated_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [True]
    keep.extend(bool(np.linalg.norm(points[index] - points[index - 1]) > 1e-9) for index in range(1, len(points)))
    return points[np.array(keep, dtype=bool)]


def _bernstein(u: float) -> tuple[float, float, float, float]:
    return (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _as_points_array(points: Sequence[Point] | np.ndarray) -> np.ndarray:
    if isinstance(points, np.ndarray):
        array = points.astype(np.float64, copy=False)
    else:
        array = np.array([point.as_tuple() if isinstance(point, Point) else point for point in points], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Bezier fitting input must have shape (n, 2).")
    return array


def _point_to_array(point: Point) -> np.ndarray:
    return np.array(point.as_tuple(), dtype=np.float64)


def _point_from_array(point: np.ndarray) -> Point:
    return Point(float(point[0]), float(point[1]))


def _line_from_array(start: np.ndarray, end: np.ndarray) -> Line:
    return Line(_point_from_array(start), _point_from_array(end))


def _bezier_from_arrays(
    start: np.ndarray,
    control1: np.ndarray,
    control2: np.ndarray,
    end: np.ndarray,
) -> BezierCurve:
    return BezierCurve(
        start=_point_from_array(start),
        control1=_point_from_array(control1),
        control2=_point_from_array(control2),
        end=_point_from_array(end),
    )
