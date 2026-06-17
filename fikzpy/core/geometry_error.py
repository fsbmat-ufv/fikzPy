"""Reusable geometric error metrics for primitive fitting."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from collections.abc import Sequence
from typing import Any

import numpy as np

from fikzpy.core.semantic_geometry import Point2D


_EPSILON = 1e-12


@dataclass(frozen=True)
class GeometryBounds:
    """Axis-aligned bounds and scale diagnostics for a point set."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width: float
    height: float
    diagonal: float

    def to_dict(self) -> dict[str, float]:
        """Return serializable bounds diagnostics."""
        return dict(self.__dict__)


def points_to_array(points: Sequence[Point2D] | np.ndarray) -> np.ndarray:
    """Copy finite two-dimensional points into an ``(n, 2)`` float array."""
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=np.float64).copy()
    else:
        array = np.array([_point_pair(point) for point in points], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite.")
    return array


def array_to_points(points: np.ndarray, *, precision: int | None = None) -> tuple[Point2D, ...]:
    """Convert an ``(n, 2)`` array to immutable semantic points."""
    array = points_to_array(points)
    if precision is not None:
        array = np.round(array, int(precision))
    return tuple(Point2D(float(x), float(y)) for x, y in array)


def geometry_bounds(points: Sequence[Point2D] | np.ndarray) -> GeometryBounds:
    """Return a deterministic bounding box and diagonal scale."""
    array = points_to_array(points)
    if len(array) == 0:
        raise ValueError("points must not be empty.")
    min_x = float(np.min(array[:, 0]))
    min_y = float(np.min(array[:, 1]))
    max_x = float(np.max(array[:, 0]))
    max_y = float(np.max(array[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    return GeometryBounds(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        width=width,
        height=height,
        diagonal=hypot(width, height),
    )


def geometric_scale(points: Sequence[Point2D] | np.ndarray) -> float:
    """Return the bounding-box diagonal used to normalize fitting errors."""
    bounds = geometry_bounds(points)
    return max(bounds.diagonal, 1.0)


def path_length(points: Sequence[Point2D] | np.ndarray, *, closed: bool = False) -> float:
    """Return the total polyline length."""
    array = points_to_array(points)
    if len(array) < 2:
        return 0.0
    diffs = np.diff(array, axis=0)
    total = float(np.linalg.norm(diffs, axis=1).sum())
    if closed:
        total += float(np.linalg.norm(array[0] - array[-1]))
    return total


def closure_error(points: Sequence[Point2D] | np.ndarray) -> float:
    """Return the endpoint gap of an ordered path."""
    array = points_to_array(points)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(array[0] - array[-1]))


def rms_error(errors: Sequence[float] | np.ndarray) -> float:
    """Return root-mean-square error for scalar distances."""
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(sqrt(float(np.mean(values * values))))


def max_error(errors: Sequence[float] | np.ndarray) -> float:
    """Return maximum absolute error for scalar distances."""
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def normalize_error(error: float, scale: float, *, enabled: bool = True) -> float:
    """Normalize an error by geometric scale when requested."""
    number = float(error)
    if not isfinite(number):
        return float("inf")
    if not enabled:
        return max(0.0, number)
    return max(0.0, number) / max(float(scale), 1.0)


def point_distances_to_line(
    points: Sequence[Point2D] | np.ndarray,
    start: Point2D | np.ndarray,
    end: Point2D | np.ndarray,
) -> np.ndarray:
    """Return orthogonal distances from points to the infinite line."""
    array = points_to_array(points)
    start_array = np.asarray(_point_pair(start), dtype=np.float64)
    end_array = np.asarray(_point_pair(end), dtype=np.float64)
    direction = end_array - start_array
    length = float(np.linalg.norm(direction))
    if length <= _EPSILON:
        return np.linalg.norm(array - start_array, axis=1)
    offsets = array - start_array
    cross = direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
    return np.abs(cross) / length


def point_distances_to_segment(
    points: Sequence[Point2D] | np.ndarray,
    start: Point2D | np.ndarray,
    end: Point2D | np.ndarray,
) -> np.ndarray:
    """Return distances from points to a finite line segment."""
    array = points_to_array(points)
    start_array = np.asarray(_point_pair(start), dtype=np.float64)
    end_array = np.asarray(_point_pair(end), dtype=np.float64)
    direction = end_array - start_array
    denominator = float(np.dot(direction, direction))
    if denominator <= _EPSILON:
        return np.linalg.norm(array - start_array, axis=1)
    ratios = ((array - start_array) @ direction) / denominator
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = start_array + ratios[:, None] * direction
    return np.linalg.norm(array - projections, axis=1)


def point_distances_to_polyline(
    points: Sequence[Point2D] | np.ndarray,
    polyline: Sequence[Point2D] | np.ndarray,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Return nearest distances from points to a polyline."""
    array = points_to_array(points)
    line = points_to_array(polyline)
    if len(line) == 0:
        raise ValueError("polyline must not be empty.")
    if len(line) == 1:
        return np.linalg.norm(array - line[0], axis=1)
    segments = [(line[index], line[index + 1]) for index in range(len(line) - 1)]
    if closed:
        segments.append((line[-1], line[0]))
    distances = np.vstack([point_distances_to_segment(array, start, end) for start, end in segments])
    return np.min(distances, axis=0)


def circle_radial_errors(
    points: Sequence[Point2D] | np.ndarray,
    center: Point2D | np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return signed radial distance errors to a circle."""
    array = points_to_array(points)
    center_array = np.asarray(_point_pair(center), dtype=np.float64)
    distances = np.linalg.norm(array - center_array, axis=1)
    return distances - float(radius)


def ellipse_distance_errors(
    points: Sequence[Point2D] | np.ndarray,
    center: Point2D | np.ndarray,
    radius_x: float,
    radius_y: float,
    rotation: float,
) -> np.ndarray:
    """Return approximate geometric errors to a rotated ellipse.

    The metric maps each point into the ellipse frame, compares the normalized
    radial coordinate to one, and scales by the average semi-axis length.
    """
    array = points_to_array(points)
    center_array = np.asarray(_point_pair(center), dtype=np.float64)
    rx = float(radius_x)
    ry = float(radius_y)
    if rx <= 0.0 or ry <= 0.0:
        raise ValueError("ellipse radii must be positive.")
    angle = float(rotation)
    c = cos(angle)
    s = sin(angle)
    shifted = array - center_array
    local_x = c * shifted[:, 0] + s * shifted[:, 1]
    local_y = -s * shifted[:, 0] + c * shifted[:, 1]
    normalized_radius = np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
    return (normalized_radius - 1.0) * ((rx + ry) / 2.0)


def angular_coverage(
    points: Sequence[Point2D] | np.ndarray,
    center: Point2D | np.ndarray,
) -> float:
    """Return angular coverage as a fraction of a full turn."""
    array = points_to_array(points)
    if len(array) < 2:
        return 0.0
    center_array = np.asarray(_point_pair(center), dtype=np.float64)
    angles = np.mod(np.arctan2(array[:, 1] - center_array[1], array[:, 0] - center_array[0]), 2.0 * pi)
    angles = np.sort(angles)
    if len(angles) == 1:
        return 0.0
    gaps = np.diff(angles)
    wrap_gap = angles[0] + 2.0 * pi - angles[-1]
    largest_gap = max(float(np.max(gaps, initial=0.0)), float(wrap_gap))
    return max(0.0, min(1.0, (2.0 * pi - largest_gap) / (2.0 * pi)))


def signed_area(points: Sequence[Point2D] | np.ndarray) -> float:
    """Return the signed area of an ordered polygonal loop."""
    array = points_to_array(points)
    if len(array) < 3:
        return 0.0
    x = array[:, 0]
    y = array[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def traversal_direction(points: Sequence[Point2D] | np.ndarray) -> str:
    """Return clockwise, counterclockwise, or indeterminate traversal."""
    area = signed_area(points)
    if abs(area) <= _EPSILON:
        return "indeterminate"
    return "counterclockwise" if area > 0.0 else "clockwise"


def evaluate_cubic_bezier(
    control_points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate a cubic Bezier curve at one or more parameters."""
    p0, p1, p2, p3 = (np.asarray(point, dtype=np.float64) for point in control_points)
    u = np.asarray(parameters, dtype=np.float64)[:, None]
    return (
        ((1.0 - u) ** 3) * p0
        + 3.0 * u * ((1.0 - u) ** 2) * p1
        + 3.0 * (u**2) * (1.0 - u) * p2
        + (u**3) * p3
    )


def point_distances_to_cubic_bezier(
    points: Sequence[Point2D] | np.ndarray,
    control_points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    samples: int = 80,
) -> np.ndarray:
    """Return approximate nearest distances from points to a cubic Bezier."""
    array = points_to_array(points)
    sample_count = max(8, int(samples))
    curve = evaluate_cubic_bezier(control_points, np.linspace(0.0, 1.0, sample_count))
    distances = []
    for point in array:
        distances.append(float(np.min(np.linalg.norm(curve - point, axis=1))))
    return np.array(distances, dtype=np.float64)


def simplify_polyline_rdp(
    points: Sequence[Point2D] | np.ndarray,
    tolerance: float,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Simplify a polyline with Ramer-Douglas-Peucker."""
    array = points_to_array(points)
    if len(array) <= 2:
        return array.copy()
    tol = max(0.0, float(tolerance))
    if tol <= 0.0:
        return array.copy()
    if closed:
        working = np.vstack([array, array[0]])
        simplified = _rdp_open(working, tol)
        if len(simplified) > 1 and np.linalg.norm(simplified[0] - simplified[-1]) <= _EPSILON:
            simplified = simplified[:-1]
        if len(simplified) < 3:
            return array.copy()
        return simplified
    return _rdp_open(array, tol)


def discrete_turn_angles(points: Sequence[Point2D] | np.ndarray, *, closed: bool = False) -> np.ndarray:
    """Return interior turn angles in degrees for an ordered path."""
    array = points_to_array(points)
    if len(array) < 3:
        return np.array([], dtype=np.float64)
    angles: list[float] = []
    count = len(array)
    indices = range(count) if closed else range(1, count - 1)
    for index in indices:
        prev_point = array[(index - 1) % count]
        point = array[index]
        next_point = array[(index + 1) % count]
        first = point - prev_point
        second = next_point - point
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= _EPSILON or second_norm <= _EPSILON:
            angles.append(0.0)
            continue
        cosine = float(np.dot(first, second) / (first_norm * second_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angle = float(np.degrees(np.arccos(cosine)))
        angles.append(angle)
    return np.array(angles, dtype=np.float64)


def corner_indices(
    points: Sequence[Point2D] | np.ndarray,
    *,
    threshold_degrees: float,
    closed: bool = False,
) -> tuple[int, ...]:
    """Return deterministic indices whose turn angle exceeds a threshold."""
    array = points_to_array(points)
    if len(array) < 3:
        return ()
    threshold = max(0.0, float(threshold_degrees))
    angles = discrete_turn_angles(array, closed=closed)
    if closed:
        return tuple(index for index, angle in enumerate(angles) if angle >= threshold)
    return tuple(index + 1 for index, angle in enumerate(angles) if angle >= threshold)


def _rdp_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()
    distances = point_distances_to_segment(points, points[0], points[-1])
    split_index = int(np.argmax(distances))
    split_distance = float(distances[split_index])
    if split_distance <= tolerance:
        return np.vstack([points[0], points[-1]])
    left = _rdp_open(points[: split_index + 1], tolerance)
    right = _rdp_open(points[split_index:], tolerance)
    return np.vstack([left[:-1], right])


def _point_pair(point: Point2D | np.ndarray | Sequence[float] | Any) -> tuple[float, float]:
    if isinstance(point, Point2D):
        return point.x, point.y
    if isinstance(point, np.ndarray):
        values = point.tolist()
    else:
        values = point
    if not isinstance(values, Sequence) or len(values) != 2:
        raise TypeError("point must be Point2D or an x/y pair.")
    x = float(values[0])
    y = float(values[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("point coordinates must be finite.")
    return x, y


__all__ = [
    "GeometryBounds",
    "angular_coverage",
    "array_to_points",
    "circle_radial_errors",
    "closure_error",
    "corner_indices",
    "discrete_turn_angles",
    "ellipse_distance_errors",
    "evaluate_cubic_bezier",
    "geometric_scale",
    "geometry_bounds",
    "max_error",
    "normalize_error",
    "path_length",
    "point_distances_to_cubic_bezier",
    "point_distances_to_line",
    "point_distances_to_polyline",
    "point_distances_to_segment",
    "points_to_array",
    "rms_error",
    "signed_area",
    "simplify_polyline_rdp",
    "traversal_direction",
]
