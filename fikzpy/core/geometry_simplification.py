"""Deterministic geometry simplification helpers for semantic primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

import numpy as np

from fikzpy.core.geometry_error import array_to_points, corner_indices, max_error, normalize_error
from fikzpy.core.geometry_error import point_distances_to_line, point_distances_to_polyline
from fikzpy.core.geometry_error import points_to_array, rms_error, signed_area, simplify_polyline_rdp
from fikzpy.core.semantic_geometry import Point2D


_EPSILON = 1e-12


class GeometrySimplificationError(ValueError):
    """Raised when geometry cannot be simplified safely."""


@dataclass(frozen=True)
class DeduplicationResult:
    """Result of consecutive point deduplication."""

    points: tuple[Point2D, ...]
    removed_count: int
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return compact diagnostics."""
        return {
            "point_count": len(self.points),
            "removed_count": self.removed_count,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class SimplificationResult:
    """Result of conservative polyline simplification."""

    points: tuple[Point2D, ...]
    input_count: int
    output_count: int
    max_error: float
    rms_error: float
    normalized_error: float
    fixed_indices: tuple[int, ...]
    topology_preserved: bool
    orientation_preserved: bool
    changed: bool
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return compact diagnostics without full point arrays."""
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "max_error": self.max_error,
            "rms_error": self.rms_error,
            "normalized_error": self.normalized_error,
            "fixed_indices": list(self.fixed_indices),
            "topology_preserved": self.topology_preserved,
            "orientation_preserved": self.orientation_preserved,
            "changed": self.changed,
            "rejected_reason": self.rejected_reason,
        }


def point_distance(first: Point2D, second: Point2D) -> float:
    """Return Euclidean distance between two semantic points."""
    return hypot(second.x - first.x, second.y - first.y)


def points_close(first: Point2D, second: Point2D, tolerance: float) -> bool:
    """Return whether two points are within a non-negative tolerance."""
    return point_distance(first, second) <= max(0.0, float(tolerance))


def sanitize_points(points: Sequence[Point2D], *, minimum: int = 0) -> tuple[Point2D, ...]:
    """Copy finite ``Point2D`` values and reject invalid coordinates."""
    copied = tuple(_coerce_point(point) for point in points)
    if len(copied) < int(minimum):
        raise GeometrySimplificationError(f"geometry requires at least {minimum} points.")
    return copied


def deduplicate_consecutive_points(
    points: Sequence[Point2D],
    tolerance: float,
    *,
    closed: bool = False,
    preserve_terminal_points: bool = True,
) -> DeduplicationResult:
    """Remove only consecutive duplicate points within ``tolerance``.

    The first point is always preserved. For open paths the final point is
    preserved even when it duplicates the previous coordinate, matching the
    conservative rules for Issue 8 cleanup.
    """
    copied = sanitize_points(points)
    if len(copied) <= 1:
        return DeduplicationResult(copied, 0, False)
    limit = max(0.0, float(tolerance))
    deduplicated: list[Point2D] = [copied[0]]
    removed = 0
    last_index = len(copied) - 1
    for index, point in enumerate(copied[1:], start=1):
        is_terminal = index == last_index
        if (
            preserve_terminal_points
            and is_terminal
            and not closed
            and point_distance(point, deduplicated[-1]) <= limit
        ):
            deduplicated.append(point)
            continue
        if point_distance(point, deduplicated[-1]) <= limit:
            removed += 1
            continue
        deduplicated.append(point)
    if closed and len(deduplicated) > 3 and point_distance(deduplicated[0], deduplicated[-1]) <= limit:
        removed += 1
        deduplicated.pop()
    return DeduplicationResult(tuple(deduplicated), removed, removed > 0)


def simplify_polyline_preserving_features(
    points: Sequence[Point2D],
    tolerance: float,
    *,
    closed: bool = False,
    scale: float = 1.0,
    preserve_corners: bool = True,
    corner_angle_threshold: float = 35.0,
    fixed_indices: Iterable[int] = (),
    preserve_topology: bool = True,
    decimal_precision: int | None = None,
) -> SimplificationResult:
    """Simplify an ordered path while keeping endpoints, corners, and junctions.

    The implementation applies Ramer-Douglas-Peucker to sections separated by
    fixed feature points. The candidate is accepted only when it keeps enough
    points, does not create a new self-intersection, and preserves closed-path
    orientation.
    """
    copied = sanitize_points(points)
    if len(copied) <= (3 if closed else 2):
        return _unchanged(copied, closed=closed, scale=scale)
    absolute_tolerance = max(0.0, float(tolerance))
    if absolute_tolerance <= 0.0:
        return _unchanged(copied, closed=closed, scale=scale)

    fixed = _feature_indices(
        copied,
        closed=closed,
        preserve_corners=preserve_corners,
        threshold=corner_angle_threshold,
        extra=fixed_indices,
    )
    if fixed:
        simplified_array = _simplify_with_fixed_indices(copied, absolute_tolerance, closed=closed, fixed_indices=fixed)
    else:
        simplified_array = simplify_polyline_rdp(copied, absolute_tolerance, closed=closed)
    if decimal_precision is not None:
        simplified_array = np.round(simplified_array, int(decimal_precision))
    minimum = 3 if closed else 2
    if len(simplified_array) < minimum:
        return _rejected(copied, "simplification would make geometry degenerate", closed=closed, scale=scale, fixed=fixed)

    simplified = array_to_points(simplified_array)
    if len(simplified) >= len(copied):
        return _unchanged(copied, closed=closed, scale=scale, fixed=fixed)

    errors = point_distances_to_polyline(copied, simplified, closed=closed)
    local_max = max_error(errors)
    local_rms = rms_error(errors)
    normalized = normalize_error(local_max, scale)
    topology_preserved = True
    orientation_preserved = True
    if preserve_topology:
        topology_preserved = not _creates_self_intersection(copied, simplified, closed=closed)
    if closed:
        orientation_preserved = _orientation_preserved(copied, simplified)
    if not topology_preserved:
        return _rejected(
            copied,
            "simplification would introduce a self-intersection",
            closed=closed,
            scale=scale,
            fixed=fixed,
            max_error_value=local_max,
            rms_error_value=local_rms,
            normalized_error_value=normalized,
            topology_preserved=False,
            orientation_preserved=orientation_preserved,
        )
    if not orientation_preserved:
        return _rejected(
            copied,
            "simplification would change closed-path orientation",
            closed=closed,
            scale=scale,
            fixed=fixed,
            max_error_value=local_max,
            rms_error_value=local_rms,
            normalized_error_value=normalized,
            topology_preserved=topology_preserved,
            orientation_preserved=False,
        )
    return SimplificationResult(
        points=simplified,
        input_count=len(copied),
        output_count=len(simplified),
        max_error=local_max,
        rms_error=local_rms,
        normalized_error=normalized,
        fixed_indices=fixed,
        topology_preserved=True,
        orientation_preserved=True,
        changed=True,
    )


def line_like_endpoints(
    points: Sequence[Point2D],
    tolerance: float,
    *,
    closed: bool = False,
) -> tuple[Point2D, Point2D, float, float] | None:
    """Return endpoints and errors when an open point sequence is line-like."""
    copied = sanitize_points(points, minimum=2)
    if closed:
        return None
    start = copied[0]
    end = copied[-1]
    if points_close(start, end, _EPSILON):
        return None
    errors = point_distances_to_line(copied, start, end)
    local_max = max_error(errors)
    if local_max > max(0.0, float(tolerance)):
        return None
    return start, end, local_max, rms_error(errors)


def is_degenerate_point_set(points: Sequence[Point2D], tolerance: float) -> bool:
    """Return whether all points collapse to one coordinate."""
    copied = sanitize_points(points)
    if not copied:
        return True
    first = copied[0]
    return all(points_close(first, point, tolerance) for point in copied[1:])


def polygon_has_self_intersection(points: Sequence[Point2D], *, closed: bool = True) -> bool:
    """Return whether a polyline or polygon contains a proper self-intersection."""
    copied = sanitize_points(points)
    segments = _segments(copied, closed=closed)
    for first_index, first in enumerate(segments):
        for second_index, second in enumerate(segments[first_index + 1 :], start=first_index + 1):
            if _segments_adjacent(first_index, second_index, len(segments), closed=closed):
                continue
            if _segments_intersect(first[0], first[1], second[0], second[1]):
                return True
    return False


def _feature_indices(
    points: tuple[Point2D, ...],
    *,
    closed: bool,
    preserve_corners: bool,
    threshold: float,
    extra: Iterable[int],
) -> tuple[int, ...]:
    count = len(points)
    fixed: set[int] = {int(index) for index in extra if 0 <= int(index) < count}
    if preserve_corners:
        fixed.update(corner_indices(points, threshold_degrees=threshold, closed=closed))
    if closed:
        fixed.add(0)
    else:
        fixed.update({0, count - 1})
    return tuple(sorted(fixed))


def _simplify_with_fixed_indices(
    points: tuple[Point2D, ...],
    tolerance: float,
    *,
    closed: bool,
    fixed_indices: tuple[int, ...],
) -> np.ndarray:
    array = points_to_array(points)
    if not fixed_indices:
        return simplify_polyline_rdp(array, tolerance, closed=closed)
    output: list[np.ndarray] = []
    if closed:
        fixed = list(fixed_indices)
        if len(fixed) == 1:
            return simplify_polyline_rdp(array, tolerance, closed=True)
        for offset, start_index in enumerate(fixed):
            end_index = fixed[(offset + 1) % len(fixed)]
            if end_index <= start_index:
                section = np.vstack([array[start_index:], array[: end_index + 1]])
            else:
                section = array[start_index : end_index + 1]
            simplified = simplify_polyline_rdp(section, tolerance, closed=False)
            if output:
                output.extend(simplified[1:])
            else:
                output.extend(simplified)
        if len(output) > 1 and np.linalg.norm(output[0] - output[-1]) <= _EPSILON:
            output.pop()
    else:
        fixed = sorted(set(fixed_indices) | {0, len(points) - 1})
        for start_index, end_index in zip(fixed, fixed[1:], strict=False):
            if end_index <= start_index:
                continue
            section = array[start_index : end_index + 1]
            simplified = simplify_polyline_rdp(section, tolerance, closed=False)
            if output:
                output.extend(simplified[1:])
            else:
                output.extend(simplified)
    if not output:
        return array.copy()
    return np.asarray(output, dtype=np.float64)


def _creates_self_intersection(
    original: tuple[Point2D, ...],
    simplified: tuple[Point2D, ...],
    *,
    closed: bool,
) -> bool:
    return not polygon_has_self_intersection(original, closed=closed) and polygon_has_self_intersection(
        simplified,
        closed=closed,
    )


def _orientation_preserved(original: tuple[Point2D, ...], simplified: tuple[Point2D, ...]) -> bool:
    original_area = signed_area(original)
    simplified_area = signed_area(simplified)
    if abs(original_area) <= _EPSILON or abs(simplified_area) <= _EPSILON:
        return False
    return (original_area > 0.0) == (simplified_area > 0.0)


def _segments(points: tuple[Point2D, ...], *, closed: bool) -> tuple[tuple[Point2D, Point2D], ...]:
    if len(points) < 2:
        return ()
    segments = [(points[index], points[index + 1]) for index in range(len(points) - 1)]
    if closed and len(points) > 2:
        segments.append((points[-1], points[0]))
    return tuple(segments)


def _segments_adjacent(first: int, second: int, count: int, *, closed: bool) -> bool:
    if abs(first - second) <= 1:
        return True
    return closed and {first, second} == {0, count - 1}


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    def orientation(p: Point2D, q: Point2D, r: Point2D) -> float:
        return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
        return (
            min(p.x, r.x) - _EPSILON <= q.x <= max(p.x, r.x) + _EPSILON
            and min(p.y, r.y) - _EPSILON <= q.y <= max(p.y, r.y) + _EPSILON
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < -_EPSILON and o3 * o4 < -_EPSILON:
        return True
    if abs(o1) <= _EPSILON and on_segment(a, c, b):
        return True
    if abs(o2) <= _EPSILON and on_segment(a, d, b):
        return True
    if abs(o3) <= _EPSILON and on_segment(c, a, d):
        return True
    if abs(o4) <= _EPSILON and on_segment(c, b, d):
        return True
    return False


def _unchanged(
    points: tuple[Point2D, ...],
    *,
    closed: bool,
    scale: float,
    fixed: tuple[int, ...] = (),
) -> SimplificationResult:
    return SimplificationResult(
        points=points,
        input_count=len(points),
        output_count=len(points),
        max_error=0.0,
        rms_error=0.0,
        normalized_error=0.0,
        fixed_indices=fixed,
        topology_preserved=True,
        orientation_preserved=True if not closed else abs(signed_area(points)) > _EPSILON,
        changed=False,
    )


def _rejected(
    points: tuple[Point2D, ...],
    reason: str,
    *,
    closed: bool,
    scale: float,
    fixed: tuple[int, ...] = (),
    max_error_value: float = 0.0,
    rms_error_value: float = 0.0,
    normalized_error_value: float | None = None,
    topology_preserved: bool = True,
    orientation_preserved: bool = True,
) -> SimplificationResult:
    normalized = normalize_error(max_error_value, scale) if normalized_error_value is None else normalized_error_value
    return SimplificationResult(
        points=points,
        input_count=len(points),
        output_count=len(points),
        max_error=max_error_value,
        rms_error=rms_error_value,
        normalized_error=normalized,
        fixed_indices=fixed,
        topology_preserved=topology_preserved,
        orientation_preserved=orientation_preserved if closed else True,
        changed=False,
        rejected_reason=reason,
    )


def _coerce_point(point: Any) -> Point2D:
    if isinstance(point, Point2D):
        return Point2D(point.x, point.y)
    if isinstance(point, Sequence) and len(point) == 2:
        x = float(point[0])
        y = float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise GeometrySimplificationError("point coordinates must be finite.")
        return Point2D(x, y)
    raise TypeError("points must be Point2D or x/y pairs.")


__all__ = [
    "DeduplicationResult",
    "GeometrySimplificationError",
    "SimplificationResult",
    "deduplicate_consecutive_points",
    "is_degenerate_point_set",
    "line_like_endpoints",
    "point_distance",
    "points_close",
    "polygon_has_self_intersection",
    "sanitize_points",
    "simplify_polyline_preserving_features",
]
