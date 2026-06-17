"""Conservative path and primitive merging helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import acos, degrees, hypot, isfinite
from typing import Any

import numpy as np

from fikzpy.core.geometry_error import point_distances_to_line, points_to_array
from fikzpy.core.geometry_error import max_error, rms_error
from fikzpy.core.geometry_simplification import deduplicate_consecutive_points, point_distance
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, LinePrimitive, Point2D, PointPrimitive
from fikzpy.core.semantic_geometry import PolylinePrimitive, Primitive, PrimitiveGroup, SemanticGeometry


_EPSILON = 1e-12


class PathJoinKind(Enum):
    """Endpoint orientation used to join two open paths."""

    NONE = "none"
    END_TO_START = "end_to_start"
    END_TO_END_REVERSE_SECOND = "end_to_end_reverse_second"
    START_TO_START_REVERSE_FIRST = "start_to_start_reverse_first"
    START_TO_END = "start_to_end"


@dataclass(frozen=True)
class MergeOutcome:
    """Result of a conservative merge attempt."""

    accepted: bool
    primitive: SemanticGeometry | None = None
    join_kind: PathJoinKind = PathJoinKind.NONE
    local_error: float = float("inf")
    rms_error: float = float("inf")
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return compact diagnostics."""
        return {
            "accepted": self.accepted,
            "join_kind": self.join_kind.value,
            "local_error": _finite_or_none(self.local_error),
            "rms_error": _finite_or_none(self.rms_error),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


def compatible_styles(first: SemanticGeometry, second: SemanticGeometry, *, preserve_styles: bool = True) -> bool:
    """Return whether two primitives may be merged without style changes."""
    if not preserve_styles:
        return True
    if isinstance(first, PrimitiveGroup) or isinstance(second, PrimitiveGroup):
        return False
    return _style_signature(first) == _style_signature(second)


def try_merge_collinear_lines(
    first: LinePrimitive,
    second: LinePrimitive,
    *,
    endpoint_tolerance: float,
    angle_tolerance: float,
    distance_tolerance: float,
    preserve_styles: bool = True,
    decimal_precision: int = 6,
    metadata: Mapping[str, Any] | None = None,
) -> MergeOutcome:
    """Merge two line segments when they are adjacent or overlapping."""
    if not compatible_styles(first, second, preserve_styles=preserve_styles):
        return MergeOutcome(False, reason="styles differ")
    first_vector = _vector(first.start, first.end)
    second_vector = _vector(second.start, second.end)
    if _length(first_vector) <= _EPSILON or _length(second_vector) <= _EPSILON:
        return MergeOutcome(False, reason="degenerate line")
    angle = _undirected_angle(first_vector, second_vector)
    if angle > max(0.0, float(angle_tolerance)):
        return MergeOutcome(False, reason="line directions differ", metadata={"angle": angle})

    points = (first.start, first.end, second.start, second.end)
    distances = point_distances_to_line(points, first.start, first.end)
    local_error = max_error(distances)
    local_rms = rms_error(distances)
    if local_error > max(0.0, float(distance_tolerance)):
        return MergeOutcome(False, reason="parallel or offset line", local_error=local_error, rms_error=local_rms)
    if not _segments_touch_or_overlap(first, second, max(0.0, float(endpoint_tolerance))):
        return MergeOutcome(False, reason="line endpoints are not compatible")

    direction = np.asarray(first.end.as_tuple(), dtype=np.float64) - np.asarray(first.start.as_tuple(), dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), _EPSILON)
    origin = np.asarray(first.start.as_tuple(), dtype=np.float64)
    array = points_to_array(points)
    projections = (array - origin) @ direction
    start = origin + float(np.min(projections)) * direction
    end = origin + float(np.max(projections)) * direction
    merged = LinePrimitive(
        start=_rounded_point(start, decimal_precision),
        end=_rounded_point(end, decimal_precision),
        stroke=first.stroke,
        fill=first.fill,
        opacity=first.opacity,
        confidence=_merged_confidence(first, second),
        error=max(float(first.error or 0.0), float(second.error or 0.0), local_error),
        metadata=dict(metadata or {}),
    )
    return MergeOutcome(True, merged, local_error=local_error, rms_error=local_rms)


def try_join_open_paths(
    first: LinePrimitive | PolylinePrimitive,
    second: LinePrimitive | PolylinePrimitive,
    *,
    endpoint_tolerance: float,
    join_angle_tolerance: float,
    preserve_styles: bool = True,
    decimal_precision: int = 6,
    metadata: Mapping[str, Any] | None = None,
) -> MergeOutcome:
    """Join two open line/polyline primitives using the best endpoint orientation."""
    if not compatible_styles(first, second, preserve_styles=preserve_styles):
        return MergeOutcome(False, reason="styles differ")
    first_points = _open_points(first)
    second_points = _open_points(second)
    if first_points is None or second_points is None:
        return MergeOutcome(False, reason="closed paths are not joined")
    candidates: list[tuple[float, PathJoinKind, tuple[Point2D, ...], tuple[Point2D, ...]]] = []
    variants = (
        (PathJoinKind.END_TO_START, first_points, second_points),
        (PathJoinKind.END_TO_END_REVERSE_SECOND, first_points, tuple(reversed(second_points))),
        (PathJoinKind.START_TO_START_REVERSE_FIRST, tuple(reversed(first_points)), second_points),
        (PathJoinKind.START_TO_END, tuple(reversed(first_points)), tuple(reversed(second_points))),
    )
    for kind, left, right in variants:
        gap = point_distance(left[-1], right[0])
        if gap > max(0.0, float(endpoint_tolerance)):
            continue
        angle = _join_angle(left, right)
        if angle > max(0.0, float(join_angle_tolerance)):
            continue
        candidates.append((gap, kind, left, right))
    if not candidates:
        return MergeOutcome(False, reason="no compatible endpoint orientation")
    gap, kind, left, right = sorted(candidates, key=lambda item: (item[0], item[1].value))[0]
    joined = deduplicate_consecutive_points((*left, *right), endpoint_tolerance, closed=False).points
    rounded = tuple(_round_point(point, decimal_precision) for point in joined)
    primitive = PolylinePrimitive(
        points=rounded,
        closed=False,
        stroke=first.stroke,
        fill=first.fill,
        opacity=first.opacity,
        confidence=_merged_confidence(first, second),
        error=max(float(first.error or 0.0), float(second.error or 0.0), gap),
        metadata=dict(metadata or {}),
    )
    return MergeOutcome(True, primitive, join_kind=kind, local_error=gap, rms_error=gap)


def duplicate_primitives(
    first: SemanticGeometry,
    second: SemanticGeometry,
    *,
    tolerance: float,
    preserve_styles: bool = True,
) -> bool:
    """Return whether two primitives are geometrically duplicate enough to remove one."""
    if isinstance(first, PrimitiveGroup) or isinstance(second, PrimitiveGroup):
        return False
    if not compatible_styles(first, second, preserve_styles=preserve_styles):
        return False
    if _has_translucency(first) or _has_translucency(second):
        return False
    limit = max(0.0, float(tolerance))
    if isinstance(first, PointPrimitive) and isinstance(second, PointPrimitive):
        return point_distance(first.point, second.point) <= limit
    if isinstance(first, LinePrimitive) and isinstance(second, LinePrimitive):
        return _same_point_pair((first.start, first.end), (second.start, second.end), limit)
    if isinstance(first, CirclePrimitive) and isinstance(second, CirclePrimitive):
        return point_distance(first.center, second.center) <= limit and abs(first.radius - second.radius) <= limit
    if isinstance(first, EllipsePrimitive) and isinstance(second, EllipsePrimitive):
        return (
            point_distance(first.center, second.center) <= limit
            and abs(first.radius_x - second.radius_x) <= limit
            and abs(first.radius_y - second.radius_y) <= limit
            and abs(first.rotation - second.rotation) <= max(limit, 1e-9)
        )
    if isinstance(first, PolylinePrimitive) and isinstance(second, PolylinePrimitive):
        return first.closed == second.closed and _same_point_sequence(first.points, second.points, limit)
    if isinstance(first, ClosedShapePrimitive) and isinstance(second, ClosedShapePrimitive):
        return _same_point_sequence(first.points, second.points, limit)
    if isinstance(first, BezierPrimitive) and isinstance(second, BezierPrimitive):
        first_points = (first.start, first.control1, first.control2, first.end)
        second_points = (second.start, second.control1, second.control2, second.end)
        reversed_second = (second.end, second.control2, second.control1, second.start)
        return _same_ordered_points(first_points, second_points, limit) or _same_ordered_points(
            first_points,
            reversed_second,
            limit,
        )
    return False


def primitive_endpoints(primitive: SemanticGeometry) -> tuple[Point2D, Point2D] | None:
    """Return open path endpoints when available."""
    if isinstance(primitive, LinePrimitive):
        return primitive.start, primitive.end
    if isinstance(primitive, PolylinePrimitive) and not primitive.closed:
        return primitive.points[0], primitive.points[-1]
    if isinstance(primitive, BezierPrimitive):
        return primitive.start, primitive.end
    return None


def _open_points(primitive: LinePrimitive | PolylinePrimitive) -> tuple[Point2D, ...] | None:
    if isinstance(primitive, LinePrimitive):
        return primitive.start, primitive.end
    if primitive.closed:
        return None
    return tuple(primitive.points)


def _segments_touch_or_overlap(first: LinePrimitive, second: LinePrimitive, tolerance: float) -> bool:
    endpoints = (
        (first.start, second.start),
        (first.start, second.end),
        (first.end, second.start),
        (first.end, second.end),
    )
    if any(point_distance(a, b) <= tolerance for a, b in endpoints):
        return True
    direction = np.asarray(first.end.as_tuple(), dtype=np.float64) - np.asarray(first.start.as_tuple(), dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), _EPSILON)
    origin = np.asarray(first.start.as_tuple(), dtype=np.float64)

    def interval(line: LinePrimitive) -> tuple[float, float]:
        values = []
        for point in (line.start, line.end):
            values.append(float((np.asarray(point.as_tuple(), dtype=np.float64) - origin) @ direction))
        return min(values), max(values)

    a0, a1 = interval(first)
    b0, b1 = interval(second)
    return max(a0, b0) <= min(a1, b1) + tolerance


def _join_angle(first: tuple[Point2D, ...], second: tuple[Point2D, ...]) -> float:
    if len(first) < 2 or len(second) < 2:
        return 180.0
    incoming = _vector(first[-2], first[-1])
    outgoing = _vector(second[0], second[1])
    return _directed_angle(incoming, outgoing)


def _vector(start: Point2D, end: Point2D) -> tuple[float, float]:
    return end.x - start.x, end.y - start.y


def _length(vector: tuple[float, float]) -> float:
    return hypot(vector[0], vector[1])


def _directed_angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_length = _length(first)
    second_length = _length(second)
    if first_length <= _EPSILON or second_length <= _EPSILON:
        return 180.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _undirected_angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    angle = _directed_angle(first, second)
    return min(angle, abs(180.0 - angle))


def _style_signature(primitive: SemanticGeometry) -> tuple[Any, ...]:
    stroke = getattr(primitive, "stroke", None)
    fill = getattr(primitive, "fill", None)
    metadata = dict(getattr(primitive, "metadata", {}))
    resolved = metadata.get("resolved_style") if isinstance(metadata.get("resolved_style"), Mapping) else {}
    return (
        stroke,
        fill,
        getattr(primitive, "opacity", None),
        resolved.get("fill_rule"),
    )


def _has_translucency(primitive: SemanticGeometry) -> bool:
    opacity = getattr(primitive, "opacity", None)
    stroke = getattr(primitive, "stroke", None)
    fill = getattr(primitive, "fill", None)
    return bool(
        (opacity is not None and opacity < 1.0)
        or (stroke is not None and stroke.opacity is not None and stroke.opacity < 1.0)
        or (fill is not None and fill.opacity is not None and fill.opacity < 1.0)
    )


def _same_point_pair(first: tuple[Point2D, Point2D], second: tuple[Point2D, Point2D], tolerance: float) -> bool:
    return (
        point_distance(first[0], second[0]) <= tolerance
        and point_distance(first[1], second[1]) <= tolerance
    ) or (
        point_distance(first[0], second[1]) <= tolerance
        and point_distance(first[1], second[0]) <= tolerance
    )


def _same_point_sequence(first: Sequence[Point2D], second: Sequence[Point2D], tolerance: float) -> bool:
    if len(first) != len(second):
        return False
    return _same_ordered_points(first, second, tolerance) or _same_ordered_points(first, tuple(reversed(second)), tolerance)


def _same_ordered_points(first: Sequence[Point2D], second: Sequence[Point2D], tolerance: float) -> bool:
    if len(first) != len(second):
        return False
    return all(point_distance(a, b) <= tolerance for a, b in zip(first, second, strict=True))


def _merged_confidence(first: SemanticGeometry, second: SemanticGeometry) -> float | None:
    values = [value for value in (getattr(first, "confidence", None), getattr(second, "confidence", None)) if value is not None]
    return min(values) if values else None


def _rounded_point(array: np.ndarray, precision: int) -> Point2D:
    return Point2D(round(float(array[0]), precision), round(float(array[1]), precision))


def _round_point(point: Point2D, precision: int) -> Point2D:
    return Point2D(round(point.x, precision), round(point.y, precision))


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if isfinite(number) else None


__all__ = [
    "MergeOutcome",
    "PathJoinKind",
    "compatible_styles",
    "duplicate_primitives",
    "primitive_endpoints",
    "try_join_open_paths",
    "try_merge_collinear_lines",
]
