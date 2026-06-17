"""Cubic Bezier fitting for semantic primitive recognition."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from collections.abc import Sequence
from typing import Any

import numpy as np

from fikzpy.core.geometry_error import evaluate_cubic_bezier, max_error, path_length
from fikzpy.core.geometry_error import point_distances_to_cubic_bezier, points_to_array, rms_error
from fikzpy.core.semantic_geometry import Point2D


_EPSILON = 1e-12


@dataclass(frozen=True)
class CubicBezierSegment:
    """One fitted cubic Bezier segment."""

    start: Point2D
    control1: Point2D
    control2: Point2D
    end: Point2D
    max_error: float
    rms_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return compact segment diagnostics."""
        return {
            "start": self.start.to_dict(),
            "control1": self.control1.to_dict(),
            "control2": self.control2.to_dict(),
            "end": self.end.to_dict(),
            "max_error": self.max_error,
            "rms_error": self.rms_error,
        }


@dataclass(frozen=True)
class CubicBezierFitResult:
    """Result of fitting one or more cubic Bezier segments."""

    accepted: bool
    segments: tuple[CubicBezierSegment, ...]
    max_error: float
    rms_error: float
    input_point_count: int
    segment_count: int
    recursion_depth_used: int
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing input samples."""
        return {
            "accepted": self.accepted,
            "segment_count": self.segment_count,
            "max_error": self.max_error,
            "rms_error": self.rms_error,
            "input_point_count": self.input_point_count,
            "recursion_depth_used": self.recursion_depth_used,
            "rejected_reason": self.rejected_reason,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def fit_cubic_beziers(
    points: Sequence[Point2D] | np.ndarray,
    tolerance: float,
    *,
    closed: bool = False,
    maximum_segments: int = 8,
    recursion_depth: int = 12,
    minimum_points: int = 6,
    minimum_length: float = 1e-6,
    straightness_tolerance: float = 1e-5,
    control_point_epsilon: float = 1e-7,
) -> CubicBezierFitResult:
    """Fit few cubic Beziers to an ordered point sequence.

    The implementation follows the deterministic structure of Schneider-style
    fitting: chord-length parameters, endpoint tangents, least-squares control
    distances, optional Newton reparameterization, then recursive splitting at
    the point of maximum error.
    """
    pts = _prepare_points(points, closed=closed)
    if len(pts) < max(4, int(minimum_points)):
        return _rejected("insufficient points", pts)
    if path_length(pts, closed=False) < max(0.0, float(minimum_length)):
        return _rejected("path is too short", pts)
    if _max_line_distance(pts) <= max(0.0, float(straightness_tolerance)):
        return _rejected("line represents the points better", pts)

    state = _FitState(
        tolerance=max(0.0, float(tolerance)),
        maximum_segments=max(1, int(maximum_segments)),
        recursion_depth=max(0, int(recursion_depth)),
        minimum_length=max(0.0, float(minimum_length)),
        control_point_epsilon=max(0.0, float(control_point_epsilon)),
        straightness_tolerance=max(0.0, float(straightness_tolerance)),
    )
    segments, depth_used, reason = _fit_recursive(pts, state, depth=0)
    if reason is not None:
        return _rejected(reason, pts, recursion_depth_used=depth_used)
    if not segments:
        return _rejected("no stable cubic segments produced", pts, recursion_depth_used=depth_used)
    if len(segments) > state.maximum_segments:
        return _rejected("maximum Bezier segment count exceeded", pts, recursion_depth_used=depth_used)

    errors = _piecewise_errors(pts, segments)
    fitted = tuple(
        _segment_from_arrays(curve, _segment_errors(pts, curve))
        for curve in segments
    )
    return CubicBezierFitResult(
        accepted=True,
        segments=fitted,
        max_error=max_error(errors),
        rms_error=rms_error(errors),
        input_point_count=len(pts),
        segment_count=len(fitted),
        recursion_depth_used=depth_used,
    )


def sample_cubic_bezier(
    start: Point2D,
    control1: Point2D,
    control2: Point2D,
    end: Point2D,
    *,
    samples: int = 32,
) -> tuple[Point2D, ...]:
    """Sample a semantic cubic Bezier at deterministic parameter values."""
    count = max(2, int(samples))
    controls = (
        np.array(start.as_tuple(), dtype=np.float64),
        np.array(control1.as_tuple(), dtype=np.float64),
        np.array(control2.as_tuple(), dtype=np.float64),
        np.array(end.as_tuple(), dtype=np.float64),
    )
    values = evaluate_cubic_bezier(controls, np.linspace(0.0, 1.0, count))
    return tuple(Point2D(float(x), float(y)) for x, y in values)


@dataclass(frozen=True)
class _FitState:
    tolerance: float
    maximum_segments: int
    recursion_depth: int
    minimum_length: float
    control_point_epsilon: float
    straightness_tolerance: float


def _fit_recursive(
    points: np.ndarray,
    state: _FitState,
    *,
    depth: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], int, str | None]:
    if len(points) < 4:
        return [], depth, "recursive segment has insufficient points"
    if path_length(points, closed=False) < state.minimum_length:
        return [], depth, "recursive segment is too short"
    if _max_line_distance(points) <= state.straightness_tolerance:
        return [], depth, "line represents a recursive segment better"

    tangent_start = _unit(points[1] - points[0])
    tangent_end = _unit(points[-2] - points[-1])
    if np.linalg.norm(tangent_start) <= _EPSILON or np.linalg.norm(tangent_end) <= _EPSILON:
        return [], depth, "unstable endpoint tangents"

    parameters = _chord_length_parameters(points)
    curve = _generate_cubic(points, parameters, tangent_start, tangent_end)
    curve = _reparameterized_curve(points, curve, tangent_start, tangent_end)
    errors = np.linalg.norm(evaluate_cubic_bezier(curve, _chord_length_parameters(points)) - points, axis=1)
    split_index = int(np.argmax(errors[1:-1]) + 1) if len(errors) > 2 else 1
    current_max_error = float(errors[split_index]) if len(errors) > 2 else 0.0

    if current_max_error <= state.tolerance and _valid_curve(curve, state):
        return [curve], depth, None

    if depth >= state.recursion_depth:
        return [], depth, "Bezier recursion depth exceeded"
    if split_index <= 1 or split_index >= len(points) - 2:
        return [], depth, "maximum error split is unstable"

    left, left_depth, left_reason = _fit_recursive(points[: split_index + 1], state, depth=depth + 1)
    if left_reason is not None:
        return [], max(depth, left_depth), left_reason
    right, right_depth, right_reason = _fit_recursive(points[split_index:], state, depth=depth + 1)
    if right_reason is not None:
        return [], max(left_depth, right_depth), right_reason
    combined = left + right
    if len(combined) > state.maximum_segments:
        return [], max(left_depth, right_depth), "maximum Bezier segment count exceeded"
    return combined, max(left_depth, right_depth), None


def _generate_cubic(
    points: np.ndarray,
    parameters: np.ndarray,
    tangent_start: np.ndarray,
    tangent_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p0 = points[0]
    p3 = points[-1]
    matrix = np.zeros((2, 2), dtype=np.float64)
    vector = np.zeros(2, dtype=np.float64)

    for point, u in zip(points, parameters, strict=True):
        b0, b1, b2, b3 = _bernstein(float(u))
        a1 = tangent_start * b1
        a2 = tangent_end * b2
        target = point - ((b0 + b1) * p0 + (b2 + b3) * p3)
        matrix[0, 0] += np.dot(a1, a1)
        matrix[0, 1] += np.dot(a1, a2)
        matrix[1, 0] += np.dot(a1, a2)
        matrix[1, 1] += np.dot(a2, a2)
        vector[0] += np.dot(a1, target)
        vector[1] += np.dot(a2, target)

    chord = float(np.linalg.norm(p3 - p0))
    fallback_alpha = chord / 3.0
    try:
        alpha_start, alpha_end = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        alpha_start = fallback_alpha
        alpha_end = fallback_alpha
    if not isfinite(float(alpha_start)) or alpha_start <= _EPSILON:
        alpha_start = fallback_alpha
    if not isfinite(float(alpha_end)) or alpha_end <= _EPSILON:
        alpha_end = fallback_alpha
    return p0, p0 + tangent_start * alpha_start, p3 + tangent_end * alpha_end, p3


def _reparameterized_curve(
    points: np.ndarray,
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tangent_start: np.ndarray,
    tangent_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters = _chord_length_parameters(points)
    refined = np.array([_newton_parameter(curve, point, float(u)) for point, u in zip(points, parameters, strict=True)])
    if not np.all(np.isfinite(refined)):
        return curve
    refined = np.maximum.accumulate(np.clip(refined, 0.0, 1.0))
    if refined[-1] <= refined[0]:
        return curve
    refined[0] = 0.0
    refined[-1] = 1.0
    return _generate_cubic(points, refined, tangent_start, tangent_end)


def _newton_parameter(
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    point: np.ndarray,
    parameter: float,
) -> float:
    p0, p1, p2, p3 = curve
    u = float(parameter)
    q = evaluate_cubic_bezier(curve, np.array([u], dtype=np.float64))[0]
    q1 = 3.0 * ((1.0 - u) ** 2) * (p1 - p0) + 6.0 * (1.0 - u) * u * (p2 - p1) + 3.0 * (u**2) * (p3 - p2)
    q2 = 6.0 * (1.0 - u) * (p2 - 2.0 * p1 + p0) + 6.0 * u * (p3 - 2.0 * p2 + p1)
    numerator = float(np.dot(q - point, q1))
    denominator = float(np.dot(q1, q1) + np.dot(q - point, q2))
    if abs(denominator) <= _EPSILON:
        return u
    return u - numerator / denominator


def _piecewise_errors(
    points: np.ndarray,
    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    distances = np.vstack([point_distances_to_cubic_bezier(points, curve, samples=96) for curve in curves])
    return np.min(distances, axis=0)


def _segment_errors(
    points: np.ndarray,
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    return point_distances_to_cubic_bezier(points, curve, samples=96)


def _segment_from_arrays(
    curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    errors: np.ndarray,
) -> CubicBezierSegment:
    return CubicBezierSegment(
        start=_point(curve[0]),
        control1=_point(curve[1]),
        control2=_point(curve[2]),
        end=_point(curve[3]),
        max_error=max_error(errors),
        rms_error=rms_error(errors),
    )


def _prepare_points(points: Sequence[Point2D] | np.ndarray, *, closed: bool) -> np.ndarray:
    array = points_to_array(points)
    if len(array) == 0:
        return array
    keep = [True]
    keep.extend(bool(np.linalg.norm(array[index] - array[index - 1]) > _EPSILON) for index in range(1, len(array)))
    array = array[np.array(keep, dtype=bool)]
    if closed and len(array) >= 2 and np.linalg.norm(array[0] - array[-1]) > _EPSILON:
        array = np.vstack([array, array[0]])
    return array


def _max_line_distance(points: np.ndarray) -> float:
    start = points[0]
    end = points[-1]
    baseline = end - start
    length = float(np.linalg.norm(baseline))
    if length <= _EPSILON:
        return float(np.linalg.norm(points - start, axis=1).max(initial=0.0))
    offsets = points - start
    cross = baseline[0] * offsets[:, 1] - baseline[1] * offsets[:, 0]
    return float(np.abs(cross).max(initial=0.0) / length)


def _chord_length_parameters(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(distances.sum())
    if total <= _EPSILON:
        return np.linspace(0.0, 1.0, len(points))
    return np.concatenate([[0.0], np.cumsum(distances) / total])


def _valid_curve(curve: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], state: _FitState) -> bool:
    p0, p1, p2, p3 = curve
    if float(np.linalg.norm(p3 - p0)) < state.minimum_length:
        return False
    first_span = float(np.linalg.norm(p1 - p0))
    second_span = float(np.linalg.norm(p2 - p3))
    if first_span <= state.control_point_epsilon or second_span <= state.control_point_epsilon:
        return False
    return True


def _bernstein(u: float) -> tuple[float, float, float, float]:
    return (1.0 - u) ** 3, 3.0 * u * (1.0 - u) ** 2, 3.0 * (u**2) * (1.0 - u), u**3


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _point(point: np.ndarray) -> Point2D:
    return Point2D(float(point[0]), float(point[1]))


def _rejected(
    reason: str,
    points: np.ndarray,
    *,
    recursion_depth_used: int = 0,
) -> CubicBezierFitResult:
    return CubicBezierFitResult(
        accepted=False,
        segments=(),
        max_error=float("inf"),
        rms_error=float("inf"),
        input_point_count=len(points),
        segment_count=0,
        recursion_depth_used=recursion_depth_used,
        rejected_reason=reason,
    )


__all__ = [
    "CubicBezierFitResult",
    "CubicBezierSegment",
    "fit_cubic_beziers",
    "sample_cubic_bezier",
]
