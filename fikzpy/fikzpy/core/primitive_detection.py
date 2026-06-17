"""Experimental primitive detection hooks for simple TikZ shapes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fikzpy.core.contour_detector import Contour, simplify_polyline


@dataclass(frozen=True)
class Primitive:
    """A detected geometric primitive in image coordinates."""

    kind: str
    params: dict[str, float | list[tuple[float, float]]]
    confidence: float


def detect_primitive(contour: Contour, *, min_confidence: float = 0.85) -> Primitive | None:
    """Detect a simple primitive when the contour is unambiguous."""
    candidates = [
        detect_line(contour),
        detect_rectangle(contour),
        detect_ellipse(contour),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None

    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= min_confidence else None


def detect_line(contour: Contour) -> Primitive | None:
    """Detect a straight open line."""
    if contour.closed or len(contour.points) < 2:
        return None

    points = contour.points
    start = points[0]
    end = points[-1]
    baseline = end - start
    length = float(np.linalg.norm(baseline))
    if length == 0.0:
        return None

    distances = _point_line_distances(points, start, end)
    max_distance = float(distances.max(initial=0.0))
    confidence = max(0.0, 1.0 - max_distance / max(length, 1.0))
    if confidence < 0.9:
        return None

    return Primitive(
        kind="line",
        params={"points": [_point_tuple(start), _point_tuple(end)]},
        confidence=confidence,
    )


def detect_rectangle(contour: Contour) -> Primitive | None:
    """Detect a clear four-corner rectangle."""
    if not contour.closed or len(contour.points) < 4:
        return None

    points = simplify_polyline(contour.points, epsilon_ratio=0.03, closed=True)
    if len(points) != 4:
        return None

    angles = [_corner_angle(points, index) for index in range(4)]
    angle_error = max(abs(angle - 90.0) for angle in angles)
    if angle_error > 12.0:
        return None

    confidence = max(0.0, 1.0 - angle_error / 45.0)
    return Primitive(
        kind="rectangle",
        params={"points": [_point_tuple(point) for point in points]},
        confidence=confidence,
    )


def detect_ellipse(contour: Contour) -> Primitive | None:
    """Detect an ellipse-like closed contour."""
    if not contour.closed or len(contour.points) < 5:
        return None

    points = contour.points.astype(np.float32)
    try:
        (center_x, center_y), (axis_a, axis_b), angle = cv2.fitEllipse(points)
    except cv2.error:
        return None

    if axis_a <= 0 or axis_b <= 0:
        return None

    ellipse_area = np.pi * (axis_a / 2.0) * (axis_b / 2.0)
    contour_area = abs(cv2.contourArea(points.reshape(-1, 1, 2)))
    if ellipse_area <= 0 or contour_area <= 0:
        return None

    area_ratio = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)
    if area_ratio < 0.82:
        return None

    kind = "circle" if min(axis_a, axis_b) / max(axis_a, axis_b) > 0.9 else "ellipse"
    return Primitive(
        kind=kind,
        params={
            "center_x": float(center_x),
            "center_y": float(center_y),
            "axis_a": float(axis_a),
            "axis_b": float(axis_b),
            "angle": float(angle),
        },
        confidence=float(area_ratio),
    )


def _point_line_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    baseline = end - start
    denominator = float(np.linalg.norm(baseline))
    if denominator == 0.0:
        return np.full(len(points), np.inf)

    offsets = points - start
    cross = baseline[0] * offsets[:, 1] - baseline[1] * offsets[:, 0]
    return np.abs(cross) / denominator


def _corner_angle(points: np.ndarray, index: int) -> float:
    previous_point = points[(index - 1) % len(points)]
    current_point = points[index]
    next_point = points[(index + 1) % len(points)]
    first = previous_point - current_point
    second = next_point - current_point
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _point_tuple(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])
