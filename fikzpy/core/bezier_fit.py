"""Small utilities for turning polylines into cubic Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
