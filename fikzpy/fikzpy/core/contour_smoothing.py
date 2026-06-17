"""Path smoothing helpers for traced contours."""

from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour, simplify_polyline
from fikzpy.core.contour_cleaning import contour_length


def smooth_contours(
    contours: list[Contour],
    *,
    iterations: int = 1,
    simplify_epsilon: float | None = None,
) -> list[Contour]:
    """Smooth contour points and optionally simplify them afterward."""
    smoothed: list[Contour] = []

    for contour in contours:
        points = smooth_polyline(contour.points, closed=contour.closed, iterations=iterations)
        if simplify_epsilon is not None:
            points = simplify_polyline(points, epsilon_ratio=simplify_epsilon, closed=contour.closed)
        if len(points) < 2:
            continue

        updated = Contour(points=points, closed=contour.closed, area=contour.area)
        smoothed.append(
            Contour(
                points=updated.points,
                closed=updated.closed,
                area=updated.area,
                perimeter=contour_length(updated),
            )
        )

    smoothed.sort(key=contour_length, reverse=True)
    return smoothed


def smooth_polyline(points: np.ndarray, *, closed: bool = False, iterations: int = 1) -> np.ndarray:
    """Reduce pixel stair-stepping while preserving open-path endpoints."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4 or iterations <= 0:
        return pts.copy()

    for _ in range(iterations):
        if closed:
            previous_points = np.roll(pts, 1, axis=0)
            next_points = np.roll(pts, -1, axis=0)
            pts = (previous_points + 2.0 * pts + next_points) / 4.0
        else:
            smoothed = pts.copy()
            smoothed[1:-1] = (pts[:-2] + 2.0 * pts[1:-1] + pts[2:]) / 4.0
            pts = smoothed

    return pts
