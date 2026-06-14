"""Conservative merging of nearby open contours."""

from __future__ import annotations

import math

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.contour_cleaning import contour_length


def merge_nearby_contours(
    contours: list[Contour],
    *,
    max_distance: float = 2.5,
    max_angle: float = 35.0,
) -> list[Contour]:
    """Merge open contours whose endpoints are close and direction-compatible."""
    remaining = list(contours)
    changed = True

    while changed:
        changed = False
        for left_index in range(len(remaining)):
            left = remaining[left_index]
            if left.closed or len(left.points) < 2:
                continue

            merge_candidate = None
            for right_index in range(left_index + 1, len(remaining)):
                right = remaining[right_index]
                if right.closed or len(right.points) < 2:
                    continue

                merged = _try_merge(left, right, max_distance=max_distance, max_angle=max_angle)
                if merged is not None:
                    merge_candidate = (right_index, merged)
                    break

            if merge_candidate is None:
                continue

            right_index, merged = merge_candidate
            remaining[left_index] = merged
            del remaining[right_index]
            changed = True
            break

    remaining.sort(key=contour_length, reverse=True)
    return remaining


def _try_merge(
    left: Contour,
    right: Contour,
    *,
    max_distance: float,
    max_angle: float,
) -> Contour | None:
    left_points = left.points
    right_points = right.points
    candidates = [
        (left_points, right_points),
        (left_points, right_points[::-1]),
        (left_points[::-1], right_points),
        (left_points[::-1], right_points[::-1]),
    ]

    best: tuple[float, np.ndarray] | None = None
    for first, second in candidates:
        distance = float(np.linalg.norm(first[-1] - second[0]))
        if distance > max_distance:
            continue

        angle = _join_angle(first, second)
        if angle > max_angle:
            continue

        joined = np.vstack([first, second[1:]])
        score = distance + angle / 180.0
        if best is None or score < best[0]:
            best = (score, joined)

    if best is None:
        return None

    points = best[1]
    return Contour(points=points, closed=False, area=0.0, perimeter=contour_length(Contour(points, closed=False)))


def _join_angle(first: np.ndarray, second: np.ndarray) -> float:
    first_direction = first[-1] - first[-2]
    second_direction = second[1] - second[0]
    return _angle_between(first_direction, second_direction)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 180.0

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))
