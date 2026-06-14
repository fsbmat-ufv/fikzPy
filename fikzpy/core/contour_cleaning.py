"""Filtering utilities for traced contours."""

from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.vectorization_config import ContourCleaningConfig


def contour_length(contour: Contour) -> float:
    """Return the polyline length of a contour."""
    points = contour.points
    if len(points) < 2:
        return 0.0

    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if contour.closed and len(points) > 2:
        length += float(np.linalg.norm(points[0] - points[-1]))
    return length


def filter_contours(
    contours: list[Contour],
    config: ContourCleaningConfig | None = None,
) -> list[Contour]:
    """Remove contours that are too small to be useful TikZ paths."""
    config = config or ContourCleaningConfig()
    filtered = [
        contour
        for contour in contours
        if len(contour.points) >= config.min_points and contour_length(contour) >= config.min_length
    ]

    if config.deduplicate:
        filtered = remove_duplicate_contours(filtered, max_distance=config.duplicate_distance)

    filtered.sort(key=contour_length, reverse=True)
    return filtered


def remove_duplicate_contours(contours: list[Contour], *, max_distance: float = 1.0) -> list[Contour]:
    """Remove nearly identical contours using rounded point signatures."""
    kept: list[Contour] = []
    signatures: list[np.ndarray] = []

    for contour in contours:
        signature = _normalized_signature(contour)
        if any(_signature_distance(signature, other) <= max_distance for other in signatures):
            continue
        kept.append(contour)
        signatures.append(signature)

    return kept


def _normalized_signature(contour: Contour) -> np.ndarray:
    points = np.asarray(contour.points, dtype=np.float64)
    if len(points) == 0:
        return points

    if contour.closed:
        start_index = int(np.lexsort((points[:, 1], points[:, 0]))[0])
        points = np.roll(points, -start_index, axis=0)
    else:
        reversed_points = points[::-1]
        if tuple(reversed_points[0]) < tuple(points[0]):
            points = reversed_points

    return points


def _signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second) or len(first) == 0:
        return float("inf")
    return float(np.linalg.norm(first - second, axis=1).mean())
