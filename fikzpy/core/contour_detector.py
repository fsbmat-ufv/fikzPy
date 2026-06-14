"""Contour detection and simplification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Contour:
    """A simplified image contour represented by image-space points."""

    points: np.ndarray
    closed: bool = True
    area: float = 0.0
    perimeter: float = 0.0

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Contour points must have shape (n, 2).")
        object.__setattr__(self, "points", points)

    @property
    def is_drawable(self) -> bool:
        """Return True when the contour has enough points for a TikZ path."""
        return len(self.points) >= 2


def _as_cv_contour(contour: np.ndarray) -> np.ndarray:
    points = np.asarray(contour)
    if points.ndim == 2 and points.shape[1] == 2:
        return points.reshape(-1, 1, 2).astype(np.float32)
    if points.ndim == 3 and points.shape[1:] == (1, 2):
        return points.astype(np.float32)
    raise ValueError("Expected a contour with shape (n, 2) or (n, 1, 2).")


def simplify_contour(contour: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
    """Simplify a contour with the Douglas-Peucker algorithm."""
    cv_contour = _as_cv_contour(contour)
    perimeter = cv2.arcLength(cv_contour, True)
    epsilon = max(float(epsilon_ratio), 0.0) * perimeter
    simplified = cv2.approxPolyDP(cv_contour, epsilon, True)
    points = simplified.reshape(-1, 2).astype(np.float64)

    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    return points


def detect_contours_from_edges(
    edges: np.ndarray,
    *,
    simplify_epsilon: float = 0.01,
    min_area: float = 8.0,
    min_perimeter: float = 8.0,
) -> list[Contour]:
    """Detect, filter, simplify, and sort contours from a binary edge image."""
    if edges.ndim != 2:
        raise ValueError("Edge image must be a single-channel array.")

    edge_image = edges.astype(np.uint8, copy=False)
    found, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours: list[Contour] = []
    for raw in found:
        area = float(abs(cv2.contourArea(raw)))
        perimeter = float(cv2.arcLength(raw, True))
        if area < min_area and perimeter < min_perimeter:
            continue

        points = simplify_contour(raw, simplify_epsilon)
        if len(points) < 2:
            continue

        contours.append(Contour(points=points, closed=True, area=area, perimeter=perimeter))

    contours.sort(key=lambda item: (item.area, item.perimeter), reverse=True)
    return contours


def contours_to_polylines(contours: Iterable[Contour]) -> list[np.ndarray]:
    """Return contours in OpenCV polyline format."""
    polylines: list[np.ndarray] = []
    for contour in contours:
        if contour.is_drawable:
            polylines.append(np.rint(contour.points).astype(np.int32).reshape(-1, 1, 2))
    return polylines
