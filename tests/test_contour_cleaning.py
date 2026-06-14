from __future__ import annotations

import numpy as np

from fikzpy.core.contour_cleaning import contour_length, filter_contours, remove_duplicate_contours
from fikzpy.core.contour_detector import Contour
from fikzpy.core.vectorization_config import ContourCleaningConfig


def test_contour_length_handles_open_polyline() -> None:
    contour = Contour(points=np.array([[0, 0], [3, 4]], dtype=float), closed=False)

    assert contour_length(contour) == 5.0


def test_filter_contours_removes_short_paths() -> None:
    short = Contour(points=np.array([[0, 0], [1, 0]], dtype=float), closed=False)
    long = Contour(points=np.array([[0, 0], [5, 0]], dtype=float), closed=False)

    filtered = filter_contours([short, long], ContourCleaningConfig(min_length=3, min_points=2))

    assert filtered == [long]


def test_remove_duplicate_contours_keeps_first_similar_path() -> None:
    first = Contour(points=np.array([[0, 0], [5, 0], [10, 0]], dtype=float), closed=False)
    duplicate = Contour(points=np.array([[0.2, 0.1], [5.1, 0], [10.1, -0.1]], dtype=float), closed=False)

    filtered = remove_duplicate_contours([first, duplicate], max_distance=0.5)

    assert filtered == [first]
