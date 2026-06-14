from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.contour_merging import merge_nearby_contours


def test_merge_nearby_contours_joins_collinear_segments() -> None:
    first = Contour(points=np.array([[0, 0], [5, 0]], dtype=float), closed=False)
    second = Contour(points=np.array([[6, 0], [10, 0]], dtype=float), closed=False)

    merged = merge_nearby_contours([first, second], max_distance=2, max_angle=10)

    assert len(merged) == 1
    assert np.allclose(merged[0].points[0], [0, 0])
    assert np.allclose(merged[0].points[-1], [10, 0])


def test_merge_nearby_contours_rejects_sharp_turn() -> None:
    first = Contour(points=np.array([[0, 0], [5, 0]], dtype=float), closed=False)
    second = Contour(points=np.array([[6, 0], [6, 5]], dtype=float), closed=False)

    merged = merge_nearby_contours([first, second], max_distance=2, max_angle=10)

    assert len(merged) == 2


def test_merge_nearby_contours_does_not_merge_closed_paths() -> None:
    first = Contour(points=np.array([[0, 0], [5, 0], [5, 5]], dtype=float), closed=True)
    second = Contour(points=np.array([[5, 0], [10, 0]], dtype=float), closed=False)

    merged = merge_nearby_contours([first, second], max_distance=2, max_angle=10)

    assert len(merged) == 2
