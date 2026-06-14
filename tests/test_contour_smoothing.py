from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.contour_smoothing import smooth_contours, smooth_polyline


def test_smooth_polyline_preserves_open_endpoints() -> None:
    points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]], dtype=float)

    smoothed = smooth_polyline(points, closed=False, iterations=1)

    assert np.allclose(smoothed[0], points[0])
    assert np.allclose(smoothed[-1], points[-1])
    assert not np.allclose(smoothed[1], points[1])


def test_smooth_contours_can_simplify_after_smoothing() -> None:
    contour = Contour(
        points=np.array([[0, 0], [1, 0.2], [2, -0.2], [3, 0], [4, 0]], dtype=float),
        closed=False,
    )

    smoothed = smooth_contours([contour], iterations=1, simplify_epsilon=0.01)

    assert len(smoothed) == 1
    assert len(smoothed[0].points) >= 2
