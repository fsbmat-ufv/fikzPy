from __future__ import annotations

import numpy as np

from fikzpy.core.bezier_fit import can_use_bezier, catmull_rom_to_bezier


def test_catmull_rom_to_bezier_returns_segments() -> None:
    points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)

    segments = catmull_rom_to_bezier(points)

    assert len(segments) == 3
    assert np.allclose(segments[0].start, points[0])
    assert np.allclose(segments[-1].end, points[-1])


def test_closed_bezier_has_one_segment_per_point() -> None:
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    segments = catmull_rom_to_bezier(points, closed=True)

    assert len(segments) == len(points)
    assert np.allclose(segments[-1].end, points[0])


def test_can_use_bezier_requires_enough_points() -> None:
    assert not can_use_bezier(np.array([[0, 0], [1, 1], [2, 0]], dtype=float))
    assert can_use_bezier(np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float))
