from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.vector_exporter import count_vector_objects, flatten_vector_objects
from fikzpy.core.vector_objects import BezierCurve, Line, Point
from fikzpy.core.vector_pipeline import contours_to_local_bezier_objects, fit_contours_to_vector_objects


def test_fit_contours_to_vector_objects_reduces_arc_vs_local_beziers() -> None:
    angles = np.linspace(0, np.pi / 2, 32)
    points = np.column_stack([50 + 20 * np.cos(angles), 50 + 20 * np.sin(angles)])
    contour = Contour(points=points, closed=False)
    options = TikzOptions(width_units=10, use_bezier=True)

    local = contours_to_local_bezier_objects([contour], (100, 100, 3), options)
    fitted = fit_contours_to_vector_objects([contour], (100, 100, 3), options)

    assert count_vector_objects(fitted.objects).total < count_vector_objects(local).total
    assert count_vector_objects(fitted.objects).bezier_curves <= 3


def test_fit_contours_to_vector_objects_keeps_straight_contour_as_line() -> None:
    points = np.column_stack([np.linspace(10, 90, 24), np.full(24, 50)])
    contour = Contour(points=points, closed=False)

    fitted = fit_contours_to_vector_objects([contour], (100, 100, 3), TikzOptions(width_units=10))

    stats = count_vector_objects(fitted.objects)
    assert stats.lines == 1
    assert isinstance(fitted.objects[0], Line)


def test_fit_contours_to_vector_objects_preserves_bezier_endpoints() -> None:
    xs = np.linspace(10, 90, 32)
    ys = 50 + 15 * np.sin(np.linspace(0, np.pi, 32))
    contour = Contour(points=np.column_stack([xs, ys]), closed=False)

    fitted = fit_contours_to_vector_objects([contour], (100, 100, 3), TikzOptions(width_units=10))
    primitives = flatten_vector_objects(fitted.objects)
    curves = [item for item in primitives if isinstance(item, BezierCurve)]

    assert curves
    assert _start_point(primitives).distance_to(Point(1.0, 5.0)) < 1e-9
    assert _end_point(primitives).distance_to(Point(9.0, 5.0)) < 1e-9


def _start_point(items):
    item = items[0]
    if isinstance(item, Line | BezierCurve):
        return item.start
    return item.points[0]


def _end_point(items):
    item = items[-1]
    if isinstance(item, Line | BezierCurve):
        return item.end
    return item.points[-1]
