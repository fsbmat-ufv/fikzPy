"""Convert detected contours into internal vector objects."""

from __future__ import annotations

import numpy as np

from fikzpy.core.bezier_fit import can_use_bezier, catmull_rom_to_bezier
from fikzpy.core.contour_detector import Contour
from fikzpy.core.tikz_generator import TikzOptions, image_point_to_tikz
from fikzpy.core.vector_objects import BezierCurve, Line, PathGroup, Point, Polyline, VectorObject


def contours_to_vector_objects(
    contours: list[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions,
) -> tuple[VectorObject, ...]:
    """Convert contours to vector objects without changing contour extraction."""
    objects: list[VectorObject] = []
    for contour in contours:
        points = _contour_points_to_tikz_points(contour, image_shape, options)
        if len(points) == 2:
            objects.append(Line(points[0], points[1]))
        elif can_use_bezier(np.array([point.as_tuple() for point in points]), min_points=options.bezier_min_points):
            objects.append(_points_to_bezier_group(points, closed=contour.closed, options=options))
        else:
            objects.append(Polyline(points, closed=contour.closed))
    return tuple(objects)


def _contour_points_to_tikz_points(
    contour: Contour,
    image_shape: tuple[int, ...],
    options: TikzOptions,
) -> tuple[Point, ...]:
    points = [
        Point.from_pair(tuple(image_point_to_tikz(point, image_shape, width_units=options.width_units)))
        for point in contour.points
    ]
    return tuple(points)


def _points_to_bezier_group(points: tuple[Point, ...], *, closed: bool, options: TikzOptions) -> PathGroup:
    array = np.array([point.as_tuple() for point in points], dtype=np.float64)
    segments = catmull_rom_to_bezier(array, closed=closed, tension=options.bezier_tension)
    curves = tuple(
        BezierCurve(
            start=Point.from_pair(tuple(segment.start)),
            control1=Point.from_pair(tuple(segment.control1)),
            control2=Point.from_pair(tuple(segment.control2)),
            end=Point.from_pair(tuple(segment.end)),
        )
        for segment in segments
    )
    return PathGroup(curves)
