"""Convert detected contours into internal vector objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fikzpy.core.bezier_fit import can_use_bezier, catmull_rom_to_bezier, fit_cubic_beziers
from fikzpy.core.bezier_fit import simplify_points_for_bezier_fit
from fikzpy.core.contour_detector import Contour
from fikzpy.core.contour_cleaning import contour_length
from fikzpy.core.tikz_generator import TikzOptions, image_point_to_tikz
from fikzpy.core.vector_objects import BezierCurve, Line, PathGroup, Point, Polyline, VectorObject


@dataclass(frozen=True)
class VectorPipelineResult:
    """Vector objects plus fitting metrics for diagnostics."""

    objects: tuple[VectorObject, ...]
    input_points: int
    simplified_points: int

    @property
    def geometric_reduction(self) -> float:
        """Return percentage reduction from input points to vector objects."""
        if self.input_points == 0:
            return 0.0
        return max(0.0, 100.0 * (1.0 - len(self.objects) / float(self.input_points)))


def contours_to_vector_objects(
    contours: list[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions,
) -> tuple[VectorObject, ...]:
    """Convert contours to fitted vector objects without changing extraction."""
    return fit_contours_to_vector_objects(contours, image_shape, options).objects


def fit_contours_to_vector_objects(
    contours: list[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions,
) -> VectorPipelineResult:
    """Convert contours to global fitted vector objects and diagnostics."""
    objects: list[VectorObject] = []
    input_points = 0
    simplified_points = 0
    for contour in contours:
        points = _contour_points_to_tikz_points(contour, image_shape, options)
        input_points += len(points)
        simplified = _simplify_points_for_contour(points, contour)
        simplified_points += len(simplified)
        fitted = _points_to_fitted_objects(simplified, closed=contour.closed)
        if len(fitted) == 1:
            objects.append(fitted[0])
        elif fitted:
            objects.append(PathGroup(fitted))
    return VectorPipelineResult(tuple(objects), input_points=input_points, simplified_points=simplified_points)


def contours_to_local_bezier_objects(
    contours: list[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions,
) -> tuple[VectorObject, ...]:
    """Convert contours with the previous local Catmull-Rom Bezier strategy."""
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


def _points_to_fitted_objects(points: tuple[Point, ...], *, closed: bool) -> tuple[VectorObject, ...]:
    if len(points) < 2:
        return ()

    path_length = _point_path_length(points, closed=closed)
    error_tolerance = max(0.01, min(0.08, path_length * 0.015))
    straightness_tolerance = max(0.006, min(0.03, path_length * 0.004))
    min_bezier_length = max(0.04, min(0.12, path_length * 0.02))
    fitted = fit_cubic_beziers(
        points,
        error_tolerance=error_tolerance,
        closed=closed,
        min_bezier_length=min_bezier_length,
        min_points_for_bezier=4,
        straightness_tolerance=straightness_tolerance,
        simplify_tolerance=0.0,
    )
    return tuple(fitted)


def _simplify_points_for_contour(points: tuple[Point, ...], contour: Contour) -> tuple[Point, ...]:
    path_length = contour_length(contour)
    tolerance = max(0.001, min(0.005, path_length * 0.001))
    simplified = simplify_points_for_bezier_fit(points, tolerance=tolerance, closed=contour.closed)
    if len(simplified) < 2:
        return points
    return tuple(Point.from_pair(tuple(point)) for point in simplified)


def _point_path_length(points: tuple[Point, ...], *, closed: bool) -> float:
    total = sum(first.distance_to(second) for first, second in zip(points, points[1:]))
    if closed and len(points) > 2:
        total += points[-1].distance_to(points[0])
    return total
