from __future__ import annotations

import pytest

from fikzpy.core.vector_objects import Circle, Line, PathGroup, Point, Polyline


def test_point_distance_to_other_point() -> None:
    start = Point(0, 0)
    end = Point(3, 4)

    assert start.distance_to(end) == 5


def test_polyline_length_supports_open_and_closed_paths() -> None:
    points = (Point(0, 0), Point(3, 0), Point(3, 4))

    assert Polyline(points).length == 7
    assert Polyline(points, closed=True).length == 12


def test_polyline_requires_at_least_two_points() -> None:
    with pytest.raises(ValueError):
        Polyline((Point(0, 0),))


def test_circle_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        Circle(Point(0, 0), 0)


def test_path_group_flatten_returns_nested_primitives() -> None:
    line = Line(Point(0, 0), Point(1, 0))
    nested = PathGroup((PathGroup((line,), name="inner"),), name="outer")

    assert nested.flatten() == (line,)
