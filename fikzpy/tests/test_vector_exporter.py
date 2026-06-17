from __future__ import annotations

from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.vector_exporter import count_vector_objects, generate_tikz_from_vector_objects
from fikzpy.core.vector_objects import BezierCurve, Circle, Line, PathGroup, Point, Polyline


def test_vector_exporter_emits_bezier_controls_and_marker() -> None:
    curve = BezierCurve(
        start=Point(0, 0),
        control1=Point(1, 1),
        control2=Point(2, 1),
        end=Point(3, 0),
    )

    code = generate_tikz_from_vector_objects([curve], options=TikzOptions(), diagnostic_marker=True)

    assert "% FIKZPY VECTOR MODE" in code
    assert "VECTOR MODE" in code
    assert ".. controls" in code


def test_count_vector_objects_counts_flat_primitives() -> None:
    objects = [
        Line(Point(0, 0), Point(1, 0)),
        Polyline((Point(0, 0), Point(1, 1))),
        Circle(Point(0, 0), 1),
    ]

    stats = count_vector_objects(objects)

    assert stats.total == 3
    assert stats.lines == 1
    assert stats.polylines == 1
    assert stats.circles == 1


def test_vector_exporter_serializes_connected_path_group_as_one_draw() -> None:
    group = PathGroup(
        (
            Line(Point(0, 0), Point(1, 0)),
            BezierCurve(
                start=Point(1, 0),
                control1=Point(1.5, 0.5),
                control2=Point(2.5, 0.5),
                end=Point(3, 0),
            ),
        )
    )

    code = generate_tikz_from_vector_objects([group], options=TikzOptions())

    assert code.count("\\draw") == 1
    assert "-- (1,0)" in code
    assert ".. controls" in code
