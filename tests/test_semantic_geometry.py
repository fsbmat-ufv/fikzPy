from __future__ import annotations

from dataclasses import FrozenInstanceError
import subprocess
import sys
from pathlib import Path

import pytest

import fikzpy.core.semantic_geometry as semantic_geometry
from fikzpy.core.semantic_geometry import (
    BezierPrimitive,
    CirclePrimitive,
    ClosedShapePrimitive,
    EllipsePrimitive,
    FillStyle,
    LinePrimitive,
    Point2D,
    PointPrimitive,
    PolylinePrimitive,
    PrimitiveGroup,
    RGBColor,
    StrokeStyle,
)


def _points() -> tuple[Point2D, Point2D, Point2D, Point2D]:
    return Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)


def test_create_each_semantic_primitive() -> None:
    p0, p1, p2, p3 = _points()

    primitives = (
        PointPrimitive(p0),
        LinePrimitive(p0, p1),
        PolylinePrimitive((p0, p1, p2)),
        CirclePrimitive(p0, 2.0),
        EllipsePrimitive(p0, 2.0, 1.0, rotation=15.0),
        BezierPrimitive(p0, p1, p2, p3),
        ClosedShapePrimitive((p0, p1, p2)),
    )

    assert [item.to_dict()["type"] for item in primitives] == [
        "point",
        "line",
        "polyline",
        "circle",
        "ellipse",
        "bezier",
        "closed_shape",
    ]


def test_coordinates_are_preserved_in_primitives() -> None:
    start = Point2D(1.25, -2.5)
    end = Point2D(3.5, 4.75)
    line = LinePrimitive(start, end)

    assert line.start.as_tuple() == (1.25, -2.5)
    assert line.end.as_tuple() == (3.5, 4.75)
    assert line.to_dict()["start"] == {"x": 1.25, "y": -2.5}
    assert line.to_dict()["end"] == {"x": 3.5, "y": 4.75}


def test_stroke_color_width_fill_opacity_confidence_error_and_metadata_are_serialized() -> None:
    p0, p1, *_ = _points()
    stroke = StrokeStyle(
        color=RGBColor(10, 20, 30),
        width=0.75,
        opacity=0.8,
        line_cap="round",
        line_join="bevel",
        dash_pattern=(1.0, 2.0),
    )
    fill = FillStyle(RGBColor(200, 210, 220), opacity=0.35)

    line = LinePrimitive(
        p0,
        p1,
        stroke=stroke,
        fill=fill,
        opacity=0.9,
        confidence=0.95,
        error=0.01,
        metadata={"source": "unit-test"},
    )

    serialized = line.to_dict()
    assert serialized["stroke"]["color"] == {"red": 10, "green": 20, "blue": 30}
    assert serialized["stroke"]["width"] == 0.75
    assert serialized["stroke"]["opacity"] == 0.8
    assert serialized["stroke"]["dash_pattern"] == [1.0, 2.0]
    assert serialized["fill"] == {"color": {"red": 200, "green": 210, "blue": 220}, "opacity": 0.35}
    assert serialized["opacity"] == 0.9
    assert serialized["confidence"] == 0.95
    assert serialized["error"] == 0.01
    assert serialized["metadata"] == {"source": "unit-test"}


def test_point_color_and_style_are_immutable_and_comparable() -> None:
    first = Point2D(1, 2)
    second = Point2D(1.0, 2.0)
    color = RGBColor(1, 2, 3)
    style = StrokeStyle(color=color, width=1.5)

    assert first == second
    assert style == StrokeStyle(color=RGBColor(1, 2, 3), width=1.5)
    with pytest.raises(FrozenInstanceError):
        first.x = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        color.red = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        style.width = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (lambda: Point2D(float("nan"), 0), ValueError),
        (lambda: RGBColor(-1, 0, 0), ValueError),
        (lambda: RGBColor(256, 0, 0), ValueError),
        (lambda: RGBColor(True, 0, 0), TypeError),
        (lambda: StrokeStyle(width=0), ValueError),
        (lambda: StrokeStyle(opacity=1.1), ValueError),
        (lambda: StrokeStyle(dash_pattern=()), ValueError),
        (lambda: StrokeStyle(dash_pattern=(-1.0,)), ValueError),
        (lambda: FillStyle(RGBColor(0, 0, 0), opacity=-0.1), ValueError),
        (lambda: LinePrimitive(Point2D(0, 0), Point2D(0, 0)), ValueError),
        (lambda: LinePrimitive(Point2D(0, 0), "bad"), TypeError),
        (lambda: PolylinePrimitive((Point2D(0, 0),)), ValueError),
        (lambda: CirclePrimitive(Point2D(0, 0), 0), ValueError),
        (lambda: EllipsePrimitive(Point2D(0, 0), 1.0, -1.0), ValueError),
        (lambda: ClosedShapePrimitive((Point2D(0, 0), Point2D(1, 0)), closed=True), ValueError),
        (
            lambda: ClosedShapePrimitive((Point2D(0, 0), Point2D(1, 0), Point2D(1, 1)), closed=False),
            ValueError,
        ),
        (lambda: LinePrimitive(Point2D(0, 0), Point2D(1, 0), confidence=1.5), ValueError),
        (lambda: LinePrimitive(Point2D(0, 0), Point2D(1, 0), error=-0.01), ValueError),
        (lambda: LinePrimitive(Point2D(0, 0), Point2D(1, 0), metadata={1: "bad"}), TypeError),
        (lambda: PrimitiveGroup(("not-a-primitive",)), TypeError),
    ],
)
def test_invalid_values_are_rejected(factory, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        factory()


def test_grouping_primitives_and_flattening_nested_groups() -> None:
    p0, p1, p2, *_ = _points()
    line = LinePrimitive(p0, p1)
    circle = CirclePrimitive(p2, 1.25)
    nested = PrimitiveGroup((PrimitiveGroup((line,), name="inner"), circle), name="outer")

    assert nested.flatten() == (line, circle)
    assert nested.to_dict()["type"] == "group"
    assert nested.to_dict()["items"][0]["items"][0]["type"] == "line"
    assert nested.to_dict()["items"][1]["type"] == "circle"


def test_semantic_geometry_has_no_tikz_generation_methods() -> None:
    line = LinePrimitive(Point2D(0, 0), Point2D(1, 0))
    serialized = line.to_dict()

    assert not hasattr(line, "to_tikz")
    assert "\\draw" not in repr(serialized)
    assert "tikzpicture" not in repr(serialized).lower()


def test_semantic_geometry_source_does_not_import_gui_or_image_processing_backends() -> None:
    source = Path(semantic_geometry.__file__).read_text(encoding="utf-8")

    assert "PySide6" not in source
    assert "fikzpy.gui" not in source
    assert "cv2" not in source
    assert "opencv" not in source.lower()
    assert "compile_latex" not in source


def test_importing_semantic_geometry_does_not_start_gui_or_application() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.semantic_geometry; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
