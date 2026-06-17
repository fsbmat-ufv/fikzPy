from __future__ import annotations

import json
from math import cos, pi, sin
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import fikzpy.core.primitive_fitting as primitive_fitting
from fikzpy.core.centerline_pipeline import CenterlinePath, PathClosureType
from fikzpy.core.geometry_error import angular_coverage, circle_radial_errors
from fikzpy.core.geometry_error import ellipse_distance_errors, point_distances_to_line
from fikzpy.core.primitive_fitting import PrimitiveFitKind, PrimitiveFittingConfig
from fikzpy.core.primitive_fitting import PrimitiveFittingError, fit_primitive, fit_primitives
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PolylinePrimitive, PrimitiveGroup, RGBColor, StrokeStyle
from fikzpy.core.svg_semantic_parser import parse_svg_to_primitives
from fikzpy.core.vectorization_config import config_for_mode


def _point(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y))


def _line_points(start: tuple[float, float], end: tuple[float, float], count: int = 24) -> list[Point2D]:
    xs = np.linspace(start[0], end[0], count)
    ys = np.linspace(start[1], end[1], count)
    return [_point(x, y) for x, y in zip(xs, ys, strict=True)]


def _circle_points(
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 10.0,
    count: int = 72,
    start: float = 0.0,
    stop: float = 2 * pi,
    endpoint: bool = False,
    noise: float = 0.0,
) -> list[Point2D]:
    angles = np.linspace(start, stop, count, endpoint=endpoint)
    points = []
    for index, angle in enumerate(angles):
        local_radius = radius + noise * sin(index * 2.3)
        points.append(_point(center[0] + local_radius * cos(angle), center[1] + local_radius * sin(angle)))
    return points


def _ellipse_points(
    *,
    center: tuple[float, float] = (2.0, -1.0),
    radius_x: float = 7.0,
    radius_y: float = 3.0,
    rotation: float = 0.0,
    count: int = 80,
    start: float = 0.0,
    stop: float = 2 * pi,
    endpoint: bool = False,
    noise: float = 0.0,
) -> list[Point2D]:
    angles = np.linspace(start, stop, count, endpoint=endpoint)
    points = []
    for index, angle in enumerate(angles):
        x = (radius_x + noise * sin(index * 1.7)) * cos(angle)
        y = (radius_y + noise * cos(index * 1.3)) * sin(angle)
        points.append(_point(center[0] + x * cos(rotation) - y * sin(rotation), center[1] + x * sin(rotation) + y * cos(rotation)))
    return points


def _quadratic_points(count: int = 32) -> list[Point2D]:
    xs = np.linspace(0.0, 1.0, count)
    return [_point(x, x * x) for x in xs]


def _cubic_points(count: int = 36) -> list[Point2D]:
    xs = np.linspace(-1.0, 1.0, count)
    return [_point(x, x * x * x - 0.2 * x) for x in xs]


def _sine_points(count: int = 64) -> list[Point2D]:
    xs = np.linspace(0.0, 2.0 * pi, count)
    return [_point(x, sin(x)) for x in xs]


def _fit(points, **config_kwargs):
    return fit_primitive(points, PrimitiveFittingConfig(**config_kwargs))


def _svg(body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">{body}</svg>'


def test_point_isolated_and_near_point_cloud_fit_to_point() -> None:
    isolated = fit_primitive([_point(2, 3)])
    cloud = fit_primitive([_point(1, 1), _point(1.0002, 0.9999), _point(0.9998, 1.0001)])

    assert isolated.selected_kind is PrimitiveFitKind.POINT
    assert cloud.selected_kind is PrimitiveFitKind.POINT
    assert cloud.metrics.point_fits == 1


@pytest.mark.parametrize(
    ("start", "end"),
    [((0, 4), (12, 4)), ((3, -2), (3, 9)), ((-2, -1), (8, 7))],
)
def test_horizontal_vertical_and_diagonal_lines_fit_to_line(start: tuple[int, int], end: tuple[int, int]) -> None:
    result = fit_primitive(_line_points(start, end))

    assert result.selected_kind is PrimitiveFitKind.LINE
    assert isinstance(result.primitives[0], LinePrimitive)
    assert result.normalized_error <= 1e-9


def test_line_with_small_noise_still_fits_line() -> None:
    xs = np.linspace(0.0, 10.0, 32)
    points = [_point(x, 0.02 * sin(index)) for index, x in enumerate(xs)]

    result = fit_primitive(points)

    assert result.selected_kind is PrimitiveFitKind.LINE
    assert result.metrics.line_fits == 1


def test_line_with_noise_above_tolerance_is_not_classified_as_line() -> None:
    xs = np.linspace(0.0, 10.0, 28)
    points = [_point(x, 0.4 * sin(2.0 * x)) for x in xs]

    result = fit_primitive(points, PrimitiveFittingConfig(line_error_tolerance=0.005, bezier_error_tolerance=0.005))

    assert result.selected_kind is not PrimitiveFitKind.LINE
    assert any(candidate.kind is PrimitiveFitKind.LINE and not candidate.accepted for candidate in result.candidates)


def test_polyline_with_one_corner_and_many_corners_preserves_corners() -> None:
    one_corner = [_point(0, 0), _point(5, 0), _point(5, 4)]
    many_corners = [_point(0, 0), _point(4, 0), _point(4, 3), _point(1, 3), _point(1, 1)]

    first = fit_primitive(one_corner)
    second = fit_primitive(many_corners)

    assert first.selected_kind is PrimitiveFitKind.POLYLINE
    assert second.selected_kind is PrimitiveFitKind.POLYLINE
    assert first.metadata["corners"]
    assert len(second.metadata["corners"]) >= 3


def test_perfect_and_noisy_circles_fit_to_circle() -> None:
    perfect = fit_primitive(ClosedShapePrimitive(_circle_points()))
    noisy = fit_primitive(ClosedShapePrimitive(_circle_points(noise=0.03)))

    assert perfect.selected_kind is PrimitiveFitKind.CIRCLE
    assert noisy.selected_kind is PrimitiveFitKind.CIRCLE
    assert isinstance(perfect.primitives[0], CirclePrimitive)


def test_incomplete_or_insufficient_coverage_circle_is_not_full_circle() -> None:
    half_arc = _circle_points(count=32, start=0.0, stop=pi, endpoint=True)
    short_arc = _circle_points(count=16, start=0.0, stop=pi / 5.0, endpoint=True)

    half = fit_primitive(half_arc)
    short = fit_primitive(short_arc)

    assert half.selected_kind is not PrimitiveFitKind.CIRCLE
    assert short.selected_kind is not PrimitiveFitKind.CIRCLE


def test_perfect_rotated_and_noisy_ellipses_fit_to_ellipse() -> None:
    perfect = fit_primitive(ClosedShapePrimitive(_ellipse_points()))
    rotated = fit_primitive(ClosedShapePrimitive(_ellipse_points(rotation=0.6)))
    noisy = fit_primitive(ClosedShapePrimitive(_ellipse_points(rotation=0.25, noise=0.025)))

    assert perfect.selected_kind is PrimitiveFitKind.ELLIPSE
    assert rotated.selected_kind is PrimitiveFitKind.ELLIPSE
    assert noisy.selected_kind is PrimitiveFitKind.ELLIPSE
    assert isinstance(rotated.primitives[0], EllipsePrimitive)


def test_circle_is_preferred_to_unnecessary_ellipse() -> None:
    result = fit_primitive(ClosedShapePrimitive(_circle_points(radius=8.0)))

    assert result.selected_kind is PrimitiveFitKind.CIRCLE
    assert any(candidate.kind is PrimitiveFitKind.ELLIPSE for candidate in result.candidates)


def test_ellipse_too_elongated_and_insufficient_coverage_are_rejected() -> None:
    elongated = fit_primitive(ClosedShapePrimitive(_ellipse_points(radius_x=100.0, radius_y=1.0)))
    partial = fit_primitive(_ellipse_points(start=0.0, stop=pi, endpoint=True))

    assert elongated.selected_kind is not PrimitiveFitKind.ELLIPSE
    assert partial.selected_kind is not PrimitiveFitKind.ELLIPSE


@pytest.mark.parametrize("points", [_quadratic_points(), _cubic_points(), _sine_points()])
def test_quadratic_cubic_s_and_sine_curves_fit_to_few_beziers(points: list[Point2D]) -> None:
    result = fit_primitive(
        points,
        PrimitiveFittingConfig(
            line_error_tolerance=0.001,
            polyline_error_tolerance=0.001,
            bezier_error_tolerance=0.03,
            maximum_bezier_segments=10,
        ),
    )

    assert result.selected_kind is PrimitiveFitKind.BEZIER
    assert 1 <= result.metrics.bezier_segment_count <= 4


def test_long_curve_uses_few_beziers_not_one_per_segment() -> None:
    points = _sine_points(80)

    result = fit_primitive(points, PrimitiveFittingConfig(line_error_tolerance=0.001, polyline_error_tolerance=0.001, bezier_error_tolerance=0.03))

    assert result.selected_kind is PrimitiveFitKind.BEZIER
    assert result.metrics.bezier_segment_count < len(points) // 8


def test_increasing_tolerance_reduces_bezier_count_and_strict_tolerance_improves_fidelity() -> None:
    points = _sine_points(64)
    strict = fit_primitive(points, PrimitiveFittingConfig(line_error_tolerance=0.001, polyline_error_tolerance=0.001, bezier_error_tolerance=0.01, maximum_bezier_segments=10))
    loose = fit_primitive(points, PrimitiveFittingConfig(line_error_tolerance=0.001, polyline_error_tolerance=0.001, bezier_error_tolerance=0.15, maximum_bezier_segments=10))

    assert loose.metrics.bezier_segment_count <= strict.metrics.bezier_segment_count
    assert strict.normalized_error <= loose.normalized_error + 1e-9


def test_degenerate_bezier_is_rejected_and_line_is_not_converted_to_bezier() -> None:
    tiny = [_point(0, 0), _point(0.01, 0.002), _point(0.02, 0.0), _point(0.03, 0.001)]
    degenerate = fit_primitive(tiny, PrimitiveFittingConfig(point_error_tolerance=1e-6, minimum_line_length=0.1, line_error_tolerance=1e-6))
    line = fit_primitive(_line_points((0, 0), (10, 0)))

    assert degenerate.selected_kind is not PrimitiveFitKind.BEZIER
    assert line.selected_kind is PrimitiveFitKind.LINE
    assert line.metrics.bezier_segment_count == 0


def test_closed_irregular_path_falls_back_without_open_gap() -> None:
    points = [_point(0, 0), _point(4, 0), _point(5, 2), _point(3, 5), _point(0, 4)]

    result = fit_primitive(ClosedShapePrimitive(points))

    assert result.selected_kind in {PrimitiveFitKind.POLYLINE, PrimitiveFitKind.CLOSED_FREEFORM}
    assert result.primitives[0].to_dict().get("closed") is True


def test_closed_circular_and_closed_elliptical_paths_use_simple_shapes() -> None:
    circle = fit_primitive(ClosedShapePrimitive(_circle_points(radius=5.0)))
    ellipse = fit_primitive(ClosedShapePrimitive(_ellipse_points(radius_x=8.0, radius_y=2.0, rotation=0.4)))

    assert circle.selected_kind is PrimitiveFitKind.CIRCLE
    assert ellipse.selected_kind is PrimitiveFitKind.ELLIPSE


def test_shape_with_corner_and_curve_preserves_corner_as_polyline() -> None:
    xs = np.linspace(0.0, 1.0, 18)
    points = [_point(0, 0), _point(2, 0), *[_point(2 + x * x, x) for x in xs[1:]]]

    result = fit_primitive(points)

    assert result.selected_kind is PrimitiveFitKind.POLYLINE
    assert result.metadata["corners"]


def test_multiple_subpaths_and_primitive_group_are_supported() -> None:
    parsed = parse_svg_to_primitives(_svg('<path d="M0 0 L10 0 M20 0 L30 0" stroke="black" fill="none"/>'))
    group = PrimitiveGroup((PolylinePrimitive(_line_points((0, 0), (5, 0))), PolylinePrimitive(_line_points((0, 2), (5, 2)))))

    parsed_result = fit_primitive(parsed)
    group_result = fit_primitive(group)

    assert parsed_result.selected_kind is PrimitiveFitKind.GROUP
    assert group_result.selected_kind is PrimitiveFitKind.GROUP
    assert parsed_result.output_primitive_count == 2


def test_style_fill_and_metadata_are_preserved() -> None:
    stroke = StrokeStyle(color=RGBColor(10, 20, 30), width=2.5, opacity=0.8, line_cap="round")
    fill = FillStyle(RGBColor(200, 100, 50), opacity=0.4)
    line = PolylinePrimitive(_line_points((0, 0), (10, 0)), stroke=stroke, metadata={"id": "guide", "tracer": "synthetic"})
    circle = ClosedShapePrimitive(_circle_points(radius=4.0), stroke=stroke, fill=fill, metadata={"svg_id": "c1"})

    line_result = fit_primitive(line)
    circle_result = fit_primitive(circle)

    assert line_result.primitives[0].stroke == stroke
    assert line_result.primitives[0].metadata["id"] == "guide"
    assert line_result.primitives[0].metadata["tracer"] == "synthetic"
    assert circle_result.primitives[0].fill == fill
    assert circle_result.primitives[0].metadata["svg_id"] == "c1"


def test_centerline_path_and_polyline_and_closed_shape_inputs_are_supported() -> None:
    centerline = CenterlinePath(
        id="p0",
        points=tuple(_line_points((0, 0), (8, 0))),
        start_node_id="n0",
        end_node_id="n1",
        closure=PathClosureType.OPEN,
        component_id=0,
        metadata={"centerline": True},
    )
    polyline = PolylinePrimitive(_line_points((0, 0), (0, 8)))
    closed = ClosedShapePrimitive(_circle_points(radius=3.0))

    assert fit_primitive(centerline).selected_kind is PrimitiveFitKind.LINE
    assert fit_primitive(polyline).selected_kind is PrimitiveFitKind.LINE
    assert fit_primitive(closed).selected_kind is PrimitiveFitKind.CIRCLE


def test_existing_semantic_primitives_are_preserved_by_default() -> None:
    line = LinePrimitive(_point(0, 0), _point(1, 0))
    circle = CirclePrimitive(_point(0, 0), 2)
    ellipse = EllipsePrimitive(_point(0, 0), 3, 1)
    bezier = BezierPrimitive(_point(0, 0), _point(1, 0), _point(2, 0), _point(3, 0))

    for primitive in (line, circle, ellipse, bezier):
        result = fit_primitive(primitive)
        assert result.selected_kind is PrimitiveFitKind.EXISTING
        assert result.primitives[0] is primitive


def test_existing_bezier_can_be_replaced_only_when_configured() -> None:
    bezier = BezierPrimitive(_point(0, 0), _point(1, 0), _point(2, 0), _point(3, 0))

    preserved = fit_primitive(bezier)
    replaced = fit_primitive(bezier, PrimitiveFittingConfig(allow_primitive_replacement=True))

    assert preserved.selected_kind is PrimitiveFitKind.EXISTING
    assert replaced.selected_kind is PrimitiveFitKind.LINE


@pytest.mark.parametrize(
    "bad_input",
    [
        object(),
        [],
        [_point(0, 0), (float("nan"), 1.0)],
        [_point(0, 0), (float("inf"), 1.0)],
    ],
)
def test_invalid_nan_infinite_and_empty_inputs_are_rejected(bad_input) -> None:
    with pytest.raises((TypeError, ValueError, PrimitiveFittingError)):
        fit_primitive(bad_input)


def test_duplicate_points_and_zero_length_path_are_handled() -> None:
    duplicated = fit_primitive([_point(0, 0), _point(0, 0), _point(5, 0)])
    zero = fit_primitive([_point(2, 2), _point(2, 2), _point(2, 2)])

    assert duplicated.selected_kind is PrimitiveFitKind.LINE
    assert duplicated.metadata["duplicate_count"] >= 1
    assert zero.selected_kind is PrimitiveFitKind.POINT


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PrimitiveFittingConfig(line_error_tolerance=-1),
        lambda: PrimitiveFittingConfig(minimum_points_for_circle=0),
        lambda: PrimitiveFittingConfig(maximum_axis_ratio=0.5),
        lambda: PrimitiveFittingConfig(ambiguity_margin=1.5),
        lambda: PrimitiveFittingConfig(maximum_bezier_segments=0),
    ],
)
def test_invalid_configuration_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_ambiguous_result_alternative_confidence_and_to_dict_are_recorded() -> None:
    result = fit_primitive(ClosedShapePrimitive(_circle_points(radius=6.0)))
    data = result.to_dict()

    assert result.selected_kind is PrimitiveFitKind.CIRCLE
    assert result.ambiguous
    assert result.alternative_kind is PrimitiveFitKind.ELLIPSE
    assert 0.0 <= result.confidence <= 1.0
    assert data["selected_kind"] == "circle"
    assert "array(" not in repr(data)


def test_result_is_deterministic() -> None:
    points = _sine_points(48)
    config = PrimitiveFittingConfig(line_error_tolerance=0.001, polyline_error_tolerance=0.001, bezier_error_tolerance=0.03)

    first = fit_primitive(points, config).to_dict()
    second = fit_primitive(points, config).to_dict()

    assert first == second


def test_fit_primitives_api_returns_independent_results() -> None:
    results = fit_primitives([_line_points((0, 0), (5, 0)), ClosedShapePrimitive(_circle_points(radius=3.0))])

    assert [result.selected_kind for result in results] == [PrimitiveFitKind.LINE, PrimitiveFitKind.CIRCLE]


def test_geometry_error_helpers_are_consistent() -> None:
    line_points = _line_points((0, 0), (10, 0))
    circle_points = _circle_points(radius=5.0)
    ellipse_points = _ellipse_points(radius_x=6.0, radius_y=2.0)

    assert point_distances_to_line(line_points, _point(0, 0), _point(10, 0)).max() == pytest.approx(0.0)
    assert np.abs(circle_radial_errors(circle_points, _point(0, 0), 5.0)).max() < 1e-9
    assert np.abs(ellipse_distance_errors(ellipse_points, _point(2, -1), 6.0, 2.0, 0.0)).max() < 1e-9
    assert angular_coverage(circle_points, _point(0, 0)) > 0.95


def test_importing_primitive_fitting_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.primitive_fitting; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_primitive_fitting_does_not_generate_tikz_call_svg2tikz_or_execute_tracers() -> None:
    sources = [
        Path(primitive_fitting.__file__).read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/geometry_error.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/bezier_fitting.py").read_text(encoding="utf-8").lower(),
    ]

    for source in sources:
        assert "\\\\draw" not in source
        assert "tikzpicture" not in source
        assert "svg2tikz" not in source
        assert "trace_image(" not in source
        assert "subprocess" not in source


def test_visual_and_svg_parser_are_preserved_and_not_connected_to_classic() -> None:
    assert config_for_mode("classic").mode == "classic"
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("contours").mode == "contours"

    parsed = parse_svg_to_primitives(_svg('<line x1="0" y1="0" x2="1" y2="0"/>'))
    assert parsed.primitives[0].to_dict()["type"] == "line"

    for path in ("fikzpy/core/visual_pipeline.py", "fikzpy/core/visual_postprocessor.py", "fikzpy/core/tikz_pipeline.py", "fikzpy/core/image_processor.py"):
        assert "primitive_fitting" not in Path(path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("svg_kind", "body", "expected_kind"),
    [
        ("potrace", "potrace_circle", PrimitiveFitKind.CIRCLE),
        ("autotrace", '<path stroke="black" fill="none" d="M0 0 C3 2 7 2 10 0"/>', PrimitiveFitKind.GROUP),
        ("vtracer", '<g opacity=".8"><path fill="red" d="M0 0 L10 0 L10 10 Z"/><path fill="blue" d="M20 0 L30 0 L30 10 Z"/></g>', PrimitiveFitKind.GROUP),
    ],
)
def test_synthetic_tracer_svg_outputs_can_be_fitted(svg_kind: str, body: str, expected_kind: PrimitiveFitKind) -> None:
    if body == "potrace_circle":
        point_text = " ".join(f"{point.x:.6g},{point.y:.6g}" for point in _circle_points(radius=10.0, count=32))
        body = f'<polygon fill="black" points="{point_text}"/>'
    parsed = parse_svg_to_primitives(_svg(body))
    result = fit_primitive(parsed)

    assert result.selected_kind is PrimitiveFitKind.GROUP
    assert result.output_primitive_count >= 1
    if expected_kind is PrimitiveFitKind.CIRCLE:
        assert result.primitives[0].flatten()[0].to_dict()["type"] == "circle"
    assert svg_kind in {"potrace", "autotrace", "vtracer"}


def test_error_limits_prevent_bad_circle_and_bad_ellipse_recognition() -> None:
    bad_circle = ClosedShapePrimitive(_circle_points(radius=8.0, noise=2.0))
    bad_ellipse = _ellipse_points(start=0.0, stop=pi / 2.0, endpoint=True)

    assert fit_primitive(bad_circle).selected_kind is not PrimitiveFitKind.CIRCLE
    assert fit_primitive(bad_ellipse).selected_kind is not PrimitiveFitKind.ELLIPSE


def test_rectangle_corners_and_small_teeth_are_preserved() -> None:
    rectangle = ClosedShapePrimitive([_point(0, 0), _point(4, 0), _point(4, 3), _point(0, 3)])
    teeth = [_point(0, 0), _point(1, 0), _point(1.2, 0.25), _point(1.4, 0), _point(2, 0), _point(2, 1), _point(0, 1)]

    rect_result = fit_primitive(rectangle)
    teeth_result = fit_primitive(ClosedShapePrimitive(teeth), PrimitiveFittingConfig(polyline_error_tolerance=0.0001))

    assert rect_result.selected_kind is PrimitiveFitKind.POLYLINE
    assert len(rect_result.primitives[0].points) == 4
    assert teeth_result.selected_kind is PrimitiveFitKind.POLYLINE
    assert len(teeth_result.primitives[0].points) >= len(teeth)


def test_style_is_copied_to_multiple_beziers() -> None:
    stroke = StrokeStyle(color=RGBColor(90, 10, 200), width=1.7)
    polyline = PolylinePrimitive(_sine_points(64), stroke=stroke, metadata={"source": "style-copy"})

    result = fit_primitive(polyline, PrimitiveFittingConfig(line_error_tolerance=0.001, polyline_error_tolerance=0.001, bezier_error_tolerance=0.03, maximum_bezier_segments=10))

    assert result.selected_kind is PrimitiveFitKind.BEZIER
    assert result.metrics.bezier_segment_count > 1
    assert all(isinstance(item, BezierPrimitive) and item.stroke == stroke for item in result.primitives)
    assert all(item.metadata["source"] == "style-copy" for item in result.primitives)


def test_baseline_reports_are_read_only_sources_for_synthetic_regression_cases() -> None:
    baseline = Path("examples/classic_semantic_baseline")
    svg_report = json.loads((baseline / "svg_parser_report.json").read_text(encoding="utf-8"))
    centerline_report = json.loads((baseline / "centerline_report.json").read_text(encoding="utf-8"))

    diagnostics = {
        "line_art_bw": fit_primitive(_line_points((0, 0), (10, 0))).selected_kind.value,
        "geometric_diagram": fit_primitive(ClosedShapePrimitive(_circle_points(radius=5.0))).selected_kind.value,
        "noisy_grayscale": fit_primitive(_line_points((0, 0), (6, 3))).selected_kind.value,
    }

    assert svg_report
    assert centerline_report
    assert diagnostics == {"line_art_bw": "line", "geometric_diagram": "circle", "noisy_grayscale": "line"}
