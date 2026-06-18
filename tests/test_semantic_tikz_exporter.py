from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest

from fikzpy.core.centerline_pipeline import extract_centerlines
from fikzpy.core.geometry_optimization import optimize_fit_results, optimize_primitives
from fikzpy.core.primitive_fitting import fit_primitive, fit_primitives
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PointPrimitive, PolylinePrimitive, PrimitiveGroup, RGBColor, StrokeStyle
from fikzpy.core.semantic_tikz_exporter import SemanticTikzExporter, TikzCodeOutputMode
from fikzpy.core.semantic_tikz_exporter import TikzExportConfig, TikzExportError, export_primitives_to_tikz
from fikzpy.core.svg_semantic_parser import parse_svg_to_primitives
from fikzpy.core.vectorization_config import config_for_mode


def p(x: float, y: float) -> Point2D:
    return Point2D(x, y)


def export(items, **kwargs):
    return export_primitives_to_tikz(items, TikzExportConfig(**kwargs))


def red_stroke(width: float = 1.0) -> StrokeStyle:
    return StrokeStyle(RGBColor(255, 0, 0), width=width)


def blue_fill(opacity: float | None = None) -> FillStyle:
    return FillStyle(RGBColor(0, 0, 255), opacity=opacity)


def test_empty_input_exports_empty_body() -> None:
    result = export([])
    assert result.code == ""
    assert result.metrics.input_primitive_count == 0
    assert result.deterministic_hash


def test_point_primitive_uses_small_circle_marker() -> None:
    result = export([PointPrimitive(p(1, 2))])
    assert result.code == r"\draw (1,2) circle[radius=0.5pt];"
    assert result.circle_count == 1


def test_line_primitive_uses_line_syntax() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0))])
    assert result.code == r"\draw (0,0) -- (1,0);"
    assert ".. controls" not in result.code


def test_short_polyline_stays_on_one_line() -> None:
    result = export([PolylinePrimitive((p(0, 0), p(1, 0), p(1, 1)))])
    assert result.code == r"\draw (0,0) -- (1,0) -- (1,1);"


def test_long_polyline_breaks_consistently() -> None:
    primitive = PolylinePrimitive(tuple(p(index, 0) for index in range(6)))
    result = export([primitive], max_points_per_line=2)
    assert result.code.splitlines()[0] == r"\draw"
    assert "  -- (5,0);" in result.code


def test_circle_uses_circle_syntax_not_polyline() -> None:
    result = export([CirclePrimitive(p(2, 3), 4)])
    assert "circle[radius=4]" in result.code
    assert "--" not in result.code


def test_ellipse_without_rotation_uses_ellipse_syntax() -> None:
    result = export([EllipsePrimitive(p(0, 0), 2, 1)])
    assert "ellipse[x radius=2, y radius=1]" in result.code
    assert ".. controls" not in result.code


def test_rotated_ellipse_uses_rotate_around_option() -> None:
    result = export([EllipsePrimitive(p(0, 0), 2, 1, rotation=30)])
    assert r"\draw[rotate around={30:(0,0)}]" in result.code
    assert "ellipse" in result.code


def test_bezier_uses_tikz_controls_syntax() -> None:
    primitive = BezierPrimitive(p(0, 0), p(1, 2), p(2, 2), p(3, 0))
    result = export([primitive])
    assert r".. controls (1,2) and (2,2) .. (3,0)" in result.code


def test_contiguous_bezier_sequence_can_share_one_draw_command() -> None:
    first = BezierPrimitive(p(0, 0), p(1, 1), p(2, 1), p(3, 0))
    second = BezierPrimitive(p(3, 0), p(4, -1), p(5, -1), p(6, 0))
    result = export([first, second])
    assert result.draw_count == 1
    assert result.code.count(".. controls") == 2
    assert result.metrics.bezier_segments_written == 2


def test_closed_shape_corrupted_open_flag_warns_but_uses_cycle() -> None:
    shape = _unsafe_closed_shape(closed=False)
    result = export([shape])
    assert "-- cycle" in result.code
    assert [warning.code for warning in result.warnings] == ["open_closed_shape"]


def test_closed_shape_uses_cycle_without_redundant_repeated_point() -> None:
    shape = ClosedShapePrimitive((p(0, 0), p(1, 0), p(1, 1), p(0, 0)))
    result = export([shape])
    assert result.code == r"\draw (0,0) -- (1,0) -- (1,1) -- cycle;"
    assert result.code.count("(0,0)") == 1


def test_filled_closed_shape_preserves_fill() -> None:
    shape = ClosedShapePrimitive((p(0, 0), p(1, 0), p(1, 1)), fill=blue_fill())
    result = export([shape])
    assert "fill={rgb,255:red,0;green,0;blue,255}" in result.code
    assert result.fill_count == 1


def test_fill_none_is_not_emitted() -> None:
    result = export([CirclePrimitive(p(0, 0), 1, fill=None)])
    assert "fill=" not in result.code


def test_draw_none_with_fill_is_exported_safely() -> None:
    stroke = StrokeStyle(opacity=0.0)
    shape = ClosedShapePrimitive((p(0, 0), p(1, 0), p(0, 1)), stroke=stroke, fill=blue_fill())
    result = export([shape])
    assert "draw=none" in result.code
    assert "fill={rgb,255:red,0;green,0;blue,255}" in result.code


def test_invisible_primitive_is_skipped_with_warning() -> None:
    primitive = LinePrimitive(p(0, 0), p(1, 0), stroke=StrokeStyle(opacity=0.0))
    result = export([primitive])
    assert result.code == ""
    assert result.metrics.skipped_primitive_count == 1
    assert result.warnings[0].code == "invisible_primitive"


def test_default_black_stroke_is_omitted_by_default() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0))])
    assert result.code.startswith(r"\draw (")
    assert "draw=black" not in result.code
    assert "{rgb" not in result.code


def test_colored_stroke_is_emitted() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0), stroke=red_stroke())])
    assert r"\draw[draw={rgb,255:red,255;green,0;blue,0}]" in result.code


def test_colored_fill_is_emitted() -> None:
    result = export([CirclePrimitive(p(0, 0), 1, fill=blue_fill())])
    assert "fill={rgb,255:red,0;green,0;blue,255}" in result.code


def test_overall_opacity_fill_opacity_and_stroke_opacity_are_emitted() -> None:
    result = export(
        [
            CirclePrimitive(p(0, 0), 1, opacity=0.5),
            CirclePrimitive(p(2, 0), 1, fill=blue_fill(0.25)),
            LinePrimitive(p(0, 1), p(1, 1), stroke=StrokeStyle(opacity=0.75)),
        ]
    )
    assert "opacity=0.5" in result.code
    assert "fill opacity=0.25" in result.code
    assert "draw opacity=0.75" in result.code


def test_line_width_cap_join_and_dash_pattern_are_emitted() -> None:
    stroke = StrokeStyle(width=0.4, line_cap="round", line_join="bevel", dash_pattern=(2, 1))
    result = export([LinePrimitive(p(0, 0), p(1, 0), stroke=stroke)])
    assert "line width=0.4pt" in result.code
    assert "line cap=round" in result.code
    assert "line join=bevel" in result.code
    assert "dash pattern=on 2pt off 1pt" in result.code


def test_primitive_group_is_preserved_as_scope() -> None:
    group = PrimitiveGroup((LinePrimitive(p(0, 0), p(1, 0)),), name="axis", metadata={"id": "g1"})
    result = export([group])
    assert r"\begin{scope}[axis]" in result.code
    assert r"\end{scope}" in result.code


def test_primitive_group_can_be_flattened() -> None:
    group = PrimitiveGroup((LinePrimitive(p(0, 0), p(1, 0)),), name="axis")
    result = export([group], preserve_groups=False)
    assert r"\begin{scope}" not in result.code
    assert result.code == r"\draw (0,0) -- (1,0);"


def test_empty_group_warns() -> None:
    result = export([PrimitiveGroup((), name="empty")])
    assert result.code == ""
    assert result.warnings[0].code == "empty_group"


def test_metadata_comments_are_optional_and_escaped() -> None:
    primitive = LinePrimitive(p(0, 0), p(1, 0), metadata={"note": "50% cafe \u00e1"})
    with_comments = export([primitive], emit_comments=True, include_metadata_comments=True)
    without_comments = export([primitive], emit_comments=False, include_metadata_comments=True)
    assert "% primitive: line" in with_comments.code
    assert r"50\% cafe \u00e1" in with_comments.code
    assert "%" not in without_comments.code


def test_tikzpicture_and_scope_wrappers() -> None:
    result = export(
        [LinePrimitive(p(0, 0), p(1, 0))],
        include_tikzpicture_environment=True,
        include_scope_environment=True,
    )
    assert result.code.splitlines()[0] == r"\begin{tikzpicture}[scale=1]"
    assert r"\begin{scope}[line cap=round, line join=round]" in result.code
    assert result.code.splitlines()[-1] == r"\end{tikzpicture}"


def test_figonly_output_mode_has_only_body() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0))], code_output_mode=TikzCodeOutputMode.FIG_ONLY)
    assert r"\begin{tikzpicture}" not in result.code
    assert result.code.startswith(r"\draw")


def test_named_styles_are_defined_only_for_reused_styles() -> None:
    primitives = [
        LinePrimitive(p(0, 0), p(1, 0), stroke=red_stroke(0.4)),
        LinePrimitive(p(0, 1), p(1, 1), stroke=red_stroke(0.4)),
    ]
    result = export(primitives, define_common_styles=True)
    assert r"\tikzset{" in result.code
    assert "fikzStyle0/.style=" in result.code
    assert result.code.count(r"\draw[fikzStyle0]") == 2


def test_style_grouping_uses_scope_for_consecutive_compatible_styles() -> None:
    primitives = [
        LinePrimitive(p(0, 0), p(1, 0), stroke=red_stroke()),
        LinePrimitive(p(0, 1), p(1, 1), stroke=red_stroke()),
    ]
    result = export(primitives, group_styles=True)
    assert r"\begin{scope}[draw={rgb,255:red,255;green,0;blue,0}]" in result.code
    assert result.metrics.groups_written == 1


def test_omit_default_styles_can_be_disabled() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0))], omit_default_styles=False)
    assert "draw={rgb,255:red,0;green,0;blue,0}" in result.code
    assert "line width=1pt" in result.code


def test_coordinate_rounding_and_negative_zero_cleanup() -> None:
    result = export([PointPrimitive(p(-0.0001, 1.23456))], coordinate_precision=2)
    assert "(0,1.23)" in result.code


def test_scale_y_inversion_origin_and_units() -> None:
    scaled = export([LinePrimitive(p(1, 1), p(2, 2))], scale=2)
    inverted = export([PointPrimitive(p(2, 3))], invert_y_axis=True, image_height=10)
    origin = export([PointPrimitive(p(12, 23))], coordinate_origin=(10, 20))
    units = export([PointPrimitive(p(1, 2))], unit="cm")
    assert "(2,2) -- (4,4)" in scaled.code
    assert "(2,7)" in inverted.code
    assert "(2,3)" in origin.code
    assert "(1cm,2cm)" in units.code


def test_y_inversion_without_image_height_warns_and_keeps_coordinates() -> None:
    result = export([PointPrimitive(p(2, 3))], invert_y_axis=True)
    assert "(2,3)" in result.code
    assert result.warnings[0].code == "y_axis_image_height_missing"


def test_result_to_dict_hash_and_determinism() -> None:
    primitive = CirclePrimitive(p(0, 0), 1, stroke=red_stroke())
    first = export([primitive])
    second = export([primitive])
    assert first.to_dict()["deterministic_hash"] == first.deterministic_hash
    assert first.deterministic_hash == second.deterministic_hash
    assert first.code == second.code
    assert [warning.to_dict() for warning in first.warnings] == [warning.to_dict() for warning in second.warnings]


def test_strict_and_tolerant_modes_for_invisible_primitive() -> None:
    primitive = LinePrimitive(p(0, 0), p(1, 0), stroke=StrokeStyle(opacity=0.0))
    tolerant = export([primitive], strict=False)
    assert tolerant.warnings
    with pytest.raises(TikzExportError):
        export([primitive], strict=True)


def test_invalid_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        TikzExportConfig(coordinate_precision=-1)


def test_nan_and_infinite_coordinates_are_skipped_in_tolerant_mode_and_raise_in_strict_mode() -> None:
    primitive = _unsafe_point_primitive(_unsafe_point(math.nan, math.inf))
    tolerant = export([primitive], strict=False)
    assert tolerant.code == ""
    assert tolerant.warnings[0].code == "geometry_degenerate"
    with pytest.raises(TikzExportError):
        export([primitive], strict=True)


def test_geometry_optimization_result_is_accepted() -> None:
    optimized = optimize_primitives([PolylinePrimitive((p(0, 0), p(1, 0)), closed=False)])
    result = export(optimized)
    assert result.code == r"\draw (0,0) -- (1,0);"


def test_primitive_fit_result_and_list_are_accepted() -> None:
    fit = fit_primitive((p(0, 0), p(1, 0), p(2, 0)))
    one = export(fit)
    many = export([fit, fit])
    assert r"\draw" in one.code
    assert many.draw_count == 2


def test_svg_parse_result_is_accepted() -> None:
    parsed = parse_svg_to_primitives('<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="2"/></svg>')
    result = export(parsed)
    assert "circle[radius=2]" in result.code


def test_centerline_result_is_converted_to_polylines_without_exporter_tracing() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 1:4] = 255
    centerline = extract_centerlines(mask)
    result = export(centerline)
    assert r"\draw" in result.code
    assert result.metrics.input_primitive_count >= 1


def test_preserves_draw_order() -> None:
    result = export([CirclePrimitive(p(0, 0), 1), LinePrimitive(p(0, 0), p(1, 0))])
    assert result.code.splitlines()[0].startswith(r"\draw (0,0) circle")
    assert result.code.splitlines()[1] == r"\draw (0,0) -- (1,0);"


def test_exporter_source_does_not_call_external_bridges_tracers_gui_or_raw_path_output() -> None:
    source = Path("fikzpy/core/semantic_tikz_exporter.py").read_text(encoding="utf-8").lower()
    assert "svg2tikz" not in source
    assert "trace_image" not in source
    assert "run_cli_tracer" not in source
    assert "qapplication" not in source
    assert "pyqt" not in source
    result = export([LinePrimitive(p(0, 0), p(1, 0))])
    assert r"\path" not in result.code


def test_classic_visual_and_contours_modes_remain_available() -> None:
    assert config_for_mode("classic").mode == "classic"
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("contours").mode == "contours"


def test_code_non_empty_and_minimal_example_shape() -> None:
    result = export([LinePrimitive(p(0, 0), p(1, 0))], include_tikzpicture_environment=True)
    assert result.code
    assert result.code.startswith(r"\begin{tikzpicture}")
    assert r"\draw" in result.code


def test_semantic_shapes_are_not_degraded_to_less_semantic_paths() -> None:
    circle = export([CirclePrimitive(p(0, 0), 1)])
    ellipse = export([EllipsePrimitive(p(0, 0), 2, 1)])
    line = export([LinePrimitive(p(0, 0), p(1, 0))])
    assert "circle" in circle.code and "--" not in circle.code
    assert "ellipse" in ellipse.code and ".. controls" not in ellipse.code
    assert "--" in line.code and ".. controls" not in line.code


def test_closed_path_uses_cycle_and_has_no_obvious_redundant_points() -> None:
    shape = ClosedShapePrimitive((p(0, 0), p(1, 0), p(1, 1), p(0, 0)))
    result = export([shape])
    assert "-- cycle" in result.code
    assert result.code.count("(0,0)") == 1


def test_unknown_type_has_clear_error() -> None:
    with pytest.raises(TypeError, match="Unsupported input"):
        export([object()])


def test_fill_rule_holes_and_subpaths_emit_warnings() -> None:
    shape = ClosedShapePrimitive(
        (p(0, 0), p(2, 0), p(2, 2), p(0, 2)),
        fill=blue_fill(),
        metadata={"resolved_style": {"fill_rule": "mystery"}, "holes": 1, "subpaths": 2},
    )
    result = export([shape])
    assert {warning.code for warning in result.warnings} >= {"fill_rule_partial", "holes_partial", "subpaths_partial"}


@pytest.mark.parametrize(
    ("svg_text", "expected"),
    [
        (
            '<svg viewBox="0 0 10 10"><path d="M0 0 L10 0 L10 10 Z" fill="#ff0000" stroke="none"/></svg>',
            ("cycle", "fill="),
        ),
        (
            '<svg viewBox="0 0 10 10"><polyline points="0,0 5,0 10,0" fill="none" stroke="#000000"/></svg>',
            (r"\draw", "--"),
        ),
        (
            '<svg viewBox="0 0 20 20"><circle cx="5" cy="5" r="3"/><ellipse cx="12" cy="5" rx="4" ry="2"/><path d="M0 12 C3 18 7 18 10 12" fill="none" stroke="#000"/></svg>',
            ("circle", "ellipse", ".. controls"),
        ),
    ],
)
def test_isolated_synthetic_svg_fit_optimize_export_flow(svg_text: str, expected: tuple[str, ...]) -> None:
    parsed = parse_svg_to_primitives(svg_text)
    fitted = fit_primitive(parsed)
    optimized = optimize_fit_results(fitted)
    result = export(optimized)
    assert r"\draw" in result.code
    for text in expected:
        assert text in result.code
    assert r"\path" not in result.code


def test_adjusted_and_optimized_primitives_export_more_semantically_than_raw_path() -> None:
    fitted = fit_primitives(
        [
            CirclePrimitive(p(0, 0), 5),
            EllipsePrimitive(p(10, 0), 4, 2),
            BezierPrimitive(p(0, 10), p(2, 12), p(4, 12), p(6, 10)),
        ]
    )
    optimized = optimize_fit_results(fitted)
    result = export(optimized)
    raw_equivalent = r"\path (0,0) -- (1,0) -- (2,1) -- (3,2) -- (4,3) -- (5,4) -- cycle;"
    assert "circle" in result.code
    assert "ellipse" in result.code
    assert ".. controls" in result.code
    assert len(result.code) < len(raw_equivalent) * 4


def test_importing_exporter_does_not_start_gui() -> None:
    assert isinstance(SemanticTikzExporter().export([LinePrimitive(p(0, 0), p(1, 0))]).code, str)


def _unsafe_point(x: float, y: float) -> Point2D:
    point = object.__new__(Point2D)
    object.__setattr__(point, "x", x)
    object.__setattr__(point, "y", y)
    return point


def _unsafe_point_primitive(point: Point2D) -> PointPrimitive:
    primitive = object.__new__(PointPrimitive)
    object.__setattr__(primitive, "point", point)
    _assign_common(primitive)
    return primitive


def _unsafe_closed_shape(*, closed: bool) -> ClosedShapePrimitive:
    primitive = object.__new__(ClosedShapePrimitive)
    object.__setattr__(primitive, "points", (p(0, 0), p(1, 0), p(0, 1)))
    object.__setattr__(primitive, "closed", closed)
    _assign_common(primitive)
    return primitive


def _assign_common(primitive) -> None:
    object.__setattr__(primitive, "stroke", StrokeStyle())
    object.__setattr__(primitive, "fill", None)
    object.__setattr__(primitive, "opacity", None)
    object.__setattr__(primitive, "confidence", None)
    object.__setattr__(primitive, "error", None)
    object.__setattr__(primitive, "metadata", {})
