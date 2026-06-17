from __future__ import annotations

import json
from math import pi, sin
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import fikzpy.core.geometry_optimization as geometry_optimization
from fikzpy.core.centerline_pipeline import CenterlinePath, PathClosureType
from fikzpy.core.geometry_error import signed_area
from fikzpy.core.geometry_optimization import DuplicateHandling, GeometryOptimizationConfig
from fikzpy.core.geometry_optimization import GeometryOptimizationError, OptimizationOperationKind
from fikzpy.core.geometry_optimization import OptimizationStatus, optimize_fit_results, optimize_primitives
from fikzpy.core.primitive_fitting import fit_primitive
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PointPrimitive, PolylinePrimitive, PrimitiveGroup
from fikzpy.core.semantic_geometry import RGBColor, StrokeStyle
from fikzpy.core.svg_semantic_parser import parse_svg_to_primitives
from fikzpy.core.vectorization_config import config_for_mode


def _point(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y))


def _line(x1: float, y1: float, x2: float, y2: float, **kwargs) -> LinePrimitive:
    return LinePrimitive(_point(x1, y1), _point(x2, y2), **kwargs)


def _poly(points: list[tuple[float, float]], **kwargs) -> PolylinePrimitive:
    return PolylinePrimitive(tuple(_point(x, y) for x, y in points), **kwargs)


def _closed(points: list[tuple[float, float]], **kwargs) -> ClosedShapePrimitive:
    return ClosedShapePrimitive(tuple(_point(x, y) for x, y in points), **kwargs)


def _bezier(offset: float = 0.0, **kwargs) -> BezierPrimitive:
    return BezierPrimitive(
        _point(0 + offset, 0),
        _point(1 + offset, 1),
        _point(2 + offset, 1),
        _point(3 + offset, 0),
        **kwargs,
    )


def _smooth_beziers() -> list[BezierPrimitive]:
    return [
        BezierPrimitive(_point(0, 0), _point(1, 1), _point(2, 1), _point(3, 0)),
        BezierPrimitive(_point(3, 0), _point(4, -1), _point(5, -1), _point(6, 0)),
    ]


def _sine_polyline(count: int = 48) -> PolylinePrimitive:
    xs = np.linspace(0.0, 2.0 * pi, count)
    return PolylinePrimitive(tuple(_point(x, sin(x) * 0.2) for x in xs))


def _svg(body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">{body}</svg>'


def test_empty_list_point_and_valid_line_are_preserved() -> None:
    empty = optimize_primitives([])
    point = optimize_primitives([PointPrimitive(_point(1, 2))])
    line = optimize_primitives([_line(0, 0, 2, 0)])

    assert empty.status is OptimizationStatus.UNCHANGED
    assert empty.metrics.input_primitive_count == 0
    assert isinstance(point.primitives[0], PointPrimitive)
    assert isinstance(line.primitives[0], LinePrimitive)


def test_degenerate_line_can_convert_remove_preserve_or_raise() -> None:
    tiny = _line(0, 0, 1e-9, 0)

    converted = optimize_primitives([tiny])
    removed = optimize_primitives([tiny], GeometryOptimizationConfig(degenerate_handling=DuplicateHandling.REMOVE))
    preserved = optimize_primitives([tiny], GeometryOptimizationConfig(degenerate_handling=DuplicateHandling.PRESERVE_WITH_WARNING))

    assert isinstance(converted.primitives[0], PointPrimitive)
    assert converted.metrics.degenerate_primitives_converted == 1
    assert removed.primitives == ()
    assert preserved.warnings
    with pytest.raises(GeometryOptimizationError):
        optimize_primitives([tiny], GeometryOptimizationConfig(strict=True))


def test_two_and_three_collinear_lines_are_merged_across_passes() -> None:
    result = optimize_primitives([_line(0, 0, 1, 0), _line(1, 0, 2, 0), _line(2, 0, 3, 0)])

    assert result.metrics.output_primitive_count == 1
    assert result.metrics.collinear_lines_merged == 2
    assert isinstance(result.primitives[0], LinePrimitive)
    assert result.metrics.optimization_passes >= 2


def test_lines_with_small_gap_merge_but_excessive_gap_does_not() -> None:
    config = GeometryOptimizationConfig(maximum_endpoint_distance=0.15, collinear_distance_tolerance=0.01)
    small_gap = optimize_primitives([_line(0, 0, 1, 0), _line(1.05, 0, 2, 0)], config)
    large_gap = optimize_primitives([_line(0, 0, 1, 0), _line(1.5, 0, 2, 0)], config)

    assert small_gap.metrics.output_primitive_count == 1
    assert large_gap.metrics.output_primitive_count == 2


@pytest.mark.parametrize(
    "primitives",
    [
        [_line(0, 0, 1, 0), _line(1, 0, 1, 1)],
        [_line(0, 0, 1, 0), _line(0, 1, 1, 1)],
        [_line(0, 0, 2, 0), _line(1, -1, 1, 1)],
    ],
)
def test_line_corners_parallel_and_crossing_lines_are_not_merged(primitives: list[LinePrimitive]) -> None:
    result = optimize_primitives(primitives)

    assert result.metrics.output_primitive_count == 2
    assert result.metrics.collinear_lines_merged == 0


def test_line_style_difference_rejects_merge() -> None:
    red = StrokeStyle(color=RGBColor(255, 0, 0))
    blue = StrokeStyle(color=RGBColor(0, 0, 255))
    result = optimize_primitives([_line(0, 0, 1, 0, stroke=red), _line(1, 0, 2, 0, stroke=blue)])

    assert result.metrics.output_primitive_count == 2
    assert result.metrics.style_rejections >= 1


def test_duplicate_and_reversed_lines_are_removed_conservatively() -> None:
    config = GeometryOptimizationConfig(merge_collinear_lines=False)
    duplicate = optimize_primitives([_line(0, 0, 1, 0), _line(0, 0, 1, 0)], config)
    reversed_line = optimize_primitives([_line(0, 0, 1, 0), _line(1, 0, 0, 0)], config)

    assert duplicate.metrics.duplicate_primitives_removed == 1
    assert reversed_line.metrics.duplicate_primitives_removed == 1


def test_duplicate_removal_preserves_style_fill_and_translucent_overlap() -> None:
    fill = FillStyle(RGBColor(10, 20, 30), opacity=0.4)
    stroke = StrokeStyle(color=RGBColor(10, 10, 10), width=2.0, opacity=0.5, dash_pattern=(1.0, 2.0))
    opaque = _closed([(0, 0), (2, 0), (2, 2), (0, 2)], fill=FillStyle(RGBColor(10, 20, 30)))
    translucent_a = CirclePrimitive(_point(0, 0), 2, stroke=stroke, fill=fill)
    translucent_b = CirclePrimitive(_point(0, 0), 2, stroke=stroke, fill=fill)

    removed = optimize_primitives([opaque, opaque], GeometryOptimizationConfig(merge_collinear_lines=False))
    preserved = optimize_primitives([translucent_a, translucent_b])

    assert removed.metrics.duplicate_primitives_removed == 1
    assert preserved.metrics.duplicate_primitives_removed == 0
    assert preserved.primitives[0].stroke == stroke
    assert preserved.primitives[0].fill == fill


def test_polyline_duplicate_points_collinear_conversion_and_open_state() -> None:
    duplicated = optimize_primitives([_poly([(0, 0), (0, 0), (1, 0), (2, 0)])])

    assert isinstance(duplicated.primitives[0], LinePrimitive)
    assert duplicated.metrics.duplicate_points_removed == 1
    assert any(op.kind is OptimizationOperationKind.CONVERT_TO_LINE for op in duplicated.operations)


def test_polyline_corner_many_redundant_points_and_small_detail_are_preserved() -> None:
    cornered = _poly([(0, 0), (1, 0), (1, 1)])
    redundant = _poly([(float(x), 0.001 * sin(x)) for x in np.linspace(0, 10, 60)])
    tooth = _poly([(0, 0), (1, 0), (1.15, 0.25), (1.3, 0), (2, 0), (2, 1)])

    corner_result = optimize_primitives([cornered])
    redundant_result = optimize_primitives([redundant], GeometryOptimizationConfig(polyline_tolerance=0.01))
    tooth_result = optimize_primitives([tooth], GeometryOptimizationConfig(polyline_tolerance=0.2))

    assert isinstance(corner_result.primitives[0], PolylinePrimitive)
    assert len(corner_result.primitives[0].points) == 3
    assert redundant_result.metrics.point_reduction > 0
    assert len(tooth_result.primitives[0].points) >= 5


def test_open_and_closed_polyline_simplification_preserves_endpoints_closure_and_orientation() -> None:
    open_poly = _sine_polyline(40)
    closed_shape = _closed([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)])

    open_result = optimize_primitives([open_poly], GeometryOptimizationConfig(polyline_tolerance=0.05))
    closed_result = optimize_primitives([closed_shape], GeometryOptimizationConfig(polyline_tolerance=0.1))

    assert isinstance(open_result.primitives[0], PolylinePrimitive)
    assert open_result.primitives[0].points[0] == open_poly.points[0]
    assert open_result.primitives[0].points[-1] == open_poly.points[-1]
    assert isinstance(closed_result.primitives[0], ClosedShapePrimitive)
    assert signed_area(closed_shape.points) * signed_area(closed_result.primitives[0].points) > 0


def test_simplification_preserves_junction_metadata_and_rejects_too_much_error() -> None:
    junction = _poly(
        [(0, 0), (1, 0.03), (2, 0), (3, 0.2), (4, 0)],
        metadata={"junction_indices": [2], "id": "centerline-arm"},
    )
    strict = optimize_primitives(
        [_sine_polyline(64)],
        GeometryOptimizationConfig(polyline_tolerance=0.00001, normalized_error_budget=0.0, cumulative_error_budget=0.0),
    )
    result = optimize_primitives([junction], GeometryOptimizationConfig(polyline_tolerance=0.2))

    assert any(point == _point(2, 0) for point in result.primitives[0].points)
    assert strict.metrics.point_reduction == 0


def test_compatible_polylines_merge_and_incompatible_polylines_do_not() -> None:
    config = GeometryOptimizationConfig(maximum_endpoint_distance=0.05, maximum_join_angle=20)
    compatible = optimize_primitives(
        [_poly([(0, 0), (1, 0), (2, 0.1)]), _poly([(2, 0.1), (3, 0.2), (4, 0.3)])],
        config,
    )
    angled = optimize_primitives([_poly([(0, 0), (1, 0), (2, 0)]), _poly([(2, 0), (2, 1), (2, 2)])], config)

    assert compatible.metrics.polylines_merged == 1
    assert angled.metrics.polylines_merged == 0


def test_polyline_merge_rejects_style_and_topology_incompatibility() -> None:
    red = StrokeStyle(color=RGBColor(255, 0, 0))
    blue = StrokeStyle(color=RGBColor(0, 0, 255))
    style_result = optimize_primitives(
        [_poly([(0, 0), (1, 0), (2, 0.1)], stroke=red), _poly([(2, 0.1), (3, 0.2), (4, 0.3)], stroke=blue)],
        GeometryOptimizationConfig(maximum_endpoint_distance=0.1, maximum_join_angle=20),
    )
    topology_result = optimize_primitives(
        [
            _poly([(0, 0), (1, 0), (2, 0.1)], metadata={"start_node_id": "n1"}),
            _poly([(2, 0.1), (3, 0.2), (4, 0.3)]),
        ],
        GeometryOptimizationConfig(maximum_endpoint_distance=0.1, maximum_join_angle=20),
    )

    assert style_result.metrics.style_rejections >= 1
    assert topology_result.metrics.topology_rejections >= 1


def test_bezier_degenerate_straight_compatible_corner_and_style_cases() -> None:
    degenerate = BezierPrimitive(_point(1, 1), _point(1, 1), _point(1, 1), _point(1, 1))
    straight = BezierPrimitive(_point(0, 0), _point(1, 0), _point(2, 0), _point(3, 0))
    smooth = optimize_primitives(
        _smooth_beziers(),
        GeometryOptimizationConfig(bezier_merge_tolerance=0.05, maximum_combined_bezier_error=0.05, normalized_error_budget=0.05, cumulative_error_budget=0.05),
    )
    corner = optimize_primitives(
        [
            BezierPrimitive(_point(0, 0), _point(1, 0), _point(2, 0), _point(3, 0)),
            BezierPrimitive(_point(3, 0), _point(3, 1), _point(3, 2), _point(3, 3)),
        ]
    )
    styled = optimize_primitives(
        [_bezier(stroke=StrokeStyle(color=RGBColor(1, 0, 0))), _bezier(3, stroke=StrokeStyle(color=RGBColor(0, 0, 1)))]
    )

    assert isinstance(optimize_primitives([degenerate]).primitives[0], PointPrimitive)
    assert isinstance(optimize_primitives([straight]).primitives[0], LinePrimitive)
    assert smooth.metrics.bezier_sequences_merged == 1
    assert smooth.metrics.output_bezier_count == 1
    assert corner.metrics.bezier_sequences_merged == 0
    assert styled.metrics.bezier_sequences_merged == 0


def test_three_beziers_can_reduce_and_bad_merge_is_rejected_by_error_budget() -> None:
    curves = _smooth_beziers() + [BezierPrimitive(_point(6, 0), _point(7, 1), _point(8, 1), _point(9, 0))]
    merged = optimize_primitives(
        curves,
        GeometryOptimizationConfig(bezier_merge_tolerance=0.08, maximum_combined_bezier_error=0.08, normalized_error_budget=0.08, cumulative_error_budget=0.08),
    )
    rejected = optimize_primitives(
        _smooth_beziers(),
        GeometryOptimizationConfig(bezier_merge_tolerance=0.001, maximum_combined_bezier_error=0.001, normalized_error_budget=0.001, cumulative_error_budget=0.001),
    )

    assert merged.metrics.output_bezier_count < 3
    assert rejected.metrics.bezier_sequences_merged == 0
    assert rejected.metrics.operations_rejected >= 1


def test_circle_ellipse_duplicates_and_simple_shapes_are_not_polygonized() -> None:
    circle_a = CirclePrimitive(_point(0, 0), 2)
    circle_b = CirclePrimitive(_point(0, 0), 2)
    ellipse_a = EllipsePrimitive(_point(0, 0), 3, 1, 10)
    ellipse_b = EllipsePrimitive(_point(0, 0), 3, 1, 10)
    styled_circle = CirclePrimitive(_point(0, 0), 2, stroke=StrokeStyle(color=RGBColor(1, 2, 3)))

    result = optimize_primitives([circle_a, circle_b, ellipse_a, ellipse_b, styled_circle])

    assert result.metrics.duplicate_primitives_removed == 2
    assert any(isinstance(item, CirclePrimitive) for item in result.primitives)
    assert any(isinstance(item, EllipsePrimitive) for item in result.primitives)
    assert len([item for item in result.primitives if isinstance(item, CirclePrimitive)]) == 2


def test_closed_shape_preserves_hole_metadata_and_avoids_new_self_intersection() -> None:
    shape = _closed(
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (3, 2), (3, 3), (0, 3)],
        metadata={"subpath": 1, "fill_rule": "evenodd"},
        fill=FillStyle(RGBColor(10, 20, 30)),
    )
    result = optimize_primitives([shape], GeometryOptimizationConfig(polyline_tolerance=0.5))

    assert isinstance(result.primitives[0], ClosedShapePrimitive)
    assert result.primitives[0].metadata["subpath"] == 1
    assert result.primitives[0].metadata["fill_rule"] == "evenodd"
    assert not result.warnings


def test_primitive_group_nested_groups_preserved_and_can_be_flattened_explicitly() -> None:
    inner = PrimitiveGroup((_line(0, 0, 1, 0), _line(1, 0, 2, 0)), name="inner")
    outer = PrimitiveGroup((inner, _line(0, 1, 1, 1)), name="outer", metadata={"id": "g"})

    preserved = optimize_primitives(outer)
    flattened = optimize_primitives(outer, GeometryOptimizationConfig(preserve_groups=False))

    assert isinstance(preserved.primitives[0], PrimitiveGroup)
    assert preserved.primitives[0].name == "outer"
    assert not any(isinstance(item, PrimitiveGroup) for item in flattened.primitives)


def test_draw_order_preserved_and_real_reductions_are_recorded() -> None:
    first = _line(0, 0, 1, 0, metadata={"id": "first"})
    second = _line(10, 0, 11, 0, metadata={"id": "second"})
    duplicate = _line(0, 0, 1, 0, metadata={"id": "duplicate"})
    result = optimize_primitives([first, second, duplicate], GeometryOptimizationConfig(merge_collinear_lines=False))

    assert [item.metadata.get("id") for item in result.primitives] == ["first", "second"]
    assert result.metrics.primitive_reduction > 0
    assert result.metrics.primitive_reduction_ratio > 0


def test_styles_metadata_and_optimization_history_are_preserved_on_merges() -> None:
    stroke = StrokeStyle(color=RGBColor(10, 20, 30), width=2.5, line_cap="round", line_join="bevel", dash_pattern=(2.0, 1.0))
    first = _line(0, 0, 1, 0, stroke=stroke, metadata={"id": "a", "tracer": "synthetic"})
    second = _line(1, 0, 2, 0, stroke=stroke, metadata={"id": "b"})
    result = optimize_primitives([first, second])
    merged = result.primitives[0]

    assert merged.stroke == stroke
    assert merged.metadata["tracer"] == "synthetic"
    assert "optimization_history" in merged.metadata
    assert set(merged.metadata["merged_source_ids"]) >= {"a", "b"}


def test_error_budget_and_cumulative_budget_control_simplification() -> None:
    polyline = _sine_polyline(80)
    loose = optimize_primitives([polyline], GeometryOptimizationConfig(polyline_tolerance=0.1, normalized_error_budget=0.1, cumulative_error_budget=0.1))
    blocked = optimize_primitives([polyline], GeometryOptimizationConfig(polyline_tolerance=0.1, normalized_error_budget=0.00001, cumulative_error_budget=0.00001))

    assert loose.metrics.point_reduction > blocked.metrics.point_reduction
    assert blocked.metrics.operations_rejected >= 1


def test_multiple_passes_stop_and_limit_passes() -> None:
    polylines = [
        _poly([(0, 0), (0.5, 0.12), (1, 0)]),
        _poly([(1, 0), (1.5, -0.12), (2, 0)]),
        _poly([(2, 0), (2.5, 0.12), (3, 0)]),
        _poly([(3, 0), (3.5, -0.12), (4, 0)]),
    ]
    config = GeometryOptimizationConfig(maximum_endpoint_distance=0.05, maximum_join_angle=20, simplify_polylines=False)
    one_pass = optimize_primitives(polylines, GeometryOptimizationConfig(maximum_endpoint_distance=0.05, maximum_join_angle=20, simplify_polylines=False, maximum_optimization_passes=1))
    full = optimize_primitives(polylines, config)

    assert one_pass.metrics.optimization_passes == 1
    assert one_pass.metrics.output_primitive_count > full.metrics.output_primitive_count
    assert full.metrics.optimization_passes <= full.configuration.maximum_optimization_passes


def test_invalid_configuration_nan_infinite_and_bad_input_are_rejected() -> None:
    with pytest.raises(ValueError):
        GeometryOptimizationConfig(polyline_tolerance=-1)
    with pytest.raises(ValueError):
        GeometryOptimizationConfig(maximum_optimization_passes=0)
    with pytest.raises(TypeError):
        GeometryOptimizationConfig(strict="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Point2D(float("nan"), 0)
    with pytest.raises(ValueError):
        Point2D(float("inf"), 0)
    with pytest.raises(TypeError):
        optimize_primitives(object())


def test_determinism_to_dict_and_hash_are_stable() -> None:
    primitives = [_line(0, 0, 1, 0), _line(1, 0, 2, 0), _sine_polyline(32)]
    first = optimize_primitives(primitives)
    second = optimize_primitives(primitives)

    assert first.to_dict() == second.to_dict()
    assert first.deterministic_hash == second.deterministic_hash
    assert "array(" not in repr(first.to_dict())


def test_fit_result_sequence_svg_parse_result_and_centerline_path_inputs_are_supported() -> None:
    fit_result = fit_primitive([_point(0, 0), _point(1, 0), _point(2, 0)])
    svg_result = parse_svg_to_primitives(_svg('<polyline points="0,0 1,0 2,0" stroke="black" fill="none"/>'))
    centerline = CenterlinePath(
        id="p0",
        points=(_point(0, 0), _point(1, 0), _point(2, 0)),
        start_node_id="n0",
        end_node_id="n1",
        closure=PathClosureType.OPEN,
        component_id=0,
        metadata={"source": "test"},
    )

    assert optimize_fit_results(fit_result).original_summary["source_type"] == "PrimitiveFitResult"
    assert optimize_fit_results([fit_result]).original_summary["source_type"] == "PrimitiveFitResultSequence"
    assert optimize_primitives(svg_result).original_summary["source_type"] == "SvgParseResult"
    assert optimize_primitives(centerline).original_summary["source_type"] == "CenterlinePath"


def test_no_tikz_svg2tikz_tracer_or_gui_dependency_and_modes_are_preserved() -> None:
    sources = [
        Path(geometry_optimization.__file__).read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/path_merging.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/geometry_simplification.py").read_text(encoding="utf-8").lower(),
    ]
    for source in sources:
        assert "\\\\draw" not in source
        assert "tikzpicture" not in source
        assert "svg2tikz" not in source
        assert "trace_image(" not in source
        assert "subprocess" not in source
        assert "pyside6" not in source

    assert config_for_mode("classic").mode == "classic"
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("contours").mode == "contours"
    for path in ("fikzpy/core/visual_pipeline.py", "fikzpy/core/visual_postprocessor.py", "fikzpy/core/tikz_pipeline.py", "fikzpy/core/image_processor.py"):
        assert "geometry_optimization" not in Path(path).read_text(encoding="utf-8")


def test_importing_optimizer_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.geometry_optimization; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_fidelity_over_reduction_preserves_rectangle_teeth_s_curve_circle_and_ellipse() -> None:
    rectangle = _closed([(0, 0), (4, 0), (4, 3), (0, 3)])
    tooth = _poly([(0, 0), (1, 0), (1.2, 0.3), (1.4, 0), (2, 0)])
    s_curve = _sine_polyline(64)
    circle = CirclePrimitive(_point(0, 0), 3)
    ellipse = EllipsePrimitive(_point(0, 0), 4, 1.5)
    result = optimize_primitives([rectangle, tooth, s_curve, circle, ellipse], GeometryOptimizationConfig(polyline_tolerance=0.2))

    assert isinstance(result.primitives[0], ClosedShapePrimitive)
    assert len(result.primitives[0].points) == 4
    assert isinstance(result.primitives[1], PolylinePrimitive)
    assert len(result.primitives[1].points) >= 4
    assert isinstance(result.primitives[-2], CirclePrimitive)
    assert isinstance(result.primitives[-1], EllipsePrimitive)


@pytest.mark.parametrize(
    ("name", "primitives"),
    [
        ("line_art", [_line(0, 0, 2, 0), _line(2, 0, 4, 0), _poly([(4, 0), (5, 0.2), (6, 0)])]),
        ("potrace", [ClosedShapePrimitive((_point(0, 0), _point(10, 0), _point(10, 10), _point(0, 10)))]),
        ("autotrace", [_poly([(0, 0), (1, 0), (2, 0)]), _bezier(2)]),
        ("vtracer", [CirclePrimitive(_point(1, 1), 2), CirclePrimitive(_point(1, 1), 2)]),
    ],
)
def test_synthetic_line_art_potrace_autotrace_and_vtracer_cases(name: str, primitives: list) -> None:
    result = optimize_primitives(primitives, GeometryOptimizationConfig(polyline_tolerance=0.05))

    assert result.success
    assert result.metrics.output_primitive_count <= result.metrics.input_primitive_count
    assert name in {"line_art", "potrace", "autotrace", "vtracer"}


def test_baseline_reports_are_read_only_sources_and_optimization_report_exists() -> None:
    baseline = Path("examples/classic_semantic_baseline")
    primitive_report = json.loads((baseline / "primitive_fitting_report.json").read_text(encoding="utf-8"))
    svg_report = json.loads((baseline / "svg_parser_report.json").read_text(encoding="utf-8"))
    centerline_report = json.loads((baseline / "centerline_report.json").read_text(encoding="utf-8"))
    optimization_report = json.loads((baseline / "geometry_optimization_report.json").read_text(encoding="utf-8"))

    assert primitive_report["issue"] == "Issue 7"
    assert svg_report
    assert centerline_report["images"]
    assert optimization_report["issue"] == "Issue 8"
    assert len(optimization_report["cases"]) >= 10
