from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from fikzpy.core.centerline_pipeline import extract_centerlines
from fikzpy.core.complexity_metrics import compute_complexity_metrics
from fikzpy.core.fidelity_score import FidelityScoreConfig, FidelityScoreError, compute_fidelity_score
from fikzpy.core.geometry_optimization import optimize_primitives
from fikzpy.core.primitive_fitting import fit_primitive, fit_primitives
from fikzpy.core.raster_metrics import RasterMetricsConfig, compare_rasters, image_to_rgb_array, raster_summary
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PolylinePrimitive, PrimitiveGroup, RGBColor, StrokeStyle
from fikzpy.core.semantic_rasterizer import SemanticRasterizationConfig, rasterize_semantic_primitives
from fikzpy.core.semantic_tikz_exporter import export_primitives_to_tikz
from fikzpy.core.svg_semantic_parser import parse_svg_to_primitives
from fikzpy.core.vectorization_config import config_for_mode
from fikzpy.core.visual_validation import ValidationCase, ValidationReport, ValidationStatus
from fikzpy.core.visual_validation import VisualValidationConfig, VisualValidationError, validate_semantic_output


def p(x: float, y: float) -> Point2D:
    return Point2D(x, y)


def white(size: tuple[int, int] = (40, 40)) -> np.ndarray:
    return np.full((size[1], size[0], 3), 255, dtype=np.uint8)


def black(size: tuple[int, int] = (40, 40)) -> np.ndarray:
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def draw_line_image(shift: int = 0) -> np.ndarray:
    image = white()
    cv2.line(image, (5, 20 + shift), (35, 20 + shift), (0, 0, 0), 1, cv2.LINE_8)
    return image


def filled_rectangle_image(missing: bool = False) -> np.ndarray:
    image = white()
    if not missing:
        cv2.rectangle(image, (8, 8), (31, 31), (0, 0, 0), -1, cv2.LINE_8)
        cv2.rectangle(image, (15, 15), (23, 23), (255, 255, 255), -1, cv2.LINE_8)
    return image


def circle_image() -> np.ndarray:
    image = white()
    cv2.circle(image, (20, 20), 10, (0, 0, 0), 1, cv2.LINE_8)
    return image


def ellipse_image() -> np.ndarray:
    image = white()
    cv2.ellipse(image, (20, 20), (12, 7), 0, 0, 360, (0, 0, 0), 1, cv2.LINE_8)
    return image


def polyline_image() -> np.ndarray:
    image = white()
    points = np.array([[4, 30], [14, 10], [26, 30], [36, 12]], dtype=np.int32)
    cv2.polylines(image, [points], False, (0, 0, 0), 1, cv2.LINE_8)
    return image


def _sample_bezier(primitive: BezierPrimitive, samples: int) -> list[Point2D]:
    output: list[Point2D] = []
    for index in range(samples + 1):
        t = index / samples
        omt = 1.0 - t
        x = omt**3 * primitive.start.x + 3 * omt**2 * t * primitive.control1.x + 3 * omt * t**2 * primitive.control2.x + t**3 * primitive.end.x
        y = omt**3 * primitive.start.y + 3 * omt**2 * t * primitive.control1.y + 3 * omt * t**2 * primitive.control2.y + t**3 * primitive.end.y
        output.append(p(x, y))
    return output


def bezier_points() -> tuple[Point2D, ...]:
    primitive = BezierPrimitive(p(5, 28), p(12, 5), p(28, 5), p(35, 28))
    return tuple(_sample_bezier(primitive, 48))


def bezier_image() -> np.ndarray:
    image = white()
    points = np.array([(round(point.x), round(point.y)) for point in bezier_points()], dtype=np.int32)
    cv2.polylines(image, [points], False, (0, 0, 0), 1, cv2.LINE_8)
    return image


def mixed_monochrome_source() -> np.ndarray:
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (38, 34), (15, 13), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_8)
    cv2.ellipse(image, (82, 34), (15, 13), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_8)
    cv2.rectangle(image, (26, 75), (50, 104), (0, 0, 0), -1, cv2.LINE_8)
    cv2.rectangle(image, (70, 75), (94, 104), (0, 0, 0), -1, cv2.LINE_8)
    cv2.circle(image, (38, 42), 7, (255, 255, 255), -1, cv2.LINE_8)
    cv2.circle(image, (82, 42), 7, (255, 255, 255), -1, cv2.LINE_8)
    for x in (35, 41, 79, 85):
        cv2.circle(image, (x, 40), 1, (0, 0, 0), -1, cv2.LINE_8)
    cv2.line(image, (23, 58), (53, 58), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (67, 58), (97, 58), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (53, 82), (67, 82), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (20, 88), (5, 102), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (100, 88), (115, 102), (0, 0, 0), 1, cv2.LINE_8)
    return image


def mixed_good_primitives() -> list:
    black_fill = FillStyle(RGBColor(0, 0, 0))
    white_fill = FillStyle(RGBColor(255, 255, 255))
    no_stroke = StrokeStyle(opacity=0.0)
    return [
        EllipsePrimitive(p(38, 34), 15, 13, fill=black_fill),
        EllipsePrimitive(p(82, 34), 15, 13, fill=black_fill),
        ClosedShapePrimitive((p(26, 75), p(50, 75), p(50, 104), p(26, 104)), fill=black_fill),
        ClosedShapePrimitive((p(70, 75), p(94, 75), p(94, 104), p(70, 104)), fill=black_fill),
        CirclePrimitive(p(38, 42), 7, stroke=no_stroke, fill=white_fill),
        CirclePrimitive(p(82, 42), 7, stroke=no_stroke, fill=white_fill),
        CirclePrimitive(p(35, 40), 1, fill=black_fill),
        CirclePrimitive(p(41, 40), 1, fill=black_fill),
        CirclePrimitive(p(79, 40), 1, fill=black_fill),
        CirclePrimitive(p(85, 40), 1, fill=black_fill),
        LinePrimitive(p(23, 58), p(53, 58)),
        LinePrimitive(p(67, 58), p(97, 58)),
        LinePrimitive(p(53, 82), p(67, 82)),
        LinePrimitive(p(20, 88), p(5, 102)),
        LinePrimitive(p(100, 88), p(115, 102)),
    ]


def mixed_bad_primitives() -> list:
    return [
        EllipsePrimitive(p(38, 42), 10, 8),
        EllipsePrimitive(p(82, 42), 10, 8),
        LinePrimitive(p(23, 58), p(53, 58)),
        LinePrimitive(p(67, 58), p(97, 58)),
        LinePrimitive(p(20, 88), p(5, 102)),
        LinePrimitive(p(100, 88), p(115, 102)),
    ]


def test_empty_white_and_black_images_compare_deterministically() -> None:
    assert validate_semantic_output(white(), []).accepted
    black_result = compute_fidelity_score(black(), black())
    assert black_result.accepted
    assert black_result.raster_metrics.foreground_recall == 1.0
    first = compute_fidelity_score(white(), white()).to_dict()
    second = compute_fidelity_score(white(), white()).to_dict()
    assert first == second


@pytest.mark.parametrize(
    ("image", "primitive", "expected"),
    [
        (draw_line_image(), LinePrimitive(p(5, 20), p(35, 20)), "line"),
        (circle_image(), CirclePrimitive(p(20, 20), 10), "circle"),
        (filled_rectangle_image(), ClosedShapePrimitive((p(8, 8), p(31, 8), p(31, 31), p(8, 31)), fill=FillStyle(RGBColor(0, 0, 0))), "closed_shape"),
        (ellipse_image(), EllipsePrimitive(p(20, 20), 12, 7), "ellipse"),
        (polyline_image(), PolylinePrimitive((p(4, 30), p(14, 10), p(26, 30), p(36, 12))), "polyline"),
        (bezier_image(), BezierPrimitive(p(5, 28), p(12, 5), p(28, 5), p(35, 28)), "bezier"),
    ],
)
def test_semantic_rasterizer_supports_core_primitive_types(image: np.ndarray, primitive, expected: str) -> None:
    result = validate_semantic_output(
        image,
        [primitive],
        config=VisualValidationConfig(minimum_acceptable_score=0.35, minimum_fidelity_score=0.25),
    )
    assert result.fidelity_score.raster_metrics.rendered_foreground_pixels > 0
    assert expected in result.complexity_metrics.primitive_type_counts


def test_perfect_nearly_perfect_and_bad_comparisons() -> None:
    perfect = compute_fidelity_score(draw_line_image(), draw_line_image())
    shifted = compute_fidelity_score(draw_line_image(), draw_line_image(shift=1), config=FidelityScoreConfig(minimum_acceptable_score=0.1))
    bad = compute_fidelity_score(draw_line_image(), white(), config=FidelityScoreConfig(minimum_acceptable_score=0.1))

    assert perfect.overall_score > shifted.overall_score > bad.overall_score
    assert not bad.accepted
    assert "invisible_output" in bad.regression_flags or "near_empty_output" in bad.regression_flags


def test_iou_precision_recall_f1_rmse_and_edge_metrics_are_reported() -> None:
    metrics = compare_rasters(draw_line_image(), draw_line_image(shift=1))
    assert 0.0 <= metrics.foreground_iou < 1.0
    assert 0.0 <= metrics.foreground_precision <= 1.0
    assert 0.0 <= metrics.foreground_recall <= 1.0
    assert 0.0 <= metrics.foreground_f1 <= 1.0
    assert metrics.root_mean_squared_error > 0.0
    assert 0.0 <= metrics.edge_overlap <= 1.0
    assert 0.0 <= metrics.edge_precision <= 1.0
    assert 0.0 <= metrics.edge_recall <= 1.0


def test_filled_region_dark_mass_thin_stroke_small_detail_and_geometry_metrics() -> None:
    source = filled_rectangle_image()
    missing = white()
    outline_only = white()
    cv2.rectangle(outline_only, (8, 8), (31, 31), (0, 0, 0), 1, cv2.LINE_8)

    missing_metrics = compare_rasters(source, missing)
    outline_metrics = compare_rasters(source, outline_only)

    assert missing_metrics.filled_region_recall == 0.0
    assert missing_metrics.dark_mass_preservation_ratio == 0.0
    assert outline_metrics.double_outline_penalty > 0.0
    assert 0.0 <= outline_metrics.thin_stroke_recall <= 1.0
    assert 0.0 <= outline_metrics.small_detail_recall <= 1.0
    assert missing_metrics.connected_component_difference >= 1
    assert missing_metrics.bounding_box_difference == 1.0
    assert missing_metrics.centroid_shift == 1.0


def test_composite_score_weights_and_thresholds_are_interpretable() -> None:
    default = compute_fidelity_score(draw_line_image(), draw_line_image(shift=1), config=FidelityScoreConfig(minimum_acceptable_score=0.1))
    weighted = compute_fidelity_score(
        draw_line_image(),
        draw_line_image(shift=1),
        config=FidelityScoreConfig(weights={"visual_fidelity": 0.90, "semantic_compactness": 0.02}, minimum_acceptable_score=0.1),
    )
    assert 0.0 <= default.overall_score <= 1.0
    assert 0.0 <= weighted.overall_score <= 1.0
    assert compute_fidelity_score(draw_line_image(), draw_line_image(), config=FidelityScoreConfig(minimum_acceptable_score=0.99)).accepted
    assert not compute_fidelity_score(draw_line_image(), white(), config=FidelityScoreConfig(minimum_acceptable_score=0.99)).accepted


def test_complexity_metrics_for_primitives_groups_and_tikz_code() -> None:
    group = PrimitiveGroup(
        (
            LinePrimitive(p(0, 0), p(10, 0), metadata={"label": "baseline"}),
            PrimitiveGroup((CirclePrimitive(p(5, 5), 3),), name="nested"),
        ),
        name="outer",
        metadata={"source": "test"},
    )
    tikz = "\\draw[draw=black] (0,0) -- (1,0);\n\\draw[draw=black] (0,0) circle[radius=1];"
    metrics = compute_complexity_metrics(group, tikz)
    assert metrics.primitive_count == 2
    assert metrics.group_count == 2
    assert metrics.max_group_depth == 2
    assert metrics.point_count >= 3
    assert metrics.linear_segment_count == 1
    assert metrics.tikz_draw_commands == 2
    assert metrics.tikz_repeated_style_count == 1
    assert metrics.tikz_semantic_primitive_count == 1


def test_compact_semantic_tikz_scores_better_than_raw_like_path() -> None:
    semantic = compute_complexity_metrics(tikz_code="\\draw (0,0) circle[radius=10];\n\\draw (0,0) ellipse[x radius=8, y radius=4];")
    raw = compute_complexity_metrics(tikz_code="\\path (0,0) -- (1,0) -- (2,0) -- (3,0) -- (4,0) -- (5,0) -- (6,0) -- (7,0) -- (8,0) -- (9,0) -- (10,0);")
    assert semantic.semantic_compactness_score > raw.semantic_compactness_score
    assert raw.tikz_raw_path_penalty > semantic.tikz_raw_path_penalty


def test_export_result_fit_result_svg_result_optimization_and_centerline_inputs_are_supported() -> None:
    line = LinePrimitive(p(5, 20), p(35, 20))
    export_result = export_primitives_to_tikz([line])
    optimized = optimize_primitives([line])
    fit_result = fit_primitive((p(5, 20), p(35, 20)))
    fit_results = fit_primitives([(p(5, 20), p(35, 20))])
    svg_result = parse_svg_to_primitives('<svg viewBox="0 0 40 40"><line x1="5" y1="20" x2="35" y2="20"/></svg>')
    centerline = extract_centerlines((draw_line_image()[:, :, 0] < 128).astype(np.uint8))

    assert compute_complexity_metrics(export_result).tikz_draw_commands == 1
    for primitives in (optimized, fit_result, fit_results, svg_result, centerline):
        result = validate_semantic_output(
            draw_line_image(),
            primitives,
            tikz_result=export_result if primitives is optimized else None,
            config=VisualValidationConfig(minimum_acceptable_score=0.1, minimum_fidelity_score=0.1),
        )
        assert result.fidelity_score.raster_metrics.rendered_foreground_pixels > 0


def test_to_dict_hash_and_validation_report_are_deterministic() -> None:
    result = validate_semantic_output(draw_line_image(), [LinePrimitive(p(5, 20), p(35, 20))])
    repeated = validate_semantic_output(draw_line_image(), [LinePrimitive(p(5, 20), p(35, 20))])
    report = ValidationReport("Issue 10", (ValidationCase("line", result, "perfect line"),))
    data = result.to_dict()

    assert data == repeated.to_dict()
    assert result.deterministic_hash == repeated.deterministic_hash
    assert report.to_dict()["cases"][0]["name"] == "line"
    assert data["status"] == ValidationStatus.ACCEPTED.value


def test_invalid_config_unknown_type_nan_infinity_strict_and_tolerant_modes() -> None:
    with pytest.raises(ValueError):
        VisualValidationConfig(foreground_threshold=999)
    with pytest.raises(TypeError):
        rasterize_semantic_primitives(object())
    nan_image = np.array([[np.nan, np.inf], [0.0, 1.0]], dtype=float)
    normalized = image_to_rgb_array(nan_image)
    assert normalized.dtype == np.uint8
    bad = compute_fidelity_score(draw_line_image(), white(), config=FidelityScoreConfig(minimum_acceptable_score=0.1))
    assert not bad.accepted
    with pytest.raises(FidelityScoreError):
        compute_fidelity_score(draw_line_image(), white(), config=FidelityScoreConfig(strict=True))
    with pytest.raises(VisualValidationError):
        validate_semantic_output(draw_line_image(), [], config=VisualValidationConfig(strict=True))
    tolerant = validate_semantic_output(draw_line_image(), [], config=VisualValidationConfig(strict=False))
    assert not tolerant.accepted


def test_external_renderer_warning_latex_not_required_and_debug_images(tmp_path: Path) -> None:
    result = validate_semantic_output(
        draw_line_image(),
        [LinePrimitive(p(5, 20), p(35, 20))],
        config=VisualValidationConfig(use_external_tikz_renderer=True),
    )
    assert any(warning.code == "external_renderer_unavailable" for warning in result.warnings)
    assert not any(tmp_path.iterdir())

    debug_dir = tmp_path / "debug"
    validate_semantic_output(
        draw_line_image(),
        [LinePrimitive(p(5, 20), p(35, 20))],
        config=VisualValidationConfig(save_debug_images=True, debug_output_dir=debug_dir),
    )
    assert (debug_dir / "source.png").exists()
    assert (debug_dir / "rendered.png").exists()


def test_mixed_monochrome_good_case_preserves_filled_regions_and_details() -> None:
    source = mixed_monochrome_source()
    result = validate_semantic_output(
        source,
        mixed_good_primitives(),
        config=VisualValidationConfig(minimum_acceptable_score=0.45, minimum_fidelity_score=0.35),
    )
    assert 0.12 <= result.source_summary["foreground_ratio"] <= 0.24
    assert result.fidelity_score.raster_metrics.dark_mass_preservation_ratio > 0.75
    assert result.fidelity_score.raster_metrics.large_dark_region_recall > 0.75
    assert result.fidelity_score.raster_metrics.thin_stroke_recall > 0.40


def test_mixed_monochrome_bad_case_rejects_lost_dark_mass_and_outline_only_output() -> None:
    source = mixed_monochrome_source()
    bad = validate_semantic_output(
        source,
        mixed_bad_primitives(),
        config=VisualValidationConfig(minimum_acceptable_score=0.45, minimum_fidelity_score=0.35),
    )
    flags = set(bad.regression_flags)
    assert not bad.accepted
    assert {"missing_large_dark_regions", "low_filled_region_recall", "dark_mass_loss"} & flags
    assert "dark_regions_lost" in bad.rejection_reasons
    assert bad.rendered_summary["foreground_ratio"] < bad.source_summary["foreground_ratio"] * 0.35


def test_output_with_only_thin_lines_is_rejected_when_source_has_filled_mass() -> None:
    outline = [
        PolylinePrimitive((p(8, 8), p(31, 8), p(31, 31), p(8, 31)), closed=True),
    ]
    result = validate_semantic_output(
        filled_rectangle_image(),
        outline,
        config=VisualValidationConfig(minimum_acceptable_score=0.30, minimum_fidelity_score=0.20),
    )
    assert not result.accepted
    assert "missing_large_dark_regions" in result.regression_flags


def test_lineart_black_mass_with_white_cutouts_sets_overfill_flags() -> None:
    source = circle_image()
    black_fill = FillStyle(RGBColor.black())
    white_fill = FillStyle(RGBColor(255, 255, 255))
    hidden_white_stroke = StrokeStyle(RGBColor(255, 255, 255), width=0.1, opacity=0.0)
    primitives = [
        ClosedShapePrimitive((p(5, 5), p(35, 5), p(35, 35), p(5, 35)), fill=black_fill, stroke=StrokeStyle(width=1.0)),
        ClosedShapePrimitive((p(10, 10), p(18, 10), p(18, 18), p(10, 18)), fill=white_fill, stroke=hidden_white_stroke),
        ClosedShapePrimitive((p(22, 10), p(30, 10), p(30, 18), p(22, 18)), fill=white_fill, stroke=hidden_white_stroke),
        ClosedShapePrimitive((p(10, 22), p(18, 22), p(18, 30), p(10, 30)), fill=white_fill, stroke=hidden_white_stroke),
        ClosedShapePrimitive((p(22, 22), p(30, 22), p(30, 30), p(22, 30)), fill=white_fill, stroke=hidden_white_stroke),
    ]
    tikz = export_primitives_to_tikz(primitives)
    result = validate_semantic_output(source, primitives, tikz)

    flags = set(result.regression_flags)
    assert not result.accepted
    assert {"excessive_filled_area", "artificial_black_mass", "overfilled_lineart", "excessive_white_cutouts"} <= flags
    assert "lineart_has_excessive_filled_area" in result.rejection_reasons
    assert "lineart_uses_excessive_white_cutouts" in result.rejection_reasons
    assert result.metrics.lineart_fill_metrics["source_is_line_art"]
    assert result.metrics.filled_region_metrics["white_cutout_count"] == 4


def test_pipeline_remains_isolated_from_app_and_scores_exported_semantic_output() -> None:
    svg = '<svg viewBox="0 0 40 40"><circle cx="20" cy="20" r="10" fill="none" stroke="black"/></svg>'
    parsed = parse_svg_to_primitives(svg)
    fitted = fit_primitives(parsed.primitives)
    optimized = optimize_primitives(fitted)
    tikz = export_primitives_to_tikz(optimized)
    result = validate_semantic_output(circle_image(), optimized, tikz)

    assert "\\draw" in tikz.code
    assert "circle" in tikz.code
    assert result.fidelity_score.semantic_score > 0.70
    assert result.complexity_metrics.tikz_raw_path_penalty == 0.0


def test_no_converter_tracer_or_gui_dependency_and_modes_are_preserved() -> None:
    sources = [
        Path("fikzpy/core/visual_validation.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/fidelity_score.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/raster_metrics.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/complexity_metrics.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/semantic_rasterizer.py").read_text(encoding="utf-8").lower(),
    ]
    for source in sources:
        assert "svg2tikz" not in source
        assert "trace_image(" not in source
        assert "run_cli_tracer" not in source
        assert "subprocess" not in source
        assert "pyside6" not in source
        assert "qapplication" not in source

    assert config_for_mode("classic").mode == "classic"
    assert config_for_mode("visual").mode == "visual"
    assert config_for_mode("contours").mode == "contours"
    for path in ("fikzpy/core/visual_pipeline.py", "fikzpy/core/visual_postprocessor.py", "fikzpy/core/tikz_pipeline.py", "fikzpy/core/image_processor.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert "visual_validation" not in text
        assert "fidelity_score" not in text


def test_importing_validation_modules_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.visual_validation; "
        "import fikzpy.core.semantic_rasterizer; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_raster_summary_and_report_json_fixture_exist() -> None:
    summary = raster_summary(draw_line_image(), RasterMetricsConfig())
    assert summary["foreground_pixels"] > 0
    path = Path("examples/classic_semantic_baseline/visual_validation_report.json")
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["issue"] == "Issue 10"
        assert {"perfect_line", "mixed_monochrome_bad", "tikz_raw_like_penalty"} <= {case["name"] for case in report["cases"]}
