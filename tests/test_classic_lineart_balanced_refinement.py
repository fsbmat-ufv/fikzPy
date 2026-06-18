from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fikzpy.core.classic_pipeline_config import ClassicPipelineStrategy, ClassicSemanticConfig
from fikzpy.core.classic_semantic_pipeline import run_classic_semantic_pipeline
from fikzpy.core.lineart_continuity import LineArtContinuityMetrics, compute_lineart_continuity_metrics
from fikzpy.core.lineart_continuity import validate_lineart_balance
from fikzpy.core.semantic_geometry import ClosedShapePrimitive, FillStyle, LinePrimitive, Point2D, RGBColor
from fikzpy.core.tikz_pipeline import build_tikz_from_image
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.tikz_generator import TikzOptions


def p(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y))


def line_art_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.line(image, (12, 20), (84, 20), (0, 0, 0), 1)
    cv2.line(image, (20, 40), (78, 70), (0, 0, 0), 1)
    cv2.circle(image, (48, 56), 18, (0, 0, 0), 1)
    return image


def filled_rectangle_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 18), (72, 76), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 34), (56, 52), (255, 255, 255), -1)
    return image


def silhouette_image() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.circle(image, (48, 30), 16, (0, 0, 0), -1)
    cv2.rectangle(image, (34, 46), (62, 80), (0, 0, 0), -1)
    return image


def mixed_monochrome_image() -> np.ndarray:
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.circle(image, (42, 34), 18, (0, 0, 0), -1)
    cv2.circle(image, (86, 36), 16, (0, 0, 0), -1)
    cv2.rectangle(image, (24, 70), (52, 104), (0, 0, 0), -1)
    cv2.rectangle(image, (76, 68), (105, 105), (0, 0, 0), -1)
    cv2.circle(image, (42, 39), 10, (255, 255, 255), -1)
    cv2.circle(image, (86, 41), 9, (255, 255, 255), -1)
    cv2.line(image, (30, 40), (37, 40), (0, 0, 0), 1)
    cv2.line(image, (46, 40), (52, 40), (0, 0, 0), 1)
    cv2.line(image, (38, 48), (47, 50), (0, 0, 0), 1)
    cv2.line(image, (79, 41), (84, 41), (0, 0, 0), 1)
    cv2.line(image, (90, 41), (96, 41), (0, 0, 0), 1)
    cv2.line(image, (20, 62), (58, 62), (0, 0, 0), 1)
    cv2.line(image, (72, 61), (110, 61), (0, 0, 0), 1)
    cv2.line(image, (60, 76), (72, 90), (0, 0, 0), 1)
    cv2.line(image, (64, 88), (70, 96), (0, 0, 0), 1)
    return image


def dinosaur_lineart_image() -> np.ndarray:
    """Synthetic line-art dinosaur: long external contour, head, teeth, legs,
    belly, internal lines, and a tail, drawn entirely in thin black strokes."""
    image = np.full((160, 220, 3), 255, dtype=np.uint8)
    body = np.array(
        [
            [30, 120], [40, 90], [60, 70], [90, 55], [120, 50], [150, 55],
            [170, 50], [185, 60], [195, 75], [190, 95], [175, 110], [160, 120],
            [150, 140], [140, 150], [120, 150], [110, 140], [95, 150],
            [85, 150], [70, 140], [55, 130], [40, 130], [30, 120],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [body], True, (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (185, 60), (205, 55), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (205, 55), (195, 75), (0, 0, 0), 1, cv2.LINE_8)
    for x in range(190, 204, 4):
        cv2.line(image, (x, 58), (x, 66), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (60, 140), (55, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (100, 148), (98, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (130, 150), (128, 158), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (60, 110), (140, 120), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (80, 90), (90, 110), (0, 0, 0), 1, cv2.LINE_8)
    cv2.line(image, (110, 80), (115, 105), (0, 0, 0), 1, cv2.LINE_8)
    cv2.circle(image, (175, 68), 2, (0, 0, 0), -1, cv2.LINE_8)
    return image


def _flat_continuity(**overrides) -> LineArtContinuityMetrics:
    base = {
        "components_before": 8,
        "components_after": 8,
        "lost_component_count": 0,
        "endpoint_count": 4,
        "junction_count": 2,
        "path_count": 8,
        "broken_path_count": 0,
        "average_path_length": 40.0,
        "contour_coverage": 0.95,
        "edge_recall": 0.95,
        "foreground_recall": 0.95,
        "skeleton_fragmentation": 0.0,
        "contour_bbox_coverage": 0.95,
        "external_contour_preservation": 0.95,
    }
    base.update(overrides)
    return LineArtContinuityMetrics(**base)


def test_default_config_lineart_fields_are_balanced() -> None:
    config = ClassicSemanticConfig()
    assert 0.35 <= config.line_art_stroke_width <= 0.55
    assert config.line_art_stroke_width != 1.0
    assert config.enable_lineart_outline_recovery is True
    assert config.reject_overfilled_lineart is True
    assert config.reject_underdrawn_lineart is True
    with pytest.raises(ValueError):
        ClassicSemanticConfig(lineart_min_edge_recall=1.5)
    with pytest.raises(ValueError):
        ClassicSemanticConfig(line_art_stroke_width=0.0)


def test_simple_line_art_accepted_with_fine_stroke_and_no_fill() -> None:
    result = run_classic_semantic_pipeline(line_art_image())
    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.accepted
    assert result.metrics.filled_region_primitives == 0
    assert result.metrics.white_cutout_count == 0
    assert "fill=" not in result.tikz_code
    assert "line width=1pt" not in result.tikz_code
    assert "line width=0.45pt" in result.tikz_code or "line width=.45pt" in result.tikz_code


def test_filled_rectangle_and_silhouette_still_use_fill() -> None:
    rectangle = run_classic_semantic_pipeline(filled_rectangle_image())
    assert rectangle.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert rectangle.metrics.filled_region_primitives > 0
    assert "fill=" in rectangle.tikz_code

    silhouette = run_classic_semantic_pipeline(silhouette_image())
    assert silhouette.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert silhouette.metrics.filled_region_primitives > 0
    assert silhouette.accepted


def test_mixed_monochrome_real_continues_accepted_and_bad_continues_rejected() -> None:
    good = run_classic_semantic_pipeline(mixed_monochrome_image())
    assert good.strategy_used is ClassicPipelineStrategy.MIXED_MONOCHROME
    assert good.accepted
    assert good.metrics.filled_region_primitives > 0
    assert good.metrics.thin_stroke_primitives > 0

    bad_primitives = [
        LinePrimitive(p(30, 40), p(37, 40)),
        LinePrimitive(p(46, 40), p(52, 40)),
        LinePrimitive(p(20, 62), p(58, 62)),
    ]
    from fikzpy.core.visual_validation import VisualValidationConfig, validate_semantic_output

    bad = validate_semantic_output(
        mixed_monochrome_image(),
        bad_primitives,
        config=VisualValidationConfig(minimum_acceptable_score=0.45, minimum_fidelity_score=0.35),
    )
    assert not bad.accepted


def test_dinosaur_synthetic_good_is_accepted_as_lineart() -> None:
    result = run_classic_semantic_pipeline(dinosaur_lineart_image())
    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.accepted
    assert result.metrics.filled_region_primitives == 0
    assert result.metrics.white_cutout_count == 0
    assert result.metrics.edge_recall > 0.5
    assert result.metrics.contour_coverage > 0.5
    assert "artificial_black_mass" not in result.metrics.lineart_regression_flags
    assert "underdrawn_lineart" not in result.metrics.lineart_regression_flags


def test_dinosaur_synthetic_overfilled_is_rejected() -> None:
    source = dinosaur_lineart_image()
    continuity = _flat_continuity()
    overfilled_primitive = ClosedShapePrimitive(
        (p(10, 10), p(200, 10), p(200, 150), p(10, 150)),
        fill=FillStyle(RGBColor(0, 0, 0)),
    )
    result = validate_lineart_balance(
        (cv2.cvtColor(source, cv2.COLOR_RGB2GRAY) < 200),
        [overfilled_primitive],
        continuity,
        max_filled_area_ratio_for_lineart=0.06,
        max_white_cutout_ratio_for_lineart=0.03,
        lineart_min_edge_recall=0.35,
        lineart_min_foreground_recall=0.55,
        lineart_min_contour_coverage=0.55,
        lineart_max_fragmentation_ratio=0.6,
        lineart_preserve_external_contour=True,
        reject_overfilled_lineart=True,
        reject_underdrawn_lineart=True,
    )
    assert not result.accepted
    assert "overfilled_lineart_rejected" in result.rejection_reasons
    assert {"artificial_black_mass", "overfilled_lineart", "excessive_filled_area"} & set(result.flags)


def test_dinosaur_synthetic_underdrawn_is_rejected() -> None:
    source = dinosaur_lineart_image()
    underdrawn_continuity = _flat_continuity(
        foreground_recall=0.10,
        contour_coverage=0.12,
        edge_recall=0.08,
        skeleton_fragmentation=0.9,
        external_contour_preservation=0.10,
        lost_component_count=5,
        components_before=8,
    )
    sparse_primitives = [LinePrimitive(p(30, 40), p(37, 40))]
    result = validate_lineart_balance(
        (cv2.cvtColor(source, cv2.COLOR_RGB2GRAY) < 200),
        sparse_primitives,
        underdrawn_continuity,
        max_filled_area_ratio_for_lineart=0.06,
        max_white_cutout_ratio_for_lineart=0.03,
        lineart_min_edge_recall=0.35,
        lineart_min_foreground_recall=0.55,
        lineart_min_contour_coverage=0.55,
        lineart_max_fragmentation_ratio=0.6,
        lineart_preserve_external_contour=True,
        reject_overfilled_lineart=True,
        reject_underdrawn_lineart=True,
    )
    assert not result.accepted
    assert "underdrawn_lineart_rejected" in result.rejection_reasons
    assert "underdrawn_lineart" in result.flags
    assert {"lost_contour_structure", "low_edge_recall"} & set(result.flags)
    assert "lost_external_contour" in result.flags
    assert "missing_internal_details" in result.flags


def test_good_lineart_does_not_get_overfill_or_underdraw_flags() -> None:
    result = run_classic_semantic_pipeline(dinosaur_lineart_image())
    flags = set(result.metrics.lineart_regression_flags)
    assert not flags & {"artificial_black_mass", "overfilled_lineart", "excessive_filled_area"}
    assert "underdrawn_lineart" not in flags


def test_outline_recovery_triggers_without_fill_or_external_calls() -> None:
    source = dinosaur_lineart_image()
    config = ClassicSemanticConfig(
        lineart_min_edge_recall=0.95,
        lineart_min_foreground_recall=0.55,
        lineart_min_contour_coverage=0.55,
    )
    result = run_classic_semantic_pipeline(source, config)
    assert result.metrics.outline_recovery_count >= 1
    assert "fill=" not in result.tikz_code
    assert "svg2tikz" not in result.tikz_code.lower()


def test_outline_recovery_disabled_falls_back_to_thin_strokes_only() -> None:
    source = dinosaur_lineart_image()
    config = ClassicSemanticConfig(
        enable_lineart_outline_recovery=False,
        lineart_min_edge_recall=0.95,
    )
    result = run_classic_semantic_pipeline(source, config)
    assert result.metrics.outline_recovery_count == 0


def test_no_near_empty_output_for_good_dinosaur() -> None:
    result = run_classic_semantic_pipeline(dinosaur_lineart_image())
    assert result.metrics.tikz_draw_commands > 0
    assert result.metrics.optimized_primitive_count > 0


def test_pipeline_is_deterministic_for_lineart_outline_recovery() -> None:
    source = dinosaur_lineart_image()
    config = ClassicSemanticConfig(lineart_min_edge_recall=0.95)
    first = run_classic_semantic_pipeline(source, config)
    second = run_classic_semantic_pipeline(source, config)
    assert first.deterministic_hash == second.deterministic_hash
    assert first.tikz_code == second.tikz_code
    assert first.metrics.to_dict() == second.metrics.to_dict()


def test_to_dict_includes_new_lineart_metrics() -> None:
    result = run_classic_semantic_pipeline(dinosaur_lineart_image())
    data = result.metrics.to_dict()
    for key in (
        "tikz_fill_commands",
        "outline_recovery_count",
        "white_cutout_count",
        "edge_recall",
        "contour_coverage",
        "fragmentation_ratio",
        "lineart_regression_flags",
    ):
        assert key in data
    json.dumps(result.to_dict(), sort_keys=True)


def test_visual_and_contours_modes_remain_unaffected() -> None:
    visual = build_tikz_from_image(filled_rectangle_image(), ProcessingSettings(vectorization_mode="visual"), TikzOptions())
    assert visual.effective_mode == "visual"
    assert visual.classic_semantic_result is None
    assert visual.visual_stats.used_svg2tikz

    contours = build_tikz_from_image(filled_rectangle_image(), ProcessingSettings(vectorization_mode="contours"), TikzOptions())
    assert contours.effective_mode == "contours"
    assert contours.classic_semantic_result is None


def test_classic_semantic_modules_avoid_svg2tikz_and_external_tracers() -> None:
    modules = [
        Path("fikzpy/core/lineart_continuity.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/classic_semantic_pipeline.py").read_text(encoding="utf-8").lower(),
    ]
    for source in modules:
        assert "svg2tikz" not in source
        assert "subprocess" not in source
        assert "potrace" not in source
        assert "vtracer" not in source
        assert "pyside6" not in source


def test_compute_lineart_continuity_metrics_handles_empty_mask() -> None:
    empty_mask = np.zeros((40, 40), dtype=bool)
    metrics = compute_lineart_continuity_metrics(empty_mask, None)
    assert metrics.foreground_recall == 1.0
    assert metrics.components_before == 0
