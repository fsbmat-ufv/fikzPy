from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from fikzpy.core.classic_pipeline_config import ClassicFallbackPolicy, ClassicPipelineStrategy
from fikzpy.core.classic_pipeline_config import ClassicSemanticConfig, ClassicValidationPolicy
from fikzpy.core.classic_semantic_pipeline import ClassicSemanticPipeline, detect_mixed_monochrome_image
from fikzpy.core.classic_semantic_pipeline import run_classic_semantic_pipeline
from fikzpy.core.filled_region_extraction import FilledRegionDecisionKind, extract_filled_regions
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.lineart_diagnostics import StrokeFillClassification, analyze_line_art_mask
from fikzpy.core.mixed_monochrome_pipeline import split_foreground_layers
from fikzpy.core.semantic_geometry import ClosedShapePrimitive, FillStyle, LinePrimitive, Point2D, PolylinePrimitive, RGBColor
from fikzpy.core.semantic_geometry import StrokeStyle
from fikzpy.core.semantic_tikz_exporter import export_primitives_to_tikz
from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.tikz_pipeline import build_tikz_from_image
from fikzpy.core.visual_validation import VisualValidationConfig, validate_semantic_output


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
    image = np.full((128, 168, 3), 255, dtype=np.uint8)
    body = np.array(
        [(20, 78), (32, 50), (62, 35), (104, 38), (132, 55), (148, 75), (132, 90), (96, 92), (72, 104), (44, 100)],
        dtype=np.int32,
    )
    cv2.polylines(image, [body], True, (0, 0, 0), 2)
    cv2.circle(image, (117, 55), 3, (0, 0, 0), 1)
    for x in (132, 138, 144):
        cv2.line(image, (x, 68), (x - 3, 75), (0, 0, 0), 1)
    for x in (47, 58, 90):
        cv2.line(image, (x, 96), (x - 5, 116), (0, 0, 0), 2)
        cv2.line(image, (x - 5, 116), (x + 6, 116), (0, 0, 0), 1)
    cv2.line(image, (70, 56), (102, 61), (0, 0, 0), 1)
    cv2.line(image, (67, 70), (102, 75), (0, 0, 0), 1)
    cv2.line(image, (24, 75), (7, 68), (0, 0, 0), 2)
    cv2.line(image, (8, 68), (18, 62), (0, 0, 0), 1)
    return image


def closed_contour_lineart_image() -> np.ndarray:
    image = np.full((112, 112, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (56, 56), (36, 26), 0, 0, 360, (0, 0, 0), 2)
    cv2.circle(image, (45, 50), 4, (0, 0, 0), 1)
    cv2.line(image, (30, 66), (82, 68), (0, 0, 0), 1)
    cv2.line(image, (44, 78), (40, 94), (0, 0, 0), 2)
    cv2.line(image, (68, 78), (74, 94), (0, 0, 0), 2)
    return image


def color_regions_image() -> np.ndarray:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[10:40, 8:38] = (220, 30, 30)
    image[42:72, 42:72] = (30, 70, 220)
    cv2.rectangle(image, (18, 18), (30, 30), (0, 0, 0), -1)
    return image


def bad_thin_only_primitives() -> list[LinePrimitive]:
    return [
        LinePrimitive(p(30, 40), p(37, 40)),
        LinePrimitive(p(46, 40), p(52, 40)),
        LinePrimitive(p(38, 48), p(47, 50)),
        LinePrimitive(p(20, 62), p(58, 62)),
        LinePrimitive(p(72, 61), p(110, 61)),
    ]


def bad_overfilled_dinosaur_primitives() -> tuple[ClosedShapePrimitive, ...]:
    white_fill = FillStyle(RGBColor(255, 255, 255))
    white_cutout_stroke = StrokeStyle(RGBColor(255, 255, 255), width=0.1, opacity=0.0)
    return (
        ClosedShapePrimitive(
            (p(18, 34), p(150, 34), p(158, 98), p(26, 108)),
            stroke=StrokeStyle(RGBColor.black(), width=1.0),
            fill=FillStyle(RGBColor.black()),
            metadata={"source_layer": "filled_region"},
        ),
        ClosedShapePrimitive(
            (p(38, 52), p(118, 48), p(128, 78), p(42, 84)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (p(44, 86), p(78, 84), p(76, 96), p(42, 98)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (p(92, 84), p(132, 82), p(130, 94), p(88, 98)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (p(108, 50), p(124, 50), p(124, 62), p(108, 62)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
    )


def test_default_config_invalid_config_empty_white_and_black_inputs() -> None:
    config = ClassicSemanticConfig()
    assert config.enable_semantic_classic
    assert config.strategy is ClassicPipelineStrategy.AUTO
    assert config.fallback_policy is ClassicFallbackPolicy.REJECT_RESULT
    assert config.prefer_lineart_when_ambiguous
    assert config.line_art_stroke_width == pytest.approx(0.4)
    assert config.mixed_line_art_stroke_width == pytest.approx(0.45)
    assert config.filled_region_draw_outline is False
    assert config.reject_overfilled_lineart
    assert config.validation_config().validate_lineart_fill_usage
    assert config.lineart_config().minimum_fill_ratio_for_filled_region >= 0.24
    strict_config = ClassicSemanticConfig(lineart_filled_region_strictness=1.5)
    assert strict_config.lineart_config().minimum_fill_ratio_for_filled_region == pytest.approx(0.36)
    with pytest.raises(ValueError):
        ClassicSemanticConfig(minimum_acceptance_score=1.5)
    with pytest.raises(ValueError):
        run_classic_semantic_pipeline(np.zeros((0, 0, 3), dtype=np.uint8))

    white = np.full((48, 48, 3), 255, dtype=np.uint8)
    white_result = run_classic_semantic_pipeline(
        white,
        replace(config, validation_policy=ClassicValidationPolicy.DISABLED),
    )
    assert white_result.tikz_code.strip() == r"\begin{tikzpicture}[scale=1]" + "\n  \\begin{scope}[line cap=round, line join=round]\n  \\end{scope}\n\\end{tikzpicture}"
    assert not white_result.accepted

    black = np.zeros((48, 48, 3), dtype=np.uint8)
    black_result = run_classic_semantic_pipeline(black)
    assert black_result.strategy_used in {ClassicPipelineStrategy.BINARY_OUTLINE, ClassicPipelineStrategy.MIXED_MONOCHROME}
    assert "\\draw" in black_result.tikz_code


def test_strategy_detection_line_art_binary_mixed_and_color() -> None:
    assert detect_mixed_monochrome_image(line_art_image()).strategy is ClassicPipelineStrategy.LINE_ART
    assert detect_mixed_monochrome_image(dinosaur_lineart_image()).strategy is ClassicPipelineStrategy.LINE_ART
    assert detect_mixed_monochrome_image(filled_rectangle_image()).strategy is ClassicPipelineStrategy.BINARY_OUTLINE
    mixed_decision = detect_mixed_monochrome_image(mixed_monochrome_image())
    assert mixed_decision.strategy is ClassicPipelineStrategy.MIXED_MONOCHROME
    assert mixed_decision.split is not None
    assert mixed_decision.split.filled_count > 0
    assert mixed_decision.split.thin_count > 0
    color_decision = detect_mixed_monochrome_image(color_regions_image())
    assert color_decision.strategy is ClassicPipelineStrategy.COLOR_REGIONS


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (ClassicPipelineStrategy.LINE_ART, "thin_stroke"),
        (ClassicPipelineStrategy.BINARY_OUTLINE, "filled_region"),
        (ClassicPipelineStrategy.MIXED_MONOCHROME, "mixed_monochrome"),
    ],
)
def test_manual_strategies_route_to_expected_extractors(strategy: ClassicPipelineStrategy, expected: str) -> None:
    result = run_classic_semantic_pipeline(
        mixed_monochrome_image(),
        ClassicSemanticConfig(strategy=strategy, auto_detect_strategy=False),
    )
    assert result.strategy_used is strategy
    if strategy is ClassicPipelineStrategy.LINE_ART:
        assert result.metrics.thin_stroke_primitives > 0
    elif strategy is ClassicPipelineStrategy.BINARY_OUTLINE:
        assert result.metrics.filled_region_primitives > 0
    else:
        assert "centerline" in result.extraction_result.strategy.value
        assert "filled_regions" in result.extraction_result.strategy.value
        assert result.metrics.thin_stroke_primitives > 0
        assert result.metrics.filled_region_primitives > 0


def test_mixed_monochrome_pipeline_preserves_black_regions_thin_strokes_and_tikz_semantics() -> None:
    source = mixed_monochrome_image()
    assert 0.16 <= float(np.mean(source[:, :, 0] < 128)) <= 0.24
    result = run_classic_semantic_pipeline(source)

    assert result.strategy_used is ClassicPipelineStrategy.MIXED_MONOCHROME
    assert result.accepted
    assert result.rejection_reasons == ()
    assert result.metrics.filled_region_primitives > 0
    assert result.metrics.thin_stroke_primitives > 0
    assert result.metrics.dark_mass_preservation_ratio > 0.75
    assert result.metrics.filled_region_recall > 0.70
    assert result.metrics.thin_stroke_recall > 0.40
    assert "\\draw" in result.tikz_code
    assert "fill=" in result.tikz_code
    assert "cycle" in result.tikz_code
    assert "svg2tikz" not in result.tikz_code.lower()
    assert result.raw_primitives and isinstance(result.raw_primitives[0], type(result.raw_primitives[0]))
    flattened = [primitive for group in result.raw_primitives if hasattr(group, "flatten") for primitive in group.flatten()]
    assert any(dict(primitive.metadata).get("source_layer") == "filled_region" for primitive in flattened)
    assert any(dict(primitive.metadata).get("source_layer") == "thin_stroke" for primitive in flattened)
    assert any("component_id" in dict(primitive.metadata) for primitive in flattened)


def test_bad_classic_like_output_is_rejected_for_mixed_monochrome_regression() -> None:
    source = mixed_monochrome_image()
    bad = validate_semantic_output(
        source,
        bad_thin_only_primitives(),
        config=VisualValidationConfig(minimum_acceptable_score=0.45, minimum_fidelity_score=0.35),
    )
    assert not bad.accepted
    assert bad.rendered_summary["foreground_ratio"] < 0.04
    assert bad.source_summary["foreground_ratio"] > 0.16
    assert {"missing_large_dark_regions", "low_filled_region_recall", "dark_mass_loss"} & set(bad.regression_flags)


def test_rejection_contains_reasons_when_validation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validate(*args, **kwargs):
        real = validate_semantic_output(*args, **kwargs)
        return replace(
            real,
            accepted=False,
            status=real.status.__class__.REJECTED,
            rejection_reasons=("missing_large_dark_regions", "near_empty_output"),
            regression_flags=("missing_large_dark_regions", "near_empty_output"),
        )

    monkeypatch.setattr("fikzpy.core.classic_semantic_pipeline.validate_semantic_output", fake_validate)
    result = run_classic_semantic_pipeline(mixed_monochrome_image())
    assert not result.accepted
    assert "missing_large_dark_regions" in result.rejection_reasons
    assert "near_empty_output" in result.rejection_reasons


def test_pipeline_calls_fitting_optimization_exporter_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    import fikzpy.core.classic_semantic_pipeline as pipeline

    real_fit = pipeline.fit_primitives
    real_optimize = pipeline.optimize_fit_results
    real_export = pipeline.export_primitives_to_tikz
    real_validate = pipeline.validate_semantic_output

    def fit_spy(*args, **kwargs):
        calls.append("fit")
        return real_fit(*args, **kwargs)

    def optimize_spy(*args, **kwargs):
        calls.append("optimize")
        return real_optimize(*args, **kwargs)

    def export_spy(*args, **kwargs):
        calls.append("export")
        return real_export(*args, **kwargs)

    def validate_spy(*args, **kwargs):
        calls.append("validate")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "fit_primitives", fit_spy)
    monkeypatch.setattr(pipeline, "optimize_fit_results", optimize_spy)
    monkeypatch.setattr(pipeline, "export_primitives_to_tikz", export_spy)
    monkeypatch.setattr(pipeline, "validate_semantic_output", validate_spy)

    result = run_classic_semantic_pipeline(filled_rectangle_image())
    assert result.tikz_code
    assert calls == ["fit", "optimize", "export", "validate"]


def test_to_dict_hash_determinism_warnings_metrics_and_json_serializable() -> None:
    config = ClassicSemanticConfig(prefer_external_filled_region_backend=True, allow_external_tracers=False)
    first = run_classic_semantic_pipeline(mixed_monochrome_image(), config)
    second = run_classic_semantic_pipeline(mixed_monochrome_image(), config)
    assert first.deterministic_hash == second.deterministic_hash
    assert first.tikz_code == second.tikz_code
    assert first.metrics.to_dict() == second.metrics.to_dict()
    assert any(warning.code == "external_filled_region_backend_disabled" for warning in first.warnings)
    data = first.to_dict()
    assert data["metrics"]["optimized_primitive_count"] > 0
    assert data["stage_results"]
    json.dumps(data, sort_keys=True)


def test_classic_gui_route_uses_semantic_pipeline_visual_and_contours_preserved() -> None:
    classic = build_tikz_from_image(line_art_image(), ProcessingSettings(vectorization_mode="classic"), TikzOptions())
    assert classic.effective_mode == "classic"
    assert classic.classic_semantic_result is not None
    assert "\\draw" in classic.tikz_code
    assert "svg2tikz" not in classic.tikz_code.lower()
    assert classic.processing_result.reconstruction_bgr.shape == line_art_image().shape

    visual = build_tikz_from_image(filled_rectangle_image(), ProcessingSettings(vectorization_mode="visual"), TikzOptions())
    assert visual.effective_mode == "visual"
    assert visual.classic_semantic_result is None
    assert visual.visual_stats.paths > 0
    assert visual.visual_stats.used_svg2tikz

    contours = build_tikz_from_image(filled_rectangle_image(), ProcessingSettings(vectorization_mode="contours"), TikzOptions())
    assert contours.effective_mode == "contours"
    assert contours.classic_semantic_result is None
    assert contours.processing_result.contours


def test_import_isolated_no_gui_latex_external_tracer_or_converter_dependency() -> None:
    modules = [
        Path("fikzpy/core/classic_semantic_pipeline.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/mixed_monochrome_pipeline.py").read_text(encoding="utf-8").lower(),
        Path("fikzpy/core/filled_region_extraction.py").read_text(encoding="utf-8").lower(),
    ]
    for source in modules:
        assert "svg2tikz" not in source
        assert "run_cli_tracer" not in source
        assert "trace_image(" not in source
        assert "subprocess" not in source
        assert "pyside6" not in source
        assert "pdflatex" not in source
        assert "vtracer" not in source
        assert "potrace" not in source

    script = (
        "import sys; "
        "import fikzpy.core.classic_semantic_pipeline; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_gui_source_keeps_classic_visual_and_contours_labels() -> None:
    source = Path("fikzpy/gui/main_window.py").read_text(encoding="utf-8")
    assert '"Classic", "classic"' in source
    assert '"Visual", "visual"' in source
    assert '"Contornos", "contours"' in source


def test_binary_filled_region_uses_fill_and_line_art_uses_centerline_metadata() -> None:
    binary = run_classic_semantic_pipeline(filled_rectangle_image())
    assert binary.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert binary.metrics.filled_region_primitives > 0
    assert "fill=" in binary.tikz_code
    assert "cycle" in binary.tikz_code

    line = run_classic_semantic_pipeline(line_art_image())
    assert line.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert line.metrics.thin_stroke_primitives > 0
    assert "\\draw" in line.tikz_code


def test_dinosaur_lineart_prefers_centerline_strokes_and_light_tikz() -> None:
    result = run_classic_semantic_pipeline(dinosaur_lineart_image())

    assert result.accepted
    assert result.rejection_reasons == ()
    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.metrics.line_art_confidence > 0.75
    assert result.metrics.mixed_monochrome_confidence < 0.05
    assert result.metrics.thin_stroke_primitives >= 10
    assert result.metrics.filled_region_primitives == 0
    assert result.metrics.thin_stroke_primitives > result.metrics.filled_region_primitives
    assert result.metrics.white_cutout_count == 0
    assert result.metrics.filled_area_ratio == 0.0
    assert result.metrics.dark_mass_preservation_ratio < 1.5
    assert result.metrics.tikz_draw_commands >= 10
    assert result.tikz_code.count("\\draw") >= 10
    assert "line width=0.4pt" in result.tikz_code
    assert "line width=1pt" not in result.tikz_code
    assert "fill={rgb,255:red,0;green,0;blue,0}" not in result.tikz_code
    assert "fill={rgb,255:red,255;green,255;blue,255}" not in result.tikz_code
    assert "svg2tikz" not in result.tikz_code.lower()
    assert result.to_dict()["decision"]["lineart_diagnostics"]["line_art_confidence"] > 0.75


def test_closed_contour_lineart_and_large_hollow_components_stay_thin_strokes() -> None:
    source = closed_contour_lineart_image()
    result = run_classic_semantic_pipeline(source)

    assert result.accepted
    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.metrics.filled_region_primitives == 0
    assert result.metrics.thin_stroke_primitives > 0
    assert result.metrics.line_art_confidence > 0.65

    mask = source[:, :, 0] < 128
    diagnostics = analyze_line_art_mask(mask, ClassicSemanticConfig().lineart_config())
    assert diagnostics.large_hollow_component_count >= 1
    assert diagnostics.filled_component_count == 0
    assert any(
        item.classification is StrokeFillClassification.THIN_STROKE and item.decision_reason == "skeleton_explains_component"
        for item in diagnostics.component_metrics
    )


def test_split_foreground_layers_uses_fill_ratio_skeleton_ratio_and_compactness() -> None:
    drawing = np.zeros((120, 160), dtype=np.uint8)
    mask = drawing
    cv2.rectangle(mask, (12, 12), (62, 58), 1, -1)
    cv2.rectangle(mask, (92, 14), (148, 70), 1, 2)
    cv2.line(mask, (20, 96), (146, 102), 1, 2)
    mask = drawing.astype(bool)

    result = split_foreground_layers(mask, filled_region_min_area=48, filled_region_min_ratio=0.22, thin_stroke_max_width=3.2)
    summaries = result.component_summaries
    solid = max(summaries, key=lambda item: item["fill_ratio"])
    hollow_or_stroke = [item for item in summaries if item is not solid]

    assert result.filled_count == 1
    assert result.thin_count >= 2
    assert solid["decision"] == "filled_region"
    assert solid["compactness"] >= ClassicSemanticConfig().minimum_compactness_for_filled_region
    assert solid["skeleton_ratio"] <= ClassicSemanticConfig().maximum_skeleton_ratio_for_filled_region
    assert solid["fill_ratio"] >= ClassicSemanticConfig().minimum_fill_ratio_for_filled_region
    assert all(item["decision"] == "thin_stroke" for item in hollow_or_stroke)
    assert any(item["reason"] in {"skeleton_explains_component", "low_component_fill_ratio"} for item in hollow_or_stroke)


def test_filled_region_candidates_preserve_decision_metadata() -> None:
    drawing = np.zeros((96, 128), dtype=np.uint8)
    mask = drawing
    cv2.rectangle(mask, (14, 18), (58, 70), 1, -1)
    cv2.rectangle(mask, (78, 22), (116, 72), 1, 2)
    mask = drawing.astype(bool)

    result = extract_filled_regions(mask)
    accepted = [candidate for candidate in result.candidates if candidate.decision is FilledRegionDecisionKind.ACCEPTED and not candidate.hole]
    rejected = [candidate for candidate in result.candidates if candidate.decision is FilledRegionDecisionKind.REJECTED and not candidate.hole]

    assert result.region_count >= 1
    assert accepted
    assert rejected
    assert accepted[0].fill_ratio >= ClassicSemanticConfig().minimum_fill_ratio_for_filled_region
    assert accepted[0].skeleton_ratio <= ClassicSemanticConfig().maximum_skeleton_ratio_for_filled_region
    assert rejected[0].reason in {
        "component_skeleton_ratio_too_high",
        "skeleton_explains_component",
        "low_fill_ratio",
        "component_too_thin",
        "low_compactness",
    }


def test_bad_dinosaur_black_mass_with_white_cutouts_is_rejected() -> None:
    source = dinosaur_lineart_image()
    primitives = bad_overfilled_dinosaur_primitives()
    tikz_result = export_primitives_to_tikz(primitives)
    result = validate_semantic_output(source, primitives, tikz_result)

    assert not result.accepted
    assert tikz_result.metrics.filled_paths_written >= 5
    assert "\\draw[fill={rgb,255:red,0;green,0;blue,0}]" in tikz_result.code
    assert tikz_result.code.count("draw=none, fill={rgb,255:red,255;green,255;blue,255}") >= 4
    flags = set(result.regression_flags)
    assert "excessive_filled_area" in flags
    assert "artificial_black_mass" in flags
    assert {"overfilled_lineart", "lineart_converted_to_silhouette"} <= flags
    assert "excessive_white_cutouts" in flags
    assert "lineart_has_excessive_filled_area" in result.rejection_reasons
    assert "lineart_uses_excessive_white_cutouts" in result.rejection_reasons
    assert result.metrics.filled_region_metrics["filled_area_ratio"] > 0.30
    assert result.metrics.filled_region_metrics["white_cutout_count"] > ClassicSemanticConfig().max_white_cutout_count_for_lineart
    assert result.metrics.lineart_fill_metrics["source_is_line_art"]


def test_color_regions_conservative_warning_and_silhouette_acceptance() -> None:
    color = run_classic_semantic_pipeline(color_regions_image())
    assert color.strategy_used is ClassicPipelineStrategy.COLOR_REGIONS
    assert any(warning.code == "color_regions_conservative" for warning in color.warnings)

    silhouette = run_classic_semantic_pipeline(silhouette_image())
    assert silhouette.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert silhouette.metrics.filled_region_primitives > 0
    assert silhouette.accepted


def test_fill_styles_are_preserved_in_validator_for_good_manual_primitives() -> None:
    source = filled_rectangle_image()
    primitive = PolylinePrimitive(
        (p(24, 18), p(72, 18), p(72, 76), p(24, 76)),
        closed=True,
        stroke=StrokeStyle(RGBColor.black(), width=1.0),
        fill=FillStyle(RGBColor.black()),
    )
    result = validate_semantic_output(source, [primitive], config=VisualValidationConfig(minimum_acceptable_score=0.3))
    assert result.fidelity_score.raster_metrics.dark_mass_preservation_ratio > 0.5
