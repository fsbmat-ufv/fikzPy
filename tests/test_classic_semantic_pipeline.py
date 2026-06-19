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
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.semantic_geometry import FillStyle, LinePrimitive, Point2D, PolylinePrimitive, RGBColor
from fikzpy.core.semantic_geometry import StrokeStyle
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


def test_default_config_invalid_config_empty_white_and_black_inputs() -> None:
    config = ClassicSemanticConfig()
    assert config.enable_semantic_classic
    assert config.strategy is ClassicPipelineStrategy.AUTO
    assert config.fallback_policy is ClassicFallbackPolicy.REJECT_RESULT
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


def test_color_regions_conservative_warning_and_silhouette_acceptance() -> None:
    color = run_classic_semantic_pipeline(color_regions_image())
    assert color.strategy_used is ClassicPipelineStrategy.COLOR_REGIONS
    assert any(warning.code == "color_regions_conservative" for warning in color.warnings)

    silhouette = run_classic_semantic_pipeline(silhouette_image())
    assert silhouette.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert silhouette.metrics.filled_region_primitives > 0
    assert silhouette.accepted


def test_dino_photo_classic_line_art_is_not_overfilled_into_silhouette() -> None:
    image_path = Path(__file__).resolve().parent / "25.jpg"
    if not image_path.exists():
        pytest.skip("tests/25.jpg is not present in this checkout")

    result = run_classic_semantic_pipeline(str(image_path))

    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.metrics.filled_region_primitives <= 3
    assert result.metrics.tikz_fill_commands <= 3

    if result.metrics.filled_region_primitives >= 30 or result.metrics.tikz_fill_commands >= 30:
        assert result.metrics.lineart_regression_flags

    first_draw_index = result.tikz_code.find("\\draw")
    assert first_draw_index != -1
    first_draw_block = result.tikz_code[first_draw_index : result.tikz_code.find("\\draw", first_draw_index + 1)]
    assert "fill={rgb,255:red,0;green,0;blue,0}" not in first_draw_block


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


def test_classic_line_art_strategy_forces_strokes_with_no_fill_on_dino_photo() -> None:
    image_path = Path(__file__).resolve().parent / "25.jpg"
    if not image_path.exists():
        pytest.skip("tests/25.jpg is not present in this checkout")

    config = ClassicSemanticConfig(vectorization_strategy="line_art")
    result = run_classic_semantic_pipeline(str(image_path), config)

    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART
    assert result.metrics.tikz_fill_commands == 0
    # Strategy may still be rejected as underdrawn, but it must never become a filled silhouette.
    assert "fill={rgb,255:red,0;green,0;blue,0}" not in result.tikz_code


def test_classic_filled_strategy_generates_fill_for_simple_silhouette() -> None:
    config = ClassicSemanticConfig(vectorization_strategy="filled")
    result = run_classic_semantic_pipeline(silhouette_image(), config)

    assert result.strategy_used is ClassicPipelineStrategy.BINARY_OUTLINE
    assert result.metrics.filled_region_primitives > 0
    assert result.metrics.thin_stroke_primitives == 0
    assert "fill=" in result.tikz_code
    assert result.accepted or result.rejection_reasons


def test_classic_auto_strategy_prefers_line_art_when_ambiguous() -> None:
    config = ClassicSemanticConfig(vectorization_strategy="auto")
    result = run_classic_semantic_pipeline(line_art_image(), config)

    assert result.strategy_used is ClassicPipelineStrategy.LINE_ART


def test_classic_auto_strategy_warns_instead_of_silent_fallback_on_mixed_complex_image() -> None:
    image_path = Path(__file__).resolve().parent / "8.jpg"
    if not image_path.exists():
        pytest.skip("tests/8.jpg is not present in this checkout")

    config = ClassicSemanticConfig(vectorization_strategy="auto")
    result = run_classic_semantic_pipeline(str(image_path), config)

    if not result.accepted:
        assert any(warning.code == "classic_auto_recommend_visual" for warning in result.warnings)
    assert "visual" not in result.tikz_code.lower()
