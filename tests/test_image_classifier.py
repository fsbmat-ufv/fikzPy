from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import fikzpy.core.image_classifier as image_classifier
from fikzpy.core.image_classifier import ImageCategory, ImageClassifierConfig, classify_image


def _white_canvas(width: int = 120, height: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_white_image_with_thin_black_lines_is_line_art() -> None:
    image = _white_canvas()
    cv2.line(image, (12, 20), (108, 20), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (24, 76), (96, 30), (0, 0, 0), 1, cv2.LINE_AA)

    result = classify_image(image)

    assert result.category is ImageCategory.LINE_ART
    assert not result.manual_override
    assert result.metrics.foreground_ratio < 0.08


def test_simple_black_and_white_stroke_drawing_is_line_art() -> None:
    image = _white_canvas()
    cv2.circle(image, (58, 48), 24, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (30, 48), (86, 48), (0, 0, 0), 1, cv2.LINE_AA)

    result = classify_image(image)

    assert result.category is ImageCategory.LINE_ART
    assert result.metrics.edge_to_foreground_ratio > 0.25


def test_filled_black_circle_is_binary_outline() -> None:
    image = _white_canvas()
    cv2.circle(image, (60, 50), 32, (0, 0, 0), -1, cv2.LINE_AA)

    result = classify_image(image)

    assert result.category is ImageCategory.BINARY_OUTLINE
    assert result.metrics.foreground_ratio > 0.20


def test_large_black_silhouette_is_binary_outline() -> None:
    image = _white_canvas()
    cv2.rectangle(image, (18, 16), (102, 86), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (38, 30), 18, (0, 0, 0), -1, cv2.LINE_AA)

    result = classify_image(image)

    assert result.category is ImageCategory.BINARY_OUTLINE
    assert result.metrics.dark_pixel_ratio > 0.40


def test_rgb_icon_with_filled_regions_is_color_regions() -> None:
    image = _white_canvas(150, 100)
    cv2.rectangle(image, (10, 20), (48, 78), (0, 0, 255), -1)
    cv2.rectangle(image, (56, 20), (94, 78), (0, 180, 0), -1)
    cv2.rectangle(image, (102, 20), (140, 78), (255, 0, 0), -1)

    result = classify_image(image)

    assert result.category is ImageCategory.COLOR_REGIONS
    assert result.metrics.colored_pixel_ratio > 0.35
    assert result.metrics.effective_color_count >= 4


def test_low_saturation_color_image_is_not_color_regions() -> None:
    image = np.full((90, 120, 3), 238, dtype=np.uint8)
    image[:, :60] = (164, 168, 170)
    image[:, 60:] = (190, 185, 181)

    result = classify_image(image)

    assert result.category is not ImageCategory.COLOR_REGIONS
    assert result.metrics.colored_pixel_ratio == 0.0


def test_grayscale_image_with_many_tones_is_handled() -> None:
    gradient = np.tile(np.linspace(30, 230, 120, dtype=np.uint8), (80, 1))

    result = classify_image(gradient)

    assert result.category in {ImageCategory.LINE_ART, ImageCategory.BINARY_OUTLINE}
    assert result.metrics.tonal_level_count > 2
    assert result.metrics.mean_saturation == 0.0


def test_totally_white_image_is_blank_line_art() -> None:
    result = classify_image(_white_canvas())

    assert result.category is ImageCategory.LINE_ART
    assert result.metrics.uniform
    assert "blank line art" in " ".join(result.reasons)


def test_totally_black_image_is_filled_binary_outline() -> None:
    result = classify_image(np.zeros((60, 80, 3), dtype=np.uint8))

    assert result.category is ImageCategory.BINARY_OUTLINE
    assert result.metrics.uniform
    assert result.metrics.dark_pixel_ratio == 1.0


def test_rgba_transparency_is_composited_over_white() -> None:
    image = np.zeros((80, 100, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    image[20:60, 30:70, :3] = (0, 0, 255)
    image[20:60, 30:70, 3] = 255

    result = classify_image(image)

    assert result.category is ImageCategory.COLOR_REGIONS
    assert 0.15 < result.metrics.foreground_ratio < 0.25


def test_ambiguous_result_marks_alternative_category() -> None:
    image = _white_canvas()
    cv2.rectangle(image, (18, 18), (58, 58), (0, 0, 0), -1)
    cv2.line(image, (66, 22), (110, 68), (0, 0, 0), 1, cv2.LINE_AA)
    config = ImageClassifierConfig(ambiguity_margin=0.35)

    result = classify_image(image, config=config)

    assert result.ambiguous
    assert result.alternative_category is not None
    assert result.alternative_category is not result.category


def test_manual_override_returns_requested_category_with_metrics() -> None:
    image = _white_canvas()
    cv2.circle(image, (60, 50), 30, (0, 0, 0), -1)

    result = classify_image(image, override=ImageCategory.LINE_ART)

    assert result.category is ImageCategory.LINE_ART
    assert result.manual_override
    assert not result.ambiguous
    assert result.confidence == 1.0
    assert result.metrics.foreground_ratio > 0.0


def test_result_is_deterministic_for_same_image_and_config() -> None:
    image = _white_canvas()
    cv2.circle(image, (60, 50), 28, (0, 0, 0), 2, cv2.LINE_AA)
    config = ImageClassifierConfig()

    first = classify_image(image, config=config).to_dict()
    second = classify_image(image, config=config).to_dict()

    assert first == second


def test_result_to_dict_contains_diagnostics() -> None:
    result = classify_image(_white_canvas())
    serialized = result.to_dict()

    assert serialized["category"] == "line_art"
    assert 0.0 <= serialized["confidence"] <= 1.0
    assert "metrics" in serialized
    assert "reasons" in serialized
    assert "alternative_category" in serialized
    assert "scores" in serialized


@pytest.mark.parametrize(
    "image",
    [
        np.array([], dtype=np.uint8),
        np.zeros((3, 3, 2), dtype=np.uint8),
        np.array([[float("nan")]], dtype=np.float64),
        np.array([[float("inf")]], dtype=np.float64),
        np.array([[-1]], dtype=np.int16),
        np.array([[300]], dtype=np.int16),
    ],
)
def test_invalid_image_values_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        classify_image(image)


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_image(_white_canvas(), override="not-a-category")


def test_path_input_is_supported(tmp_path: Path) -> None:
    image = _white_canvas()
    cv2.line(image, (10, 10), (90, 70), (0, 0, 0), 1)
    path = tmp_path / "line.png"
    cv2.imwrite(str(path), image)

    result = classify_image(path)

    assert result.category is ImageCategory.LINE_ART


def test_importing_classifier_does_not_start_gui_or_application() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.image_classifier; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_classifier_source_does_not_call_external_tracers_or_svg_conversion() -> None:
    source = Path(image_classifier.__file__).read_text(encoding="utf-8").lower()

    for forbidden in ("potrace", "autotrace", "vtracer", "svg2tikz", "subprocess"):
        assert forbidden not in source
