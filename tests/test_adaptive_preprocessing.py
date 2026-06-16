from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import fikzpy.core.adaptive_preprocessing as adaptive_preprocessing
import fikzpy.core.threshold_selector as threshold_selector
from fikzpy.core.adaptive_preprocessing import DenoiseMethod, PreprocessingConfig, preprocess_image
from fikzpy.core.image_classifier import ImageCategory, classify_image
from fikzpy.core.threshold_selector import ThresholdSelectionConfig, select_best_threshold


def _white(width: int = 120, height: int = 90) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_black_thin_lines_on_white_preprocess_to_line_mask() -> None:
    image = _white()
    cv2.line(image, (12, 20), (106, 20), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (24, 70), (94, 32), (0, 0, 0), 1, cv2.LINE_AA)

    result = preprocess_image(image, category=ImageCategory.LINE_ART)

    assert result.binary_mask[20, 60] == 255
    assert 0.0 < result.metrics.foreground_ratio < 0.12
    assert result.category is ImageCategory.LINE_ART


def test_weak_gray_lines_on_white_are_recovered() -> None:
    image = _white()
    cv2.line(image, (10, 44), (110, 44), (205, 205, 205), 1, cv2.LINE_AA)

    result = preprocess_image(image, category=ImageCategory.LINE_ART)

    assert result.binary_mask[44, 60] == 255
    assert result.threshold_result.selected.metrics.edge_coverage > 0.0


def test_white_lines_on_black_support_light_foreground() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.line(image, (10, 40), (110, 40), (255, 255, 255), 2, cv2.LINE_AA)
    config = PreprocessingConfig(foreground_is_dark=False)

    result = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert result.binary_mask[40, 60] == 255
    assert result.config.threshold_selection.foreground_is_dark is False


def test_filled_silhouette_produces_binary_outline_mask() -> None:
    image = _white()
    cv2.circle(image, (60, 45), 28, (0, 0, 0), -1, cv2.LINE_AA)

    result = preprocess_image(image, category=ImageCategory.BINARY_OUTLINE)

    assert result.metrics.foreground_ratio > 0.20
    assert result.metrics.component_count == 1


def test_uneven_background_can_use_illumination_correction() -> None:
    gradient = np.tile(np.linspace(210, 255, 140, dtype=np.uint8), (90, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB)
    cv2.line(image, (16, 52), (126, 52), (70, 70, 70), 2, cv2.LINE_AA)
    config = PreprocessingConfig(enable_illumination_correction=True, illumination_blur_kernel_size=31)

    result = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert result.binary_mask[52, 70] == 255
    assert result.corrected.shape == result.grayscale.shape


def test_salt_and_pepper_noise_can_be_removed_conservatively() -> None:
    image = _white()
    cv2.line(image, (10, 45), (110, 45), (0, 0, 0), 1)
    for y, x in ((5, 5), (12, 88), (70, 22), (80, 100), (30, 40)):
        image[y, x] = (0, 0, 0)
    config = PreprocessingConfig(
        denoise_method=DenoiseMethod.NONE,
        preserve_small_details=False,
        minimum_component_area=3,
        maximum_noise_removal_ratio=0.20,
    )

    result = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert result.metrics.removed_component_count > 0
    assert result.binary_mask[45, 60] == 255


def test_blurred_line_remains_detectable() -> None:
    image = _white()
    cv2.line(image, (12, 45), (108, 45), (0, 0, 0), 2)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    result = preprocess_image(image, category=ImageCategory.LINE_ART)

    assert result.binary_mask[45, 60] == 255


def test_morphological_close_can_bridge_small_gap() -> None:
    image = _white()
    cv2.line(image, (16, 45), (56, 45), (0, 0, 0), 1)
    cv2.line(image, (59, 45), (104, 45), (0, 0, 0), 1)
    config = PreprocessingConfig(morphological_close_iterations=1, morphology_kernel_size=3)

    result = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert result.binary_mask[45, 57] == 255


def test_uniform_white_image_is_handled_without_nan() -> None:
    result = preprocess_image(_white(), category=ImageCategory.LINE_ART)

    assert result.uniform
    assert result.metrics.foreground_ratio == 0.0
    assert "empty foreground mask" in result.warnings
    assert np.isfinite(result.threshold_result.score)


def test_uniform_black_image_is_handled_without_nan() -> None:
    result = preprocess_image(np.zeros((50, 60, 3), dtype=np.uint8), category=ImageCategory.BINARY_OUTLINE)

    assert result.uniform
    assert result.metrics.foreground_ratio == 1.0
    assert "full foreground mask" in result.warnings


def test_uniform_gray_image_is_handled() -> None:
    result = preprocess_image(np.full((50, 60), 128, dtype=np.uint8))

    assert result.uniform
    assert result.grayscale.shape == (50, 60)
    assert result.binary_mask.dtype == np.uint8


def test_rgba_transparency_is_composited_over_configured_background() -> None:
    image = np.zeros((70, 100, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    image[30:34, 12:88, :3] = (0, 0, 0)
    image[30:34, 12:88, 3] = 255

    result = preprocess_image(image, category=ImageCategory.LINE_ART)

    assert result.binary_mask[32, 50] == 255
    assert result.original[0, 0].tolist() == [255, 255, 255]


def test_color_image_uses_grayscale_diagnostic_only() -> None:
    image = _white(150, 80)
    cv2.rectangle(image, (10, 20), (45, 60), (255, 0, 0), -1)
    cv2.rectangle(image, (55, 20), (90, 60), (0, 255, 0), -1)
    cv2.rectangle(image, (100, 20), (135, 60), (0, 0, 255), -1)

    result = preprocess_image(image, category=ImageCategory.COLOR_REGIONS)

    assert result.category is ImageCategory.COLOR_REGIONS
    assert "color category uses grayscale diagnostic preprocessing only" in result.warnings


@pytest.mark.parametrize("bad", [np.array([]), np.array([[np.nan]]), np.array([[np.inf]])])
def test_invalid_images_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_image(bad)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PreprocessingConfig(autocontrast_low_percentile=90, autocontrast_high_percentile=10),
        lambda: PreprocessingConfig(denoise_method="not-a-method"),
        lambda: PreprocessingConfig(connectivity=6),
        lambda: ThresholdSelectionConfig(threshold_methods=("unknown",)),
        lambda: ThresholdSelectionConfig(min_foreground_ratio=0.8, max_foreground_ratio=0.1),
    ],
)
def test_invalid_configuration_is_rejected(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_preprocessing_is_deterministic() -> None:
    image = _white()
    cv2.circle(image, (60, 45), 22, (0, 0, 0), 2, cv2.LINE_AA)
    config = PreprocessingConfig()

    first = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)
    second = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert first.to_dict() == second.to_dict()
    assert np.array_equal(first.binary_mask, second.binary_mask)


def test_threshold_ranking_contains_candidates_in_score_order() -> None:
    gray = np.full((80, 100), 255, dtype=np.uint8)
    cv2.circle(gray, (50, 40), 20, 0, -1)

    result = select_best_threshold(gray)

    scores = [candidate.score for candidate in result.candidates]
    assert len(scores) >= 3
    assert scores == sorted(scores, reverse=True)
    assert result.selected is result.candidates[0]


def test_result_serialization_uses_array_summaries() -> None:
    result = preprocess_image(_white())
    serialized = result.to_dict()

    assert "sha256" in serialized["binary_mask"]
    assert "shape" in serialized["grayscale"]
    assert "ranking" in serialized["threshold_selection"]
    assert "array(" not in repr(serialized)


def test_preserve_small_details_keeps_tiny_component_by_default() -> None:
    image = _white()
    cv2.line(image, (10, 45), (110, 45), (0, 0, 0), 1)
    image[20, 20] = (0, 0, 0)
    config = PreprocessingConfig(minimum_component_area=8, preserve_small_details=True)

    result = preprocess_image(image, config=config, category=ImageCategory.LINE_ART)

    assert result.binary_mask[20, 20] == 255


def test_noise_cleanup_respects_maximum_removal_ratio() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 2:18] = 255
    mask[1, 1] = 255
    mask[2, 2] = 255
    config = PreprocessingConfig(
        minimum_component_area=4,
        preserve_small_details=False,
        maximum_noise_removal_ratio=0.06,
    )

    cleaned, removed_components, removed_ratio = adaptive_preprocessing.remove_small_components_conservatively(
        mask,
        config,
    )

    assert removed_components <= 1
    assert removed_ratio <= 0.06
    assert cleaned[10, 10] == 255


def test_modules_do_not_generate_tikz_or_call_external_tracers() -> None:
    sources = [
        Path(adaptive_preprocessing.__file__).read_text(encoding="utf-8").lower(),
        Path(threshold_selector.__file__).read_text(encoding="utf-8").lower(),
    ]
    forbidden = ("\\\\draw", "tikzpicture", "potrace", "autotrace", "vtracer", "svg2tikz")

    for source in sources:
        for token in forbidden:
            assert token not in source


def test_modules_do_not_use_later_line_art_graph_tools() -> None:
    sources = [
        Path(adaptive_preprocessing.__file__).read_text(encoding="utf-8").lower(),
        Path(threshold_selector.__file__).read_text(encoding="utf-8").lower(),
    ]

    for source in sources:
        assert "skeletonize" not in source
        assert "sknw" not in source


def test_importing_preprocessing_does_not_start_gui() -> None:
    code = (
        "import sys; "
        "import fikzpy.core.adaptive_preprocessing; "
        "import fikzpy.core.threshold_selector; "
        "assert 'PySide6' not in sys.modules; "
        "assert 'fikzpy.gui.main_window' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_baseline_images_are_read_only_and_select_methods() -> None:
    baseline = Path("examples/classic_semantic_baseline")
    methods: dict[str, str] = {}
    for image_path in sorted(baseline.glob("*.png")):
        category = classify_image(image_path).category
        result = preprocess_image(image_path, category=category)
        methods[image_path.name] = result.method

    assert set(methods) == {
        "geometric_diagram.png",
        "line_art_bw.png",
        "noisy_grayscale.png",
        "silhouette_bw.png",
        "simple_color.png",
    }
    assert all(method for method in methods.values())


def test_category_changes_threshold_selection_weights() -> None:
    config = PreprocessingConfig()
    line_result = preprocess_image(_white(), config=config, category=ImageCategory.LINE_ART)
    binary_result = preprocess_image(_white(), config=config, category=ImageCategory.BINARY_OUTLINE)

    assert line_result.config.threshold_selection.max_foreground_ratio < binary_result.config.threshold_selection.max_foreground_ratio
