"""Deterministic image classification for future semantic vectorization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STRATEGY_NAME = "classic_semantic_image_classifier_v1"


class ImageCategory(Enum):
    """High-level routing categories for future semantic vectorization."""

    LINE_ART = "line_art"
    BINARY_OUTLINE = "binary_outline"
    COLOR_REGIONS = "color_regions"


@dataclass(frozen=True)
class ImageClassifierConfig:
    """Tunable thresholds for deterministic image classification."""

    color_quantization_levels: int = 6
    tonal_quantization_levels: int = 8
    min_color_bin_ratio: float = 0.005
    min_tonal_bin_ratio: float = 0.01
    saturation_threshold: float = 0.18
    colored_pixel_ratio_threshold: float = 0.05
    color_channel_delta_threshold: float = 18.0
    dominant_background_threshold: float = 0.65
    foreground_dark_threshold: float = 180.0
    background_distance_threshold: float = 36.0
    line_art_foreground_max_ratio: float = 0.14
    binary_outline_foreground_min_ratio: float = 0.16
    line_art_edge_foreground_min_ratio: float = 0.35
    filled_edge_foreground_max_ratio: float = 0.25
    color_regions_min_effective_colors: int = 4
    edge_threshold: float = 24.0
    ambiguity_margin: float = 0.10
    minimum_confidence: float = 0.05
    uniform_contrast_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.color_quantization_levels < 2:
            raise ValueError("color_quantization_levels must be at least 2.")
        if self.tonal_quantization_levels < 2:
            raise ValueError("tonal_quantization_levels must be at least 2.")
        if self.color_regions_min_effective_colors < 2:
            raise ValueError("color_regions_min_effective_colors must be at least 2.")
        for name in (
            "min_color_bin_ratio",
            "min_tonal_bin_ratio",
            "saturation_threshold",
            "colored_pixel_ratio_threshold",
            "dominant_background_threshold",
            "line_art_foreground_max_ratio",
            "binary_outline_foreground_min_ratio",
            "line_art_edge_foreground_min_ratio",
            "filled_edge_foreground_max_ratio",
            "ambiguity_margin",
            "minimum_confidence",
            "uniform_contrast_threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in (
            "color_channel_delta_threshold",
            "foreground_dark_threshold",
            "background_distance_threshold",
            "edge_threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class ImageClassificationMetrics:
    """Numeric diagnostics used by the image classifier."""

    width: int
    height: int
    effective_color_count: int
    tonal_level_count: int
    mean_saturation: float
    chromatic_variation: float
    colored_pixel_ratio: float
    dominant_background_ratio: float
    dark_pixel_ratio: float
    foreground_ratio: float
    edge_density: float
    edge_to_foreground_ratio: float
    estimated_stroke_thickness: float
    contrast: float
    uniform: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a plain diagnostic dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "effective_color_count": self.effective_color_count,
            "tonal_level_count": self.tonal_level_count,
            "mean_saturation": self.mean_saturation,
            "chromatic_variation": self.chromatic_variation,
            "colored_pixel_ratio": self.colored_pixel_ratio,
            "dominant_background_ratio": self.dominant_background_ratio,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "foreground_ratio": self.foreground_ratio,
            "edge_density": self.edge_density,
            "edge_to_foreground_ratio": self.edge_to_foreground_ratio,
            "estimated_stroke_thickness": self.estimated_stroke_thickness,
            "contrast": self.contrast,
            "uniform": self.uniform,
        }


@dataclass(frozen=True)
class ImageClassificationResult:
    """Structured classification result for diagnostics and routing."""

    category: ImageCategory
    confidence: float
    metrics: ImageClassificationMetrics
    reasons: tuple[str, ...]
    ambiguous: bool = False
    manual_override: bool = False
    alternative_category: ImageCategory | None = None
    strategy: str = STRATEGY_NAME
    scores: Mapping[ImageCategory, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.category, ImageCategory):
            raise TypeError("category must be an ImageCategory.")
        if self.alternative_category is not None and not isinstance(self.alternative_category, ImageCategory):
            raise TypeError("alternative_category must be an ImageCategory or None.")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "scores", dict(self.scores))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostic representation."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "manual_override": self.manual_override,
            "alternative_category": self.alternative_category.value if self.alternative_category else None,
            "strategy": self.strategy,
            "metrics": self.metrics.to_dict(),
            "reasons": list(self.reasons),
            "scores": {category.value: score for category, score in self.scores.items()},
        }


def classify_image(
    image: str | Path | np.ndarray | object,
    config: ImageClassifierConfig | None = None,
    override: ImageCategory | str | None = None,
) -> ImageClassificationResult:
    """Classify an image for future semantic vectorization routing."""
    classifier_config = config or ImageClassifierConfig()
    rgb = _image_to_rgb_array(image)
    metrics = _compute_metrics(rgb, classifier_config)
    scores = _score_categories(metrics, classifier_config)
    auto_category, alternative_category = _rank_categories(scores)
    ambiguous = _is_ambiguous(scores, auto_category, alternative_category, classifier_config)

    selected_category = auto_category
    manual_override = override is not None
    reasons = _reasons_for_category(selected_category, metrics, classifier_config)

    if manual_override:
        selected_category = _coerce_category(override)
        reasons = (
            f"manual override selected {selected_category.value}",
            *_reasons_for_category(auto_category, metrics, classifier_config)[:2],
        )
        confidence = 1.0
        ambiguous = False
        alternative = auto_category if auto_category != selected_category else alternative_category
    else:
        confidence = _confidence_from_scores(scores, auto_category, alternative_category, classifier_config, ambiguous)
        alternative = alternative_category if ambiguous else None

    if metrics.uniform:
        reasons = (*reasons, _uniform_reason(metrics))
    if ambiguous and not manual_override:
        reasons = (
            *reasons,
            f"top scores are within {classifier_config.ambiguity_margin:.3f}",
        )

    return ImageClassificationResult(
        category=selected_category,
        confidence=confidence,
        metrics=metrics,
        reasons=reasons,
        ambiguous=ambiguous,
        manual_override=manual_override,
        alternative_category=alternative,
        scores=scores,
    )


def _image_to_rgb_array(image: str | Path | np.ndarray | object) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not read image: {path}")
        array = loaded
    elif isinstance(image, np.ndarray):
        array = image
    elif _looks_like_pil_image(image):
        mode = getattr(image, "mode", "")
        converted = image.convert("RGBA" if "A" in mode else "RGB")
        array = np.asarray(converted)
    else:
        try:
            array = np.asarray(image)
        except Exception as exc:  # pragma: no cover - defensive conversion guard
            raise TypeError("Unsupported image input type.") from exc

    return _normalize_image_array(array)


def _looks_like_pil_image(image: object) -> bool:
    return hasattr(image, "convert") and hasattr(image, "mode") and image.__class__.__module__.startswith("PIL")


def _normalize_image_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.size == 0:
        raise ValueError("Image must not be empty.")
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError("Image must be a 2D grayscale or 3D color array.")

    height, width, channels = values.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    if channels not in {1, 3, 4}:
        raise ValueError("Image must have 1, 3, or 4 channels.")

    numeric = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Image values must be finite.")

    max_value = float(numeric.max(initial=0.0))
    min_value = float(numeric.min(initial=0.0))
    if min_value < 0.0:
        raise ValueError("Image values must be non-negative.")
    if max_value <= 1.0 and values.dtype.kind == "f":
        numeric = numeric * 255.0
        max_value = float(numeric.max(initial=0.0))
    if max_value > 255.0:
        raise ValueError("Image values must be in the 0-255 range.")

    if channels == 1:
        rgb = np.repeat(numeric[:, :, :1], 3, axis=2)
    elif channels == 4:
        alpha = numeric[:, :, 3:4] / 255.0
        rgb = numeric[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    else:
        rgb = numeric[:, :, :3]
    return np.clip(rgb, 0.0, 255.0)


def _compute_metrics(rgb: np.ndarray, config: ImageClassifierConfig) -> ImageClassificationMetrics:
    height, width = rgb.shape[:2]
    total_pixels = float(height * width)
    gray = rgb.mean(axis=2)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    channel_delta = channel_max - channel_min
    saturation = np.divide(channel_delta, channel_max, out=np.zeros_like(channel_delta), where=channel_max > 0.0)
    colored_mask = (saturation >= config.saturation_threshold) & (channel_delta >= config.color_channel_delta_threshold)

    quantized_rgb = _quantize(rgb, config.color_quantization_levels)
    _, color_counts = np.unique(quantized_rgb.reshape(-1, 3), axis=0, return_counts=True)
    color_ratios = color_counts.astype(np.float64) / total_pixels
    effective_color_count = int(np.count_nonzero(color_ratios >= config.min_color_bin_ratio))
    dominant_background_ratio = float(color_ratios.max(initial=0.0))

    dominant_color = _dominant_quantized_color(rgb, quantized_rgb, config.color_quantization_levels)
    dominant_gray = float(np.mean(dominant_color))
    distance_from_dominant = np.linalg.norm(rgb - dominant_color, axis=2)
    dark_mask = gray <= config.foreground_dark_threshold
    if dominant_gray >= config.foreground_dark_threshold:
        foreground_mask = distance_from_dominant >= config.background_distance_threshold
    else:
        foreground_mask = dark_mask
    foreground_ratio = _ratio(foreground_mask)

    tonal = _quantize(gray[:, :, np.newaxis], config.tonal_quantization_levels).reshape(-1)
    _, tonal_counts = np.unique(tonal, return_counts=True)
    tonal_ratios = tonal_counts.astype(np.float64) / total_pixels
    tonal_level_count = int(np.count_nonzero(tonal_ratios >= config.min_tonal_bin_ratio))

    edge_density = _edge_density(gray, config.edge_threshold)
    edge_to_foreground_ratio = edge_density / max(foreground_ratio, 1.0 / total_pixels)
    estimated_stroke_thickness = foreground_ratio / edge_density if edge_density > 0.0 else (0.0 if foreground_ratio == 0.0 else float(max(width, height)))
    contrast = float((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0)
    uniform = contrast <= config.uniform_contrast_threshold and effective_color_count <= 1

    return ImageClassificationMetrics(
        width=int(width),
        height=int(height),
        effective_color_count=effective_color_count,
        tonal_level_count=tonal_level_count,
        mean_saturation=_finite_ratio(float(np.mean(saturation))),
        chromatic_variation=_finite_ratio(float(np.mean(channel_delta) / 255.0)),
        colored_pixel_ratio=_ratio(colored_mask),
        dominant_background_ratio=_finite_ratio(dominant_background_ratio),
        dark_pixel_ratio=_ratio(dark_mask),
        foreground_ratio=_finite_ratio(foreground_ratio),
        edge_density=_finite_ratio(edge_density),
        edge_to_foreground_ratio=float(max(0.0, edge_to_foreground_ratio)),
        estimated_stroke_thickness=float(max(0.0, estimated_stroke_thickness)),
        contrast=_finite_ratio(contrast),
        uniform=bool(uniform),
    )


def _quantize(values: np.ndarray, levels: int) -> np.ndarray:
    scaled = np.floor(values * float(levels) / 256.0)
    return np.clip(scaled, 0, levels - 1).astype(np.uint8)


def _dominant_quantized_color(rgb: np.ndarray, quantized_rgb: np.ndarray, levels: int) -> np.ndarray:
    flattened = quantized_rgb.reshape(-1, 3)
    colors, counts = np.unique(flattened, axis=0, return_counts=True)
    dominant_bin = colors[int(np.argmax(counts))]
    bin_size = 256.0 / float(levels)
    return (dominant_bin.astype(np.float64) + 0.5) * bin_size


def _edge_density(gray: np.ndarray, threshold: float) -> float:
    if gray.shape[0] == 1 and gray.shape[1] == 1:
        return 0.0
    horizontal = np.abs(np.diff(gray, axis=1)) >= threshold if gray.shape[1] > 1 else np.zeros((gray.shape[0], 0), dtype=bool)
    vertical = np.abs(np.diff(gray, axis=0)) >= threshold if gray.shape[0] > 1 else np.zeros((0, gray.shape[1]), dtype=bool)
    edge_count = int(np.count_nonzero(horizontal)) + int(np.count_nonzero(vertical))
    possible = horizontal.size + vertical.size
    return float(edge_count / possible) if possible else 0.0


def _score_categories(
    metrics: ImageClassificationMetrics,
    config: ImageClassifierConfig,
) -> dict[ImageCategory, float]:
    return {
        ImageCategory.LINE_ART: _score_line_art(metrics, config),
        ImageCategory.BINARY_OUTLINE: _score_binary_outline(metrics, config),
        ImageCategory.COLOR_REGIONS: _score_color_regions(metrics, config),
    }


def _score_line_art(metrics: ImageClassificationMetrics, config: ImageClassifierConfig) -> float:
    if metrics.dark_pixel_ratio >= 0.98 and metrics.uniform:
        return 0.05
    foreground_score = 1.0 - _clamp01(metrics.foreground_ratio / max(config.line_art_foreground_max_ratio, 1e-9))
    thin_score = _clamp01(metrics.edge_to_foreground_ratio / max(config.line_art_edge_foreground_min_ratio, 1e-9))
    low_color_score = 1.0 - _clamp01(metrics.colored_pixel_ratio / max(config.colored_pixel_ratio_threshold, 1e-9))
    background_score = _clamp01(metrics.dominant_background_ratio / max(config.dominant_background_threshold, 1e-9))
    contrast_score = metrics.contrast
    score = (
        0.25 * background_score
        + 0.25 * foreground_score
        + 0.20 * thin_score
        + 0.20 * low_color_score
        + 0.10 * contrast_score
    )
    return _clamp01(score)


def _score_binary_outline(metrics: ImageClassificationMetrics, config: ImageClassifierConfig) -> float:
    foreground_score = _clamp01(metrics.foreground_ratio / max(config.binary_outline_foreground_min_ratio, 1e-9))
    low_color_score = 1.0 - _clamp01(metrics.colored_pixel_ratio / max(config.colored_pixel_ratio_threshold, 1e-9))
    two_tone_score = 1.0 - _clamp01(max(0, metrics.tonal_level_count - 2) / max(config.tonal_quantization_levels - 2, 1))
    filled_score = 1.0 - _clamp01(metrics.edge_to_foreground_ratio / max(config.filled_edge_foreground_max_ratio, 1e-9))
    score = (
        0.30 * foreground_score
        + 0.25 * low_color_score
        + 0.20 * metrics.contrast
        + 0.15 * two_tone_score
        + 0.10 * filled_score
    )
    return _clamp01(score)


def _score_color_regions(metrics: ImageClassificationMetrics, config: ImageClassifierConfig) -> float:
    colored_score = _clamp01(metrics.colored_pixel_ratio / max(config.colored_pixel_ratio_threshold, 1e-9))
    saturation_score = _clamp01(metrics.mean_saturation / max(config.saturation_threshold, 1e-9))
    effective_score = _clamp01(
        (metrics.effective_color_count - 2) / max(config.color_regions_min_effective_colors - 2, 1)
    )
    foreground_score = _clamp01(metrics.foreground_ratio / 0.20)
    score = (
        0.35 * colored_score
        + 0.25 * saturation_score
        + 0.20 * effective_score
        + 0.10 * metrics.chromatic_variation
        + 0.10 * foreground_score
    )
    return _clamp01(score)


def _rank_categories(scores: Mapping[ImageCategory, float]) -> tuple[ImageCategory, ImageCategory]:
    ordered = sorted(ImageCategory, key=lambda category: (-scores[category], category.value))
    return ordered[0], ordered[1]


def _is_ambiguous(
    scores: Mapping[ImageCategory, float],
    category: ImageCategory,
    alternative: ImageCategory,
    config: ImageClassifierConfig,
) -> bool:
    return scores[category] - scores[alternative] <= config.ambiguity_margin


def _confidence_from_scores(
    scores: Mapping[ImageCategory, float],
    category: ImageCategory,
    alternative: ImageCategory,
    config: ImageClassifierConfig,
    ambiguous: bool,
) -> float:
    gap = max(0.0, scores[category] - scores[alternative])
    confidence = 0.60 * scores[category] + 0.40 * _clamp01(gap / max(config.ambiguity_margin, 1e-9))
    if ambiguous:
        confidence *= 0.72
    return _clamp01(max(config.minimum_confidence, confidence))


def _reasons_for_category(
    category: ImageCategory,
    metrics: ImageClassificationMetrics,
    config: ImageClassifierConfig,
) -> tuple[str, ...]:
    if category is ImageCategory.COLOR_REGIONS:
        return (
            f"colored pixel ratio {metrics.colored_pixel_ratio:.3f}",
            f"mean saturation {metrics.mean_saturation:.3f}",
            f"effective colors {metrics.effective_color_count}",
        )
    if category is ImageCategory.BINARY_OUTLINE:
        return (
            f"foreground ratio {metrics.foreground_ratio:.3f}",
            f"dark pixel ratio {metrics.dark_pixel_ratio:.3f}",
            f"tonal levels {metrics.tonal_level_count}",
        )
    return (
        f"dominant background ratio {metrics.dominant_background_ratio:.3f}",
        f"foreground ratio {metrics.foreground_ratio:.3f}",
        f"edge to foreground ratio {metrics.edge_to_foreground_ratio:.3f}",
    )


def _uniform_reason(metrics: ImageClassificationMetrics) -> str:
    if metrics.dark_pixel_ratio >= 0.98:
        return "uniform dark image treated as a filled binary region"
    if metrics.foreground_ratio <= 0.001:
        return "uniform bright image treated as blank line art"
    return "uniform image classified with low-detail metrics"


def _coerce_category(value: ImageCategory | str | None) -> ImageCategory:
    if isinstance(value, ImageCategory):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for category in ImageCategory:
            if normalized in {category.value, category.name.lower()}:
                return category
    raise ValueError(f"Unsupported image category override: {value!r}")


def _ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return _finite_ratio(float(np.count_nonzero(mask) / mask.size))


def _finite_ratio(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return _clamp01(float(value))


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "ImageCategory",
    "ImageClassificationMetrics",
    "ImageClassificationResult",
    "ImageClassifierConfig",
    "classify_image",
]
