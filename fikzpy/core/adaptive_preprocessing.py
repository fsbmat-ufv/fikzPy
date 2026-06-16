"""Adaptive preprocessing utilities for the future semantic Classic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fikzpy.core.diagnostics import log_event
from fikzpy.core.image_classifier import ImageCategory
from fikzpy.core.threshold_selector import ThresholdSelectionConfig, ThresholdSelectionResult
from fikzpy.core.threshold_selector import ThresholdScoreWeights, select_best_threshold


class DenoiseMethod(Enum):
    """Supported conservative denoising methods."""

    NONE = "none"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for isolated adaptive preprocessing."""

    enable_autocontrast: bool = True
    autocontrast_low_percentile: float = 1.0
    autocontrast_high_percentile: float = 99.0
    autocontrast_max_gain: float = 3.0
    enable_illumination_correction: bool = False
    illumination_blur_kernel_size: int = 31
    illumination_strength: float = 0.65
    denoise_method: DenoiseMethod | str = DenoiseMethod.NONE
    median_kernel_size: int = 3
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 35.0
    gaussian_kernel_size: int = 3
    morphological_close_iterations: int = 0
    morphological_open_iterations: int = 0
    morphology_kernel_size: int = 3
    minimum_component_area: int = 3
    preserve_small_details: bool = True
    maximum_noise_removal_ratio: float = 0.015
    connectivity: int = 8
    foreground_is_dark: bool = True
    alpha_background: tuple[int, int, int] = (255, 255, 255)
    threshold_selection: ThresholdSelectionConfig = field(default_factory=ThresholdSelectionConfig)

    def __post_init__(self) -> None:
        _validate_bool("enable_autocontrast", self.enable_autocontrast)
        _validate_bool("enable_illumination_correction", self.enable_illumination_correction)
        _validate_percentile("autocontrast_low_percentile", self.autocontrast_low_percentile)
        _validate_percentile("autocontrast_high_percentile", self.autocontrast_high_percentile)
        if self.autocontrast_low_percentile >= self.autocontrast_high_percentile:
            raise ValueError("autocontrast_low_percentile must be lower than autocontrast_high_percentile.")
        _validate_non_negative("autocontrast_max_gain", self.autocontrast_max_gain)
        if self.autocontrast_max_gain < 1.0:
            raise ValueError("autocontrast_max_gain must be at least 1.")
        object.__setattr__(self, "illumination_blur_kernel_size", _odd_at_least(self.illumination_blur_kernel_size, 3))
        _validate_ratio("illumination_strength", self.illumination_strength)
        object.__setattr__(self, "denoise_method", _coerce_denoise_method(self.denoise_method))
        object.__setattr__(self, "median_kernel_size", _odd_at_least(self.median_kernel_size, 1))
        object.__setattr__(self, "bilateral_diameter", max(1, int(self.bilateral_diameter)))
        _validate_non_negative("bilateral_sigma_color", self.bilateral_sigma_color)
        _validate_non_negative("bilateral_sigma_space", self.bilateral_sigma_space)
        object.__setattr__(self, "gaussian_kernel_size", _odd_at_least(self.gaussian_kernel_size, 1))
        _validate_iteration_count("morphological_close_iterations", self.morphological_close_iterations)
        _validate_iteration_count("morphological_open_iterations", self.morphological_open_iterations)
        object.__setattr__(self, "morphology_kernel_size", _odd_at_least(self.morphology_kernel_size, 1))
        if int(self.minimum_component_area) < 0:
            raise ValueError("minimum_component_area must be non-negative.")
        object.__setattr__(self, "minimum_component_area", int(self.minimum_component_area))
        _validate_bool("preserve_small_details", self.preserve_small_details)
        _validate_ratio("maximum_noise_removal_ratio", self.maximum_noise_removal_ratio)
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8.")
        _validate_bool("foreground_is_dark", self.foreground_is_dark)
        if len(self.alpha_background) != 3:
            raise ValueError("alpha_background must contain three channels.")
        for value in self.alpha_background:
            if int(value) < 0 or int(value) > 255:
                raise ValueError("alpha_background channels must be in the 0-255 range.")
        object.__setattr__(self, "alpha_background", tuple(int(value) for value in self.alpha_background))
        if not isinstance(self.threshold_selection, ThresholdSelectionConfig):
            raise TypeError("threshold_selection must be ThresholdSelectionConfig.")

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary."""
        return {
            "enable_autocontrast": self.enable_autocontrast,
            "autocontrast_low_percentile": self.autocontrast_low_percentile,
            "autocontrast_high_percentile": self.autocontrast_high_percentile,
            "autocontrast_max_gain": self.autocontrast_max_gain,
            "enable_illumination_correction": self.enable_illumination_correction,
            "illumination_blur_kernel_size": self.illumination_blur_kernel_size,
            "illumination_strength": self.illumination_strength,
            "denoise_method": self.denoise_method.value,
            "median_kernel_size": self.median_kernel_size,
            "bilateral_diameter": self.bilateral_diameter,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "gaussian_kernel_size": self.gaussian_kernel_size,
            "morphological_close_iterations": self.morphological_close_iterations,
            "morphological_open_iterations": self.morphological_open_iterations,
            "morphology_kernel_size": self.morphology_kernel_size,
            "minimum_component_area": self.minimum_component_area,
            "preserve_small_details": self.preserve_small_details,
            "maximum_noise_removal_ratio": self.maximum_noise_removal_ratio,
            "connectivity": self.connectivity,
            "foreground_is_dark": self.foreground_is_dark,
            "alpha_background": list(self.alpha_background),
            "threshold_selection": self.threshold_selection.to_dict(),
        }


@dataclass(frozen=True)
class PreprocessingMetrics:
    """Scalar diagnostics for the final preprocessed mask."""

    foreground_ratio: float
    component_count: int
    tiny_component_count: int
    largest_component_ratio: float
    removed_component_count: int
    removed_pixel_ratio: float
    contrast: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a diagnostic dictionary."""
        return {
            "foreground_ratio": self.foreground_ratio,
            "component_count": self.component_count,
            "tiny_component_count": self.tiny_component_count,
            "largest_component_ratio": self.largest_component_ratio,
            "removed_component_count": self.removed_component_count,
            "removed_pixel_ratio": self.removed_pixel_ratio,
            "contrast": self.contrast,
        }


@dataclass(frozen=True)
class PreprocessingResult:
    """Outputs and diagnostics from adaptive preprocessing."""

    original: np.ndarray
    grayscale: np.ndarray
    corrected: np.ndarray
    smoothed: np.ndarray
    binary_mask: np.ndarray
    threshold_result: ThresholdSelectionResult
    metrics: PreprocessingMetrics
    warnings: tuple[str, ...]
    config: PreprocessingConfig
    category: ImageCategory | None = None
    uniform: bool = False

    @property
    def threshold(self) -> float | None:
        """Return the selected threshold value when available."""
        return self.threshold_result.threshold

    @property
    def method(self) -> str:
        """Return the selected threshold method."""
        return self.threshold_result.method

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing full image arrays."""
        return {
            "original": _array_summary(self.original),
            "grayscale": _array_summary(self.grayscale),
            "corrected": _array_summary(self.corrected),
            "smoothed": _array_summary(self.smoothed),
            "binary_mask": _array_summary(self.binary_mask),
            "threshold": self.threshold,
            "method": self.method,
            "threshold_selection": self.threshold_result.to_dict(),
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
            "config": self.config.to_dict(),
            "category": self.category.value if self.category is not None else None,
            "uniform": self.uniform,
        }


def preprocess_image(
    image: str | Path | np.ndarray | object,
    config: PreprocessingConfig | None = None,
    category: ImageCategory | str | None = None,
) -> PreprocessingResult:
    """Run isolated adaptive preprocessing and threshold selection."""
    effective_config = config or PreprocessingConfig()
    effective_category = _coerce_category(category)
    base_selection = replace(
        effective_config.threshold_selection,
        foreground_is_dark=effective_config.foreground_is_dark,
        connectivity=effective_config.connectivity,
    )
    selection_config = _selection_config_for_category(base_selection, effective_category)
    effective_config = replace(effective_config, threshold_selection=selection_config)

    original = _image_to_rgb_array(image, effective_config)
    grayscale = convert_to_grayscale(original)
    corrected = apply_autocontrast(grayscale, effective_config) if effective_config.enable_autocontrast else grayscale.copy()
    if effective_config.enable_illumination_correction:
        corrected = correct_illumination(corrected, effective_config)
    smoothed = denoise_grayscale(corrected, effective_config)
    threshold_result = select_best_threshold(smoothed, selection_config)
    refined_mask, removed_components, removed_ratio = refine_mask(threshold_result.best_mask, effective_config)
    warnings = _warnings_for_result(original, grayscale, refined_mask, threshold_result, effective_category)
    metrics = _preprocessing_metrics(smoothed, refined_mask, effective_config, removed_components, removed_ratio)
    uniform = bool(np.max(grayscale) == np.min(grayscale))

    log_event("Preprocess", f"category={effective_category.value if effective_category else 'none'}")
    log_event("Preprocess", f"grayscale_shape={grayscale.shape[0]}x{grayscale.shape[1]}")
    log_event("Preprocess", f"selected_method={threshold_result.method}")
    log_event("Preprocess", f"selected_threshold={threshold_result.threshold}")

    return PreprocessingResult(
        original=original,
        grayscale=grayscale,
        corrected=corrected,
        smoothed=smoothed,
        binary_mask=refined_mask,
        threshold_result=threshold_result,
        metrics=metrics,
        warnings=warnings,
        config=effective_config,
        category=effective_category,
        uniform=uniform,
    )


def convert_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized RGB data to grayscale luminance."""
    image = _normalize_rgb_array(rgb, PreprocessingConfig())
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.rint(np.clip(gray, 0.0, 255.0)).astype(np.uint8)


def apply_autocontrast(gray: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply percentile-clipped autocontrast with a bounded gain."""
    source = _normalize_gray(gray)
    low = float(np.percentile(source, config.autocontrast_low_percentile))
    high = float(np.percentile(source, config.autocontrast_high_percentile))
    if high <= low:
        return source.copy()
    gain = min(255.0 / max(high - low, 1e-9), float(config.autocontrast_max_gain))
    midpoint = (high + low) / 2.0
    stretched = (source.astype(np.float32) - midpoint) * gain + 127.5
    return np.rint(np.clip(stretched, 0.0, 255.0)).astype(np.uint8)


def correct_illumination(gray: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Normalize slow background variation with a broad blur estimate."""
    source = _normalize_gray(gray)
    kernel = _bounded_odd_kernel(config.illumination_blur_kernel_size, source.shape)
    if kernel <= 1:
        return source.copy()
    background = cv2.GaussianBlur(source, (kernel, kernel), 0).astype(np.float32)
    source_float = source.astype(np.float32)
    normalized = source_float - background + float(np.mean(background))
    blended = (1.0 - config.illumination_strength) * source_float + config.illumination_strength * normalized
    return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)


def denoise_grayscale(gray: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply the configured conservative denoising method."""
    source = _normalize_gray(gray)
    if config.denoise_method is DenoiseMethod.NONE:
        return source.copy()
    if config.denoise_method is DenoiseMethod.MEDIAN:
        if config.median_kernel_size <= 1:
            return source.copy()
        return cv2.medianBlur(source, config.median_kernel_size)
    if config.denoise_method is DenoiseMethod.BILATERAL:
        return cv2.bilateralFilter(
            source,
            config.bilateral_diameter,
            float(config.bilateral_sigma_color),
            float(config.bilateral_sigma_space),
        )
    if config.gaussian_kernel_size <= 1:
        return source.copy()
    return cv2.GaussianBlur(source, (config.gaussian_kernel_size, config.gaussian_kernel_size), 0)


def refine_mask(mask: np.ndarray, config: PreprocessingConfig) -> tuple[np.ndarray, int, float]:
    """Apply optional conservative morphology and component cleanup."""
    result = _as_binary_mask(mask)
    kernel = np.ones((config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8)
    if config.morphological_close_iterations > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=config.morphological_close_iterations)
    if config.morphological_open_iterations > 0:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=config.morphological_open_iterations)
    cleaned, removed_components, removed_ratio = remove_small_components_conservatively(result, config)
    return cleaned, removed_components, removed_ratio


def remove_small_components_conservatively(
    mask: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, int, float]:
    """Remove tiny isolated components while respecting the configured limit."""
    binary = _as_binary_mask(mask)
    if config.minimum_component_area <= 0:
        return binary, 0, 0.0
    foreground_pixels = int(np.count_nonzero(binary))
    if foreground_pixels == 0:
        return binary, 0, 0.0

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=config.connectivity)
    if count <= 1:
        return binary, 0, 0.0

    area_limit = config.minimum_component_area
    if config.preserve_small_details:
        area_limit = 0

    removable: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= area_limit:
            removable.append((label, area))

    if not removable:
        return binary, 0, 0.0

    pixel_limit = int(np.floor(foreground_pixels * config.maximum_noise_removal_ratio))
    if pixel_limit <= 0 and not config.preserve_small_details:
        pixel_limit = 1

    cleaned = binary.copy()
    removed_pixels = 0
    removed_components = 0
    for label, area in sorted(removable, key=lambda item: item[1]):
        if pixel_limit > 0 and removed_pixels + area > pixel_limit:
            break
        cleaned[labels == label] = 0
        removed_pixels += area
        removed_components += 1

    removed_ratio = removed_pixels / max(foreground_pixels, 1)
    return cleaned, removed_components, float(removed_ratio)


def _selection_config_for_category(
    selection_config: ThresholdSelectionConfig,
    category: ImageCategory | None,
) -> ThresholdSelectionConfig:
    if category is ImageCategory.LINE_ART:
        return replace(
            selection_config,
            min_foreground_ratio=0.0005,
            max_foreground_ratio=0.35,
            score_weights=ThresholdScoreWeights(
                continuity=0.28,
                edge_coverage=0.30,
                foreground_plausibility=0.20,
                background_consistency=0.10,
                noise=0.10,
                fragmentation_penalty=0.10,
                tiny_component_penalty=0.06,
                extreme_mask_penalty=0.18,
            ),
        )
    if category is ImageCategory.BINARY_OUTLINE:
        return replace(
            selection_config,
            min_foreground_ratio=0.02,
            max_foreground_ratio=0.92,
            score_weights=ThresholdScoreWeights(
                continuity=0.20,
                edge_coverage=0.20,
                foreground_plausibility=0.30,
                background_consistency=0.14,
                noise=0.10,
                fragmentation_penalty=0.06,
                tiny_component_penalty=0.10,
                extreme_mask_penalty=0.16,
            ),
        )
    if category is ImageCategory.COLOR_REGIONS:
        return replace(selection_config, max_foreground_ratio=0.90)
    return selection_config


def _image_to_rgb_array(image: str | Path | np.ndarray | object, config: PreprocessingConfig) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise ValueError(f"Could not read image: {path}")
        if loaded.ndim == 3 and loaded.shape[2] == 3:
            loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        elif loaded.ndim == 3 and loaded.shape[2] == 4:
            loaded = cv2.cvtColor(loaded, cv2.COLOR_BGRA2RGBA)
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
    return _normalize_rgb_array(array, config)


def _normalize_rgb_array(array: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
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
    if float(numeric.min(initial=0.0)) < 0.0:
        raise ValueError("Image values must be non-negative.")
    if numeric.dtype.kind == "f" and float(numeric.max(initial=0.0)) <= 1.0:
        numeric = numeric * 255.0
    if float(numeric.max(initial=0.0)) > 255.0:
        raise ValueError("Image values must be in the 0-255 range.")

    if channels == 1:
        rgb = np.repeat(numeric[:, :, :1], 3, axis=2)
    elif channels == 4:
        alpha = numeric[:, :, 3:4] / 255.0
        background = np.array(config.alpha_background, dtype=np.float64).reshape(1, 1, 3)
        rgb = numeric[:, :, :3] * alpha + background * (1.0 - alpha)
    else:
        rgb = numeric[:, :, :3]
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    values = np.asarray(gray)
    if values.size == 0:
        raise ValueError("grayscale image must not be empty.")
    if values.ndim != 2:
        raise ValueError("grayscale image must be 2D.")
    numeric = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("grayscale image values must be finite.")
    if float(numeric.min(initial=0.0)) < 0.0:
        raise ValueError("grayscale image values must be non-negative.")
    if values.dtype.kind == "f" and float(numeric.max(initial=0.0)) <= 1.0:
        numeric = numeric * 255.0
    if float(numeric.max(initial=0.0)) > 255.0:
        raise ValueError("grayscale image values must be in the 0-255 range.")
    return np.rint(np.clip(numeric, 0.0, 255.0)).astype(np.uint8)


def _preprocessing_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
    config: PreprocessingConfig,
    removed_components: int,
    removed_ratio: float,
) -> PreprocessingMetrics:
    foreground_pixels = int(np.count_nonzero(mask))
    foreground_ratio = foreground_pixels / max(mask.size, 1)
    component_count, tiny_count, largest_ratio = _component_stats(mask, foreground_pixels, config)
    contrast = float((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0)
    return PreprocessingMetrics(
        foreground_ratio=_clamp01(foreground_ratio),
        component_count=component_count,
        tiny_component_count=tiny_count,
        largest_component_ratio=_clamp01(largest_ratio),
        removed_component_count=removed_components,
        removed_pixel_ratio=_clamp01(removed_ratio),
        contrast=_clamp01(contrast),
    )


def _component_stats(mask: np.ndarray, foreground_pixels: int, config: PreprocessingConfig) -> tuple[int, int, float]:
    if foreground_pixels == 0:
        return 0, 0, 0.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=config.connectivity)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    if not areas:
        return 0, 0, 0.0
    tiny_limit = max(1, config.minimum_component_area)
    tiny_count = sum(1 for area in areas if area <= tiny_limit)
    return len(areas), tiny_count, max(areas) / max(foreground_pixels, 1)


def _warnings_for_result(
    original: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    threshold_result: ThresholdSelectionResult,
    category: ImageCategory | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if np.max(gray) == np.min(gray):
        warnings.append("uniform image")
    if np.count_nonzero(mask) == 0:
        warnings.append("empty foreground mask")
    elif np.count_nonzero(mask) == mask.size:
        warnings.append("full foreground mask")
    if threshold_result.ambiguous:
        warnings.append("threshold selection is ambiguous")
    if category is ImageCategory.COLOR_REGIONS:
        warnings.append("color category uses grayscale diagnostic preprocessing only")
    if original.shape[0] < 4 or original.shape[1] < 4:
        warnings.append("very small image")
    return tuple(warnings)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": float(values.min(initial=0.0)),
        "max": float(values.max(initial=0.0)),
        "mean": float(values.mean()) if values.size else 0.0,
        "sha256": sha256(values.tobytes()).hexdigest(),
    }


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    return (array > 0).astype(np.uint8) * 255


def _coerce_category(category: ImageCategory | str | None) -> ImageCategory | None:
    if category is None:
        return None
    if isinstance(category, ImageCategory):
        return category
    if isinstance(category, str):
        normalized = category.strip().lower()
        for item in ImageCategory:
            if normalized in {item.value, item.name.lower()}:
                return item
    raise ValueError(f"Unsupported image category: {category!r}")


def _coerce_denoise_method(value: DenoiseMethod | str) -> DenoiseMethod:
    if isinstance(value, DenoiseMethod):
        return value
    normalized = str(value).strip().lower()
    for method in DenoiseMethod:
        if normalized == method.value:
            return method
    raise ValueError(f"Unsupported denoise method: {value!r}")


def _looks_like_pil_image(image: object) -> bool:
    return hasattr(image, "convert") and hasattr(image, "mode") and image.__class__.__module__.startswith("PIL")


def _bounded_odd_kernel(value: int, shape: tuple[int, int]) -> int:
    maximum = max(1, min(shape))
    size = min(_odd_at_least(value, 1), maximum if maximum % 2 == 1 else maximum - 1)
    return max(1, size)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_iteration_count(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_non_negative(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_percentile(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 100.0:
        raise ValueError(f"{name} must be between 0 and 100.")


def _validate_ratio(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _odd_at_least(value: int, minimum: int) -> int:
    size = max(int(value), minimum)
    return size if size % 2 == 1 else size + 1


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DenoiseMethod",
    "PreprocessingConfig",
    "PreprocessingMetrics",
    "PreprocessingResult",
    "apply_autocontrast",
    "convert_to_grayscale",
    "correct_illumination",
    "denoise_grayscale",
    "preprocess_image",
    "refine_mask",
    "remove_small_components_conservatively",
]
