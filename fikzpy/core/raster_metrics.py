"""Raster comparison metrics for semantic visual validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot, isfinite, sqrt
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class RasterComparisonMode(Enum):
    """Foreground comparison strategy."""

    DARK_FOREGROUND = "dark_foreground"


@dataclass(frozen=True)
class RasterMetricsConfig:
    """Configuration for deterministic raster comparisons."""

    target_size: tuple[int, int] | None = None
    background_color: tuple[int, int, int] = (255, 255, 255)
    foreground_threshold: int = 200
    dark_pixel_threshold: int = 200
    edge_threshold: int = 40
    filled_region_min_area: int = 64
    thin_stroke_max_width: float = 2.5
    small_detail_max_area: int = 32
    component_connectivity: int = 8
    compare_edges: bool = True
    compare_filled_regions: bool = True
    compare_thin_strokes: bool = True
    mode: RasterComparisonMode | str = RasterComparisonMode.DARK_FOREGROUND

    def __post_init__(self) -> None:
        if self.target_size is not None:
            if len(self.target_size) != 2:
                raise ValueError("target_size must contain width and height.")
            width, height = int(self.target_size[0]), int(self.target_size[1])
            if width <= 0 or height <= 0:
                raise ValueError("target_size values must be positive.")
            object.__setattr__(self, "target_size", (width, height))
        object.__setattr__(self, "background_color", _coerce_rgb_tuple("background_color", self.background_color))
        for name in ("foreground_threshold", "dark_pixel_threshold", "edge_threshold"):
            value = int(getattr(self, name))
            if value < 0 or value > 255:
                raise ValueError(f"{name} must be between 0 and 255.")
            object.__setattr__(self, name, value)
        for name in ("filled_region_min_area", "small_detail_max_area"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        thin_width = float(self.thin_stroke_max_width)
        if not isfinite(thin_width) or thin_width <= 0.0:
            raise ValueError("thin_stroke_max_width must be finite and positive.")
        object.__setattr__(self, "thin_stroke_max_width", thin_width)
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        for name in ("compare_edges", "compare_filled_regions", "compare_thin_strokes"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        object.__setattr__(self, "mode", _coerce_mode(self.mode))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return {
            "target_size": list(self.target_size) if self.target_size else None,
            "background_color": list(self.background_color),
            "foreground_threshold": self.foreground_threshold,
            "dark_pixel_threshold": self.dark_pixel_threshold,
            "edge_threshold": self.edge_threshold,
            "filled_region_min_area": self.filled_region_min_area,
            "thin_stroke_max_width": self.thin_stroke_max_width,
            "small_detail_max_area": self.small_detail_max_area,
            "component_connectivity": self.component_connectivity,
            "compare_edges": self.compare_edges,
            "compare_filled_regions": self.compare_filled_regions,
            "compare_thin_strokes": self.compare_thin_strokes,
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class RasterComparisonMetrics:
    """Scalar raster diagnostics for source/rendered image comparison."""

    width: int
    height: int
    source_foreground_pixels: int
    rendered_foreground_pixels: int
    source_foreground_ratio: float
    rendered_foreground_ratio: float
    mean_absolute_error: float
    root_mean_squared_error: float
    normalized_rmse: float
    foreground_iou: float
    foreground_precision: float
    foreground_recall: float
    foreground_f1: float
    false_positive_rate: float
    false_negative_rate: float
    structural_proxy: float
    edge_overlap: float
    edge_recall: float
    edge_precision: float
    edge_f1: float
    filled_region_recall: float
    large_dark_region_recall: float
    thin_stroke_recall: float
    small_detail_recall: float
    double_outline_penalty: float
    dark_mass_preservation_ratio: float
    foreground_fragmentation_delta: float
    connected_component_difference: int
    source_connected_components: int
    rendered_connected_components: int
    bounding_box_difference: float
    centroid_shift: float
    area_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        data = dict(self.__dict__)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = _rounded(value)
        return data


def compare_rasters(
    source_image: Any,
    rendered_image: Any,
    config: RasterMetricsConfig | None = None,
) -> RasterComparisonMetrics:
    """Compare source and rendered rasters with deterministic foreground metrics."""
    effective_config = config or RasterMetricsConfig()
    source = image_to_rgb_array(source_image, target_size=effective_config.target_size, background_color=effective_config.background_color)
    rendered = image_to_rgb_array(
        rendered_image,
        target_size=(source.shape[1], source.shape[0]),
        background_color=effective_config.background_color,
    )
    source_gray = _gray(source)
    rendered_gray = _gray(rendered)
    source_fg = source_gray <= effective_config.dark_pixel_threshold
    rendered_fg = rendered_gray <= effective_config.foreground_threshold
    source_count = int(np.count_nonzero(source_fg))
    rendered_count = int(np.count_nonzero(rendered_fg))
    total_pixels = int(source_fg.size)
    intersection = int(np.count_nonzero(source_fg & rendered_fg))
    union = int(np.count_nonzero(source_fg | rendered_fg))
    false_positive = int(np.count_nonzero(~source_fg & rendered_fg))
    false_negative = int(np.count_nonzero(source_fg & ~rendered_fg))
    true_negative = int(np.count_nonzero(~source_fg & ~rendered_fg))

    diff = np.abs(source_gray.astype(np.float32) - rendered_gray.astype(np.float32))
    mae = float(np.mean(diff))
    rmse = float(sqrt(float(np.mean(diff**2))))
    normalized_rmse = rmse / 255.0

    edge_metrics = _edge_metrics(source_gray, rendered_gray, effective_config)
    region_metrics = _region_metrics(source_fg, rendered_fg, effective_config)
    source_components = _component_count(source_fg, effective_config.component_connectivity)
    rendered_components = _component_count(rendered_fg, effective_config.component_connectivity)
    bbox_difference = _bbox_difference(source_fg, rendered_fg)
    centroid_shift = _centroid_shift(source_fg, rendered_fg)
    area_ratio = _safe_ratio(rendered_count, source_count, empty_value=1.0)
    dark_mass_ratio = area_ratio if source_count else (1.0 if rendered_count == 0 else 0.0)
    structural_proxy = _clamp01(
        (
            _iou(intersection, union) * 0.30
            + _f1(_precision(intersection, false_positive), _recall(intersection, false_negative)) * 0.25
            + (1.0 - normalized_rmse) * 0.20
            + edge_metrics["edge_f1"] * 0.15
            + region_metrics["filled_region_recall"] * 0.10
        )
    )

    return RasterComparisonMetrics(
        width=int(source.shape[1]),
        height=int(source.shape[0]),
        source_foreground_pixels=source_count,
        rendered_foreground_pixels=rendered_count,
        source_foreground_ratio=_safe_ratio(source_count, total_pixels),
        rendered_foreground_ratio=_safe_ratio(rendered_count, total_pixels),
        mean_absolute_error=mae,
        root_mean_squared_error=rmse,
        normalized_rmse=normalized_rmse,
        foreground_iou=_iou(intersection, union),
        foreground_precision=_precision(intersection, false_positive),
        foreground_recall=_recall(intersection, false_negative),
        foreground_f1=_f1(_precision(intersection, false_positive), _recall(intersection, false_negative)),
        false_positive_rate=_safe_ratio(false_positive, false_positive + true_negative),
        false_negative_rate=_safe_ratio(false_negative, false_negative + intersection),
        structural_proxy=structural_proxy,
        edge_overlap=edge_metrics["edge_overlap"],
        edge_recall=edge_metrics["edge_recall"],
        edge_precision=edge_metrics["edge_precision"],
        edge_f1=edge_metrics["edge_f1"],
        filled_region_recall=region_metrics["filled_region_recall"],
        large_dark_region_recall=region_metrics["large_dark_region_recall"],
        thin_stroke_recall=region_metrics["thin_stroke_recall"],
        small_detail_recall=region_metrics["small_detail_recall"],
        double_outline_penalty=region_metrics["double_outline_penalty"],
        dark_mass_preservation_ratio=dark_mass_ratio,
        foreground_fragmentation_delta=float(rendered_components - source_components),
        connected_component_difference=abs(rendered_components - source_components),
        source_connected_components=source_components,
        rendered_connected_components=rendered_components,
        bounding_box_difference=bbox_difference,
        centroid_shift=centroid_shift,
        area_ratio=area_ratio,
    )


def image_to_rgb_array(
    image: Any,
    *,
    target_size: tuple[int, int] | None = None,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Normalize common image inputs to an RGB uint8 array."""
    background = _coerce_rgb_tuple("background_color", background_color)
    if isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif _looks_like_pil_image(image):
        pil = image
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 4:
            pil_rgba = Image.fromarray(_normalize_uint8(array), mode="RGBA")
            base = Image.new("RGBA", pil_rgba.size, (*background, 255))
            base.alpha_composite(pil_rgba)
            pil = base.convert("RGB")
            return _resize_if_needed(np.asarray(pil, dtype=np.uint8), target_size)
        elif array.ndim != 3 or array.shape[2] not in {1, 3}:
            raise ValueError("image arrays must be grayscale, RGB, or RGBA.")
        if array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return _resize_if_needed(_normalize_uint8(array[:, :, :3]), target_size)
    if pil.mode == "RGBA":
        base = Image.new("RGBA", pil.size, (*background, 255))
        base.alpha_composite(pil.convert("RGBA"))
        pil = base.convert("RGB")
    else:
        pil = pil.convert("RGB")
    if target_size is not None and pil.size != target_size:
        pil = pil.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.uint8).copy()


def raster_summary(image: Any, config: RasterMetricsConfig | None = None) -> dict[str, Any]:
    """Return lightweight deterministic diagnostics for a raster image."""
    effective_config = config or RasterMetricsConfig()
    array = image_to_rgb_array(image, target_size=effective_config.target_size, background_color=effective_config.background_color)
    gray = _gray(array)
    foreground = gray <= effective_config.dark_pixel_threshold
    return {
        "shape": list(array.shape),
        "foreground_pixels": int(np.count_nonzero(foreground)),
        "foreground_ratio": _rounded(_safe_ratio(int(np.count_nonzero(foreground)), int(foreground.size))),
        "mean_luma": _rounded(float(np.mean(gray))),
        "sha256": _image_hash(array),
    }


def _edge_metrics(source_gray: np.ndarray, rendered_gray: np.ndarray, config: RasterMetricsConfig) -> dict[str, float]:
    if not config.compare_edges:
        return {"edge_overlap": 1.0, "edge_recall": 1.0, "edge_precision": 1.0, "edge_f1": 1.0}
    source_edges = _edges(source_gray, config.edge_threshold)
    rendered_edges = _edges(rendered_gray, config.edge_threshold)
    source_count = int(np.count_nonzero(source_edges))
    rendered_count = int(np.count_nonzero(rendered_edges))
    intersection = int(np.count_nonzero(source_edges & rendered_edges))
    union = int(np.count_nonzero(source_edges | rendered_edges))
    precision = _precision(intersection, rendered_count - intersection)
    recall = _recall(intersection, source_count - intersection)
    return {
        "edge_overlap": _iou(intersection, union),
        "edge_recall": recall,
        "edge_precision": precision,
        "edge_f1": _f1(precision, recall),
    }


def _region_metrics(source_fg: np.ndarray, rendered_fg: np.ndarray, config: RasterMetricsConfig) -> dict[str, float]:
    large_mask, small_mask = _large_and_small_component_masks(source_fg, config)
    filled_recall = _mask_recall(large_mask, rendered_fg) if config.compare_filled_regions else 1.0
    large_recall = filled_recall
    thin_mask = _thin_stroke_mask(source_fg, config) if config.compare_thin_strokes else np.zeros_like(source_fg, dtype=bool)
    thin_recall = _mask_recall(thin_mask, rendered_fg) if config.compare_thin_strokes else 1.0
    small_recall = _mask_recall(small_mask, rendered_fg)
    outline_penalty = _double_outline_penalty(large_mask, source_fg, rendered_fg, filled_recall)
    return {
        "filled_region_recall": filled_recall,
        "large_dark_region_recall": large_recall,
        "thin_stroke_recall": thin_recall,
        "small_detail_recall": small_recall,
        "double_outline_penalty": outline_penalty,
    }


def _large_and_small_component_masks(source_fg: np.ndarray, config: RasterMetricsConfig) -> tuple[np.ndarray, np.ndarray]:
    mask = source_fg.astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=config.component_connectivity)
    large = np.zeros_like(source_fg, dtype=bool)
    small = np.zeros_like(source_fg, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= config.filled_region_min_area:
            large |= labels == label
        if area <= config.small_detail_max_area:
            small |= labels == label
    return large, small


def _thin_stroke_mask(source_fg: np.ndarray, config: RasterMetricsConfig) -> np.ndarray:
    if not np.any(source_fg):
        return np.zeros_like(source_fg, dtype=bool)
    distance = cv2.distanceTransform(source_fg.astype(np.uint8), cv2.DIST_L2, 3)
    return source_fg & (distance <= float(config.thin_stroke_max_width))


def _double_outline_penalty(
    large_mask: np.ndarray,
    source_fg: np.ndarray,
    rendered_fg: np.ndarray,
    filled_recall: float,
) -> float:
    if not np.any(large_mask):
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = large_mask & ~cv2.erode(large_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary_recall = _mask_recall(boundary, rendered_fg)
    rendered_inside = int(np.count_nonzero(rendered_fg & large_mask))
    source_inside = int(np.count_nonzero(source_fg & large_mask))
    interior_ratio = _safe_ratio(rendered_inside, source_inside)
    return _clamp01(boundary_recall * (1.0 - filled_recall) * (1.0 - min(1.0, interior_ratio)))


def _edges(gray: np.ndarray, threshold: int) -> np.ndarray:
    low = max(0, int(threshold))
    high = min(255, max(low + 1, low * 2))
    return cv2.Canny(gray.astype(np.uint8), low, high) > 0


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    if not np.any(mask):
        return 0
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return int(max(0, count - 1))


def _bbox_difference(source: np.ndarray, rendered: np.ndarray) -> float:
    source_box = _bbox(source)
    rendered_box = _bbox(rendered)
    if source_box is None and rendered_box is None:
        return 0.0
    if source_box is None or rendered_box is None:
        return 1.0
    height, width = source.shape
    diagonal = max(hypot(width, height), 1.0)
    difference = sum(abs(float(a) - float(b)) for a, b in zip(source_box, rendered_box, strict=False)) / (4.0 * diagonal)
    return _clamp01(difference)


def _centroid_shift(source: np.ndarray, rendered: np.ndarray) -> float:
    source_centroid = _centroid(source)
    rendered_centroid = _centroid(rendered)
    if source_centroid is None and rendered_centroid is None:
        return 0.0
    if source_centroid is None or rendered_centroid is None:
        return 1.0
    height, width = source.shape
    return _clamp01(hypot(source_centroid[0] - rendered_centroid[0], source_centroid[1] - rendered_centroid[1]) / max(hypot(width, height), 1.0))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_recall(source_mask: np.ndarray, rendered_fg: np.ndarray) -> float:
    source_count = int(np.count_nonzero(source_mask))
    if source_count == 0:
        return 1.0
    return _safe_ratio(int(np.count_nonzero(source_mask & rendered_fg)), source_count)


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _resize_if_needed(array: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return array.astype(np.uint8, copy=True)
    width, height = int(target_size[0]), int(target_size[1])
    if array.shape[1] == width and array.shape[0] == height:
        return array.astype(np.uint8, copy=True)
    pil = Image.fromarray(array.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8).copy()


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    if data.dtype == np.uint8:
        return data.copy()
    if np.issubdtype(data.dtype, np.floating):
        max_value = 1.0 if float(np.nanmax(data)) <= 1.0 else 255.0
        data = np.nan_to_num(data, nan=0.0, posinf=max_value, neginf=0.0)
        return np.clip(np.rint(data * (255.0 / max_value)), 0, 255).astype(np.uint8)
    return np.clip(data, 0, 255).astype(np.uint8)


def _precision(intersection: int, false_positive: int) -> float:
    denominator = intersection + false_positive
    if denominator == 0:
        return 1.0
    return _safe_ratio(intersection, denominator)


def _recall(intersection: int, false_negative: int) -> float:
    denominator = intersection + false_negative
    if denominator == 0:
        return 1.0
    return _safe_ratio(intersection, denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _iou(intersection: int, union: int) -> float:
    if union == 0:
        return 1.0
    return _safe_ratio(intersection, union)


def _safe_ratio(numerator: int | float, denominator: int | float, empty_value: float = 0.0) -> float:
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return float(empty_value)
    return float(numerator) / denominator_float


def _image_hash(array: np.ndarray) -> str:
    return __import__("hashlib").sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _looks_like_pil_image(image: object) -> bool:
    return hasattr(image, "mode") and hasattr(image, "size") and hasattr(image, "convert")


def _coerce_rgb_tuple(name: str, value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three RGB channels.")
    channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"{name} channels must be between 0 and 255.")
    return channels


def _coerce_mode(value: RasterComparisonMode | str) -> RasterComparisonMode:
    if isinstance(value, RasterComparisonMode):
        return value
    normalized = str(value).strip().lower()
    for item in RasterComparisonMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported raster comparison mode: {value!r}")


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "RasterComparisonMetrics",
    "RasterComparisonMode",
    "RasterMetricsConfig",
    "compare_rasters",
    "image_to_rgb_array",
    "raster_summary",
]
