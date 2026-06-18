"""Line-art diagnostics for Classic semantic stroke/fill separation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite, pi
from typing import Any

import cv2
import numpy as np

from fikzpy.core.centerline_pipeline import skeletonize_mask


class StrokeFillClassification(Enum):
    """Per-component stroke/fill classification."""

    THIN_STROKE = "thin_stroke"
    FILLED_REGION = "filled_region"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class LineArtDiagnosticsConfig:
    """Tunable thresholds for line-art versus filled-region diagnostics."""

    component_connectivity: int = 8
    thin_stroke_max_width: float = 3.2
    filled_region_min_area: int = 48
    minimum_fill_ratio_for_filled_region: float = 0.24
    minimum_compactness_for_filled_region: float = 0.035
    maximum_skeleton_ratio_for_filled_region: float = 0.24
    line_art_foreground_ratio: float = 0.16
    mixed_solid_area_ratio: float = 0.025
    binary_solid_area_ratio: float = 0.08

    def __post_init__(self) -> None:
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        for name in (
            "thin_stroke_max_width",
            "minimum_fill_ratio_for_filled_region",
            "minimum_compactness_for_filled_region",
            "maximum_skeleton_ratio_for_filled_region",
            "line_art_foreground_ratio",
            "mixed_solid_area_ratio",
            "binary_solid_area_ratio",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if int(self.filled_region_min_area) < 1:
            raise ValueError("filled_region_min_area must be positive.")
        object.__setattr__(self, "filled_region_min_area", int(self.filled_region_min_area))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class ComponentStrokeFillMetrics:
    """Measured stroke/fill evidence for one connected foreground component."""

    component_id: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    fill_ratio: float
    compactness: float
    max_thickness: float
    mean_thickness: float
    median_thickness: float
    skeleton_ratio: float
    relative_area: float
    classification: StrokeFillClassification
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic component diagnostics."""
        return {
            "component_id": self.component_id,
            "area": self.area,
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 3), round(self.centroid[1], 3)],
            "fill_ratio": _rounded(self.fill_ratio),
            "compactness": _rounded(self.compactness),
            "max_thickness": _rounded(self.max_thickness),
            "mean_thickness": _rounded(self.mean_thickness),
            "median_thickness": _rounded(self.median_thickness),
            "skeleton_ratio": _rounded(self.skeleton_ratio),
            "relative_area": _rounded(self.relative_area),
            "classification": self.classification.value,
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class LineArtDiagnosticsResult:
    """Global line-art, mixed, and filled-region confidence diagnostics."""

    foreground_ratio: float
    skeleton_area_ratio: float
    median_component_thickness: float
    p90_component_thickness: float
    filled_component_count: int
    thin_component_count: int
    ambiguous_component_count: int
    solid_component_area_ratio: float
    thin_component_area_ratio: float
    largest_component_area_ratio: float
    large_hollow_component_count: int
    line_art_confidence: float
    mixed_monochrome_confidence: float
    binary_outline_confidence: float
    component_metrics: tuple[ComponentStrokeFillMetrics, ...]
    warnings: tuple[str, ...]
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without embedding masks."""
        return {
            "foreground_ratio": _rounded(self.foreground_ratio),
            "skeleton_area_ratio": _rounded(self.skeleton_area_ratio),
            "median_component_thickness": _rounded(self.median_component_thickness),
            "p90_component_thickness": _rounded(self.p90_component_thickness),
            "filled_component_count": self.filled_component_count,
            "thin_component_count": self.thin_component_count,
            "ambiguous_component_count": self.ambiguous_component_count,
            "solid_component_area_ratio": _rounded(self.solid_component_area_ratio),
            "thin_component_area_ratio": _rounded(self.thin_component_area_ratio),
            "largest_component_area_ratio": _rounded(self.largest_component_area_ratio),
            "large_hollow_component_count": self.large_hollow_component_count,
            "line_art_confidence": _rounded(self.line_art_confidence),
            "mixed_monochrome_confidence": _rounded(self.mixed_monochrome_confidence),
            "binary_outline_confidence": _rounded(self.binary_outline_confidence),
            "component_metrics": [item.to_dict() for item in self.component_metrics],
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


def analyze_line_art_mask(
    mask: np.ndarray,
    config: LineArtDiagnosticsConfig | None = None,
) -> LineArtDiagnosticsResult:
    """Analyze whether a binary foreground mask is line art, filled, or mixed."""
    effective_config = config or LineArtDiagnosticsConfig()
    foreground = _normalize_mask(mask)
    foreground_pixels = int(np.count_nonzero(foreground))
    total_pixels = int(foreground.size)
    warnings: list[str] = []
    if foreground_pixels == 0:
        warnings.append("empty_foreground_mask")
        return _empty_result(total_pixels, warnings)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=effective_config.component_connectivity,
    )
    components: list[ComponentStrokeFillMetrics] = []
    thickness_values: list[float] = []
    for component_id in range(1, int(count)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_mask = labels == component_id
        metrics = _component_metrics(
            component_id=component_id,
            component_mask=component_mask,
            bbox=(x, y, width, height),
            centroid=(float(centroids[component_id][0]), float(centroids[component_id][1])),
            total_pixels=total_pixels,
            config=effective_config,
        )
        components.append(metrics)
        thickness_values.append(metrics.median_thickness)

    skeleton_ratio = _safe_ratio(int(np.count_nonzero(skeletonize_mask(foreground))), foreground_pixels)
    filled_area = sum(item.area for item in components if item.classification is StrokeFillClassification.FILLED_REGION)
    thin_area = sum(item.area for item in components if item.classification is StrokeFillClassification.THIN_STROKE)
    largest_area = max((item.area for item in components), default=0)
    large_hollow = sum(
        1
        for item in components
        if item.area >= effective_config.filled_region_min_area * 4
        and item.fill_ratio < effective_config.minimum_fill_ratio_for_filled_region
        and item.skeleton_ratio > effective_config.maximum_skeleton_ratio_for_filled_region
    )
    solid_ratio = _safe_ratio(filled_area, total_pixels)
    thin_ratio = _safe_ratio(thin_area, total_pixels)
    foreground_ratio = _safe_ratio(foreground_pixels, total_pixels)
    median_thickness = float(np.median(thickness_values)) if thickness_values else 0.0
    p90_thickness = float(np.percentile(thickness_values, 90)) if thickness_values else 0.0

    sparse_score = 1.0 - _clamp01(foreground_ratio / max(effective_config.line_art_foreground_ratio, 1e-9))
    skeleton_score = _clamp01(skeleton_ratio / max(effective_config.maximum_skeleton_ratio_for_filled_region, 1e-9))
    no_solid_score = 1.0 - _clamp01(solid_ratio / max(effective_config.mixed_solid_area_ratio, 1e-9))
    thin_thickness_score = 1.0 - _clamp01(median_thickness / max(effective_config.thin_stroke_max_width * 1.6, 1e-9))
    line_art_confidence = _clamp01(
        sparse_score * 0.28
        + skeleton_score * 0.34
        + no_solid_score * 0.24
        + thin_thickness_score * 0.10
        + min(0.04, large_hollow * 0.02)
    )
    binary_outline_confidence = _clamp01(
        _clamp01(foreground_ratio / 0.18) * 0.28
        + _clamp01(solid_ratio / max(effective_config.binary_solid_area_ratio, 1e-9)) * 0.42
        + (1.0 - _clamp01(skeleton_ratio / max(effective_config.maximum_skeleton_ratio_for_filled_region, 1e-9))) * 0.20
        + _clamp01(p90_thickness / max(effective_config.thin_stroke_max_width * 2.0, 1e-9)) * 0.10
    )
    mixed_confidence = 0.0
    if solid_ratio > 0.0 and thin_ratio > 0.0:
        mixed_confidence = _clamp01(
            _clamp01(solid_ratio / max(effective_config.mixed_solid_area_ratio, 1e-9)) * 0.50
            + _clamp01(thin_ratio / max(foreground_ratio * 0.18, 1e-9)) * 0.30
            + _clamp01(len(components) / 6.0) * 0.20
        )

    payload = {
        "foreground_ratio": foreground_ratio,
        "skeleton_area_ratio": skeleton_ratio,
        "median_component_thickness": median_thickness,
        "p90_component_thickness": p90_thickness,
        "solid_component_area_ratio": solid_ratio,
        "thin_component_area_ratio": thin_ratio,
        "largest_component_area_ratio": _safe_ratio(largest_area, total_pixels),
        "components": [item.to_dict() for item in components],
        "warnings": warnings,
    }
    return LineArtDiagnosticsResult(
        foreground_ratio=foreground_ratio,
        skeleton_area_ratio=skeleton_ratio,
        median_component_thickness=median_thickness,
        p90_component_thickness=p90_thickness,
        filled_component_count=sum(1 for item in components if item.classification is StrokeFillClassification.FILLED_REGION),
        thin_component_count=sum(1 for item in components if item.classification is StrokeFillClassification.THIN_STROKE),
        ambiguous_component_count=sum(1 for item in components if item.classification is StrokeFillClassification.AMBIGUOUS),
        solid_component_area_ratio=solid_ratio,
        thin_component_area_ratio=thin_ratio,
        largest_component_area_ratio=_safe_ratio(largest_area, total_pixels),
        large_hollow_component_count=large_hollow,
        line_art_confidence=line_art_confidence,
        mixed_monochrome_confidence=mixed_confidence,
        binary_outline_confidence=binary_outline_confidence,
        component_metrics=tuple(sorted(components, key=lambda item: item.component_id)),
        warnings=tuple(warnings),
        deterministic_hash=_hash_payload(payload),
    )


def _component_metrics(
    *,
    component_id: int,
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    centroid: tuple[float, float],
    total_pixels: int,
    config: LineArtDiagnosticsConfig,
) -> ComponentStrokeFillMetrics:
    x, y, width, height = bbox
    area = int(np.count_nonzero(component_mask))
    bbox_area = max(1, width * height)
    fill_ratio = float(area / bbox_area)
    distances = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 3)
    foreground_distances = distances[component_mask]
    if foreground_distances.size:
        max_thickness = float(np.max(foreground_distances)) * 2.0
        mean_thickness = float(np.mean(foreground_distances)) * 2.0
        median_thickness = float(np.median(foreground_distances)) * 2.0
    else:
        max_thickness = mean_thickness = median_thickness = 0.0
    skeleton = skeletonize_mask(component_mask)
    skeleton_ratio = _safe_ratio(int(np.count_nonzero(skeleton)), area)
    compactness = _component_compactness(component_mask, area)
    classification, reason = _classify_component(
        area=area,
        fill_ratio=fill_ratio,
        compactness=compactness,
        max_thickness=max_thickness,
        median_thickness=median_thickness,
        skeleton_ratio=skeleton_ratio,
        config=config,
    )
    return ComponentStrokeFillMetrics(
        component_id=int(component_id),
        area=area,
        bbox=(int(x), int(y), int(width), int(height)),
        centroid=centroid,
        fill_ratio=fill_ratio,
        compactness=compactness,
        max_thickness=max_thickness,
        mean_thickness=mean_thickness,
        median_thickness=median_thickness,
        skeleton_ratio=skeleton_ratio,
        relative_area=_safe_ratio(area, total_pixels),
        classification=classification,
        decision_reason=reason,
    )


def _classify_component(
    *,
    area: int,
    fill_ratio: float,
    compactness: float,
    max_thickness: float,
    median_thickness: float,
    skeleton_ratio: float,
    config: LineArtDiagnosticsConfig,
) -> tuple[StrokeFillClassification, str]:
    large_enough = area >= config.filled_region_min_area
    dense_enough = fill_ratio >= config.minimum_fill_ratio_for_filled_region
    compact_enough = compactness >= config.minimum_compactness_for_filled_region
    skeleton_low = skeleton_ratio <= config.maximum_skeleton_ratio_for_filled_region
    thick_enough = (
        max_thickness > config.thin_stroke_max_width * 1.35
        or median_thickness > config.thin_stroke_max_width * 0.85
    )
    if large_enough and dense_enough and compact_enough and skeleton_low and thick_enough:
        return StrokeFillClassification.FILLED_REGION, "solid_component_evidence"
    if not large_enough:
        return StrokeFillClassification.THIN_STROKE, "below_filled_area_threshold"
    if skeleton_ratio > config.maximum_skeleton_ratio_for_filled_region:
        return StrokeFillClassification.THIN_STROKE, "skeleton_explains_component"
    if fill_ratio < config.minimum_fill_ratio_for_filled_region:
        return StrokeFillClassification.THIN_STROKE, "low_component_fill_ratio"
    if not thick_enough:
        return StrokeFillClassification.THIN_STROKE, "component_too_thin"
    return StrokeFillClassification.AMBIGUOUS, "ambiguous_fill_evidence"


def _component_compactness(mask: np.ndarray, area: int) -> float:
    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    if perimeter <= 0.0:
        return 0.0
    return _clamp01((4.0 * pi * float(area)) / (perimeter * perimeter))


def _empty_result(total_pixels: int, warnings: list[str]) -> LineArtDiagnosticsResult:
    payload = {"foreground_ratio": 0.0, "total_pixels": total_pixels, "warnings": warnings}
    return LineArtDiagnosticsResult(
        foreground_ratio=0.0,
        skeleton_area_ratio=0.0,
        median_component_thickness=0.0,
        p90_component_thickness=0.0,
        filled_component_count=0,
        thin_component_count=0,
        ambiguous_component_count=0,
        solid_component_area_ratio=0.0,
        thin_component_area_ratio=0.0,
        largest_component_area_ratio=0.0,
        large_hollow_component_count=0,
        line_art_confidence=1.0,
        mixed_monochrome_confidence=0.0,
        binary_outline_confidence=0.0,
        component_metrics=(),
        warnings=tuple(warnings),
        deterministic_hash=_hash_payload(payload),
    )


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("mask must not be empty.")
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.all(np.isfinite(array)):
        raise ValueError("mask values must be finite.")
    return array > 0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return float(numerator) / denominator_float


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ComponentStrokeFillMetrics",
    "LineArtDiagnosticsConfig",
    "LineArtDiagnosticsResult",
    "StrokeFillClassification",
    "analyze_line_art_mask",
]
