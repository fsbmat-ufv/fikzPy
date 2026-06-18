"""Hybrid extraction for mixed monochrome Classic images."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import cv2
import numpy as np

from fikzpy.core.centerline_pipeline import CenterlineConfig, CenterlineResult, centerline_paths_to_polylines
from fikzpy.core.centerline_pipeline import extract_centerlines
from fikzpy.core.filled_region_extraction import FilledRegionExtractionConfig, FilledRegionExtractionResult
from fikzpy.core.filled_region_extraction import extract_filled_regions
from fikzpy.core.lineart_diagnostics import LineArtDiagnosticsConfig, StrokeFillClassification
from fikzpy.core.lineart_diagnostics import analyze_line_art_mask
from fikzpy.core.semantic_geometry import PolylinePrimitive, PrimitiveGroup, StrokeStyle


class MixedMonochromeStrategy(Enum):
    """Available mixed monochrome extraction strategies."""

    HYBRID_CENTERLINE_AND_FILLED_REGIONS = "hybrid_centerline_and_filled_regions"


@dataclass(frozen=True)
class ThinStrokeExtractionResult:
    """Centerline primitives extracted from probable thin strokes."""

    primitives: tuple[PolylinePrimitive, ...]
    centerline_result: CenterlineResult | None
    thin_mask: np.ndarray
    path_count: int
    warnings: tuple[str, ...]
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without embedding full masks."""
        return {
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "centerline_result": self.centerline_result.to_dict() if self.centerline_result is not None else None,
            "thin_mask": _array_summary(self.thin_mask),
            "path_count": self.path_count,
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


@dataclass(frozen=True)
class ForegroundLayerSplit:
    """Foreground masks split into filled regions and thin strokes."""

    foreground_mask: np.ndarray
    filled_mask: np.ndarray
    thin_mask: np.ndarray
    filled_component_ids: tuple[int, ...]
    thin_component_ids: tuple[int, ...]
    ambiguous_component_ids: tuple[int, ...]
    component_summaries: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    deterministic_hash: str

    @property
    def filled_count(self) -> int:
        """Return number of components classified as filled."""
        return len(self.filled_component_ids)

    @property
    def thin_count(self) -> int:
        """Return number of components classified as thin."""
        return len(self.thin_component_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without embedding full masks."""
        return {
            "foreground_mask": _array_summary(self.foreground_mask),
            "filled_mask": _array_summary(self.filled_mask),
            "thin_mask": _array_summary(self.thin_mask),
            "filled_component_ids": list(self.filled_component_ids),
            "thin_component_ids": list(self.thin_component_ids),
            "ambiguous_component_ids": list(self.ambiguous_component_ids),
            "component_summaries": list(self.component_summaries),
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


@dataclass(frozen=True)
class MixedMonochromeResult:
    """Hybrid filled-region and thin-stroke extraction result."""

    primitives: tuple[PrimitiveGroup, ...]
    split: ForegroundLayerSplit
    filled_regions: FilledRegionExtractionResult
    thin_strokes: ThinStrokeExtractionResult
    strategy: MixedMonochromeStrategy
    warnings: tuple[str, ...]
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics."""
        return {
            "primitives": [group.to_dict() for group in self.primitives],
            "split": self.split.to_dict(),
            "filled_regions": self.filled_regions.to_dict(),
            "thin_strokes": self.thin_strokes.to_dict(),
            "strategy": self.strategy.value,
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


def split_foreground_layers(
    mask: np.ndarray,
    *,
    filled_region_min_area: int,
    filled_region_min_ratio: float,
    thin_stroke_max_width: float,
    component_connectivity: int = 8,
    minimum_fill_ratio_for_filled_region: float | None = None,
    minimum_compactness_for_filled_region: float = 0.035,
    maximum_skeleton_ratio_for_filled_region: float = 0.24,
    prefer_lineart_when_ambiguous: bool = True,
) -> ForegroundLayerSplit:
    """Split binary foreground into probable filled and thin-stroke masks."""
    foreground = _normalize_mask(mask)
    if int(filled_region_min_area) < 1:
        raise ValueError("filled_region_min_area must be positive.")
    fill_ratio_threshold = float(filled_region_min_ratio)
    thin_width = float(thin_stroke_max_width)
    if not isfinite(fill_ratio_threshold) or fill_ratio_threshold < 0.0 or fill_ratio_threshold > 1.0:
        raise ValueError("filled_region_min_ratio must be between 0 and 1.")
    if not isfinite(thin_width) or thin_width <= 0.0:
        raise ValueError("thin_stroke_max_width must be finite and positive.")
    if component_connectivity not in {4, 8}:
        raise ValueError("component_connectivity must be 4 or 8.")
    fill_ratio_for_decision = (
        float(minimum_fill_ratio_for_filled_region)
        if minimum_fill_ratio_for_filled_region is not None
        else max(fill_ratio_threshold, 0.24)
    )
    diagnostics = analyze_line_art_mask(
        foreground,
        LineArtDiagnosticsConfig(
            component_connectivity=component_connectivity,
            thin_stroke_max_width=thin_width,
            filled_region_min_area=int(filled_region_min_area),
            minimum_fill_ratio_for_filled_region=fill_ratio_for_decision,
            minimum_compactness_for_filled_region=minimum_compactness_for_filled_region,
            maximum_skeleton_ratio_for_filled_region=maximum_skeleton_ratio_for_filled_region,
        ),
    )
    component_diagnostics = {item.component_id: item for item in diagnostics.component_metrics}
    lineart_strict = (
        prefer_lineart_when_ambiguous
        and diagnostics.line_art_confidence >= 0.55
        and diagnostics.solid_component_area_ratio <= 0.02
    )

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=component_connectivity,
    )
    filled_mask = np.zeros_like(foreground, dtype=bool)
    thin_mask = np.zeros_like(foreground, dtype=bool)
    filled_ids: list[int] = []
    thin_ids: list[int] = []
    ambiguous_ids: list[int] = []
    summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for component_id in range(1, int(labels_count)):
        x, y, width, height, area = [int(value) for value in stats[component_id]]
        if area <= 0:
            continue
        bbox_area = max(1, width * height)
        fill_ratio = float(area / bbox_area)
        component_mask = labels == component_id
        thickness = _component_max_thickness(component_mask)
        component_metric = component_diagnostics.get(component_id)
        clearly_filled = (
            component_metric is not None
            and component_metric.classification is StrokeFillClassification.FILLED_REGION
            and not lineart_strict
        )
        if clearly_filled:
            filled_mask |= component_mask
            filled_ids.append(component_id)
            layer = "filled_region"
            reason = component_metric.decision_reason if component_metric is not None else "legacy_filled_evidence"
        else:
            thin_mask |= component_mask
            thin_ids.append(component_id)
            layer = "thin_stroke"
            reason = component_metric.decision_reason if component_metric is not None else "thin_by_default"
            if area >= int(filled_region_min_area) and component_metric is not None and component_metric.classification is StrokeFillClassification.AMBIGUOUS:
                ambiguous_ids.append(component_id)
                warnings.append(f"ambiguous_component_as_thin:{component_id}")
            elif lineart_strict and area >= int(filled_region_min_area):
                warnings.append(f"lineart_component_as_thin:{component_id}")
        summaries.append(
            {
                "component_id": component_id,
                "layer": layer,
                "decision": layer,
                "reason": reason,
                "area": area,
                "bbox": [x, y, width, height],
                "centroid": [round(float(centroids[component_id][0]), 3), round(float(centroids[component_id][1]), 3)],
                "fill_ratio": round(fill_ratio, 6),
                "max_thickness": round(thickness, 3),
                "compactness": round(component_metric.compactness, 6) if component_metric is not None else 0.0,
                "median_thickness": round(component_metric.median_thickness, 3) if component_metric is not None else 0.0,
                "skeleton_ratio": round(component_metric.skeleton_ratio, 6) if component_metric is not None else 0.0,
                "line_art_confidence": round(diagnostics.line_art_confidence, 6),
            }
        )

    if not np.any(foreground):
        warnings.append("empty_foreground_mask")
    if np.any(foreground) and not np.any(filled_mask):
        warnings.append("no_filled_regions_detected")
    if np.any(foreground) and not np.any(thin_mask):
        warnings.append("no_thin_strokes_detected")

    payload = {
        "foreground": _array_summary(foreground),
        "filled": _array_summary(filled_mask),
        "thin": _array_summary(thin_mask),
        "lineart_diagnostics": diagnostics.to_dict(),
        "filled_ids": filled_ids,
        "thin_ids": thin_ids,
        "ambiguous_ids": ambiguous_ids,
        "summaries": summaries,
        "warnings": warnings,
    }
    digest = _hash_payload(payload)
    return ForegroundLayerSplit(
        foreground_mask=foreground.copy(),
        filled_mask=filled_mask,
        thin_mask=thin_mask,
        filled_component_ids=tuple(filled_ids),
        thin_component_ids=tuple(thin_ids),
        ambiguous_component_ids=tuple(ambiguous_ids),
        component_summaries=tuple(summaries),
        warnings=tuple(warnings),
        deterministic_hash=digest,
    )


def extract_thin_strokes(
    mask: np.ndarray,
    config: CenterlineConfig | None = None,
    *,
    stroke_width: float = 1.0,
) -> ThinStrokeExtractionResult:
    """Extract centerline polylines from a thin-stroke mask."""
    thin_mask = _normalize_mask(mask)
    warnings: list[str] = []
    if not np.any(thin_mask):
        payload = {"mask": _array_summary(thin_mask), "primitives": [], "warnings": ["empty_thin_stroke_mask"]}
        return ThinStrokeExtractionResult(
            primitives=(),
            centerline_result=None,
            thin_mask=thin_mask.copy(),
            path_count=0,
            warnings=("empty_thin_stroke_mask",),
            deterministic_hash=_hash_payload(payload),
        )
    centerline = extract_centerlines(thin_mask, config)
    primitives: list[PolylinePrimitive] = []
    for primitive in centerline_paths_to_polylines(centerline):
        metadata = {
            **dict(primitive.metadata),
            "source_layer": "thin_stroke",
            "strategy": "centerline",
        }
        primitives.append(
            PolylinePrimitive(
                primitive.points,
                closed=primitive.closed,
                stroke=StrokeStyle(width=float(stroke_width)),
                fill=None,
                confidence=primitive.confidence,
                error=primitive.error,
                metadata=metadata,
            )
        )
    warnings.extend(centerline.warnings)
    payload = {
        "mask": _array_summary(thin_mask),
        "primitives": [primitive.to_dict() for primitive in primitives],
        "centerline": centerline.to_dict(),
        "warnings": warnings,
    }
    return ThinStrokeExtractionResult(
        primitives=tuple(primitives),
        centerline_result=centerline,
        thin_mask=thin_mask.copy(),
        path_count=len(primitives),
        warnings=tuple(warnings),
        deterministic_hash=_hash_payload(payload),
    )


def extract_mixed_monochrome_primitives(
    mask: np.ndarray,
    *,
    filled_config: FilledRegionExtractionConfig,
    centerline_config: CenterlineConfig | None = None,
    filled_region_min_area: int,
    filled_region_min_ratio: float,
    thin_stroke_max_width: float,
    component_connectivity: int = 8,
    thin_stroke_width: float = 1.0,
    minimum_fill_ratio_for_filled_region: float | None = None,
    minimum_compactness_for_filled_region: float = 0.035,
    maximum_skeleton_ratio_for_filled_region: float = 0.24,
    prefer_lineart_when_ambiguous: bool = True,
) -> MixedMonochromeResult:
    """Extract filled regions plus thin-stroke centerlines from a monochrome mask."""
    split = split_foreground_layers(
        mask,
        filled_region_min_area=filled_region_min_area,
        filled_region_min_ratio=filled_region_min_ratio,
        thin_stroke_max_width=thin_stroke_max_width,
        component_connectivity=component_connectivity,
        minimum_fill_ratio_for_filled_region=minimum_fill_ratio_for_filled_region,
        minimum_compactness_for_filled_region=minimum_compactness_for_filled_region,
        maximum_skeleton_ratio_for_filled_region=maximum_skeleton_ratio_for_filled_region,
        prefer_lineart_when_ambiguous=prefer_lineart_when_ambiguous,
    )
    filled = extract_filled_regions(split.filled_mask, filled_config)
    thin = extract_thin_strokes(split.thin_mask, centerline_config, stroke_width=thin_stroke_width)
    groups: list[PrimitiveGroup] = []
    if filled.primitives:
        groups.append(
            PrimitiveGroup(
                filled.primitives,
                name="filled_regions",
                metadata={"source_layer": "filled_region", "strategy": "mixed_monochrome"},
            )
        )
    if thin.primitives:
        groups.append(
            PrimitiveGroup(
                thin.primitives,
                name="thin_strokes",
                metadata={"source_layer": "thin_stroke", "strategy": "mixed_monochrome"},
            )
        )
    warnings = tuple((*split.warnings, *filled.warnings, *thin.warnings))
    payload = {
        "groups": [group.to_dict() for group in groups],
        "split": split.to_dict(),
        "filled": filled.to_dict(),
        "thin": thin.to_dict(),
        "warnings": warnings,
        "strategy": MixedMonochromeStrategy.HYBRID_CENTERLINE_AND_FILLED_REGIONS.value,
    }
    return MixedMonochromeResult(
        primitives=tuple(groups),
        split=split,
        filled_regions=filled,
        thin_strokes=thin,
        strategy=MixedMonochromeStrategy.HYBRID_CENTERLINE_AND_FILLED_REGIONS,
        warnings=warnings,
        deterministic_hash=_hash_payload(payload),
    )


def _component_max_thickness(component_mask: np.ndarray) -> float:
    distances = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.max(distances)) * 2.0 if distances.size else 0.0


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("mask must not be empty.")
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.all(np.isfinite(array)):
        raise ValueError("mask values must be finite.")
    return array > 0


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    data = np.asarray(array)
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "nonzero": int(np.count_nonzero(data)),
        "sha256": sha256(np.ascontiguousarray(data.astype(np.uint8)).tobytes()).hexdigest(),
    }


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ForegroundLayerSplit",
    "MixedMonochromeResult",
    "MixedMonochromeStrategy",
    "ThinStrokeExtractionResult",
    "extract_mixed_monochrome_primitives",
    "extract_thin_strokes",
    "split_foreground_layers",
]
