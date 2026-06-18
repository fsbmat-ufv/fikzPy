"""Filled-region extraction for the semantic Classic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import cv2
import numpy as np

from fikzpy.core.semantic_geometry import ClosedShapePrimitive, FillStyle, Point2D, RGBColor, StrokeStyle


class FilledRegionExtractionError(ValueError):
    """Raised when filled-region extraction cannot continue."""


class FilledRegionBackend(Enum):
    """Filled-region extraction backend."""

    OPENCV_CONTOURS = "opencv_contours"


class FilledRegionDecisionKind(Enum):
    """Accepted/rejected filled-region candidate decision."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class FilledRegionExtractionConfig:
    """Configuration for deterministic filled-region extraction."""

    minimum_area: int = 48
    minimum_fill_ratio: float = 0.22
    minimum_compactness: float = 0.035
    maximum_skeleton_ratio: float = 0.24
    minimum_median_thickness: float = 2.2
    contour_simplify_epsilon: float = 0.006
    component_connectivity: int = 8
    preserve_holes: bool = True
    stroke_width: float = 1.0
    draw_outline: bool = True
    fill_color: RGBColor = field(default_factory=RGBColor.black)
    hole_fill_color: RGBColor = field(default_factory=lambda: RGBColor(255, 255, 255))
    backend: FilledRegionBackend | str = FilledRegionBackend.OPENCV_CONTOURS
    strict: bool = False

    def __post_init__(self) -> None:
        if int(self.minimum_area) < 1:
            raise ValueError("minimum_area must be positive.")
        object.__setattr__(self, "minimum_area", int(self.minimum_area))
        for name in (
            "minimum_fill_ratio",
            "minimum_compactness",
            "maximum_skeleton_ratio",
            "minimum_median_thickness",
            "contour_simplify_epsilon",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if self.minimum_fill_ratio > 1.0:
            raise ValueError("minimum_fill_ratio must not exceed 1.")
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        if not isinstance(self.preserve_holes, bool):
            raise TypeError("preserve_holes must be a bool.")
        if not isinstance(self.draw_outline, bool):
            raise TypeError("draw_outline must be a bool.")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a bool.")
        width = float(self.stroke_width)
        if not isfinite(width) or width <= 0.0:
            raise ValueError("stroke_width must be finite and positive.")
        object.__setattr__(self, "stroke_width", width)
        if not isinstance(self.fill_color, RGBColor):
            raise TypeError("fill_color must be RGBColor.")
        if not isinstance(self.hole_fill_color, RGBColor):
            raise TypeError("hole_fill_color must be RGBColor.")
        object.__setattr__(self, "backend", _coerce_backend(self.backend))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return {
            "minimum_area": self.minimum_area,
            "minimum_fill_ratio": self.minimum_fill_ratio,
            "minimum_compactness": self.minimum_compactness,
            "maximum_skeleton_ratio": self.maximum_skeleton_ratio,
            "minimum_median_thickness": self.minimum_median_thickness,
            "contour_simplify_epsilon": self.contour_simplify_epsilon,
            "component_connectivity": self.component_connectivity,
            "preserve_holes": self.preserve_holes,
            "stroke_width": self.stroke_width,
            "draw_outline": self.draw_outline,
            "fill_color": self.fill_color.to_dict(),
            "hole_fill_color": self.hole_fill_color.to_dict(),
            "backend": self.backend.value,
            "strict": self.strict,
        }


@dataclass(frozen=True)
class FilledRegionCandidate:
    """One contour candidate and the decision used by filled extraction."""

    component_id: int
    parent_component_id: int | None
    hole: bool
    area: float
    foreground_area: int
    bbox: tuple[int, int, int, int]
    fill_ratio: float
    compactness: float
    estimated_thickness: float
    median_thickness: float
    skeleton_ratio: float
    decision: FilledRegionDecisionKind
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic candidate diagnostics."""
        return {
            "component_id": self.component_id,
            "parent_component_id": self.parent_component_id,
            "hole": self.hole,
            "area": round(float(self.area), 3),
            "foreground_area": self.foreground_area,
            "bbox": list(self.bbox),
            "fill_ratio": round(float(self.fill_ratio), 6),
            "compactness": round(float(self.compactness), 6),
            "estimated_thickness": round(float(self.estimated_thickness), 3),
            "median_thickness": round(float(self.median_thickness), 3),
            "skeleton_ratio": round(float(self.skeleton_ratio), 6),
            "decision": self.decision.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FilledRegionExtractionResult:
    """Filled shape primitives and diagnostics."""

    primitives: tuple[ClosedShapePrimitive, ...]
    filled_mask: np.ndarray
    region_count: int
    hole_count: int
    component_count: int
    candidates: tuple[FilledRegionCandidate, ...]
    warnings: tuple[str, ...]
    config: FilledRegionExtractionConfig
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without full masks."""
        return {
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "filled_mask": _array_summary(self.filled_mask),
            "region_count": self.region_count,
            "hole_count": self.hole_count,
            "component_count": self.component_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "config": self.config.to_dict(),
            "deterministic_hash": self.deterministic_hash,
        }


def extract_filled_regions(
    mask: np.ndarray,
    config: FilledRegionExtractionConfig | None = None,
) -> FilledRegionExtractionResult:
    """Extract filled dark regions as closed semantic shapes."""
    effective_config = config or FilledRegionExtractionConfig()
    try:
        normalized = _normalize_mask(mask)
        primitives, region_count, hole_count, candidates, warnings = _contours_to_primitives(normalized, effective_config)
        payload = {
            "mask": _array_summary(normalized),
            "primitives": [primitive.to_dict() for primitive in primitives],
            "region_count": region_count,
            "hole_count": hole_count,
            "component_count": _component_count(normalized, effective_config.component_connectivity),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "warnings": warnings,
            "config": effective_config.to_dict(),
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        return FilledRegionExtractionResult(
            primitives=tuple(primitives),
            filled_mask=normalized.copy(),
            region_count=region_count,
            hole_count=hole_count,
            component_count=payload["component_count"],
            candidates=tuple(candidates),
            warnings=tuple(warnings),
            config=effective_config,
            deterministic_hash=digest,
        )
    except Exception as exc:
        if effective_config.strict:
            raise FilledRegionExtractionError(str(exc)) from exc
        raise


def _contours_to_primitives(
    mask: np.ndarray,
    config: FilledRegionExtractionConfig,
) -> tuple[list[ClosedShapePrimitive], int, int, list[FilledRegionCandidate], list[str]]:
    warnings: list[str] = []
    if not np.any(mask):
        return [], 0, 0, [], ["empty_filled_region_mask"]
    contours, hierarchy = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], 0, 0, [], ["no_contours_found"]
    hierarchy_rows = hierarchy[0]
    primitives: list[ClosedShapePrimitive] = []
    candidates: list[FilledRegionCandidate] = []
    region_count = 0
    hole_count = 0
    outer_indices = [index for index, row in enumerate(hierarchy_rows) if int(row[3]) < 0]
    outer_indices.sort(key=lambda index: _contour_sort_key(contours[index]))
    for outer_index in outer_indices:
        primitive, candidate = _primitive_from_contour(mask, contours[outer_index], outer_index, None, False, config)
        candidates.append(candidate)
        if primitive is None:
            warnings.append(f"skipped_filled_region:{outer_index}:{candidate.reason}")
            continue
        primitives.append(primitive)
        region_count += 1
        if config.preserve_holes:
            child = int(hierarchy_rows[outer_index][2])
            child_indices: list[int] = []
            while child >= 0:
                child_indices.append(child)
                child = int(hierarchy_rows[child][0])
            child_indices.sort(key=lambda index: _contour_sort_key(contours[index]))
            for child_index in child_indices:
                hole, hole_candidate = _primitive_from_contour(mask, contours[child_index], child_index, outer_index, True, config)
                candidates.append(hole_candidate)
                if hole is None:
                    warnings.append(f"skipped_small_hole:{child_index}")
                    continue
                primitives.append(hole)
                hole_count += 1
        elif int(hierarchy_rows[outer_index][2]) >= 0:
            warnings.append("holes_preserved_as_metadata_only")
    return primitives, region_count, hole_count, candidates, warnings


def _primitive_from_contour(
    mask: np.ndarray,
    contour: np.ndarray,
    contour_index: int,
    parent_index: int | None,
    is_hole: bool,
    config: FilledRegionExtractionConfig,
) -> tuple[ClosedShapePrimitive | None, FilledRegionCandidate]:
    area = float(abs(cv2.contourArea(contour)))
    candidate = _candidate_from_contour(mask, contour, contour_index, parent_index, is_hole, area, config)
    if area < config.minimum_area:
        return None, candidate
    if not is_hole and candidate.decision is FilledRegionDecisionKind.REJECTED:
        return None, candidate
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(0.5, perimeter * config.contour_simplify_epsilon)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = tuple(Point2D(float(x), float(y)) for x, y in approx.reshape(-1, 2))
    if len(points) < 3:
        return None, candidate
    x, y, width, height = cv2.boundingRect(contour)
    bbox_area = max(1.0, float(width * height))
    moments = cv2.moments(contour)
    centroid = (
        float(moments["m10"] / moments["m00"]) if moments["m00"] else float(x + width / 2.0),
        float(moments["m01"] / moments["m00"]) if moments["m00"] else float(y + height / 2.0),
    )
    metadata = {
        "source_layer": "filled_region_hole" if is_hole else "filled_region",
        "component_id": int(contour_index),
        "parent_component_id": parent_index,
        "area": round(area, 3),
        "foreground_area": candidate.foreground_area,
        "bbox": [int(x), int(y), int(width), int(height)],
        "centroid": [round(centroid[0], 3), round(centroid[1], 3)],
        "fill_ratio": round(candidate.fill_ratio, 6),
        "compactness": round(candidate.compactness, 6),
        "estimated_thickness": round(candidate.estimated_thickness, 3),
        "median_thickness": round(candidate.median_thickness, 3),
        "skeleton_ratio": round(candidate.skeleton_ratio, 6),
        "decision": candidate.decision.value,
        "decision_reason": candidate.reason,
        "strategy": "filled_region_extraction",
        "hole": bool(is_hole),
    }
    fill = FillStyle(config.hole_fill_color if is_hole else config.fill_color)
    stroke = StrokeStyle(config.fill_color, width=config.stroke_width, opacity=None if config.draw_outline else 0.0)
    if is_hole:
        stroke = StrokeStyle(config.hole_fill_color, width=config.stroke_width, opacity=0.0)
    return ClosedShapePrimitive(points, stroke=stroke, fill=fill, metadata=metadata), candidate


def _candidate_from_contour(
    mask: np.ndarray,
    contour: np.ndarray,
    contour_index: int,
    parent_index: int | None,
    is_hole: bool,
    area: float,
    config: FilledRegionExtractionConfig,
) -> FilledRegionCandidate:
    x, y, width, height = cv2.boundingRect(contour)
    bbox_area = max(1, int(width * height))
    roi = mask[y : y + height, x : x + width]
    foreground_area = int(np.count_nonzero(roi))
    fill_ratio = float(foreground_area / bbox_area)
    compactness = _compactness_from_contour(contour, foreground_area)
    distances = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 3)
    foreground_distances = distances[roi > 0]
    if foreground_distances.size:
        estimated_thickness = float(np.max(foreground_distances)) * 2.0
        median_thickness = float(np.median(foreground_distances)) * 2.0
    else:
        estimated_thickness = 0.0
        median_thickness = 0.0
    skeleton_ratio = _skeleton_ratio(roi)
    decision = FilledRegionDecisionKind.ACCEPTED
    reason = "accepted"
    if area < config.minimum_area:
        decision = FilledRegionDecisionKind.REJECTED
        reason = "area_below_minimum"
    elif not is_hole:
        if fill_ratio < config.minimum_fill_ratio:
            decision = FilledRegionDecisionKind.REJECTED
            reason = "component_fill_ratio_below_minimum"
        elif compactness < config.minimum_compactness:
            decision = FilledRegionDecisionKind.REJECTED
            reason = "component_compactness_below_minimum"
        elif skeleton_ratio > config.maximum_skeleton_ratio:
            decision = FilledRegionDecisionKind.REJECTED
            reason = "component_skeleton_ratio_too_high"
        elif median_thickness < config.minimum_median_thickness:
            decision = FilledRegionDecisionKind.REJECTED
            reason = "component_thickness_below_minimum"
    return FilledRegionCandidate(
        component_id=int(contour_index),
        parent_component_id=parent_index,
        hole=bool(is_hole),
        area=float(area),
        foreground_area=foreground_area,
        bbox=(int(x), int(y), int(width), int(height)),
        fill_ratio=fill_ratio,
        compactness=compactness,
        estimated_thickness=estimated_thickness,
        median_thickness=median_thickness,
        skeleton_ratio=skeleton_ratio,
        decision=decision,
        reason=reason,
    )


def _compactness_from_contour(contour: np.ndarray, foreground_area: int) -> float:
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 4.0 * np.pi * float(foreground_area) / (perimeter * perimeter))))


def _skeleton_ratio(mask: np.ndarray) -> float:
    foreground = np.asarray(mask) > 0
    foreground_area = int(np.count_nonzero(foreground))
    if foreground_area <= 0:
        return 0.0
    try:
        from fikzpy.core.centerline_pipeline import skeletonize_mask

        skeleton = skeletonize_mask(foreground)
        return float(np.count_nonzero(skeleton) / foreground_area)
    except Exception:
        return 1.0


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("filled-region mask must be 2D.")
    if array.dtype == bool:
        return array.copy()
    return array > 0


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    if not np.any(mask):
        return 0
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return int(max(0, count - 1))


def _contour_sort_key(contour: np.ndarray) -> tuple[int, int, float]:
    x, y, width, height = cv2.boundingRect(contour)
    area = float(abs(cv2.contourArea(contour)))
    return int(y), int(x), -area if width * height else 0.0


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    data = np.asarray(array)
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "nonzero": int(np.count_nonzero(data)),
        "sha256": sha256(np.ascontiguousarray(data.astype(np.uint8)).tobytes()).hexdigest(),
    }


def _coerce_backend(value: FilledRegionBackend | str) -> FilledRegionBackend:
    if isinstance(value, FilledRegionBackend):
        return value
    normalized = str(value).strip().lower()
    for item in FilledRegionBackend:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported filled-region backend: {value!r}")


__all__ = [
    "FilledRegionBackend",
    "FilledRegionCandidate",
    "FilledRegionDecisionKind",
    "FilledRegionExtractionConfig",
    "FilledRegionExtractionError",
    "FilledRegionExtractionResult",
    "extract_filled_regions",
]
