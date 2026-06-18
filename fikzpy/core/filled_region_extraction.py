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


@dataclass(frozen=True)
class FilledRegionExtractionConfig:
    """Configuration for deterministic filled-region extraction."""

    minimum_area: int = 48
    minimum_fill_ratio: float = 0.22
    contour_simplify_epsilon: float = 0.006
    component_connectivity: int = 8
    preserve_holes: bool = True
    stroke_width: float = 1.0
    fill_color: RGBColor = field(default_factory=RGBColor.black)
    hole_fill_color: RGBColor = field(default_factory=lambda: RGBColor(255, 255, 255))
    backend: FilledRegionBackend | str = FilledRegionBackend.OPENCV_CONTOURS
    strict: bool = False

    def __post_init__(self) -> None:
        if int(self.minimum_area) < 1:
            raise ValueError("minimum_area must be positive.")
        object.__setattr__(self, "minimum_area", int(self.minimum_area))
        for name in ("minimum_fill_ratio", "contour_simplify_epsilon"):
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
            "contour_simplify_epsilon": self.contour_simplify_epsilon,
            "component_connectivity": self.component_connectivity,
            "preserve_holes": self.preserve_holes,
            "stroke_width": self.stroke_width,
            "fill_color": self.fill_color.to_dict(),
            "hole_fill_color": self.hole_fill_color.to_dict(),
            "backend": self.backend.value,
            "strict": self.strict,
        }


@dataclass(frozen=True)
class FilledRegionExtractionResult:
    """Filled shape primitives and diagnostics."""

    primitives: tuple[ClosedShapePrimitive, ...]
    filled_mask: np.ndarray
    region_count: int
    hole_count: int
    component_count: int
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
        primitives, region_count, hole_count, warnings = _contours_to_primitives(normalized, effective_config)
        payload = {
            "mask": _array_summary(normalized),
            "primitives": [primitive.to_dict() for primitive in primitives],
            "region_count": region_count,
            "hole_count": hole_count,
            "component_count": _component_count(normalized, effective_config.component_connectivity),
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
) -> tuple[list[ClosedShapePrimitive], int, int, list[str]]:
    warnings: list[str] = []
    if not np.any(mask):
        return [], 0, 0, ["empty_filled_region_mask"]
    contours, hierarchy = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], 0, 0, ["no_contours_found"]
    hierarchy_rows = hierarchy[0]
    primitives: list[ClosedShapePrimitive] = []
    region_count = 0
    hole_count = 0
    outer_indices = [index for index, row in enumerate(hierarchy_rows) if int(row[3]) < 0]
    outer_indices.sort(key=lambda index: _contour_sort_key(contours[index]))
    for outer_index in outer_indices:
        primitive = _primitive_from_contour(contours[outer_index], outer_index, None, False, config)
        if primitive is None:
            warnings.append(f"skipped_small_region:{outer_index}")
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
                hole = _primitive_from_contour(contours[child_index], child_index, outer_index, True, config)
                if hole is None:
                    warnings.append(f"skipped_small_hole:{child_index}")
                    continue
                primitives.append(hole)
                hole_count += 1
        elif int(hierarchy_rows[outer_index][2]) >= 0:
            warnings.append("holes_preserved_as_metadata_only")
    return primitives, region_count, hole_count, warnings


def _primitive_from_contour(
    contour: np.ndarray,
    contour_index: int,
    parent_index: int | None,
    is_hole: bool,
    config: FilledRegionExtractionConfig,
) -> ClosedShapePrimitive | None:
    area = float(abs(cv2.contourArea(contour)))
    if area < config.minimum_area:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(0.5, perimeter * config.contour_simplify_epsilon)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = tuple(Point2D(float(x), float(y)) for x, y in approx.reshape(-1, 2))
    if len(points) < 3:
        return None
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
        "bbox": [int(x), int(y), int(width), int(height)],
        "centroid": [round(centroid[0], 3), round(centroid[1], 3)],
        "fill_ratio": round(area / bbox_area, 6),
        "strategy": "filled_region_extraction",
        "hole": bool(is_hole),
    }
    fill = FillStyle(config.hole_fill_color if is_hole else config.fill_color)
    stroke = StrokeStyle(config.fill_color, width=config.stroke_width)
    if is_hole:
        stroke = StrokeStyle(config.hole_fill_color, width=config.stroke_width, opacity=0.0)
    return ClosedShapePrimitive(points, stroke=stroke, fill=fill, metadata=metadata)


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
    "FilledRegionExtractionConfig",
    "FilledRegionExtractionError",
    "FilledRegionExtractionResult",
    "extract_filled_regions",
]
