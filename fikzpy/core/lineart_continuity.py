"""Line-art continuity metrics and conservative outline-stroke recovery.

Issue 11.6 adds a balance layer for the Classic LINE_ART strategy: it measures
whether centerline extraction preserved contour continuity and foreground
coverage, and provides a conservative contour-based fallback (stroke only,
never filled) for components where the centerline is fragmented or missing.
It also distinguishes "overfilled" line art (artificial black mass) from
"underdrawn" line art (lost contours, fragmented paths) so the Classic
pipeline can reject both failure modes instead of only guarding against one.

This module does not call any SVG-to-TikZ bridge, Visual mode, GUI code, or
external tracer process.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import cv2
import numpy as np

from fikzpy.core.centerline_pipeline import CenterlinePath, CenterlineResult, PathClosureType
from fikzpy.core.semantic_geometry import Point2D, PolylinePrimitive, PrimitiveGroup, StrokeStyle


class LineArtContinuityError(ValueError):
    """Raised when line-art continuity analysis cannot continue."""


@dataclass(frozen=True)
class LineArtContinuityMetrics:
    """Scalar continuity diagnostics comparing a mask to rendered centerlines."""

    components_before: int
    components_after: int
    lost_component_count: int
    endpoint_count: int
    junction_count: int
    path_count: int
    broken_path_count: int
    average_path_length: float
    contour_coverage: float
    edge_recall: float
    foreground_recall: float
    skeleton_fragmentation: float
    contour_bbox_coverage: float
    external_contour_preservation: float

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        data = dict(self.__dict__)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = _rounded(value)
        return data


@dataclass(frozen=True)
class LineArtContinuityDecision:
    """Outline-recovery decision derived from continuity metrics."""

    needs_outline_recovery: bool
    flags: tuple[str, ...]
    metrics: LineArtContinuityMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return serializable decision diagnostics."""
        return {
            "needs_outline_recovery": self.needs_outline_recovery,
            "flags": list(self.flags),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class OutlineRecoveryResult:
    """Conservative stroke-only outline primitives recovered for weak components."""

    primitives: tuple[PolylinePrimitive, ...]
    recovered_component_count: int
    skipped_component_count: int
    warnings: tuple[str, ...]
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics."""
        return {
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "recovered_component_count": self.recovered_component_count,
            "skipped_component_count": self.skipped_component_count,
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


@dataclass(frozen=True)
class LineArtFillMetrics:
    """Polygon-area based fill diagnostics for a line-art primitive set."""

    filled_area_ratio: float
    white_cutout_ratio: float
    filled_region_count: int
    white_cutout_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        return {
            "filled_area_ratio": _rounded(self.filled_area_ratio),
            "white_cutout_ratio": _rounded(self.white_cutout_ratio),
            "filled_region_count": self.filled_region_count,
            "white_cutout_count": self.white_cutout_count,
        }


@dataclass(frozen=True)
class LineArtBalanceResult:
    """Balanced overfilled/underdrawn validation for Classic line art."""

    accepted: bool
    flags: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    fill_metrics: LineArtFillMetrics
    continuity_metrics: LineArtContinuityMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable diagnostics."""
        return {
            "accepted": self.accepted,
            "flags": list(self.flags),
            "rejection_reasons": list(self.rejection_reasons),
            "fill_metrics": self.fill_metrics.to_dict(),
            "continuity_metrics": self.continuity_metrics.to_dict(),
        }


_OVERFILL_FLAGS = frozenset(
    {
        "excessive_filled_area",
        "artificial_black_mass",
        "overfilled_lineart",
        "lineart_converted_to_silhouette",
        "excessive_white_cutouts",
    }
)
_UNDERDRAW_FLAGS = frozenset(
    {
        "underdrawn_lineart",
        "lost_contour_structure",
        "low_edge_recall",
        "excessive_line_fragmentation",
        "lost_external_contour",
        "missing_internal_details",
    }
)


def compute_lineart_continuity_metrics(
    mask: np.ndarray,
    centerline_result: CenterlineResult | None,
    *,
    component_connectivity: int = 8,
    rendered_override: np.ndarray | None = None,
) -> LineArtContinuityMetrics:
    """Compare a binary line-art mask against rendered centerline paths.

    ``rendered_override`` lets a caller recompute coverage metrics against the
    final primitive set (thin strokes plus any outline recovery) instead of
    the raw centerline paths, so acceptance reflects the recovered output.
    """
    foreground = _normalize_mask(mask)
    height, width = foreground.shape
    paths: tuple[CenterlinePath, ...] = centerline_result.paths if centerline_result is not None else ()
    rendered = _render_paths_mask(paths, (width, height)) if rendered_override is None else np.asarray(rendered_override, dtype=bool)

    components_before = _component_count(foreground, component_connectivity)
    components_after = _component_count(rendered, component_connectivity)
    lost_component_count = max(0, components_before - components_after)

    foreground_pixels = int(np.count_nonzero(foreground))
    foreground_recall = _safe_ratio(int(np.count_nonzero(foreground & rendered)), foreground_pixels, empty_value=1.0)

    dilated_rendered = _dilate(rendered, 1)
    contour_coverage = _safe_ratio(int(np.count_nonzero(foreground & dilated_rendered)), foreground_pixels, empty_value=1.0)

    source_edges = _edge_mask(foreground)
    rendered_edges = _edge_mask(rendered)
    source_edge_pixels = int(np.count_nonzero(source_edges))
    edge_recall = _safe_ratio(int(np.count_nonzero(source_edges & rendered_edges)), source_edge_pixels, empty_value=1.0)

    external_mask = _external_contour_mask(foreground, component_connectivity)
    external_pixels = int(np.count_nonzero(external_mask))
    external_contour_preservation = _safe_ratio(
        int(np.count_nonzero(external_mask & dilated_rendered)), external_pixels, empty_value=1.0
    )

    source_bbox = _bbox(foreground)
    rendered_bbox = _bbox(rendered)
    contour_bbox_coverage = _bbox_coverage(source_bbox, rendered_bbox)

    metrics_obj = centerline_result.metrics if centerline_result is not None else None
    endpoint_count = int(metrics_obj.endpoint_count) if metrics_obj is not None else 0
    junction_count = int(metrics_obj.junction_count) if metrics_obj is not None else 0
    path_count = len(paths)
    average_path_length = float(np.mean([path.length for path in paths])) if paths else 0.0
    # A junction never adds endpoints, so legitimate drawings keep roughly two
    # endpoints per connected component (closed loops contribute zero). Extra
    # endpoints beyond that budget indicate the centerline was cut, which is
    # the signal Issue 11.6 needs to tell real junction topology apart from a
    # fragmented trace.
    expected_endpoints = 2 * components_before
    excess_endpoints = max(0, endpoint_count - expected_endpoints)
    broken_path_count = excess_endpoints // 2
    skeleton_fragmentation = _safe_ratio(excess_endpoints, max(expected_endpoints, 1))

    return LineArtContinuityMetrics(
        components_before=components_before,
        components_after=components_after,
        lost_component_count=lost_component_count,
        endpoint_count=endpoint_count,
        junction_count=junction_count,
        path_count=path_count,
        broken_path_count=broken_path_count,
        average_path_length=average_path_length,
        contour_coverage=contour_coverage,
        edge_recall=edge_recall,
        foreground_recall=foreground_recall,
        skeleton_fragmentation=skeleton_fragmentation,
        contour_bbox_coverage=contour_bbox_coverage,
        external_contour_preservation=external_contour_preservation,
    )


def decide_lineart_outline_recovery(
    metrics: LineArtContinuityMetrics,
    *,
    min_edge_recall: float,
    min_foreground_recall: float,
    min_contour_coverage: float,
    max_fragmentation_ratio: float,
    preserve_external_contour: bool,
    trigger_when_centerline_fails: bool = True,
) -> LineArtContinuityDecision:
    """Decide whether outline-stroke recovery should run for this component set."""
    flags: list[str] = []
    if metrics.edge_recall < min_edge_recall:
        flags.append("low_edge_recall")
    if metrics.foreground_recall < min_foreground_recall:
        flags.append("underdrawn_lineart")
    if metrics.contour_coverage < min_contour_coverage:
        flags.append("lost_contour_structure")
    if metrics.skeleton_fragmentation > max_fragmentation_ratio:
        flags.append("excessive_line_fragmentation")
    if preserve_external_contour and metrics.external_contour_preservation < min_contour_coverage:
        flags.append("lost_external_contour")
    if metrics.lost_component_count > 0:
        flags.append("lost_contour_structure")
    flags = list(dict.fromkeys(flags))
    needs_recovery = trigger_when_centerline_fails and bool(flags)
    return LineArtContinuityDecision(needs_outline_recovery=needs_recovery, flags=tuple(flags), metrics=metrics)


def extract_outline_strokes(
    mask: np.ndarray,
    rendered_thin_mask: np.ndarray,
    *,
    stroke_width: float,
    simplification_tolerance: float = 0.01,
    minimum_recall_to_skip: float = 0.6,
    max_components: int = 24,
    component_connectivity: int = 8,
) -> OutlineRecoveryResult:
    """Recover a conservative stroke-only outline for weakly-covered components.

    Recovered shapes never receive a fill, so this fallback cannot introduce
    artificial black mass or white cutouts; it only restores a visible
    boundary for components the centerline lost or fragmented.
    """
    foreground = _normalize_mask(mask)
    rendered = np.asarray(rendered_thin_mask, dtype=bool)
    labels_count, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=component_connectivity)
    primitives: list[PolylinePrimitive] = []
    warnings: list[str] = []
    recovered = 0
    skipped = 0
    for component_id in range(1, int(labels_count)):
        component_mask = labels == component_id
        component_pixels = int(np.count_nonzero(component_mask))
        if component_pixels == 0:
            continue
        recall = _safe_ratio(int(np.count_nonzero(component_mask & rendered)), component_pixels, empty_value=1.0)
        if recall >= minimum_recall_to_skip:
            skipped += 1
            continue
        if recovered >= max_components:
            warnings.append("outline_recovery_max_components_reached")
            break
        contour = _largest_external_contour(component_mask)
        if contour is None:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.5, perimeter * simplification_tolerance)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = tuple(Point2D(float(x), float(y)) for x, y in approx.reshape(-1, 2))
        if len(points) < 3:
            continue
        primitives.append(
            PolylinePrimitive(
                points,
                closed=True,
                stroke=StrokeStyle(width=stroke_width),
                fill=None,
                metadata={
                    "source_layer": "outline_stroke",
                    "strategy": "lineart_outline_recovery",
                    "component_id": int(component_id),
                    "covered_recall": round(recall, 6),
                },
            )
        )
        recovered += 1
    payload = {
        "primitives": [primitive.to_dict() for primitive in primitives],
        "recovered": recovered,
        "skipped": skipped,
        "warnings": warnings,
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    return OutlineRecoveryResult(
        primitives=tuple(primitives),
        recovered_component_count=recovered,
        skipped_component_count=skipped,
        warnings=tuple(warnings),
        deterministic_hash=digest,
    )


def render_polyline_primitives_mask(
    primitives: Iterable[Any],
    shape: tuple[int, int],
) -> np.ndarray:
    """Render polyline/line stroke primitives onto a boolean mask for diagnostics."""
    width, height = shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    for primitive in _flatten_primitives(primitives):
        points = getattr(primitive, "points", None)
        if points is None:
            start = getattr(primitive, "start", None)
            end = getattr(primitive, "end", None)
            if start is None or end is None:
                continue
            points = (start, end)
        pts = np.array([(round(point.x), round(point.y)) for point in points], dtype=np.int32)
        if len(pts) < 2:
            continue
        closed = bool(getattr(primitive, "closed", False))
        cv2.polylines(canvas, [pts], closed, 1, thickness=1)
    return canvas.astype(bool)


def compute_lineart_fill_metrics(primitives: Iterable[Any], image_area: float) -> LineArtFillMetrics:
    """Measure black-fill and white-cutout area ratios for a line-art candidate."""
    filled_region_count = 0
    white_cutout_count = 0
    black_area = 0.0
    white_area = 0.0
    for primitive in _flatten_primitives(primitives):
        fill = getattr(primitive, "fill", None)
        points = getattr(primitive, "points", None)
        if fill is None or points is None:
            continue
        area = _polygon_area(points)
        color = fill.color
        is_white = color.red > 200 and color.green > 200 and color.blue > 200
        if is_white:
            white_area += area
            white_cutout_count += 1
        else:
            black_area += area
            filled_region_count += 1
    safe_area = max(image_area, 1.0)
    return LineArtFillMetrics(
        filled_area_ratio=_clamp_non_negative(black_area / safe_area),
        white_cutout_ratio=_clamp_non_negative(white_area / safe_area),
        filled_region_count=filled_region_count,
        white_cutout_count=white_cutout_count,
    )


def validate_lineart_balance(
    mask: np.ndarray,
    primitives: Iterable[Any],
    continuity: LineArtContinuityMetrics,
    *,
    max_filled_area_ratio_for_lineart: float,
    max_white_cutout_ratio_for_lineart: float,
    lineart_min_edge_recall: float,
    lineart_min_foreground_recall: float,
    lineart_min_contour_coverage: float,
    lineart_max_fragmentation_ratio: float,
    lineart_preserve_external_contour: bool,
    reject_overfilled_lineart: bool,
    reject_underdrawn_lineart: bool,
) -> LineArtBalanceResult:
    """Validate a Classic line-art candidate against overfilled/underdrawn failure modes."""
    foreground = _normalize_mask(mask)
    fill_metrics = compute_lineart_fill_metrics(primitives, float(foreground.size))

    flags: list[str] = []
    if fill_metrics.filled_area_ratio > max_filled_area_ratio_for_lineart:
        flags.extend(("excessive_filled_area", "artificial_black_mass", "overfilled_lineart"))
        if fill_metrics.filled_area_ratio > max_filled_area_ratio_for_lineart * 2.0:
            flags.append("lineart_converted_to_silhouette")
    if fill_metrics.white_cutout_ratio > max_white_cutout_ratio_for_lineart:
        flags.append("excessive_white_cutouts")
    if continuity.foreground_recall < lineart_min_foreground_recall:
        flags.append("underdrawn_lineart")
    if continuity.contour_coverage < lineart_min_contour_coverage:
        flags.append("lost_contour_structure")
    if continuity.edge_recall < lineart_min_edge_recall:
        flags.append("low_edge_recall")
    if continuity.skeleton_fragmentation > lineart_max_fragmentation_ratio:
        flags.append("excessive_line_fragmentation")
    if lineart_preserve_external_contour and continuity.external_contour_preservation < lineart_min_contour_coverage:
        flags.append("lost_external_contour")
    if continuity.components_before > 0 and (continuity.lost_component_count / continuity.components_before) > 0.3:
        flags.append("missing_internal_details")
    flags = list(dict.fromkeys(flags))

    reasons: list[str] = []
    flag_set = set(flags)
    if reject_overfilled_lineart and (_OVERFILL_FLAGS & flag_set):
        reasons.append("overfilled_lineart_rejected")
    if reject_underdrawn_lineart and (_UNDERDRAW_FLAGS & flag_set):
        reasons.append("underdrawn_lineart_rejected")

    return LineArtBalanceResult(
        accepted=not reasons,
        flags=tuple(flags),
        rejection_reasons=tuple(reasons),
        fill_metrics=fill_metrics,
        continuity_metrics=continuity,
    )


def _render_paths_mask(paths: tuple[CenterlinePath, ...], shape: tuple[int, int]) -> np.ndarray:
    width, height = shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        pts = np.array([(round(point.x), round(point.y)) for point in path.points], dtype=np.int32)
        if len(pts) < 2:
            if len(pts) == 1:
                x, y = int(pts[0][0]), int(pts[0][1])
                if 0 <= y < height and 0 <= x < width:
                    canvas[y, x] = 1
            continue
        closed = path.closure is PathClosureType.CLOSED
        cv2.polylines(canvas, [pts], closed, 1, thickness=1)
    return canvas.astype(bool)


def _flatten_primitives(primitives: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    items = primitives if isinstance(primitives, (list, tuple)) else list(primitives)
    for item in items:
        if isinstance(item, PrimitiveGroup):
            output.extend(_flatten_primitives(item.items))
        else:
            output.append(item)
    return output


def _largest_external_contour(component_mask: np.ndarray) -> np.ndarray | None:
    contours, _hierarchy = cv2.findContours(
        (component_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _external_contour_mask(foreground: np.ndarray, connectivity: int) -> np.ndarray:
    if not np.any(foreground):
        return np.zeros_like(foreground, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel, iterations=1).astype(bool)
    return foreground & ~eroded


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return cv2.Canny((mask.astype(np.uint8) * 255), 40, 80) > 0


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    if not np.any(mask):
        return 0
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return int(max(0, count - 1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_coverage(source: tuple[int, int, int, int] | None, rendered: tuple[int, int, int, int] | None) -> float:
    if source is None:
        return 1.0
    if rendered is None:
        return 0.0
    sx0, sy0, sx1, sy1 = source
    rx0, ry0, rx1, ry1 = rendered
    inter_x0, inter_y0 = max(sx0, rx0), max(sy0, ry0)
    inter_x1, inter_y1 = min(sx1, rx1), min(sy1, ry1)
    inter_area = max(0, inter_x1 - inter_x0 + 1) * max(0, inter_y1 - inter_y0 + 1)
    source_area = max(1, (sx1 - sx0 + 1) * (sy1 - sy0 + 1))
    return _clamp_non_negative(min(1.0, inter_area / source_area))


def _polygon_area(points: Iterable[Point2D]) -> float:
    pts = [(point.x, point.y) for point in points]
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for index in range(len(pts)):
        x1, y1 = pts[index]
        x2, y2 = pts[(index + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise LineArtContinuityError("mask must not be empty.")
    if array.ndim != 2:
        raise LineArtContinuityError("mask must be 2D.")
    if not np.all(np.isfinite(array)):
        raise LineArtContinuityError("mask values must be finite.")
    return array > 0


def _safe_ratio(numerator: int | float, denominator: int | float, empty_value: float = 0.0) -> float:
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return float(empty_value)
    return float(numerator) / denominator_float


def _clamp_non_negative(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, float(value))


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


__all__ = [
    "LineArtBalanceResult",
    "LineArtContinuityDecision",
    "LineArtContinuityError",
    "LineArtContinuityMetrics",
    "LineArtFillMetrics",
    "OutlineRecoveryResult",
    "compute_lineart_continuity_metrics",
    "compute_lineart_fill_metrics",
    "decide_lineart_outline_recovery",
    "extract_outline_strokes",
    "render_polyline_primitives_mask",
    "validate_lineart_balance",
]
