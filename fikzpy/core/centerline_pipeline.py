"""Isolated centerline extraction pipeline for line drawings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from math import acos, degrees, hypot, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from fikzpy.core.adaptive_preprocessing import PreprocessingResult, preprocess_image
from fikzpy.core.diagnostics import log_event
from fikzpy.core.image_classifier import ImageCategory
from fikzpy.core.semantic_geometry import Point2D, PolylinePrimitive
from fikzpy.core.skeleton_graph import SkeletonEdge, SkeletonGraph, SkeletonNodeType
from fikzpy.core.skeleton_graph import build_skeleton_graph, component_pixel_counts


class PathClosureType(Enum):
    """Open or closed centerline path marker."""

    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True)
class CenterlineConfig:
    """Configuration for isolated centerline tracing."""

    connectivity: int = 8
    skeleton_method: str = "skimage_skeletonize"
    enable_spur_pruning: bool = True
    minimum_spur_length: int = 4
    relative_spur_length: float = 0.04
    preserve_small_details: bool = True
    preserve_cycles: bool = True
    merge_nearby_endpoints: bool = False
    maximum_endpoint_distance: float = 2.0
    maximum_merge_angle: float = 25.0
    minimum_path_length: int = 2
    simplify_pixel_paths: bool = True
    simplification_tolerance: float = 0.75
    use_sknw_if_available: bool = False
    strict_topology_validation: bool = False
    maximum_pruning_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8.")
        if str(self.skeleton_method) != "skimage_skeletonize":
            raise ValueError("skeleton_method must be 'skimage_skeletonize'.")
        _validate_bool("enable_spur_pruning", self.enable_spur_pruning)
        _validate_non_negative_int("minimum_spur_length", self.minimum_spur_length)
        _validate_ratio("relative_spur_length", self.relative_spur_length)
        _validate_bool("preserve_small_details", self.preserve_small_details)
        _validate_bool("preserve_cycles", self.preserve_cycles)
        _validate_bool("merge_nearby_endpoints", self.merge_nearby_endpoints)
        _validate_non_negative_float("maximum_endpoint_distance", self.maximum_endpoint_distance)
        _validate_non_negative_float("maximum_merge_angle", self.maximum_merge_angle)
        if float(self.maximum_merge_angle) > 180.0:
            raise ValueError("maximum_merge_angle must not exceed 180.")
        if int(self.minimum_path_length) < 1:
            raise ValueError("minimum_path_length must be positive.")
        _validate_bool("simplify_pixel_paths", self.simplify_pixel_paths)
        _validate_non_negative_float("simplification_tolerance", self.simplification_tolerance)
        _validate_bool("use_sknw_if_available", self.use_sknw_if_available)
        _validate_bool("strict_topology_validation", self.strict_topology_validation)
        _validate_ratio("maximum_pruning_ratio", self.maximum_pruning_ratio)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary."""
        return {
            "connectivity": self.connectivity,
            "skeleton_method": self.skeleton_method,
            "enable_spur_pruning": self.enable_spur_pruning,
            "minimum_spur_length": self.minimum_spur_length,
            "relative_spur_length": self.relative_spur_length,
            "preserve_small_details": self.preserve_small_details,
            "preserve_cycles": self.preserve_cycles,
            "merge_nearby_endpoints": self.merge_nearby_endpoints,
            "maximum_endpoint_distance": self.maximum_endpoint_distance,
            "maximum_merge_angle": self.maximum_merge_angle,
            "minimum_path_length": self.minimum_path_length,
            "simplify_pixel_paths": self.simplify_pixel_paths,
            "simplification_tolerance": self.simplification_tolerance,
            "use_sknw_if_available": self.use_sknw_if_available,
            "strict_topology_validation": self.strict_topology_validation,
            "maximum_pruning_ratio": self.maximum_pruning_ratio,
        }


@dataclass(frozen=True)
class CenterlinePath:
    """An ordered centerline trajectory extracted from a skeleton graph."""

    id: str
    points: tuple[Point2D, ...]
    start_node_id: str | None
    end_node_id: str | None
    closure: PathClosureType
    component_id: int
    is_cycle: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("path id must not be empty.")
        points = tuple(self.points)
        if not points:
            raise ValueError("path points must not be empty.")
        for index, point in enumerate(points):
            if not isinstance(point, Point2D):
                raise TypeError(f"points[{index}] must be a Point2D.")
        if not isinstance(self.closure, PathClosureType):
            raise TypeError("closure must be a PathClosureType.")
        if self.closure is PathClosureType.CLOSED and len(points) < 3:
            raise ValueError("closed paths require at least three points.")
        if int(self.component_id) < 0:
            raise ValueError("component_id must be non-negative.")
        if bool(self.is_cycle) and self.closure is not PathClosureType.CLOSED:
            raise ValueError("cycle paths must be closed.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def length(self) -> float:
        """Return Euclidean path length in pixel coordinates."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for start, end in zip(self.points, self.points[1:], strict=False):
            total += hypot(end.x - start.x, end.y - start.y)
        if self.closure is PathClosureType.CLOSED:
            total += hypot(self.points[0].x - self.points[-1].x, self.points[0].y - self.points[-1].y)
        return float(total)

    def to_polyline_primitive(self) -> PolylinePrimitive:
        """Convert this path explicitly to a semantic polyline primitive."""
        return PolylinePrimitive(
            points=self.points,
            closed=self.closure is PathClosureType.CLOSED,
            metadata={
                "source": "centerline_path",
                "path_id": self.id,
                "component_id": self.component_id,
                "is_cycle": self.is_cycle,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without generating an output document."""
        return {
            "id": self.id,
            "point_count": len(self.points),
            "first_point": self.points[0].to_dict(),
            "last_point": self.points[-1].to_dict(),
            "length": self.length,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "closure": self.closure.value,
            "component_id": self.component_id,
            "is_cycle": self.is_cycle,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CenterlineMetrics:
    """Scalar diagnostics for centerline extraction."""

    skeleton_pixel_count: int
    connected_component_count: int
    node_count: int
    endpoint_count: int
    junction_count: int
    isolated_pixel_count: int
    edge_count: int
    open_path_count: int
    closed_path_count: int
    cycle_count: int
    pixels_before_pruning: int
    pixels_after_pruning: int
    spurs_removed: int
    removed_length: int
    paths_before_merging: int
    paths_after_merging: int
    points_before_simplification: int
    points_after_simplification: int

    def to_dict(self) -> dict[str, int]:
        """Return a serializable diagnostic dictionary."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class CenterlineResult:
    """Structured output of centerline extraction."""

    skeleton: np.ndarray
    graph: SkeletonGraph
    paths: tuple[CenterlinePath, ...]
    metrics: CenterlineMetrics
    warnings: tuple[str, ...]
    config: CenterlineConfig
    preprocessing_summary: dict[str, Any] | None = None
    input_shape: tuple[int, int] | None = None
    selected_threshold: float | None = None
    selected_threshold_method: str | None = None
    skeletonization_backend: str = "skimage_skeletonize"
    graph_backend: str = "fallback"
    sknw_used: bool = False

    def __post_init__(self) -> None:
        skeleton = np.asarray(self.skeleton)
        if skeleton.ndim != 2:
            raise ValueError("skeleton must be 2D.")
        object.__setattr__(self, "skeleton", skeleton.astype(bool, copy=True))
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        if self.input_shape is not None and len(self.input_shape) != 2:
            raise ValueError("input_shape must be a 2D shape.")

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing full arrays."""
        return {
            "skeleton": _array_summary(self.skeleton),
            "graph": self.graph.to_dict(),
            "paths": [path.to_dict() for path in self.paths],
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
            "config": self.config.to_dict(),
            "preprocessing_summary": self.preprocessing_summary,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "selected_threshold": self.selected_threshold,
            "selected_threshold_method": self.selected_threshold_method,
            "skeletonization_backend": self.skeletonization_backend,
            "graph_backend": self.graph_backend,
            "sknw_used": self.sknw_used,
        }


def extract_centerlines(
    image_or_preprocessing_result: str | Path | np.ndarray | PreprocessingResult | object,
    config: CenterlineConfig | None = None,
) -> CenterlineResult:
    """Extract ordered centerline paths from a mask or line-art image."""
    effective_config = config or CenterlineConfig()
    mask, preprocessing_result = _input_to_binary_mask(image_or_preprocessing_result)
    input_shape = tuple(int(value) for value in mask.shape)
    skeleton, skeleton_backend = _skeletonize_mask_with_backend(mask, method=effective_config.skeleton_method)
    pixels_before = int(np.count_nonzero(skeleton))

    initial_graph = build_skeleton_graph(
        skeleton,
        connectivity=effective_config.connectivity,
        use_sknw_if_available=effective_config.use_sknw_if_available,
        strict=effective_config.strict_topology_validation,
    )
    pruned_skeleton, spurs_removed, removed_length, prune_warnings = _prune_spurs(
        skeleton,
        initial_graph,
        effective_config,
    )
    graph = build_skeleton_graph(
        pruned_skeleton,
        connectivity=effective_config.connectivity,
        use_sknw_if_available=effective_config.use_sknw_if_available,
        strict=effective_config.strict_topology_validation,
    )
    paths_before_merge = _paths_from_graph(graph, effective_config, pruned_skeleton)
    merged_paths = _merge_paths(paths_before_merge, effective_config)
    points_before_simplification = sum(len(path.points) for path in merged_paths)
    simplified_paths = _simplify_paths(merged_paths, effective_config)
    paths = _renumber_paths(simplified_paths)
    points_after_simplification = sum(len(path.points) for path in paths)

    metrics = _metrics_from_result(
        graph=graph,
        paths=paths,
        skeleton_pixels=int(np.count_nonzero(pruned_skeleton)),
        pixels_before_pruning=pixels_before,
        pixels_after_pruning=int(np.count_nonzero(pruned_skeleton)),
        spurs_removed=spurs_removed,
        removed_length=removed_length,
        paths_before_merging=len(paths_before_merge),
        paths_after_merging=len(merged_paths),
        points_before_simplification=points_before_simplification,
        points_after_simplification=points_after_simplification,
    )
    warnings = tuple(
        warning
        for warning in (
            *initial_graph.warnings,
            *graph.warnings,
            *prune_warnings,
            *_special_case_warnings(mask, pruned_skeleton, paths),
        )
    )
    preprocessing_summary = preprocessing_result.to_dict() if preprocessing_result is not None else None

    log_event("Centerline", f"method={skeleton_backend}")
    log_event("Centerline", f"graph_backend={graph.backend}")
    log_event("Centerline", f"skeleton_pixels={metrics.skeleton_pixel_count}")
    log_event("Centerline", f"components={metrics.connected_component_count}")
    log_event("Centerline", f"endpoints={metrics.endpoint_count}")
    log_event("Centerline", f"junctions={metrics.junction_count}")
    log_event("Centerline", f"cycles={metrics.cycle_count}")
    log_event("Centerline", f"spurs_removed={metrics.spurs_removed}")
    log_event("Centerline", f"paths={len(paths)}")
    log_event("Centerline", f"points_before={metrics.points_before_simplification}")
    log_event("Centerline", f"points_after={metrics.points_after_simplification}")

    return CenterlineResult(
        skeleton=pruned_skeleton,
        graph=graph,
        paths=paths,
        metrics=metrics,
        warnings=warnings,
        config=effective_config,
        preprocessing_summary=preprocessing_summary,
        input_shape=input_shape,
        selected_threshold=preprocessing_result.threshold if preprocessing_result is not None else None,
        selected_threshold_method=preprocessing_result.method if preprocessing_result is not None else None,
        skeletonization_backend=skeleton_backend,
        graph_backend=graph.backend,
        sknw_used=graph.backend == "sknw",
    )


def skeletonize_mask(mask: np.ndarray, *, method: str = "skimage_skeletonize") -> np.ndarray:
    """Reduce a binary foreground mask to a deterministic one-pixel skeleton."""
    skeleton, _backend = _skeletonize_mask_with_backend(mask, method=method)
    return skeleton


def _skeletonize_mask_with_backend(mask: np.ndarray, *, method: str = "skimage_skeletonize") -> tuple[np.ndarray, str]:
    if method != "skimage_skeletonize":
        raise ValueError("method must be 'skimage_skeletonize'.")
    binary = _normalize_binary_mask(mask)
    try:
        from skimage.morphology import skeletonize as skimage_skeletonize
    except Exception:
        return _skeletonize_zhang_suen(binary), "zhang_suen_fallback"
    return skimage_skeletonize(binary).astype(bool, copy=False), "skimage_skeletonize"


def centerline_paths_to_polylines(
    result_or_paths: CenterlineResult | tuple[CenterlinePath, ...] | list[CenterlinePath],
) -> tuple[PolylinePrimitive, ...]:
    """Explicitly convert centerline paths to semantic polyline primitives."""
    paths = result_or_paths.paths if isinstance(result_or_paths, CenterlineResult) else tuple(result_or_paths)
    return tuple(path.to_polyline_primitive() for path in paths)


def _input_to_binary_mask(
    image_or_preprocessing_result: str | Path | np.ndarray | PreprocessingResult | object,
) -> tuple[np.ndarray, PreprocessingResult | None]:
    if isinstance(image_or_preprocessing_result, PreprocessingResult):
        return _normalize_binary_mask(image_or_preprocessing_result.binary_mask), image_or_preprocessing_result
    if isinstance(image_or_preprocessing_result, np.ndarray) and _looks_like_binary_mask(image_or_preprocessing_result):
        return _normalize_binary_mask(image_or_preprocessing_result), None
    preprocessing_result = preprocess_image(image_or_preprocessing_result, category=ImageCategory.LINE_ART)
    return _normalize_binary_mask(preprocessing_result.binary_mask), preprocessing_result


def _looks_like_binary_mask(array: np.ndarray) -> bool:
    values = np.asarray(array)
    if values.ndim != 2 or values.size == 0:
        return False
    if not np.all(np.isfinite(values)):
        return False
    unique = np.unique(values)
    return all(float(value) in {0.0, 1.0, 255.0} for value in unique)


def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("mask must not be empty.")
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    if not np.all(np.isfinite(array)):
        raise ValueError("mask values must be finite.")
    return array > 0


def _skeletonize_zhang_suen(binary_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(binary_mask).astype(np.uint8, copy=True)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = _zhang_suen_candidates(image, step)
            if np.any(to_remove):
                image[to_remove] = 0
                changed = True
    return image.astype(bool, copy=False)


def _zhang_suen_candidates(image: np.ndarray, step: int) -> np.ndarray:
    padded = np.pad(image, 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
    neighbor_count = sum(neighbors)
    transitions = sum((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1) for index in range(8))
    if step == 0:
        condition_a = (p2 * p4 * p6) == 0
        condition_b = (p4 * p6 * p8) == 0
    else:
        condition_a = (p2 * p4 * p8) == 0
        condition_b = (p2 * p6 * p8) == 0
    return (image == 1) & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & condition_a & condition_b


def _paths_from_graph(
    graph: SkeletonGraph,
    config: CenterlineConfig,
    skeleton: np.ndarray | None = None,
) -> tuple[CenterlinePath, ...]:
    paths: list[CenterlinePath] = []
    minimum_cycle_pixels = max(6, config.minimum_path_length * 3)
    cycle_components = (
        _cycle_core_components(skeleton, config.connectivity, minimum_cycle_pixels) if skeleton is not None else ()
    )
    cycle_pixels = {pixel for component in cycle_components for pixel in component}
    for edge in graph.edges:
        if edge.pixels and all(pixel in cycle_pixels for pixel in edge.pixels):
            continue
        if not edge.is_cycle and edge.pixel_length < config.minimum_path_length:
            continue
        if edge.is_cycle and not config.preserve_cycles:
            continue
        points = tuple(_pixel_to_point(pixel) for pixel in edge.pixels)
        if len(points) < 2 and not edge.is_cycle:
            continue
        paths.append(_path_from_edge(edge, points))
    if config.preserve_cycles:
        for index, component in enumerate(cycle_components):
            ordered_pixels = _walk_closed_pixels(component, config.connectivity)
            if len(ordered_pixels) < 3:
                continue
            paths.append(
                CenterlinePath(
                    id=f"cycle{index:04d}",
                    points=tuple(_pixel_to_point(pixel) for pixel in ordered_pixels),
                    start_node_id=None,
                    end_node_id=None,
                    closure=PathClosureType.CLOSED,
                    component_id=_component_id_for_cycle_pixels(graph, ordered_pixels),
                    is_cycle=True,
                    metadata={"source": "cycle_core"},
                )
            )
    return _renumber_paths(paths)


def _path_from_edge(edge: SkeletonEdge, points: tuple[Point2D, ...]) -> CenterlinePath:
    closure = PathClosureType.CLOSED if edge.is_cycle else PathClosureType.OPEN
    return CenterlinePath(
        id=edge.id.replace("e", "p", 1),
        points=points,
        start_node_id=edge.start_node_id,
        end_node_id=edge.end_node_id,
        closure=closure,
        component_id=edge.component_id,
        is_cycle=edge.is_cycle,
        metadata={"source_edge_id": edge.id, **edge.metadata},
    )


def _pixel_to_point(pixel: tuple[int, int]) -> Point2D:
    row, col = pixel
    return Point2D(float(col), float(row))


def _prune_spurs(
    skeleton: np.ndarray,
    graph: SkeletonGraph,
    config: CenterlineConfig,
) -> tuple[np.ndarray, int, int, tuple[str, ...]]:
    pruned = np.asarray(skeleton).astype(bool, copy=True)
    if not config.enable_spur_pruning or config.preserve_small_details:
        return pruned, 0, 0, ()
    total_pixels = int(np.count_nonzero(pruned))
    if total_pixels == 0:
        return pruned, 0, 0, ()

    node_by_id = graph.node_by_id()
    component_counts = component_pixel_counts(pruned, config.connectivity)
    candidates = []
    for edge in graph.edges:
        if edge.is_cycle:
            continue
        start = node_by_id.get(edge.start_node_id)
        end = node_by_id.get(edge.end_node_id)
        if start is None or end is None:
            continue
        node_types = {start.node_type, end.node_type}
        if node_types != {SkeletonNodeType.ENDPOINT, SkeletonNodeType.JUNCTION}:
            continue
        component_pixels = component_counts.get(edge.component_id, total_pixels)
        threshold = max(float(config.minimum_spur_length), config.relative_spur_length * float(component_pixels))
        if edge.pixel_length <= threshold:
            candidates.append((edge.pixel_length, edge.id, edge))

    removed = 0
    removed_length = 0
    pixel_budget = int(np.floor(total_pixels * config.maximum_pruning_ratio))
    if pixel_budget <= 0 and candidates and config.maximum_pruning_ratio > 0.0:
        pixel_budget = 1
    warnings: list[str] = []
    for _length, _edge_id, edge in sorted(candidates):
        removable = _removable_spur_pixels(edge, node_by_id)
        if not removable:
            continue
        if removed_length + len(removable) > pixel_budget:
            warnings.append("maximum pruning ratio reached")
            break
        for row, col in removable:
            pruned[row, col] = False
        removed += 1
        removed_length += len(removable)
    return pruned, removed, removed_length, tuple(warnings)


def _removable_spur_pixels(edge: SkeletonEdge, node_by_id: dict[str, Any]) -> list[tuple[int, int]]:
    start = node_by_id[edge.start_node_id]
    end = node_by_id[edge.end_node_id]
    junction_pixel = start.pixel if start.node_type is SkeletonNodeType.JUNCTION else end.pixel
    return [pixel for pixel in edge.pixels if pixel != junction_pixel]


def _merge_paths(paths: tuple[CenterlinePath, ...], config: CenterlineConfig) -> tuple[CenterlinePath, ...]:
    if not config.merge_nearby_endpoints:
        return paths
    remaining = list(paths)
    merged: list[CenterlinePath] = []
    while remaining:
        base = remaining.pop(0)
        best_index = None
        best_pair = None
        best_distance = float("inf")
        for index, candidate in enumerate(remaining):
            pair = _compatible_endpoint_pair(base, candidate, config)
            if pair is None:
                continue
            distance = _point_distance(pair[0]["point"], pair[1]["point"])
            if distance < best_distance:
                best_distance = distance
                best_index = index
                best_pair = pair
        if best_index is None or best_pair is None:
            merged.append(base)
            continue
        other = remaining.pop(best_index)
        remaining.insert(0, _combine_paths(base, other, best_pair))
    return _renumber_paths(merged)


def _compatible_endpoint_pair(
    first: CenterlinePath,
    second: CenterlinePath,
    config: CenterlineConfig,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if first.closure is PathClosureType.CLOSED or second.closure is PathClosureType.CLOSED:
        return None
    if len(first.points) < 2 or len(second.points) < 2:
        return None
    candidates = []
    for first_side in ("start", "end"):
        first_endpoint = _endpoint_info(first, first_side)
        for second_side in ("start", "end"):
            second_endpoint = _endpoint_info(second, second_side)
            distance = _point_distance(first_endpoint["point"], second_endpoint["point"])
            if distance > config.maximum_endpoint_distance or distance == 0.0:
                continue
            if _endpoint_angles_compatible(first_endpoint, second_endpoint, config):
                candidates.append((distance, first_side, second_side, first_endpoint, second_endpoint))
    if not candidates:
        return None
    _distance, _first_side, _second_side, first_endpoint, second_endpoint = sorted(candidates, key=lambda item: item[:3])[0]
    return first_endpoint, second_endpoint


def _endpoint_info(path: CenterlinePath, side: str) -> dict[str, Any]:
    points = path.points
    if side == "start":
        endpoint = points[0]
        reference = points[min(2, len(points) - 1)]
        tangent = (endpoint.x - reference.x, endpoint.y - reference.y)
    else:
        endpoint = points[-1]
        reference = points[max(0, len(points) - 3)]
        tangent = (endpoint.x - reference.x, endpoint.y - reference.y)
    return {"path": path, "side": side, "point": endpoint, "tangent": tangent}


def _endpoint_angles_compatible(first: dict[str, Any], second: dict[str, Any], config: CenterlineConfig) -> bool:
    first_point = first["point"]
    second_point = second["point"]
    connection = (second_point.x - first_point.x, second_point.y - first_point.y)
    reverse_connection = (-connection[0], -connection[1])
    return (
        _angle_between(first["tangent"], connection) <= config.maximum_merge_angle
        and _angle_between(second["tangent"], reverse_connection) <= config.maximum_merge_angle
    )


def _combine_paths(
    first: CenterlinePath,
    second: CenterlinePath,
    pair: tuple[dict[str, Any], dict[str, Any]],
) -> CenterlinePath:
    first_endpoint, second_endpoint = pair
    first_points = first.points if first_endpoint["side"] == "end" else tuple(reversed(first.points))
    second_points = second.points if second_endpoint["side"] == "start" else tuple(reversed(second.points))
    combined_points = _deduplicate_adjacent_points((*first_points, *second_points))
    return CenterlinePath(
        id=first.id,
        points=combined_points,
        start_node_id=first.start_node_id if first_endpoint["side"] == "end" else first.end_node_id,
        end_node_id=second.end_node_id if second_endpoint["side"] == "start" else second.start_node_id,
        closure=PathClosureType.OPEN,
        component_id=min(first.component_id, second.component_id),
        is_cycle=False,
        metadata={
            "merged_from": [first.id, second.id],
            "merge_distance": _point_distance(first_endpoint["point"], second_endpoint["point"]),
        },
    )


def _simplify_paths(paths: tuple[CenterlinePath, ...], config: CenterlineConfig) -> tuple[CenterlinePath, ...]:
    if not config.simplify_pixel_paths or config.simplification_tolerance <= 0.0:
        return paths
    simplified = []
    for path in paths:
        if len(path.points) <= 2:
            simplified.append(path)
            continue
        points = _simplify_points(path.points, config.simplification_tolerance, path.closure is PathClosureType.CLOSED)
        simplified.append(replace(path, points=points, metadata={**path.metadata, "simplified": True}))
    return tuple(simplified)


def _simplify_points(points: tuple[Point2D, ...], tolerance: float, closed: bool) -> tuple[Point2D, ...]:
    if closed:
        if len(points) <= 3:
            return points
        working = (*points, points[0])
        simplified = _rdp(working, tolerance)
        if simplified and simplified[-1] == simplified[0]:
            simplified = simplified[:-1]
        return tuple(simplified if len(simplified) >= 3 else points)
    return tuple(_rdp(points, tolerance))


def _rdp(points: tuple[Point2D, ...], tolerance: float) -> tuple[Point2D, ...]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        distance = _distance_to_segment(point, start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = index
    if max_distance <= tolerance:
        return (start, end)
    left = _rdp(points[: split_index + 1], tolerance)
    right = _rdp(points[split_index:], tolerance)
    return (*left[:-1], *right)


def _renumber_paths(paths: tuple[CenterlinePath, ...] | list[CenterlinePath]) -> tuple[CenterlinePath, ...]:
    ordered = sorted(
        tuple(paths),
        key=lambda path: (
            path.component_id,
            path.closure.value,
            min((point.y, point.x) for point in path.points),
            path.length,
            path.id,
        ),
    )
    return tuple(replace(path, id=f"p{index:04d}") for index, path in enumerate(ordered))


def _metrics_from_result(
    *,
    graph: SkeletonGraph,
    paths: tuple[CenterlinePath, ...],
    skeleton_pixels: int,
    pixels_before_pruning: int,
    pixels_after_pruning: int,
    spurs_removed: int,
    removed_length: int,
    paths_before_merging: int,
    paths_after_merging: int,
    points_before_simplification: int,
    points_after_simplification: int,
) -> CenterlineMetrics:
    return CenterlineMetrics(
        skeleton_pixel_count=skeleton_pixels,
        connected_component_count=graph.component_count,
        node_count=len(graph.nodes),
        endpoint_count=graph.endpoint_count,
        junction_count=graph.junction_count,
        isolated_pixel_count=graph.isolated_pixel_count,
        edge_count=len(graph.edges),
        open_path_count=sum(1 for path in paths if path.closure is PathClosureType.OPEN),
        closed_path_count=sum(1 for path in paths if path.closure is PathClosureType.CLOSED),
        cycle_count=max(graph.cycle_count, sum(1 for path in paths if path.closure is PathClosureType.CLOSED)),
        pixels_before_pruning=pixels_before_pruning,
        pixels_after_pruning=pixels_after_pruning,
        spurs_removed=spurs_removed,
        removed_length=removed_length,
        paths_before_merging=paths_before_merging,
        paths_after_merging=paths_after_merging,
        points_before_simplification=points_before_simplification,
        points_after_simplification=points_after_simplification,
    )


def _special_case_warnings(mask: np.ndarray, skeleton: np.ndarray, paths: tuple[CenterlinePath, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    foreground = int(np.count_nonzero(mask))
    skeleton_pixels = int(np.count_nonzero(skeleton))
    if foreground == 0:
        warnings.append("empty mask")
    if foreground == mask.size:
        warnings.append("full mask")
    if skeleton_pixels == 1:
        warnings.append("isolated skeleton pixel")
    if not paths and skeleton_pixels > 0:
        warnings.append("no extractable paths")
    if min(mask.shape) <= 2:
        warnings.append("very small mask")
    return tuple(warnings)


def _cycle_core_components(
    skeleton: np.ndarray | None,
    connectivity: int,
    minimum_cycle_pixels: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if skeleton is None:
        return ()
    binary = np.asarray(skeleton) > 0
    pixels = {tuple(map(int, pixel)) for pixel in zip(*np.nonzero(binary), strict=True)}
    if len(pixels) < 3:
        return ()
    neighbors = {pixel: [item for item in _neighbor_pixels(pixel, pixels, connectivity)] for pixel in pixels}
    degrees = {pixel: len(items) for pixel, items in neighbors.items()}
    queue = [pixel for pixel, degree in degrees.items() if degree <= 1]
    removed: set[tuple[int, int]] = set()
    while queue:
        pixel = queue.pop()
        if pixel in removed:
            continue
        removed.add(pixel)
        for neighbor in neighbors[pixel]:
            if neighbor in removed:
                continue
            degrees[neighbor] -= 1
            if degrees[neighbor] == 1:
                queue.append(neighbor)
    core = pixels - removed
    if len(core) < 3:
        return ()

    components: list[tuple[tuple[int, int], ...]] = []
    unseen = set(core)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[tuple[int, int]] = []
        while stack:
            pixel = stack.pop()
            component.append(pixel)
            for neighbor in _neighbor_pixels(pixel, core, connectivity):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        if len(component) >= minimum_cycle_pixels:
            components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: (min(item), len(item))))


def _walk_closed_pixels(component: tuple[tuple[int, int], ...], connectivity: int) -> tuple[tuple[int, int], ...]:
    pixels = set(component)
    start = min(pixels)
    neighbors = {pixel: list(_neighbor_pixels(pixel, pixels, connectivity)) for pixel in pixels}
    if not neighbors[start]:
        return (start,)
    ordered = [start]
    previous = start
    current = _initial_cycle_neighbor(start, neighbors[start])
    visited_links = {frozenset((previous, current))}
    guard = 0
    while current != start and guard <= len(pixels) + 2:
        ordered.append(current)
        candidates = [pixel for pixel in neighbors[current] if pixel != previous]
        if not candidates:
            break
        direction = (current[0] - previous[0], current[1] - previous[1])
        next_pixel = _best_cycle_neighbor(current, candidates, direction, visited_links, start)
        link = frozenset((current, next_pixel))
        if link in visited_links and next_pixel != start:
            break
        visited_links.add(link)
        previous, current = current, next_pixel
        guard += 1
    return tuple(ordered)


def _initial_cycle_neighbor(start: tuple[int, int], neighbors: list[tuple[int, int]]) -> tuple[int, int]:
    return sorted(neighbors, key=lambda pixel: (pixel[0], pixel[1]))[0]


def _best_cycle_neighbor(
    current: tuple[int, int],
    candidates: list[tuple[int, int]],
    direction: tuple[int, int],
    visited_links: set[frozenset[tuple[int, int]]],
    start: tuple[int, int],
) -> tuple[int, int]:
    def score(pixel: tuple[int, int]) -> tuple[int, int, int, tuple[int, int]]:
        vector = (pixel[0] - current[0], pixel[1] - current[1])
        dot = direction[0] * vector[0] + direction[1] * vector[1]
        link_seen = frozenset((current, pixel)) in visited_links and pixel != start
        return (1 if link_seen else 0, -dot, abs(vector[0]) + abs(vector[1]), pixel)

    return sorted(candidates, key=score)[0]


def _neighbor_pixels(
    pixel: tuple[int, int],
    pixels: set[tuple[int, int]],
    connectivity: int,
) -> tuple[tuple[int, int], ...]:
    row, col = pixel
    offsets = (
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        if connectivity == 4
        else (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    )
    return tuple(sorted((row + dy, col + dx) for dy, dx in offsets if (row + dy, col + dx) in pixels))


def _component_id_for_cycle_pixels(graph: SkeletonGraph, pixels: tuple[tuple[int, int], ...]) -> int:
    if not pixels:
        return 0
    pixel_set = set(pixels)
    for edge in graph.edges:
        if any(pixel in pixel_set for pixel in edge.pixels):
            return edge.component_id
    for node in graph.nodes:
        if node.pixel in pixel_set:
            return node.component_id
    return 0


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": int(values.min(initial=0)),
        "max": int(values.max(initial=0)),
        "count": int(np.count_nonzero(values)),
        "sha256": sha256(values.astype(np.uint8, copy=False).tobytes()).hexdigest(),
    }


def _point_distance(first: Point2D, second: Point2D) -> float:
    return hypot(second.x - first.x, second.y - first.y)


def _angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_length = hypot(first[0], first[1])
    second_length = hypot(second[0], second[1])
    if first_length == 0.0 or second_length == 0.0:
        return 180.0
    dot = first[0] * second[0] + first[1] * second[1]
    cosine = max(-1.0, min(1.0, dot / (first_length * second_length)))
    return degrees(acos(cosine))


def _distance_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return _point_distance(point, start)
    ratio = ((point.x - start.x) * dx + (point.y - start.y) * dy) / denominator
    ratio = max(0.0, min(1.0, ratio))
    projected = Point2D(start.x + ratio * dx, start.y + ratio * dy)
    return _point_distance(point, projected)


def _deduplicate_adjacent_points(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    if not points:
        return points
    deduplicated = [points[0]]
    for point in points[1:]:
        if point != deduplicated[-1]:
            deduplicated.append(point)
    return tuple(deduplicated)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_non_negative_float(name: str, value: float) -> None:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_ratio(name: str, value: float) -> None:
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


__all__ = [
    "CenterlineConfig",
    "CenterlineMetrics",
    "CenterlinePath",
    "CenterlineResult",
    "PathClosureType",
    "centerline_paths_to_polylines",
    "extract_centerlines",
    "skeletonize_mask",
]
