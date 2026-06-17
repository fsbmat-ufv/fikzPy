"""Simplify and merge fitted semantic geometry without changing app flows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import hypot, isfinite
from typing import Any

import numpy as np

from fikzpy.core.bezier_fitting import fit_cubic_beziers, sample_cubic_bezier
from fikzpy.core.diagnostics import log_event
from fikzpy.core.geometry_error import geometric_scale, max_error, normalize_error, path_length
from fikzpy.core.geometry_error import point_distances_to_line, points_to_array, rms_error, signed_area
from fikzpy.core.geometry_simplification import DeduplicationResult, deduplicate_consecutive_points
from fikzpy.core.geometry_simplification import is_degenerate_point_set, line_like_endpoints
from fikzpy.core.geometry_simplification import point_distance, polygon_has_self_intersection
from fikzpy.core.geometry_simplification import simplify_polyline_preserving_features
from fikzpy.core.path_merging import MergeOutcome, PathJoinKind, duplicate_primitives, try_join_open_paths
from fikzpy.core.path_merging import try_merge_collinear_lines
from fikzpy.core.primitive_fitting import PrimitiveFitResult
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D, PointPrimitive
from fikzpy.core.semantic_geometry import PolylinePrimitive, Primitive, PrimitiveGroup, RGBColor
from fikzpy.core.semantic_geometry import SemanticGeometry, StrokeStyle


_EPSILON = 1e-12


class GeometryOptimizationError(ValueError):
    """Raised when semantic geometry cannot be optimized safely."""


class OptimizationStatus(Enum):
    """Overall optimization status."""

    UNCHANGED = "unchanged"
    OPTIMIZED = "optimized"
    FAILED = "failed"


class OptimizationOperationKind(Enum):
    """Kinds of operations recorded by the optimizer."""

    NORMALIZE = "normalize"
    REMOVE_DUPLICATE_POINT = "remove_duplicate_point"
    REMOVE_DEGENERATE = "remove_degenerate"
    CONVERT_DEGENERATE = "convert_degenerate"
    CONVERT_TO_LINE = "convert_to_line"
    MERGE_COLLINEAR_LINES = "merge_collinear_lines"
    MERGE_POLYLINES = "merge_polylines"
    SIMPLIFY_POLYLINE = "simplify_polyline"
    MERGE_BEZIERS = "merge_beziers"
    REMOVE_DUPLICATE_PRIMITIVE = "remove_duplicate_primitive"
    VALIDATE = "validate"


class DuplicateHandling(Enum):
    """Configured behavior for degenerate primitives."""

    REMOVE = "remove"
    CONVERT_TO_POINT = "convert_to_point"
    PRESERVE_WITH_WARNING = "preserve_with_warning"
    ERROR = "error"


@dataclass(frozen=True)
class GeometryOptimizationConfig:
    """Centralized parameters for Issue 8 semantic optimization."""

    remove_duplicate_points: bool = True
    duplicate_point_tolerance: float = 1e-9
    remove_degenerate_primitives: bool = True
    minimum_primitive_length: float = 1e-6
    degenerate_handling: DuplicateHandling | str = DuplicateHandling.CONVERT_TO_POINT
    merge_collinear_lines: bool = True
    collinear_angle_tolerance: float = 2.0
    collinear_distance_tolerance: float = 1e-6
    merge_adjacent_polylines: bool = True
    maximum_endpoint_distance: float = 1e-6
    maximum_join_angle: float = 8.0
    simplify_polylines: bool = True
    polyline_tolerance: float = 0.005
    preserve_corners: bool = True
    corner_angle_threshold: float = 35.0
    simplify_closed_shapes: bool = True
    preserve_topology: bool = True
    merge_bezier_segments: bool = True
    bezier_merge_tolerance: float = 0.01
    bezier_tangent_tolerance: float = 8.0
    maximum_combined_bezier_error: float = 0.015
    remove_duplicate_primitives: bool = True
    duplicate_geometry_tolerance: float = 1e-9
    preserve_groups: bool = True
    preserve_draw_order: bool = True
    preserve_metadata: bool = True
    preserve_styles: bool = True
    allow_cross_group_merging: bool = False
    cumulative_error_budget: float = 0.02
    normalized_error_budget: float = 0.02
    decimal_precision: int = 6
    strict: bool = False
    maximum_optimization_passes: int = 4

    def __post_init__(self) -> None:
        for name in (
            "remove_duplicate_points",
            "remove_degenerate_primitives",
            "merge_collinear_lines",
            "merge_adjacent_polylines",
            "simplify_polylines",
            "preserve_corners",
            "simplify_closed_shapes",
            "preserve_topology",
            "merge_bezier_segments",
            "remove_duplicate_primitives",
            "preserve_groups",
            "preserve_draw_order",
            "preserve_metadata",
            "preserve_styles",
            "allow_cross_group_merging",
            "strict",
        ):
            _validate_bool(name, getattr(self, name))
        for name in (
            "duplicate_point_tolerance",
            "minimum_primitive_length",
            "collinear_angle_tolerance",
            "collinear_distance_tolerance",
            "maximum_endpoint_distance",
            "maximum_join_angle",
            "polyline_tolerance",
            "corner_angle_threshold",
            "bezier_merge_tolerance",
            "bezier_tangent_tolerance",
            "maximum_combined_bezier_error",
            "duplicate_geometry_tolerance",
            "cumulative_error_budget",
            "normalized_error_budget",
        ):
            _validate_non_negative_float(name, getattr(self, name))
        if float(self.collinear_angle_tolerance) > 180.0:
            raise ValueError("collinear_angle_tolerance must not exceed 180.")
        if float(self.maximum_join_angle) > 180.0:
            raise ValueError("maximum_join_angle must not exceed 180.")
        if float(self.bezier_tangent_tolerance) > 180.0:
            raise ValueError("bezier_tangent_tolerance must not exceed 180.")
        if int(self.decimal_precision) < 0:
            raise ValueError("decimal_precision must be non-negative.")
        if int(self.maximum_optimization_passes) < 1:
            raise ValueError("maximum_optimization_passes must be positive.")
        object.__setattr__(self, "decimal_precision", int(self.decimal_precision))
        object.__setattr__(self, "maximum_optimization_passes", int(self.maximum_optimization_passes))
        object.__setattr__(self, "degenerate_handling", _coerce_duplicate_handling(self.degenerate_handling))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        data = dict(self.__dict__)
        data["degenerate_handling"] = self.degenerate_handling.value
        return data


@dataclass(frozen=True)
class GeometryOptimizationWarning:
    """Structured warning emitted by semantic optimization."""

    code: str
    message: str
    primitive_index: int | None = None
    operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable warning diagnostics."""
        return {
            "code": self.code,
            "message": self.message,
            "primitive_index": self.primitive_index,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class OptimizationCandidate:
    """One considered optimization candidate."""

    kind: OptimizationOperationKind
    source_primitive_ids: tuple[str, ...]
    before_count: int
    after_count: int
    local_error: float
    rms_error: float
    normalized_error: float
    accepted: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return compact candidate diagnostics."""
        return {
            "kind": self.kind.value,
            "source_primitive_ids": list(self.source_primitive_ids),
            "before_count": self.before_count,
            "after_count": self.after_count,
            "local_error": _finite_or_none(self.local_error),
            "rms_error": _finite_or_none(self.rms_error),
            "normalized_error": _finite_or_none(self.normalized_error),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class OptimizationOperation:
    """A recorded optimization operation."""

    kind: OptimizationOperationKind
    source_primitive_ids: tuple[str, ...]
    output_primitive_ids: tuple[str, ...]
    before_count: int
    after_count: int
    local_error: float
    rms_error: float
    normalized_error: float
    accepted: bool
    reason: str
    pass_number: int

    def to_dict(self) -> dict[str, Any]:
        """Return compact operation diagnostics."""
        return {
            "kind": self.kind.value,
            "source_primitive_ids": list(self.source_primitive_ids),
            "output_primitive_ids": list(self.output_primitive_ids),
            "before_count": self.before_count,
            "after_count": self.after_count,
            "local_error": _finite_or_none(self.local_error),
            "rms_error": _finite_or_none(self.rms_error),
            "normalized_error": _finite_or_none(self.normalized_error),
            "accepted": self.accepted,
            "reason": self.reason,
            "pass_number": self.pass_number,
        }


@dataclass(frozen=True)
class PrimitiveSequence:
    """A normalized primitive sequence plus lightweight source diagnostics."""

    items: tuple[SemanticGeometry, ...]
    source_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return compact sequence diagnostics."""
        return {
            "source_type": self.source_type,
            "item_count": len(self.items),
            "metadata": _compact_mapping(self.metadata),
        }


@dataclass
class GeometryOptimizationMetrics:
    """Scalar diagnostics for the optimization pass."""

    input_primitive_count: int = 0
    output_primitive_count: int = 0
    primitive_reduction: int = 0
    primitive_reduction_ratio: float = 0.0
    input_point_count: int = 0
    output_point_count: int = 0
    point_reduction: int = 0
    point_reduction_ratio: float = 0.0
    duplicate_points_removed: int = 0
    degenerate_primitives_removed: int = 0
    degenerate_primitives_converted: int = 0
    collinear_lines_merged: int = 0
    polylines_merged: int = 0
    polylines_simplified: int = 0
    polyline_points_removed: int = 0
    bezier_sequences_examined: int = 0
    bezier_sequences_merged: int = 0
    input_bezier_count: int = 0
    output_bezier_count: int = 0
    duplicate_primitives_removed: int = 0
    groups_preserved: int = 0
    groups_removed: int = 0
    operations_applied: int = 0
    operations_rejected: int = 0
    maximum_error: float = 0.0
    rms_error: float = 0.0
    normalized_error: float = 0.0
    cumulative_error: float = 0.0
    topology_rejections: int = 0
    style_rejections: int = 0
    warnings_count: int = 0
    optimization_passes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Return serializable scalar diagnostics."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class GeometryOptimizationResult:
    """Selected optimized primitive sequence and diagnostics."""

    original_summary: Mapping[str, Any]
    primitives: tuple[SemanticGeometry, ...]
    metrics: GeometryOptimizationMetrics
    warnings: tuple[GeometryOptimizationWarning, ...]
    operations: tuple[OptimizationOperation, ...]
    configuration: GeometryOptimizationConfig
    deterministic_hash: str
    success: bool
    status: OptimizationStatus

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without large point arrays."""
        return {
            "original_summary": _compact_mapping(self.original_summary),
            "primitives": [_primitive_summary(primitive) for primitive in self.primitives],
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "operations": [operation.to_dict() for operation in self.operations],
            "configuration": self.configuration.to_dict(),
            "deterministic_hash": self.deterministic_hash,
            "success": self.success,
            "status": self.status.value,
        }


class GeometryOptimizer:
    """Optimize fitted semantic primitives with conservative local operations."""

    def __init__(self, config: GeometryOptimizationConfig | None = None) -> None:
        self.config = config or GeometryOptimizationConfig()
        self.metrics = GeometryOptimizationMetrics()
        self.warnings: list[GeometryOptimizationWarning] = []
        self.operations: list[OptimizationOperation] = []
        self._cumulative_error = 0.0
        self._global_scale = 1.0

    def optimize(self, primitives: Any) -> GeometryOptimizationResult:
        """Optimize semantic primitives without mutating the input object."""
        sequence = _normalize_input(primitives, self.config)
        original_items = _copy_items(sequence.items, self.config)
        original_summary = _sequence_summary(original_items, source_type=sequence.source_type, metadata=sequence.metadata)
        self.metrics.input_primitive_count = _object_count(original_items)
        self.metrics.input_point_count = _point_count(original_items)
        self.metrics.input_bezier_count = _bezier_count(original_items)
        self._global_scale = _sequence_scale(original_items)
        current = original_items
        changed_any = False
        passes_used = 0

        try:
            for pass_number in range(1, self.config.maximum_optimization_passes + 1):
                passes_used = pass_number
                optimized, changed = self._optimize_items(current, pass_number=pass_number, path=())
                current = optimized
                changed_any = changed_any or changed
                if not changed:
                    break
            self.metrics.optimization_passes = passes_used
            current = self._validate_items(current, pass_number=passes_used)
            success = True
            status = OptimizationStatus.OPTIMIZED if changed_any else OptimizationStatus.UNCHANGED
        except Exception as exc:
            if self.config.strict:
                raise
            self.warnings.append(GeometryOptimizationWarning("optimization_failed", str(exc)))
            current = original_items
            success = False
            status = OptimizationStatus.FAILED

        self._finalize_metrics(original_items, current)
        deterministic_hash = _deterministic_hash(
            original_summary=original_summary,
            primitives=current,
            metrics=self.metrics,
            warnings=tuple(self.warnings),
            operations=tuple(self.operations),
            configuration=self.config,
            status=status,
            success=success,
        )
        result = GeometryOptimizationResult(
            original_summary=original_summary,
            primitives=current,
            metrics=self.metrics,
            warnings=tuple(self.warnings),
            operations=tuple(self.operations),
            configuration=self.config,
            deterministic_hash=deterministic_hash,
            success=success,
            status=status,
        )
        _log_result(result)
        return result

    def _optimize_items(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        current = items
        changed_any = False
        for stage in (
            self._optimize_groups,
            self._remove_duplicate_points,
            self._handle_degenerate_primitives,
            self._convert_simple_primitives,
            self._merge_collinear_lines,
            self._merge_adjacent_polylines,
            self._simplify_polylines,
            self._merge_bezier_sequences,
            self._remove_duplicate_primitives,
        ):
            current, changed = stage(current, pass_number=pass_number, path=path)
            changed_any = changed_any or changed
        return current, changed_any

    def _optimize_groups(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.preserve_groups:
            flattened: list[SemanticGeometry] = []
            changed = False
            for item in items:
                if isinstance(item, PrimitiveGroup):
                    flattened.extend(item.flatten())
                    changed = True
                else:
                    flattened.append(item)
            return tuple(flattened), changed
        changed_any = False
        output: list[SemanticGeometry] = []
        for index, item in enumerate(items):
            if not isinstance(item, PrimitiveGroup):
                output.append(item)
                continue
            self.metrics.groups_preserved += 1
            optimized_children, changed = self._optimize_items(tuple(item.items), pass_number=pass_number, path=(*path, index))
            changed_any = changed_any or changed
            if not optimized_children and self.config.remove_degenerate_primitives:
                self.metrics.groups_removed += 1
                self._record_operation(
                    OptimizationOperationKind.REMOVE_DEGENERATE,
                    (item,),
                    (),
                    before_count=1,
                    after_count=0,
                    local_error=0.0,
                    rms=0.0,
                    normalized=0.0,
                    accepted=True,
                    reason="empty group removed",
                    pass_number=pass_number,
                )
                changed_any = True
                continue
            if changed:
                group = PrimitiveGroup(
                    optimized_children,
                    name=item.name,
                    metadata=_history_metadata(item.metadata, OptimizationOperationKind.NORMALIZE, self.config),
                )
                output.append(group)
            else:
                output.append(item)
        return tuple(output), changed_any

    def _remove_duplicate_points(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.remove_duplicate_points:
            return items, False
        output: list[SemanticGeometry] = []
        changed_any = False
        for item in items:
            if isinstance(item, PolylinePrimitive):
                result = deduplicate_consecutive_points(
                    item.points,
                    self.config.duplicate_point_tolerance,
                    closed=item.closed,
                )
                replaced = self._deduplicated_polyline(item, result, pass_number)
            elif isinstance(item, ClosedShapePrimitive):
                result = deduplicate_consecutive_points(
                    item.points,
                    self.config.duplicate_point_tolerance,
                    closed=True,
                )
                replaced = self._deduplicated_closed_shape(item, result, pass_number)
            else:
                replaced = item
            changed_any = changed_any or replaced is not item
            output.append(replaced)
        return tuple(output), changed_any

    def _deduplicated_polyline(
        self,
        primitive: PolylinePrimitive,
        result: DeduplicationResult,
        pass_number: int,
    ) -> SemanticGeometry:
        if not result.changed or len(result.points) < 2:
            return primitive
        self.metrics.duplicate_points_removed += result.removed_count
        replaced = PolylinePrimitive(
            points=result.points,
            closed=primitive.closed,
            **_style_kwargs(primitive, OptimizationOperationKind.REMOVE_DUPLICATE_POINT, self.config),
        )
        self._record_operation(
            OptimizationOperationKind.REMOVE_DUPLICATE_POINT,
            (primitive,),
            (replaced,),
            before_count=len(primitive.points),
            after_count=len(result.points),
            local_error=self.config.duplicate_point_tolerance,
            rms=self.config.duplicate_point_tolerance,
            normalized=normalize_error(self.config.duplicate_point_tolerance, self._global_scale),
            accepted=True,
            reason=f"removed {result.removed_count} duplicate points",
            pass_number=pass_number,
        )
        return replaced

    def _deduplicated_closed_shape(
        self,
        primitive: ClosedShapePrimitive,
        result: DeduplicationResult,
        pass_number: int,
    ) -> SemanticGeometry:
        if not result.changed or len(result.points) < 3:
            return primitive
        self.metrics.duplicate_points_removed += result.removed_count
        replaced = ClosedShapePrimitive(
            points=result.points,
            **_style_kwargs(primitive, OptimizationOperationKind.REMOVE_DUPLICATE_POINT, self.config),
        )
        self._record_operation(
            OptimizationOperationKind.REMOVE_DUPLICATE_POINT,
            (primitive,),
            (replaced,),
            before_count=len(primitive.points),
            after_count=len(result.points),
            local_error=self.config.duplicate_point_tolerance,
            rms=self.config.duplicate_point_tolerance,
            normalized=normalize_error(self.config.duplicate_point_tolerance, self._global_scale),
            accepted=True,
            reason=f"removed {result.removed_count} duplicate points",
            pass_number=pass_number,
        )
        return replaced

    def _handle_degenerate_primitives(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.remove_degenerate_primitives:
            return items, False
        output: list[SemanticGeometry] = []
        changed_any = False
        for index, item in enumerate(items):
            reason = _degenerate_reason(item, self.config.minimum_primitive_length)
            if reason is None:
                output.append(item)
                continue
            handling = self.config.degenerate_handling
            if self.config.strict or handling is DuplicateHandling.ERROR:
                raise GeometryOptimizationError(reason)
            if handling is DuplicateHandling.PRESERVE_WITH_WARNING:
                self.warnings.append(GeometryOptimizationWarning("degenerate_preserved", reason, index, "degenerate"))
                output.append(item)
                continue
            if handling is DuplicateHandling.REMOVE:
                self.metrics.degenerate_primitives_removed += 1
                self._record_operation(
                    OptimizationOperationKind.REMOVE_DEGENERATE,
                    (item,),
                    (),
                    before_count=1,
                    after_count=0,
                    local_error=0.0,
                    rms=0.0,
                    normalized=0.0,
                    accepted=True,
                    reason=reason,
                    pass_number=pass_number,
                )
                changed_any = True
                continue
            point = _representative_point(item)
            converted = PointPrimitive(
                point=point,
                **_style_kwargs(item, OptimizationOperationKind.CONVERT_DEGENERATE, self.config),
            )
            self.metrics.degenerate_primitives_converted += 1
            self._record_operation(
                OptimizationOperationKind.CONVERT_DEGENERATE,
                (item,),
                (converted,),
                before_count=1,
                after_count=1,
                local_error=0.0,
                rms=0.0,
                normalized=0.0,
                accepted=True,
                reason=reason,
                pass_number=pass_number,
            )
            output.append(converted)
            changed_any = True
        return tuple(output), changed_any

    def _convert_simple_primitives(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        output: list[SemanticGeometry] = []
        changed_any = False
        for item in items:
            converted = self._convert_to_line_if_safe(item, pass_number)
            output.append(converted)
            changed_any = changed_any or converted is not item
        return tuple(output), changed_any

    def _convert_to_line_if_safe(self, item: SemanticGeometry, pass_number: int) -> SemanticGeometry:
        tolerance = self.config.collinear_distance_tolerance * self._global_scale
        if isinstance(item, PolylinePrimitive) and not item.closed:
            if len(item.points) == 2:
                start, end = item.points
                if point_distance(start, end) > self.config.minimum_primitive_length:
                    return self._line_from_conversion(item, start, end, 0.0, 0.0, pass_number, "polyline has two points")
            line_like = line_like_endpoints(item.points, tolerance, closed=False)
            if line_like is not None:
                start, end, local_error, local_rms = line_like
                return self._line_from_conversion(item, start, end, local_error, local_rms, pass_number, "polyline is collinear")
        if isinstance(item, BezierPrimitive):
            samples = sample_cubic_bezier(item.start, item.control1, item.control2, item.end, samples=24)
            line_like = line_like_endpoints(samples, tolerance, closed=False)
            if line_like is not None:
                start, end, local_error, local_rms = line_like
                return self._line_from_conversion(item, start, end, local_error, local_rms, pass_number, "Bezier is effectively straight")
        return item

    def _line_from_conversion(
        self,
        item: SemanticGeometry,
        start: Point2D,
        end: Point2D,
        local_error: float,
        local_rms: float,
        pass_number: int,
        reason: str,
    ) -> SemanticGeometry:
        normalized = normalize_error(local_error, self._global_scale)
        if not self._within_error_budget(normalized):
            self._record_rejection(
                OptimizationOperationKind.CONVERT_TO_LINE,
                (item,),
                reason="error budget would be exceeded",
                local_error=local_error,
                rms=local_rms,
                normalized=normalized,
                pass_number=pass_number,
            )
            return item
        line = LinePrimitive(
            start=_round_point(start, self.config.decimal_precision),
            end=_round_point(end, self.config.decimal_precision),
            **_style_kwargs(item, OptimizationOperationKind.CONVERT_TO_LINE, self.config, error=local_error),
        )
        self._accept_error(normalized, local_error)
        self._record_operation(
            OptimizationOperationKind.CONVERT_TO_LINE,
            (item,),
            (line,),
            before_count=1,
            after_count=1,
            local_error=local_error,
            rms=local_rms,
            normalized=normalized,
            accepted=True,
            reason=reason,
            pass_number=pass_number,
        )
        return line

    def _merge_collinear_lines(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.merge_collinear_lines or len(items) < 2:
            return items, False
        output: list[SemanticGeometry] = []
        changed = False
        index = 0
        while index < len(items):
            current = items[index]
            if not isinstance(current, LinePrimitive):
                output.append(current)
                index += 1
                continue
            while index + 1 < len(items) and isinstance(items[index + 1], LinePrimitive):
                following = items[index + 1]
                assert isinstance(following, LinePrimitive)
                if _topology_locked(current) or _topology_locked(following):
                    self.metrics.topology_rejections += 1
                    self._record_rejection(
                        OptimizationOperationKind.MERGE_COLLINEAR_LINES,
                        (current, following),
                        reason="topology metadata preserves a junction",
                        pass_number=pass_number,
                    )
                    break
                metadata = _merged_metadata((current, following), OptimizationOperationKind.MERGE_COLLINEAR_LINES, self.config)
                outcome = try_merge_collinear_lines(
                    current,
                    following,
                    endpoint_tolerance=self.config.maximum_endpoint_distance,
                    angle_tolerance=self.config.collinear_angle_tolerance,
                    distance_tolerance=self.config.collinear_distance_tolerance * self._global_scale,
                    preserve_styles=self.config.preserve_styles,
                    decimal_precision=self.config.decimal_precision,
                    metadata=metadata,
                )
                if not outcome.accepted:
                    if outcome.reason == "styles differ":
                        self.metrics.style_rejections += 1
                        self._record_rejection(
                            OptimizationOperationKind.MERGE_COLLINEAR_LINES,
                            (current, following),
                            reason=outcome.reason,
                            pass_number=pass_number,
                        )
                    break
                normalized = normalize_error(outcome.local_error, self._global_scale)
                if not self._within_error_budget(normalized):
                    self._record_rejection(
                        OptimizationOperationKind.MERGE_COLLINEAR_LINES,
                        (current, following),
                        reason="error budget would be exceeded",
                        local_error=outcome.local_error,
                        rms=outcome.rms_error,
                        normalized=normalized,
                        pass_number=pass_number,
                    )
                    break
                assert outcome.primitive is not None
                self._accept_error(normalized, outcome.local_error)
                self.metrics.collinear_lines_merged += 1
                self._record_operation(
                    OptimizationOperationKind.MERGE_COLLINEAR_LINES,
                    (current, following),
                    (outcome.primitive,),
                    before_count=2,
                    after_count=1,
                    local_error=outcome.local_error,
                    rms=outcome.rms_error,
                    normalized=normalized,
                    accepted=True,
                    reason="adjacent collinear lines merged",
                    pass_number=pass_number,
                )
                current = outcome.primitive
                changed = True
                index += 1
            output.append(current)
            index += 1
        return tuple(output), changed

    def _merge_adjacent_polylines(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.merge_adjacent_polylines or len(items) < 2:
            return items, False
        output: list[SemanticGeometry] = []
        changed = False
        index = 0
        allowed = (LinePrimitive, PolylinePrimitive)
        while index < len(items):
            current = items[index]
            if index + 1 >= len(items) or not isinstance(current, allowed) or not isinstance(items[index + 1], allowed):
                output.append(current)
                index += 1
                continue
            following = items[index + 1]
            assert isinstance(following, allowed)
            if _topology_locked(current) or _topology_locked(following):
                self.metrics.topology_rejections += 1
                self._record_rejection(
                    OptimizationOperationKind.MERGE_POLYLINES,
                    (current, following),
                    reason="topology metadata preserves a junction",
                    pass_number=pass_number,
                )
                output.append(current)
                index += 1
                continue
            metadata = _merged_metadata((current, following), OptimizationOperationKind.MERGE_POLYLINES, self.config)
            outcome = try_join_open_paths(
                current,
                following,
                endpoint_tolerance=self.config.maximum_endpoint_distance,
                join_angle_tolerance=self.config.maximum_join_angle,
                preserve_styles=self.config.preserve_styles,
                decimal_precision=self.config.decimal_precision,
                metadata=metadata,
            )
            if not outcome.accepted:
                if outcome.reason == "styles differ":
                    self.metrics.style_rejections += 1
                    self._record_rejection(
                        OptimizationOperationKind.MERGE_POLYLINES,
                        (current, following),
                        reason=outcome.reason,
                        pass_number=pass_number,
                    )
                output.append(current)
                index += 1
                continue
            normalized = normalize_error(outcome.local_error, self._global_scale)
            if not self._within_error_budget(normalized):
                self._record_rejection(
                    OptimizationOperationKind.MERGE_POLYLINES,
                    (current, following),
                    reason="error budget would be exceeded",
                    local_error=outcome.local_error,
                    rms=outcome.rms_error,
                    normalized=normalized,
                    pass_number=pass_number,
                )
                output.append(current)
                index += 1
                continue
            assert outcome.primitive is not None
            self._accept_error(normalized, outcome.local_error)
            self.metrics.polylines_merged += 1
            self._record_operation(
                OptimizationOperationKind.MERGE_POLYLINES,
                (current, following),
                (outcome.primitive,),
                before_count=2,
                after_count=1,
                local_error=outcome.local_error,
                rms=outcome.rms_error,
                normalized=normalized,
                accepted=True,
                reason=f"joined paths using {outcome.join_kind.value}",
                pass_number=pass_number,
            )
            output.append(outcome.primitive)
            changed = True
            index += 2
        return tuple(output), changed

    def _simplify_polylines(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        output: list[SemanticGeometry] = []
        changed_any = False
        for item in items:
            if isinstance(item, PolylinePrimitive) and self.config.simplify_polylines:
                replaced = self._simplify_path_primitive(item, closed=item.closed, pass_number=pass_number)
            elif isinstance(item, ClosedShapePrimitive) and self.config.simplify_closed_shapes:
                replaced = self._simplify_path_primitive(item, closed=True, pass_number=pass_number)
            else:
                replaced = item
            output.append(replaced)
            changed_any = changed_any or replaced is not item
        return tuple(output), changed_any

    def _simplify_path_primitive(
        self,
        item: PolylinePrimitive | ClosedShapePrimitive,
        *,
        closed: bool,
        pass_number: int,
    ) -> SemanticGeometry:
        scale = _primitive_scale(item)
        tolerance = self.config.polyline_tolerance * scale
        fixed = _metadata_fixed_indices(item)
        result = simplify_polyline_preserving_features(
            item.points,
            tolerance,
            closed=closed,
            scale=scale,
            preserve_corners=self.config.preserve_corners,
            corner_angle_threshold=self.config.corner_angle_threshold,
            fixed_indices=fixed,
            preserve_topology=self.config.preserve_topology,
            decimal_precision=self.config.decimal_precision,
        )
        if result.rejected_reason:
            if "self-intersection" in result.rejected_reason or "orientation" in result.rejected_reason:
                self.metrics.topology_rejections += 1
            self._record_rejection(
                OptimizationOperationKind.SIMPLIFY_POLYLINE,
                (item,),
                reason=result.rejected_reason,
                local_error=result.max_error,
                rms=result.rms_error,
                normalized=result.normalized_error,
                pass_number=pass_number,
            )
            return item
        if not result.changed:
            return item
        if result.normalized_error > self.config.normalized_error_budget or not self._within_error_budget(result.normalized_error):
            self._record_rejection(
                OptimizationOperationKind.SIMPLIFY_POLYLINE,
                (item,),
                reason="simplification exceeds error budget",
                local_error=result.max_error,
                rms=result.rms_error,
                normalized=result.normalized_error,
                pass_number=pass_number,
            )
            return item
        self._accept_error(result.normalized_error, result.max_error)
        if isinstance(item, ClosedShapePrimitive):
            replaced: SemanticGeometry = ClosedShapePrimitive(
                points=result.points,
                **_style_kwargs(item, OptimizationOperationKind.SIMPLIFY_POLYLINE, self.config, error=result.max_error),
            )
        else:
            replaced = PolylinePrimitive(
                points=result.points,
                closed=item.closed,
                **_style_kwargs(item, OptimizationOperationKind.SIMPLIFY_POLYLINE, self.config, error=result.max_error),
            )
        self.metrics.polylines_simplified += 1
        self.metrics.polyline_points_removed += result.input_count - result.output_count
        self._record_operation(
            OptimizationOperationKind.SIMPLIFY_POLYLINE,
            (item,),
            (replaced,),
            before_count=result.input_count,
            after_count=result.output_count,
            local_error=result.max_error,
            rms=result.rms_error,
            normalized=result.normalized_error,
            accepted=True,
            reason="polyline simplified within tolerance",
            pass_number=pass_number,
        )
        return replaced

    def _merge_bezier_sequences(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.merge_bezier_segments or len(items) < 2:
            return items, False
        output: list[SemanticGeometry] = []
        changed = False
        index = 0
        while index < len(items):
            current = items[index]
            if not isinstance(current, BezierPrimitive) or index + 1 >= len(items):
                output.append(current)
                index += 1
                continue
            run = [current]
            cursor = index + 1
            while cursor < len(items) and isinstance(items[cursor], BezierPrimitive):
                previous = run[-1]
                candidate = items[cursor]
                assert isinstance(candidate, BezierPrimitive)
                if not _beziers_compatible(previous, candidate, self.config):
                    break
                run.append(candidate)
                cursor += 1
            if len(run) < 2:
                output.append(current)
                index += 1
                continue
            self.metrics.bezier_sequences_examined += 1
            merged = self._merge_bezier_run(tuple(run), pass_number)
            if merged is None:
                output.extend(run)
            else:
                output.extend(merged)
                changed = True
            index += len(run)
        return tuple(output), changed

    def _merge_bezier_run(
        self,
        run: tuple[BezierPrimitive, ...],
        pass_number: int,
    ) -> tuple[BezierPrimitive, ...] | None:
        samples: list[Point2D] = []
        for index, primitive in enumerate(run):
            sampled = sample_cubic_bezier(primitive.start, primitive.control1, primitive.control2, primitive.end, samples=18)
            samples.extend(sampled if index == 0 else sampled[1:])
        scale = max(_points_scale(samples), self._global_scale)
        tolerance = self.config.bezier_merge_tolerance * scale
        fit = fit_cubic_beziers(
            samples,
            tolerance,
            maximum_segments=max(1, len(run) - 1),
            recursion_depth=8,
            minimum_points=6,
            minimum_length=self.config.minimum_primitive_length,
            straightness_tolerance=self.config.collinear_distance_tolerance * scale,
        )
        normalized = normalize_error(fit.max_error, scale)
        if not fit.accepted or fit.segment_count >= len(run):
            self._record_rejection(
                OptimizationOperationKind.MERGE_BEZIERS,
                run,
                reason=fit.rejected_reason or "combined Bezier sequence did not reduce segment count",
                local_error=fit.max_error,
                rms=fit.rms_error,
                normalized=normalized,
                pass_number=pass_number,
            )
            return None
        if normalized > self.config.maximum_combined_bezier_error or not self._within_error_budget(normalized):
            self._record_rejection(
                OptimizationOperationKind.MERGE_BEZIERS,
                run,
                reason="combined Bezier exceeds error budget",
                local_error=fit.max_error,
                rms=fit.rms_error,
                normalized=normalized,
                pass_number=pass_number,
            )
            return None
        metadata = _merged_metadata(run, OptimizationOperationKind.MERGE_BEZIERS, self.config)
        merged = tuple(
            BezierPrimitive(
                start=_round_point(segment.start, self.config.decimal_precision),
                control1=_round_point(segment.control1, self.config.decimal_precision),
                control2=_round_point(segment.control2, self.config.decimal_precision),
                end=_round_point(segment.end, self.config.decimal_precision),
                stroke=run[0].stroke,
                fill=run[0].fill,
                opacity=run[0].opacity,
                confidence=min((item.confidence for item in run if item.confidence is not None), default=None),
                error=fit.max_error,
                metadata={**metadata, "optimization_segment_index": index, "optimization_segments": fit.segment_count},
            )
            for index, segment in enumerate(fit.segments)
        )
        self._accept_error(normalized, fit.max_error)
        self.metrics.bezier_sequences_merged += 1
        self._record_operation(
            OptimizationOperationKind.MERGE_BEZIERS,
            run,
            merged,
            before_count=len(run),
            after_count=len(merged),
            local_error=fit.max_error,
            rms=fit.rms_error,
            normalized=normalized,
            accepted=True,
            reason="compatible Bezier sequence refit to fewer segments",
            pass_number=pass_number,
        )
        return merged

    def _remove_duplicate_primitives(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
        path: tuple[int, ...],
    ) -> tuple[tuple[SemanticGeometry, ...], bool]:
        if not self.config.remove_duplicate_primitives or len(items) < 2:
            return items, False
        output: list[SemanticGeometry] = []
        changed = False
        for item in items:
            duplicate = next(
                (
                    kept
                    for kept in output
                    if duplicate_primitives(
                        kept,
                        item,
                        tolerance=self.config.duplicate_geometry_tolerance,
                        preserve_styles=self.config.preserve_styles,
                    )
                ),
                None,
            )
            if duplicate is None:
                output.append(item)
                continue
            self.metrics.duplicate_primitives_removed += 1
            self._record_operation(
                OptimizationOperationKind.REMOVE_DUPLICATE_PRIMITIVE,
                (item,),
                (),
                before_count=1,
                after_count=0,
                local_error=0.0,
                rms=0.0,
                normalized=0.0,
                accepted=True,
                reason="duplicate primitive removed",
                pass_number=pass_number,
            )
            changed = True
        return tuple(output), changed

    def _validate_items(
        self,
        items: tuple[SemanticGeometry, ...],
        *,
        pass_number: int,
    ) -> tuple[SemanticGeometry, ...]:
        for index, item in enumerate(items):
            try:
                _validate_primitive(item, self.config)
            except Exception as exc:
                if self.config.strict:
                    raise
                self.warnings.append(GeometryOptimizationWarning("validation", str(exc), index, "validate"))
        return items

    def _within_error_budget(self, normalized_error: float) -> bool:
        normalized = 0.0 if not isfinite(float(normalized_error)) else max(0.0, float(normalized_error))
        return (
            normalized <= self.config.normalized_error_budget
            and self._cumulative_error + normalized <= self.config.cumulative_error_budget + _EPSILON
        )

    def _accept_error(self, normalized_error: float, local_error: float) -> None:
        normalized = 0.0 if not isfinite(float(normalized_error)) else max(0.0, float(normalized_error))
        local = 0.0 if not isfinite(float(local_error)) else max(0.0, float(local_error))
        self._cumulative_error += normalized
        self.metrics.cumulative_error = self._cumulative_error
        self.metrics.maximum_error = max(self.metrics.maximum_error, local)
        self.metrics.normalized_error = max(self.metrics.normalized_error, normalized)

    def _record_operation(
        self,
        kind: OptimizationOperationKind,
        source: Sequence[SemanticGeometry],
        output: Sequence[SemanticGeometry],
        *,
        before_count: int,
        after_count: int,
        local_error: float,
        rms: float,
        normalized: float,
        accepted: bool,
        reason: str,
        pass_number: int,
    ) -> None:
        operation = OptimizationOperation(
            kind=kind,
            source_primitive_ids=tuple(_primitive_identity(item, index) for index, item in enumerate(source)),
            output_primitive_ids=tuple(_primitive_identity(item, index) for index, item in enumerate(output)),
            before_count=before_count,
            after_count=after_count,
            local_error=local_error,
            rms_error=rms,
            normalized_error=normalized,
            accepted=accepted,
            reason=reason,
            pass_number=pass_number,
        )
        self.operations.append(operation)
        if accepted:
            self.metrics.operations_applied += 1
        else:
            self.metrics.operations_rejected += 1

    def _record_rejection(
        self,
        kind: OptimizationOperationKind,
        source: Sequence[SemanticGeometry],
        *,
        reason: str,
        local_error: float = float("inf"),
        rms: float = float("inf"),
        normalized: float = float("inf"),
        pass_number: int,
    ) -> None:
        self._record_operation(
            kind,
            source,
            (),
            before_count=len(source),
            after_count=len(source),
            local_error=local_error,
            rms=rms,
            normalized=normalized,
            accepted=False,
            reason=reason,
            pass_number=pass_number,
        )

    def _finalize_metrics(self, original: tuple[SemanticGeometry, ...], optimized: tuple[SemanticGeometry, ...]) -> None:
        self.metrics.output_primitive_count = _object_count(optimized)
        self.metrics.output_point_count = _point_count(optimized)
        self.metrics.output_bezier_count = _bezier_count(optimized)
        self.metrics.primitive_reduction = self.metrics.input_primitive_count - self.metrics.output_primitive_count
        self.metrics.point_reduction = self.metrics.input_point_count - self.metrics.output_point_count
        self.metrics.primitive_reduction_ratio = _ratio(self.metrics.primitive_reduction, self.metrics.input_primitive_count)
        self.metrics.point_reduction_ratio = _ratio(self.metrics.point_reduction, self.metrics.input_point_count)
        rms_values = [operation.rms_error for operation in self.operations if operation.accepted and isfinite(operation.rms_error)]
        self.metrics.rms_error = rms_error(rms_values) if rms_values else 0.0
        self.metrics.warnings_count = len(self.warnings)


def optimize_primitives(
    primitives: Any,
    config: GeometryOptimizationConfig | None = None,
) -> GeometryOptimizationResult:
    """Optimize semantic primitives without generating TikZ or touching GUI code."""
    return GeometryOptimizer(config).optimize(primitives)


def optimize_fit_results(
    fit_results: PrimitiveFitResult | Iterable[PrimitiveFitResult],
    config: GeometryOptimizationConfig | None = None,
) -> GeometryOptimizationResult:
    """Optimize primitives produced by Issue 7 fitting results."""
    return GeometryOptimizer(config).optimize(fit_results)


def _normalize_input(value: Any, config: GeometryOptimizationConfig) -> PrimitiveSequence:
    if isinstance(value, PrimitiveFitResult):
        return PrimitiveSequence(tuple(value.primitives), "PrimitiveFitResult", {"selected_kind": value.selected_kind.value})
    if _looks_like_svg_parse_result(value):
        return PrimitiveSequence(
            tuple(value.primitives),
            "SvgParseResult",
            {
                "input_hash": getattr(value, "input_hash", None),
                "source_type": getattr(value, "source_type", None),
                "tracer_metadata": getattr(value, "tracer_metadata", None),
            },
        )
    if _looks_like_centerline_result(value):
        return PrimitiveSequence(tuple(path.to_polyline_primitive() for path in value.paths), "CenterlineResult", {})
    if _looks_like_centerline_path(value):
        return PrimitiveSequence((value.to_polyline_primitive(),), "CenterlinePath", {"path_id": value.id})
    if isinstance(value, PrimitiveGroup):
        return PrimitiveSequence((value,), "PrimitiveGroup", {"name": value.name})
    if isinstance(value, _SEMANTIC_TYPES):
        return PrimitiveSequence((value,), type(value).__name__, {})
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        items = tuple(value)
        if not items:
            return PrimitiveSequence((), "empty_sequence", {})
        if all(isinstance(item, PrimitiveFitResult) for item in items):
            primitives: list[SemanticGeometry] = []
            for result in items:
                primitives.extend(result.primitives)
            return PrimitiveSequence(tuple(primitives), "PrimitiveFitResultSequence", {"fit_results": len(items)})
        if all(isinstance(item, _SEMANTIC_TYPES) for item in items):
            return PrimitiveSequence(tuple(items), "primitive_sequence", {})
    raise TypeError(f"Unsupported geometry input for optimization: {type(value).__name__}")


def _copy_items(items: Sequence[SemanticGeometry], config: GeometryOptimizationConfig) -> tuple[SemanticGeometry, ...]:
    return tuple(_copy_item(item, config) for item in items)


def _copy_item(item: SemanticGeometry, config: GeometryOptimizationConfig) -> SemanticGeometry:
    if isinstance(item, PrimitiveGroup):
        return PrimitiveGroup(_copy_items(item.items, config), name=item.name, metadata=dict(item.metadata))
    if isinstance(item, PointPrimitive):
        return PointPrimitive(item.point, **_base_kwargs(item, config))
    if isinstance(item, LinePrimitive):
        return LinePrimitive(item.start, item.end, **_base_kwargs(item, config))
    if isinstance(item, PolylinePrimitive):
        return PolylinePrimitive(tuple(item.points), closed=item.closed, **_base_kwargs(item, config))
    if isinstance(item, CirclePrimitive):
        return CirclePrimitive(item.center, item.radius, **_base_kwargs(item, config))
    if isinstance(item, EllipsePrimitive):
        return EllipsePrimitive(item.center, item.radius_x, item.radius_y, item.rotation, **_base_kwargs(item, config))
    if isinstance(item, BezierPrimitive):
        return BezierPrimitive(item.start, item.control1, item.control2, item.end, **_base_kwargs(item, config))
    if isinstance(item, ClosedShapePrimitive):
        return ClosedShapePrimitive(tuple(item.points), **_base_kwargs(item, config))
    raise TypeError(f"Unsupported semantic item: {type(item).__name__}")


def _base_kwargs(item: SemanticGeometry, config: GeometryOptimizationConfig) -> dict[str, Any]:
    return {
        "stroke": item.stroke,
        "fill": item.fill,
        "opacity": item.opacity,
        "confidence": item.confidence,
        "error": item.error,
        "metadata": dict(item.metadata) if config.preserve_metadata else {},
    }


def _style_kwargs(
    item: SemanticGeometry,
    kind: OptimizationOperationKind,
    config: GeometryOptimizationConfig,
    *,
    error: float | None = None,
) -> dict[str, Any]:
    return {
        "stroke": item.stroke,
        "fill": item.fill,
        "opacity": item.opacity,
        "confidence": item.confidence,
        "error": max(0.0, float(error)) if error is not None and isfinite(float(error)) else item.error,
        "metadata": _history_metadata(item.metadata, kind, config),
    }


def _history_metadata(
    metadata: Mapping[str, Any],
    kind: OptimizationOperationKind,
    config: GeometryOptimizationConfig,
) -> dict[str, Any]:
    if not config.preserve_metadata:
        return {"optimization_history": [kind.value]}
    copied = _compact_mapping(metadata)
    history = list(copied.get("optimization_history", [])) if isinstance(copied.get("optimization_history"), list) else []
    history.append(kind.value)
    copied["optimization_history"] = history
    return copied


def _merged_metadata(
    items: Sequence[SemanticGeometry],
    kind: OptimizationOperationKind,
    config: GeometryOptimizationConfig,
) -> dict[str, Any]:
    first = dict(getattr(items[0], "metadata", {})) if items else {}
    metadata = _history_metadata(first, kind, config)
    source_ids: list[str] = []
    for index, item in enumerate(items):
        identity = _primitive_identity(item, index)
        if identity not in source_ids:
            source_ids.append(identity)
        for key in ("id", "source_id", "path_id", "svg_id"):
            value = dict(getattr(item, "metadata", {})).get(key)
            if isinstance(value, str) and value not in source_ids:
                source_ids.append(value)
    metadata["merged_source_ids"] = source_ids[:32]
    if len(source_ids) > 32:
        metadata["merged_source_ids_truncated"] = True
    return metadata


def _degenerate_reason(item: SemanticGeometry, minimum_length: float) -> str | None:
    limit = max(0.0, float(minimum_length))
    if isinstance(item, LinePrimitive):
        if point_distance(item.start, item.end) <= limit:
            return "line length is below minimum"
    elif isinstance(item, PolylinePrimitive):
        if len(item.points) < 2 or path_length(item.points, closed=item.closed) <= limit:
            return "polyline length is below minimum"
        if is_degenerate_point_set(item.points, limit):
            return "polyline points collapse to one coordinate"
    elif isinstance(item, ClosedShapePrimitive):
        if len(item.points) < 3 or path_length(item.points, closed=True) <= limit:
            return "closed shape perimeter is below minimum"
        if abs(signed_area(item.points)) <= limit * limit:
            return "closed shape area is below minimum"
    elif isinstance(item, CirclePrimitive):
        if item.radius <= limit:
            return "circle radius is below minimum"
    elif isinstance(item, EllipsePrimitive):
        if item.radius_x <= limit or item.radius_y <= limit:
            return "ellipse axis is below minimum"
    elif isinstance(item, BezierPrimitive):
        points = (item.start, item.control1, item.control2, item.end)
        if is_degenerate_point_set(points, limit):
            return "Bezier controls collapse to one coordinate"
        if path_length(points) <= limit:
            return "Bezier control polygon length is below minimum"
    elif isinstance(item, PrimitiveGroup):
        if not item.items:
            return "primitive group is empty"
    return None


def _representative_point(item: SemanticGeometry) -> Point2D:
    points = _primitive_points(item)
    if not points:
        return Point2D(0.0, 0.0)
    array = points_to_array(points)
    center = np.mean(array, axis=0)
    return Point2D(round(float(center[0]), 6), round(float(center[1]), 6))


def _beziers_compatible(first: BezierPrimitive, second: BezierPrimitive, config: GeometryOptimizationConfig) -> bool:
    if not _styles_match(first, second, config):
        return False
    if _topology_locked(first) or _topology_locked(second):
        return False
    if point_distance(first.end, second.start) > config.maximum_endpoint_distance:
        return False
    first_tangent = (first.end.x - first.control2.x, first.end.y - first.control2.y)
    second_tangent = (second.control1.x - second.start.x, second.control1.y - second.start.y)
    return _angle_between(first_tangent, second_tangent) <= config.bezier_tangent_tolerance


def _styles_match(first: SemanticGeometry, second: SemanticGeometry, config: GeometryOptimizationConfig) -> bool:
    if not config.preserve_styles:
        return True
    return (
        getattr(first, "stroke", None) == getattr(second, "stroke", None)
        and getattr(first, "fill", None) == getattr(second, "fill", None)
        and getattr(first, "opacity", None) == getattr(second, "opacity", None)
    )


def _topology_locked(item: SemanticGeometry) -> bool:
    metadata = dict(getattr(item, "metadata", {}))
    keys = {
        "junction",
        "junction_id",
        "junction_ids",
        "preserve_junction",
        "topology_lock",
        "start_node_id",
        "end_node_id",
        "centerline_node_ids",
    }
    return any(key in metadata and metadata[key] not in (None, False, (), []) for key in keys)


def _metadata_fixed_indices(item: SemanticGeometry) -> tuple[int, ...]:
    metadata = dict(getattr(item, "metadata", {}))
    fixed: list[int] = []
    for key in ("junction_indices", "corner_indices", "fixed_indices"):
        value = metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            fixed.extend(int(index) for index in value)
    return tuple(sorted(set(fixed)))


def _validate_primitive(item: SemanticGeometry, config: GeometryOptimizationConfig) -> None:
    if isinstance(item, PrimitiveGroup):
        for child in item.items:
            _validate_primitive(child, config)
        return
    for point in _primitive_points(item):
        if not isfinite(point.x) or not isfinite(point.y):
            raise GeometryOptimizationError("primitive contains non-finite coordinates")
    if isinstance(item, ClosedShapePrimitive) and config.preserve_topology:
        if polygon_has_self_intersection(item.points, closed=True):
            raise GeometryOptimizationError("closed shape contains self-intersection")
    if isinstance(item, BezierPrimitive):
        if _degenerate_reason(item, config.minimum_primitive_length) is not None:
            raise GeometryOptimizationError("Bezier primitive is degenerate")


def _primitive_points(item: SemanticGeometry) -> tuple[Point2D, ...]:
    if isinstance(item, PointPrimitive):
        return (item.point,)
    if isinstance(item, LinePrimitive):
        return item.start, item.end
    if isinstance(item, PolylinePrimitive):
        return tuple(item.points)
    if isinstance(item, CirclePrimitive):
        return (item.center,)
    if isinstance(item, EllipsePrimitive):
        return (item.center,)
    if isinstance(item, BezierPrimitive):
        return item.start, item.control1, item.control2, item.end
    if isinstance(item, ClosedShapePrimitive):
        return tuple(item.points)
    if isinstance(item, PrimitiveGroup):
        points: list[Point2D] = []
        for child in item.items:
            points.extend(_primitive_points(child))
        return tuple(points)
    return ()


def _primitive_scale(item: SemanticGeometry) -> float:
    points = _primitive_points(item)
    if not points:
        return 1.0
    return geometric_scale(points)


def _points_scale(points: Sequence[Point2D]) -> float:
    return geometric_scale(points) if points else 1.0


def _sequence_scale(items: Sequence[SemanticGeometry]) -> float:
    points: list[Point2D] = []
    for item in items:
        points.extend(_primitive_points(item))
    if not points:
        return 1.0
    return geometric_scale(points)


def _object_count(items: Sequence[SemanticGeometry]) -> int:
    count = 0
    for item in items:
        if isinstance(item, PrimitiveGroup):
            count += _object_count(item.items)
        else:
            count += 1
    return count


def _point_count(items: Sequence[SemanticGeometry]) -> int:
    return sum(_point_count_one(item) for item in items)


def _point_count_one(item: SemanticGeometry) -> int:
    if isinstance(item, PrimitiveGroup):
        return _point_count(item.items)
    return len(_primitive_points(item))


def _bezier_count(items: Sequence[SemanticGeometry]) -> int:
    count = 0
    for item in items:
        if isinstance(item, BezierPrimitive):
            count += 1
        elif isinstance(item, PrimitiveGroup):
            count += _bezier_count(item.items)
    return count


def _sequence_summary(items: Sequence[SemanticGeometry], *, source_type: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "input_primitive_count": _object_count(items),
        "input_point_count": _point_count(items),
        "input_bezier_count": _bezier_count(items),
        "scale": _sequence_scale(items),
        "metadata": _compact_mapping(metadata),
    }


def _primitive_summary(item: SemanticGeometry) -> dict[str, Any]:
    if isinstance(item, PrimitiveGroup):
        return {
            "type": "group",
            "name": item.name,
            "items": len(item.items),
            "flattened_count": len(item.flatten()),
            "metadata": _compact_mapping(item.metadata),
        }
    summary: dict[str, Any] = {
        "type": _primitive_type(item),
        "point_count": _point_count_one(item),
        "metadata": _compact_mapping(getattr(item, "metadata", {})),
        "confidence": getattr(item, "confidence", None),
        "error": getattr(item, "error", None),
    }
    if isinstance(item, PolylinePrimitive):
        summary["closed"] = item.closed
    elif isinstance(item, ClosedShapePrimitive):
        summary["closed"] = True
    elif isinstance(item, CirclePrimitive):
        summary["radius"] = item.radius
    elif isinstance(item, EllipsePrimitive):
        summary["radius_x"] = item.radius_x
        summary["radius_y"] = item.radius_y
        summary["rotation"] = item.rotation
    return summary


def _primitive_type(item: SemanticGeometry) -> str:
    return item.to_dict().get("type", type(item).__name__) if hasattr(item, "to_dict") else type(item).__name__


def _primitive_identity(item: SemanticGeometry, fallback_index: int) -> str:
    metadata = dict(getattr(item, "metadata", {}))
    for key in ("id", "source_id", "svg_id", "path_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return f"{_primitive_type(item)}:{fallback_index}"


def _compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(mapping).items():
        if isinstance(value, Mapping):
            result[key] = _compact_mapping(value)
        elif isinstance(value, Enum):
            result[key] = value.value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result[key] = list(value)[:16]
            if len(value) > 16:
                result[f"{key}_truncated"] = True
        else:
            result[key] = repr(value)
    return result


def _deterministic_hash(
    *,
    original_summary: Mapping[str, Any],
    primitives: tuple[SemanticGeometry, ...],
    metrics: GeometryOptimizationMetrics,
    warnings: tuple[GeometryOptimizationWarning, ...],
    operations: tuple[OptimizationOperation, ...],
    configuration: GeometryOptimizationConfig,
    status: OptimizationStatus,
    success: bool,
) -> str:
    payload = {
        "original_summary": _compact_mapping(original_summary),
        "primitives": [_primitive_summary(item) for item in primitives],
        "metrics": metrics.to_dict(),
        "warnings": [warning.to_dict() for warning in warnings],
        "operations": [operation.to_dict() for operation in operations],
        "configuration": configuration.to_dict(),
        "success": success,
        "status": status.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _log_result(result: GeometryOptimizationResult) -> None:
    metrics = result.metrics
    log_event("GeometryOptimize", f"input_primitives={metrics.input_primitive_count}")
    log_event("GeometryOptimize", f"output_primitives={metrics.output_primitive_count}")
    log_event("GeometryOptimize", f"input_points={metrics.input_point_count}")
    log_event("GeometryOptimize", f"output_points={metrics.output_point_count}")
    log_event("GeometryOptimize", f"lines_merged={metrics.collinear_lines_merged}")
    log_event("GeometryOptimize", f"polylines_simplified={metrics.polylines_simplified}")
    log_event("GeometryOptimize", f"beziers_merged={metrics.bezier_sequences_merged}")
    log_event("GeometryOptimize", f"normalized_error={metrics.normalized_error:.4f}")
    log_event("GeometryOptimize", f"reduction={metrics.primitive_reduction_ratio * 100.0:.1f}%")


def _angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_length = hypot(first[0], first[1])
    second_length = hypot(second[0], second[1])
    if first_length <= _EPSILON or second_length <= _EPSILON:
        return 180.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _round_point(point: Point2D, precision: int) -> Point2D:
    return Point2D(round(point.x, precision), round(point.y, precision))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom == 0.0:
        return 0.0
    return float(numerator) / denom


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if isfinite(number) else None


def _coerce_duplicate_handling(value: DuplicateHandling | str) -> DuplicateHandling:
    if isinstance(value, DuplicateHandling):
        return value
    normalized = str(value).strip().lower()
    for item in DuplicateHandling:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported duplicate handling: {value!r}")


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_non_negative_float(name: str, value: float) -> None:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _looks_like_svg_parse_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "document_info") and hasattr(value, "metrics")


def _looks_like_centerline_result(value: Any) -> bool:
    return hasattr(value, "paths") and hasattr(value, "metrics") and type(value).__name__ == "CenterlineResult"


def _looks_like_centerline_path(value: Any) -> bool:
    return hasattr(value, "points") and hasattr(value, "closure") and type(value).__name__ == "CenterlinePath"


_SEMANTIC_TYPES = (
    PointPrimitive,
    LinePrimitive,
    PolylinePrimitive,
    CirclePrimitive,
    EllipsePrimitive,
    BezierPrimitive,
    ClosedShapePrimitive,
    PrimitiveGroup,
)


__all__ = [
    "DuplicateHandling",
    "GeometryOptimizationConfig",
    "GeometryOptimizationError",
    "GeometryOptimizationMetrics",
    "GeometryOptimizationResult",
    "GeometryOptimizationWarning",
    "GeometryOptimizer",
    "OptimizationCandidate",
    "OptimizationOperation",
    "OptimizationOperationKind",
    "OptimizationStatus",
    "PathJoinKind",
    "PrimitiveSequence",
    "optimize_fit_results",
    "optimize_primitives",
]
