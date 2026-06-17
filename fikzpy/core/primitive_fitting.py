"""Fit semantic primitives to traced geometry without changing application flows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import atan2, hypot, isfinite
from typing import Any

import numpy as np

from fikzpy.core.bezier_fitting import CubicBezierFitResult, fit_cubic_beziers, sample_cubic_bezier
from fikzpy.core.diagnostics import log_event
from fikzpy.core.geometry_error import GeometryBounds, angular_coverage, array_to_points
from fikzpy.core.geometry_error import circle_radial_errors, closure_error, corner_indices
from fikzpy.core.geometry_error import ellipse_distance_errors, geometric_scale, geometry_bounds
from fikzpy.core.geometry_error import max_error, normalize_error, path_length
from fikzpy.core.geometry_error import point_distances_to_line, point_distances_to_polyline
from fikzpy.core.geometry_error import points_to_array, rms_error, simplify_polyline_rdp
from fikzpy.core.geometry_error import traversal_direction
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D
from fikzpy.core.semantic_geometry import PointPrimitive, PolylinePrimitive, Primitive
from fikzpy.core.semantic_geometry import PrimitiveGroup, SemanticGeometry, StrokeStyle


_EPSILON = 1e-12


class PrimitiveFittingError(ValueError):
    """Raised when geometry cannot be normalized for fitting."""


class PrimitiveFitKind(Enum):
    """Candidate and selected primitive kinds."""

    EXISTING = "existing"
    POINT = "point"
    LINE = "line"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYLINE = "polyline"
    BEZIER = "bezier"
    CLOSED_FREEFORM = "closed_freeform"
    GROUP = "group"


class FitStatus(Enum):
    """Candidate evaluation status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PRESERVED = "preserved"
    FAILED = "failed"


class ClosedPathHandling(Enum):
    """How closed paths are handled by the fitter."""

    PRESERVE = "preserve"
    FIT_SIMPLE = "fit_simple"
    FREEFORM = "freeform"


@dataclass(frozen=True)
class PrimitiveFittingConfig:
    """Centralized parameters for deterministic primitive fitting."""

    enable_point_fit: bool = True
    enable_line_fit: bool = True
    enable_circle_fit: bool = True
    enable_ellipse_fit: bool = True
    enable_polyline_fit: bool = True
    enable_bezier_fit: bool = True
    point_error_tolerance: float = 1e-3
    line_error_tolerance: float = 0.01
    circle_error_tolerance: float = 0.012
    ellipse_error_tolerance: float = 0.015
    polyline_error_tolerance: float = 0.01
    bezier_error_tolerance: float = 0.015
    minimum_points_for_line: int = 2
    minimum_points_for_circle: int = 8
    minimum_points_for_ellipse: int = 12
    minimum_points_for_bezier: int = 6
    minimum_line_length: float = 1e-6
    minimum_arc_coverage: float = 0.75
    minimum_circle_coverage: float = 0.90
    minimum_ellipse_coverage: float = 0.90
    maximum_axis_ratio: float = 20.0
    collinearity_tolerance: float = 0.01
    closed_path_tolerance: float = 0.02
    corner_angle_threshold: float = 35.0
    preserve_corners: bool = True
    preserve_closed_paths: bool = True
    prefer_existing_semantic_primitives: bool = True
    allow_primitive_replacement: bool = False
    closed_path_handling: ClosedPathHandling | str = ClosedPathHandling.FIT_SIMPLE
    maximum_bezier_segments: int = 8
    bezier_recursion_depth: int = 12
    decimal_precision: int = 6
    normalize_error_by_scale: bool = True
    confidence_threshold: float = 0.15
    ambiguity_margin: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "enable_point_fit",
            "enable_line_fit",
            "enable_circle_fit",
            "enable_ellipse_fit",
            "enable_polyline_fit",
            "enable_bezier_fit",
            "preserve_corners",
            "preserve_closed_paths",
            "prefer_existing_semantic_primitives",
            "allow_primitive_replacement",
            "normalize_error_by_scale",
        ):
            _validate_bool(name, getattr(self, name))
        for name in (
            "point_error_tolerance",
            "line_error_tolerance",
            "circle_error_tolerance",
            "ellipse_error_tolerance",
            "polyline_error_tolerance",
            "bezier_error_tolerance",
            "minimum_line_length",
            "collinearity_tolerance",
            "closed_path_tolerance",
        ):
            _validate_non_negative_float(name, getattr(self, name))
        for name in (
            "minimum_arc_coverage",
            "minimum_circle_coverage",
            "minimum_ellipse_coverage",
            "confidence_threshold",
            "ambiguity_margin",
        ):
            _validate_ratio(name, getattr(self, name))
        for name in (
            "minimum_points_for_line",
            "minimum_points_for_circle",
            "minimum_points_for_ellipse",
            "minimum_points_for_bezier",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, int(getattr(self, name)))
        if float(self.maximum_axis_ratio) < 1.0 or not isfinite(float(self.maximum_axis_ratio)):
            raise ValueError("maximum_axis_ratio must be finite and at least 1.")
        if not isfinite(float(self.corner_angle_threshold)) or float(self.corner_angle_threshold) < 0.0:
            raise ValueError("corner_angle_threshold must be finite and non-negative.")
        if int(self.maximum_bezier_segments) < 1:
            raise ValueError("maximum_bezier_segments must be positive.")
        if int(self.bezier_recursion_depth) < 0:
            raise ValueError("bezier_recursion_depth must be non-negative.")
        if int(self.decimal_precision) < 0:
            raise ValueError("decimal_precision must be non-negative.")
        object.__setattr__(self, "maximum_bezier_segments", int(self.maximum_bezier_segments))
        object.__setattr__(self, "bezier_recursion_depth", int(self.bezier_recursion_depth))
        object.__setattr__(self, "decimal_precision", int(self.decimal_precision))
        object.__setattr__(self, "closed_path_handling", _coerce_closed_handling(self.closed_path_handling))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        data = dict(self.__dict__)
        data["closed_path_handling"] = self.closed_path_handling.value
        return data


@dataclass(frozen=True)
class PrimitiveFitWarning:
    """A structured primitive fitting warning."""

    code: str
    message: str
    source_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable warning diagnostics."""
        return {"code": self.code, "message": self.message, "source_type": self.source_type}


@dataclass
class PrimitiveFitMetrics:
    """Scalar diagnostics for primitive fitting."""

    inputs_processed: int = 0
    existing_primitives_preserved: int = 0
    point_fits: int = 0
    line_fits: int = 0
    circle_fits: int = 0
    ellipse_fits: int = 0
    polyline_fits: int = 0
    bezier_fits: int = 0
    closed_freeform_fallbacks: int = 0
    ambiguous_results: int = 0
    rejected_candidates: int = 0
    input_points: int = 0
    output_points: int = 0
    bezier_segment_count: int = 0
    fitting_failures: int = 0
    warnings_count: int = 0

    def add(self, other: "PrimitiveFitMetrics") -> None:
        """Accumulate another metrics object into this one."""
        for key in self.__dict__:
            setattr(self, key, int(getattr(self, key)) + int(getattr(other, key)))

    def to_dict(self) -> dict[str, int]:
        """Return serializable metrics."""
        return {key: int(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class PrimitiveCandidate:
    """One evaluated fitting candidate."""

    kind: PrimitiveFitKind
    status: FitStatus
    primitives: tuple[SemanticGeometry, ...] = ()
    error: float = float("inf")
    normalized_error: float = float("inf")
    confidence: float = 0.0
    complexity: float = float("inf")
    object_count: int = 0
    parameter_count: int = 0
    rejection_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Return whether this candidate may be selected."""
        return self.status in {FitStatus.ACCEPTED, FitStatus.PRESERVED}

    def to_dict(self) -> dict[str, Any]:
        """Return compact candidate diagnostics."""
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "accepted": self.accepted,
            "error": _finite_or_none(self.error),
            "normalized_error": _finite_or_none(self.normalized_error),
            "confidence": self.confidence,
            "complexity": _finite_or_none(self.complexity),
            "object_count": self.object_count,
            "parameter_count": self.parameter_count,
            "rejection_reason": self.rejection_reason,
            "diagnostics": _compact_mapping(self.diagnostics),
            "primitives": [_primitive_summary(item) for item in self.primitives],
        }


@dataclass(frozen=True)
class PrimitiveFitResult:
    """Selected primitive fitting result for one input geometry."""

    source_type: str
    selected_kind: PrimitiveFitKind
    primitives: tuple[SemanticGeometry, ...]
    candidates: tuple[PrimitiveCandidate, ...]
    metrics: PrimitiveFitMetrics
    warnings: tuple[PrimitiveFitWarning, ...]
    confidence: float
    ambiguous: bool
    alternative_kind: PrimitiveFitKind | None
    input_point_count: int
    output_primitive_count: int
    fitting_error: float
    normalized_error: float
    configuration: PrimitiveFittingConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without large coordinate arrays."""
        return {
            "source_type": self.source_type,
            "selected_kind": self.selected_kind.value,
            "primitives": [_primitive_summary(item) for item in self.primitives],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "alternative_kind": self.alternative_kind.value if self.alternative_kind else None,
            "input_point_count": self.input_point_count,
            "output_primitive_count": self.output_primitive_count,
            "fitting_error": _finite_or_none(self.fitting_error),
            "normalized_error": _finite_or_none(self.normalized_error),
            "configuration": self.configuration.to_dict(),
            "metadata": _compact_mapping(self.metadata),
        }


@dataclass(frozen=True)
class _NormalizedGeometry:
    points: tuple[Point2D, ...]
    closed: bool
    source_type: str
    stroke: StrokeStyle
    fill: FillStyle | None
    opacity: float | None
    metadata: Mapping[str, Any]
    bounds: GeometryBounds
    scale: float
    length: float
    closure_error: float
    duplicate_count: int
    corners: tuple[int, ...]
    source: Any

    @property
    def array(self) -> np.ndarray:
        return points_to_array(self.points)


class PrimitiveFitter:
    """Fit primitive candidates and select the simplest faithful result."""

    def __init__(self, config: PrimitiveFittingConfig | None = None) -> None:
        self.config = config or PrimitiveFittingConfig()

    def fit(self, geometry: Any) -> PrimitiveFitResult:
        """Fit semantic primitives to one geometry input."""
        if _looks_like_svg_parse_result(geometry):
            return self._fit_collection(
                tuple(geometry.primitives),
                source_type="SvgParseResult",
                metadata={
                    "source": "svg_parse_result",
                    "input_hash": getattr(geometry, "input_hash", None),
                    "tracer_metadata": getattr(geometry, "tracer_metadata", None),
                },
            )
        if isinstance(geometry, PrimitiveGroup):
            return self._fit_collection(
                tuple(geometry.items),
                source_type="PrimitiveGroup",
                metadata={**dict(geometry.metadata), "name": geometry.name},
            )
        if _looks_like_primitive_collection(geometry):
            return self._fit_collection(tuple(geometry), source_type="primitive_collection", metadata={})

        existing = self._preserve_existing_if_requested(geometry)
        if existing is not None:
            return existing

        normalized = _normalize_geometry(geometry, self.config)
        candidates = self._build_candidates(normalized)
        selected = _select_candidate(candidates)
        warnings: list[PrimitiveFitWarning] = []
        if selected is None:
            selected = self._fallback_candidate(normalized)
            warnings.append(PrimitiveFitWarning("fallback", "No fitting candidate was accepted.", normalized.source_type))

        ambiguous, alternative = _ambiguity(selected, candidates, self.config)
        if ambiguous and alternative is not None:
            warnings.append(
                PrimitiveFitWarning(
                    "ambiguous",
                    f"Selected {selected.kind.value} is close to {alternative.kind.value}.",
                    normalized.source_type,
                )
            )

        metrics = _metrics_for_result(
            selected,
            candidates,
            input_points=len(normalized.points),
            warnings_count=len(warnings),
            ambiguous=ambiguous,
        )
        confidence = selected.confidence
        if ambiguous:
            confidence = max(0.0, confidence - self.config.ambiguity_margin)

        result = PrimitiveFitResult(
            source_type=normalized.source_type,
            selected_kind=selected.kind,
            primitives=selected.primitives,
            candidates=candidates,
            metrics=metrics,
            warnings=tuple(warnings),
            confidence=confidence,
            ambiguous=ambiguous,
            alternative_kind=alternative.kind if alternative is not None else None,
            input_point_count=len(normalized.points),
            output_primitive_count=_object_count(selected.primitives),
            fitting_error=selected.error,
            normalized_error=selected.normalized_error,
            configuration=self.config,
            metadata={
                "bounds": normalized.bounds.to_dict(),
                "scale": normalized.scale,
                "length": normalized.length,
                "closed": normalized.closed,
                "duplicate_count": normalized.duplicate_count,
                "corners": list(normalized.corners),
                **dict(normalized.metadata),
            },
        )
        _log_result(result)
        return result

    def fit_many(self, geometries: Iterable[Any]) -> list[PrimitiveFitResult]:
        """Fit a sequence of independent geometries."""
        return [self.fit(geometry) for geometry in geometries]

    def _preserve_existing_if_requested(self, geometry: Any) -> PrimitiveFitResult | None:
        if isinstance(geometry, (PointPrimitive, LinePrimitive, CirclePrimitive, EllipsePrimitive)):
            return self._existing_result(geometry)
        if (
            isinstance(geometry, BezierPrimitive)
            and self.config.prefer_existing_semantic_primitives
            and not self.config.allow_primitive_replacement
        ):
            return self._existing_result(geometry)
        if (
            isinstance(geometry, (LinePrimitive, CirclePrimitive, EllipsePrimitive))
            and self.config.prefer_existing_semantic_primitives
        ):
            return self._existing_result(geometry)
        return None

    def _existing_result(self, primitive: SemanticGeometry) -> PrimitiveFitResult:
        source_type = type(primitive).__name__
        candidate = PrimitiveCandidate(
            kind=PrimitiveFitKind.EXISTING,
            status=FitStatus.PRESERVED,
            primitives=(primitive,),
            error=float(getattr(primitive, "error", 0.0) or 0.0),
            normalized_error=0.0,
            confidence=float(getattr(primitive, "confidence", 1.0) or 1.0),
            complexity=0.0,
            object_count=1,
            parameter_count=_parameter_count(primitive),
            diagnostics={"preserved_type": _primitive_type(primitive)},
        )
        metrics = PrimitiveFitMetrics(
            inputs_processed=1,
            existing_primitives_preserved=1,
            input_points=_primitive_point_count(primitive),
            output_points=_primitive_point_count(primitive),
        )
        result = PrimitiveFitResult(
            source_type=source_type,
            selected_kind=PrimitiveFitKind.EXISTING,
            primitives=(primitive,),
            candidates=(candidate,),
            metrics=metrics,
            warnings=(),
            confidence=candidate.confidence,
            ambiguous=False,
            alternative_kind=None,
            input_point_count=_primitive_point_count(primitive),
            output_primitive_count=1,
            fitting_error=candidate.error,
            normalized_error=0.0,
            configuration=self.config,
            metadata=dict(getattr(primitive, "metadata", {})),
        )
        _log_result(result)
        return result

    def _fit_collection(
        self,
        geometries: tuple[Any, ...],
        *,
        source_type: str,
        metadata: Mapping[str, Any],
    ) -> PrimitiveFitResult:
        child_results = [self.fit(item) for item in geometries]
        output_items: list[SemanticGeometry] = []
        metrics = PrimitiveFitMetrics(inputs_processed=1)
        warnings: list[PrimitiveFitWarning] = []
        candidates: list[PrimitiveCandidate] = []
        for index, child in enumerate(child_results):
            output_items.extend(child.primitives)
            metrics.add(child.metrics)
            warnings.extend(child.warnings)
            candidates.extend(child.candidates)
            if child.ambiguous:
                warnings.append(
                    PrimitiveFitWarning("child_ambiguous", f"Child result {index} is ambiguous.", child.source_type)
                )
        group = PrimitiveGroup(tuple(output_items), name=str(metadata.get("name") or source_type), metadata=dict(metadata))
        candidate = PrimitiveCandidate(
            kind=PrimitiveFitKind.GROUP,
            status=FitStatus.ACCEPTED,
            primitives=(group,),
            error=max((child.fitting_error for child in child_results), default=0.0),
            normalized_error=max((child.normalized_error for child in child_results), default=0.0),
            confidence=min((child.confidence for child in child_results), default=1.0),
            complexity=sum(candidate.complexity for candidate in candidates if candidate.accepted),
            object_count=_object_count((group,)),
            parameter_count=_parameter_count(group),
            diagnostics={"child_results": len(child_results)},
        )
        metrics.output_points = _output_point_count((group,))
        metrics.bezier_segment_count = _bezier_count((group,))
        metrics.warnings_count = len(warnings)
        result = PrimitiveFitResult(
            source_type=source_type,
            selected_kind=PrimitiveFitKind.GROUP,
            primitives=(group,),
            candidates=(candidate,),
            metrics=metrics,
            warnings=tuple(warnings),
            confidence=candidate.confidence,
            ambiguous=any(child.ambiguous for child in child_results),
            alternative_kind=None,
            input_point_count=sum(child.input_point_count for child in child_results),
            output_primitive_count=_object_count((group,)),
            fitting_error=candidate.error,
            normalized_error=candidate.normalized_error,
            configuration=self.config,
            metadata=dict(metadata),
        )
        _log_result(result)
        return result

    def _build_candidates(self, geometry: _NormalizedGeometry) -> tuple[PrimitiveCandidate, ...]:
        candidates = [
            self._point_candidate(geometry),
            self._line_candidate(geometry),
            self._circle_candidate(geometry),
            self._ellipse_candidate(geometry),
            self._polyline_candidate(geometry),
            self._bezier_candidate(geometry),
        ]
        if geometry.closed:
            candidates.append(self._closed_freeform_candidate(geometry))
        return tuple(candidates)

    def _point_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_point_fit:
            return _rejected_candidate(PrimitiveFitKind.POINT, "point fitting disabled")
        array = geometry.array
        centroid = np.mean(array, axis=0)
        distances = np.linalg.norm(array - centroid, axis=1)
        error = max_error(distances)
        normalized = normalize_error(error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        tolerance = self.config.point_error_tolerance
        accepted = len(array) == 1 or normalized <= tolerance
        if not accepted:
            return _rejected_candidate(
                PrimitiveFitKind.POINT,
                "point dispersion exceeds tolerance",
                error=error,
                normalized_error=normalized,
                diagnostics={"dispersion": error},
            )
        primitive = PointPrimitive(
            point=_rounded_point(centroid, self.config.decimal_precision),
            **_style_kwargs(geometry, PrimitiveFitKind.POINT, _confidence(normalized, tolerance), error),
        )
        return _accepted_candidate(
            PrimitiveFitKind.POINT,
            (primitive,),
            error,
            normalized,
            _confidence(normalized, tolerance),
            diagnostics={"dispersion": error},
        )

    def _line_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_line_fit:
            return _rejected_candidate(PrimitiveFitKind.LINE, "line fitting disabled")
        array = geometry.array
        if len(array) < self.config.minimum_points_for_line:
            return _rejected_candidate(PrimitiveFitKind.LINE, "insufficient points for line")
        if geometry.closed and self.config.preserve_closed_paths:
            return _rejected_candidate(PrimitiveFitKind.LINE, "closed paths are preserved unless point-like")
        if geometry.corners and self.config.preserve_corners:
            return _rejected_candidate(PrimitiveFitKind.LINE, "corners should not be flattened into a line")

        centroid = np.mean(array, axis=0)
        centered = array - centroid
        try:
            _values, vectors = np.linalg.eigh(centered.T @ centered)
        except np.linalg.LinAlgError:
            return _rejected_candidate(PrimitiveFitKind.LINE, "line PCA failed")
        direction = vectors[:, int(np.argmax(_values))]
        norm = float(np.linalg.norm(direction))
        if norm <= _EPSILON:
            return _rejected_candidate(PrimitiveFitKind.LINE, "line direction is unstable")
        direction = direction / norm
        projections = centered @ direction
        start = centroid + float(np.min(projections)) * direction
        end = centroid + float(np.max(projections)) * direction
        length = float(np.linalg.norm(end - start))
        if length < self.config.minimum_line_length:
            return _rejected_candidate(PrimitiveFitKind.LINE, "line length is below minimum")
        errors = point_distances_to_line(array, start, end)
        error = max_error(errors)
        rms = rms_error(errors)
        normalized = normalize_error(error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        stretch = geometry.length / max(length, _EPSILON)
        stretch_limit = 1.0 + 5.0 * self.config.collinearity_tolerance
        if normalized > self.config.line_error_tolerance:
            return _rejected_candidate(
                PrimitiveFitKind.LINE,
                "line error exceeds tolerance",
                error=error,
                normalized_error=normalized,
                diagnostics={"rms_error": rms, "length": length, "path_stretch": stretch},
            )
        if stretch > stretch_limit:
            return _rejected_candidate(
                PrimitiveFitKind.LINE,
                "path length indicates relevant curvature",
                error=error,
                normalized_error=normalized,
                diagnostics={"rms_error": rms, "length": length, "path_stretch": stretch},
            )
        confidence = _confidence(normalized, self.config.line_error_tolerance)
        primitive = LinePrimitive(
            start=_rounded_point(start, self.config.decimal_precision),
            end=_rounded_point(end, self.config.decimal_precision),
            **_style_kwargs(geometry, PrimitiveFitKind.LINE, confidence, error),
        )
        return _accepted_candidate(
            PrimitiveFitKind.LINE,
            (primitive,),
            error,
            normalized,
            confidence,
            diagnostics={"rms_error": rms, "length": length, "path_stretch": stretch},
        )

    def _circle_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_circle_fit:
            return _rejected_candidate(PrimitiveFitKind.CIRCLE, "circle fitting disabled")
        array = geometry.array
        if len(array) < self.config.minimum_points_for_circle:
            return _rejected_candidate(PrimitiveFitKind.CIRCLE, "insufficient points for circle")
        if not geometry.closed:
            return _rejected_candidate(PrimitiveFitKind.CIRCLE, "partial circular arcs are not emitted in this issue")
        fit = _fit_circle(array)
        if fit is None:
            return _rejected_candidate(PrimitiveFitKind.CIRCLE, "circle least-squares fit failed")
        center, radius = fit
        if radius <= _EPSILON:
            return _rejected_candidate(PrimitiveFitKind.CIRCLE, "circle radius is not positive")
        errors = np.abs(circle_radial_errors(array, center, radius))
        error = max_error(errors)
        rms = rms_error(errors)
        normalized = normalize_error(error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        coverage = angular_coverage(array, center)
        if coverage < self.config.minimum_circle_coverage:
            return _rejected_candidate(
                PrimitiveFitKind.CIRCLE,
                "circle angular coverage is insufficient",
                error=error,
                normalized_error=normalized,
                diagnostics={"coverage": coverage, "radius": radius},
            )
        if normalized > self.config.circle_error_tolerance:
            return _rejected_candidate(
                PrimitiveFitKind.CIRCLE,
                "circle radial error exceeds tolerance",
                error=error,
                normalized_error=normalized,
                diagnostics={"coverage": coverage, "radius": radius, "rms_error": rms},
            )
        confidence = _confidence(normalized, self.config.circle_error_tolerance) * coverage
        primitive = CirclePrimitive(
            center=_rounded_point(center, self.config.decimal_precision),
            radius=round(float(radius), self.config.decimal_precision),
            **_style_kwargs(geometry, PrimitiveFitKind.CIRCLE, confidence, error),
        )
        return _accepted_candidate(
            PrimitiveFitKind.CIRCLE,
            (primitive,),
            error,
            normalized,
            confidence,
            diagnostics={
                "rms_error": rms,
                "coverage": coverage,
                "radius": radius,
                "direction": traversal_direction(array),
                "closed": geometry.closed,
            },
        )

    def _ellipse_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_ellipse_fit:
            return _rejected_candidate(PrimitiveFitKind.ELLIPSE, "ellipse fitting disabled")
        array = geometry.array
        if len(array) < self.config.minimum_points_for_ellipse:
            return _rejected_candidate(PrimitiveFitKind.ELLIPSE, "insufficient points for ellipse")
        if not geometry.closed:
            return _rejected_candidate(PrimitiveFitKind.ELLIPSE, "partial elliptical arcs are not emitted in this issue")
        fit = _fit_ellipse_pca(array)
        if fit is None:
            return _rejected_candidate(PrimitiveFitKind.ELLIPSE, "ellipse fit failed")
        center, major, minor, rotation = fit
        if major <= _EPSILON or minor <= _EPSILON:
            return _rejected_candidate(PrimitiveFitKind.ELLIPSE, "ellipse axes are not positive")
        axis_ratio = major / minor
        if axis_ratio > self.config.maximum_axis_ratio:
            return _rejected_candidate(
                PrimitiveFitKind.ELLIPSE,
                "ellipse axis ratio exceeds limit",
                diagnostics={"axis_ratio": axis_ratio},
            )
        errors = np.abs(ellipse_distance_errors(array, center, major, minor, rotation))
        error = max_error(errors)
        rms = rms_error(errors)
        normalized = normalize_error(error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        coverage = angular_coverage(array, center)
        if coverage < self.config.minimum_ellipse_coverage:
            return _rejected_candidate(
                PrimitiveFitKind.ELLIPSE,
                "ellipse angular coverage is insufficient",
                error=error,
                normalized_error=normalized,
                diagnostics={"coverage": coverage, "axis_ratio": axis_ratio},
            )
        if normalized > self.config.ellipse_error_tolerance:
            return _rejected_candidate(
                PrimitiveFitKind.ELLIPSE,
                "ellipse error exceeds tolerance",
                error=error,
                normalized_error=normalized,
                diagnostics={"coverage": coverage, "axis_ratio": axis_ratio, "rms_error": rms},
            )
        confidence = _confidence(normalized, self.config.ellipse_error_tolerance) * coverage
        primitive = EllipsePrimitive(
            center=_rounded_point(center, self.config.decimal_precision),
            radius_x=round(float(major), self.config.decimal_precision),
            radius_y=round(float(minor), self.config.decimal_precision),
            rotation=round(float(np.degrees(rotation)), self.config.decimal_precision),
            **_style_kwargs(geometry, PrimitiveFitKind.ELLIPSE, confidence, error),
        )
        return _accepted_candidate(
            PrimitiveFitKind.ELLIPSE,
            (primitive,),
            error,
            normalized,
            confidence,
            diagnostics={"rms_error": rms, "coverage": coverage, "axis_ratio": axis_ratio},
        )

    def _polyline_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_polyline_fit:
            return _rejected_candidate(PrimitiveFitKind.POLYLINE, "polyline fitting disabled")
        if len(geometry.points) < (3 if geometry.closed else 2):
            return _rejected_candidate(PrimitiveFitKind.POLYLINE, "insufficient points for polyline")
        tolerance = _absolute_tolerance(self.config.polyline_error_tolerance, geometry.scale, self.config)
        simplified = simplify_polyline_rdp(geometry.array, tolerance, closed=geometry.closed)
        if len(simplified) < (3 if geometry.closed else 2):
            simplified = geometry.array
        errors = point_distances_to_polyline(geometry.array, simplified, closed=geometry.closed)
        error = max_error(errors)
        normalized = normalize_error(error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        if normalized > self.config.polyline_error_tolerance:
            simplified = geometry.array
            error = 0.0
            normalized = 0.0
        points = array_to_points(np.round(simplified, self.config.decimal_precision))
        confidence = _confidence(normalized, max(self.config.polyline_error_tolerance, _EPSILON))
        metadata_extra = {"simplified_points": len(points), "original_points": len(geometry.points)}
        primitive: SemanticGeometry
        primitive = PolylinePrimitive(
            points=points,
            closed=geometry.closed,
            **_style_kwargs(geometry, PrimitiveFitKind.POLYLINE, confidence, error, metadata_extra=metadata_extra),
        )
        return _accepted_candidate(
            PrimitiveFitKind.POLYLINE,
            (primitive,),
            error,
            normalized,
            confidence,
            diagnostics={
                "rms_error": rms_error(errors),
                "input_points": len(geometry.points),
                "output_points": len(points),
                "corners": list(geometry.corners),
            },
        )

    def _bezier_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not self.config.enable_bezier_fit:
            return _rejected_candidate(PrimitiveFitKind.BEZIER, "Bezier fitting disabled")
        if len(geometry.points) < self.config.minimum_points_for_bezier:
            return _rejected_candidate(PrimitiveFitKind.BEZIER, "insufficient points for Bezier")
        if geometry.corners and self.config.preserve_corners:
            return _rejected_candidate(PrimitiveFitKind.BEZIER, "corners are preserved as polyline geometry")
        tolerance = _absolute_tolerance(self.config.bezier_error_tolerance, geometry.scale, self.config)
        fit = fit_cubic_beziers(
            geometry.points,
            tolerance,
            closed=geometry.closed,
            maximum_segments=self.config.maximum_bezier_segments,
            recursion_depth=self.config.bezier_recursion_depth,
            minimum_points=self.config.minimum_points_for_bezier,
            minimum_length=self.config.minimum_line_length,
            straightness_tolerance=_absolute_tolerance(self.config.line_error_tolerance, geometry.scale, self.config),
        )
        if not fit.accepted:
            return _bezier_rejected_candidate(fit)
        normalized = normalize_error(fit.max_error, geometry.scale, enabled=self.config.normalize_error_by_scale)
        if normalized > self.config.bezier_error_tolerance:
            return _rejected_candidate(
                PrimitiveFitKind.BEZIER,
                "Bezier error exceeds tolerance",
                error=fit.max_error,
                normalized_error=normalized,
                diagnostics=fit.to_dict(),
            )
        confidence = _confidence(normalized, self.config.bezier_error_tolerance)
        primitives: list[BezierPrimitive] = []
        for index, segment in enumerate(fit.segments):
            primitives.append(
                BezierPrimitive(
                    start=_round_point(segment.start, self.config.decimal_precision),
                    control1=_round_point(segment.control1, self.config.decimal_precision),
                    control2=_round_point(segment.control2, self.config.decimal_precision),
                    end=_round_point(segment.end, self.config.decimal_precision),
                    **_style_kwargs(
                        geometry,
                        PrimitiveFitKind.BEZIER,
                        confidence,
                        fit.max_error,
                        segment_index=index,
                        metadata_extra={"bezier_segments": len(fit.segments)},
                    ),
                )
            )
        return _accepted_candidate(
            PrimitiveFitKind.BEZIER,
            tuple(primitives),
            fit.max_error,
            normalized,
            confidence,
            diagnostics={
                "rms_error": fit.rms_error,
                "segment_count": len(primitives),
                "recursion_depth_used": fit.recursion_depth_used,
            },
        )

    def _closed_freeform_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if not geometry.closed:
            return _rejected_candidate(PrimitiveFitKind.CLOSED_FREEFORM, "path is open")
        if len(geometry.points) < 3:
            return _rejected_candidate(PrimitiveFitKind.CLOSED_FREEFORM, "insufficient points for closed freeform")
        points = tuple(_round_point(point, self.config.decimal_precision) for point in geometry.points)
        primitive = ClosedShapePrimitive(
            points=points,
            **_style_kwargs(geometry, PrimitiveFitKind.CLOSED_FREEFORM, 1.0, 0.0),
        )
        return _accepted_candidate(
            PrimitiveFitKind.CLOSED_FREEFORM,
            (primitive,),
            0.0,
            0.0,
            1.0,
            diagnostics={"fallback": True, "input_points": len(points)},
        )

    def _fallback_candidate(self, geometry: _NormalizedGeometry) -> PrimitiveCandidate:
        if geometry.closed and len(geometry.points) >= 3:
            return self._closed_freeform_candidate(geometry)
        return self._polyline_candidate(geometry)


def fit_primitive(geometry: Any, config: PrimitiveFittingConfig | None = None) -> PrimitiveFitResult:
    """Fit semantic primitives to one geometry input."""
    return PrimitiveFitter(config).fit(geometry)


def fit_primitives(
    geometries: Iterable[Any],
    config: PrimitiveFittingConfig | None = None,
) -> list[PrimitiveFitResult]:
    """Fit semantic primitives to multiple independent geometry inputs."""
    return PrimitiveFitter(config).fit_many(geometries)


def _normalize_geometry(geometry: Any, config: PrimitiveFittingConfig) -> _NormalizedGeometry:
    source_type = type(geometry).__name__
    stroke = getattr(geometry, "stroke", StrokeStyle())
    fill = getattr(geometry, "fill", None)
    opacity = getattr(geometry, "opacity", None)
    metadata = dict(getattr(geometry, "metadata", {}))
    closed = bool(getattr(geometry, "closed", False))

    if _looks_like_centerline_path(geometry):
        points = tuple(geometry.points)
        closure = getattr(geometry, "closure", None)
        closed = str(getattr(closure, "value", closure)).lower() == "closed"
        metadata = {"source": "centerline_path", **dict(getattr(geometry, "metadata", {}))}
    elif isinstance(geometry, PolylinePrimitive):
        points = tuple(geometry.points)
        closed = geometry.closed
    elif isinstance(geometry, ClosedShapePrimitive):
        points = tuple(geometry.points)
        closed = True
    elif isinstance(geometry, BezierPrimitive):
        if not config.allow_primitive_replacement:
            points = sample_cubic_bezier(geometry.start, geometry.control1, geometry.control2, geometry.end)
        else:
            points = sample_cubic_bezier(geometry.start, geometry.control1, geometry.control2, geometry.end, samples=48)
    elif isinstance(geometry, np.ndarray):
        points = array_to_points(points_to_array(geometry))
        source_type = "PointSequence"
    elif _looks_like_point_sequence(geometry) or (
        isinstance(geometry, Sequence) and not isinstance(geometry, (str, bytes, bytearray))
    ):
        points = tuple(_coerce_point(item) for item in geometry)
        source_type = "PointSequence"
    else:
        raise TypeError(f"Unsupported geometry input for primitive fitting: {source_type}")

    cleaned, duplicate_count, inferred_closed, gap = _clean_points(points, closed, config)
    if inferred_closed:
        closed = True
    if len(cleaned) == 0:
        raise PrimitiveFittingError("geometry contains no valid points")
    bounds = geometry_bounds(cleaned)
    scale = geometric_scale(cleaned)
    length = path_length(cleaned, closed=closed)
    corners = corner_indices(cleaned, threshold_degrees=config.corner_angle_threshold, closed=closed)
    return _NormalizedGeometry(
        points=cleaned,
        closed=closed,
        source_type=source_type,
        stroke=stroke,
        fill=fill,
        opacity=opacity,
        metadata=metadata,
        bounds=bounds,
        scale=scale,
        length=length,
        closure_error=0.0 if closed else gap,
        duplicate_count=duplicate_count,
        corners=corners,
        source=geometry,
    )


def _clean_points(
    points: Sequence[Point2D],
    closed: bool,
    config: PrimitiveFittingConfig,
) -> tuple[tuple[Point2D, ...], int, bool, float]:
    copied = tuple(_coerce_point(point) for point in points)
    if not copied:
        return (), 0, False, 0.0
    deduplicated = [copied[0]]
    duplicates = 0
    for point in copied[1:]:
        if _point_distance(point, deduplicated[-1]) <= _EPSILON:
            duplicates += 1
            continue
        deduplicated.append(point)
    gap = closure_error(deduplicated) if len(deduplicated) > 1 else 0.0
    inferred_closed = False
    close_limit = _absolute_tolerance(config.closed_path_tolerance, geometric_scale(deduplicated), config)
    duplicate_limit = max(_EPSILON, _absolute_tolerance(config.point_error_tolerance, geometric_scale(deduplicated), config))
    if len(deduplicated) > 2 and not closed and gap <= close_limit:
        if _point_distance(deduplicated[0], deduplicated[-1]) <= duplicate_limit:
            deduplicated.pop()
            duplicates += 1
        inferred_closed = True
    if closed and len(deduplicated) > 1 and _point_distance(deduplicated[0], deduplicated[-1]) <= duplicate_limit:
        deduplicated.pop()
        duplicates += 1
    return tuple(deduplicated), duplicates, inferred_closed, gap


def _fit_circle(array: np.ndarray) -> tuple[np.ndarray, float] | None:
    x = array[:, 0]
    y = array[:, 1]
    matrix = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(array))])
    vector = x * x + y * y
    try:
        solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, vector, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center = np.array([solution[0], solution[1]], dtype=np.float64)
    radius_squared = float(solution[2] + solution[0] * solution[0] + solution[1] * solution[1])
    if radius_squared <= 0.0 or not isfinite(radius_squared):
        return None
    return center, float(np.sqrt(radius_squared))


def _fit_ellipse_pca(array: np.ndarray) -> tuple[np.ndarray, float, float, float] | None:
    center = np.mean(array, axis=0)
    centered = array - center
    try:
        values, vectors = np.linalg.eigh((centered.T @ centered) / max(len(array), 1))
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if values[0] <= 0.0 or values[1] <= 0.0:
        return None
    major = float(np.sqrt(2.0 * values[0]))
    minor = float(np.sqrt(2.0 * values[1]))
    if major < minor:
        major, minor = minor, major
    direction = vectors[:, 0]
    rotation = atan2(float(direction[1]), float(direction[0]))
    return center, major, minor, rotation


def _select_candidate(candidates: tuple[PrimitiveCandidate, ...]) -> PrimitiveCandidate | None:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if not accepted:
        return None
    return sorted(
        accepted,
        key=lambda candidate: (
            candidate.complexity,
            candidate.normalized_error,
            candidate.object_count,
            _kind_order(candidate.kind),
            candidate.parameter_count,
        ),
    )[0]


def _ambiguity(
    selected: PrimitiveCandidate,
    candidates: tuple[PrimitiveCandidate, ...],
    config: PrimitiveFittingConfig,
) -> tuple[bool, PrimitiveCandidate | None]:
    accepted = [
        candidate
        for candidate in candidates
        if candidate.accepted
        and candidate.kind is not selected.kind
        and not bool(candidate.diagnostics.get("fallback"))
        and candidate.kind is not PrimitiveFitKind.CLOSED_FREEFORM
    ]
    if selected.kind in {
        PrimitiveFitKind.POINT,
        PrimitiveFitKind.LINE,
        PrimitiveFitKind.CIRCLE,
        PrimitiveFitKind.ELLIPSE,
    }:
        accepted = [
            candidate
            for candidate in accepted
            if candidate.kind
            in {
                PrimitiveFitKind.POINT,
                PrimitiveFitKind.LINE,
                PrimitiveFitKind.CIRCLE,
                PrimitiveFitKind.ELLIPSE,
            }
        ]
    if not accepted:
        return False, None
    ordered = sorted(accepted, key=lambda candidate: (candidate.normalized_error, _kind_order(candidate.kind)))
    alternative = ordered[0]
    close_error = abs(alternative.normalized_error - selected.normalized_error) <= config.ambiguity_margin
    close_confidence = abs(alternative.confidence - selected.confidence) <= max(config.ambiguity_margin, 0.01)
    return bool(close_error or close_confidence), alternative if close_error or close_confidence else None


def _metrics_for_result(
    selected: PrimitiveCandidate,
    candidates: tuple[PrimitiveCandidate, ...],
    *,
    input_points: int,
    warnings_count: int,
    ambiguous: bool,
) -> PrimitiveFitMetrics:
    metrics = PrimitiveFitMetrics(
        inputs_processed=1,
        rejected_candidates=sum(1 for candidate in candidates if not candidate.accepted),
        input_points=input_points,
        output_points=_output_point_count(selected.primitives),
        bezier_segment_count=_bezier_count(selected.primitives),
        warnings_count=warnings_count,
        ambiguous_results=1 if ambiguous else 0,
    )
    if selected.kind is PrimitiveFitKind.EXISTING:
        metrics.existing_primitives_preserved = 1
    elif selected.kind is PrimitiveFitKind.POINT:
        metrics.point_fits = 1
    elif selected.kind is PrimitiveFitKind.LINE:
        metrics.line_fits = 1
    elif selected.kind is PrimitiveFitKind.CIRCLE:
        metrics.circle_fits = 1
    elif selected.kind is PrimitiveFitKind.ELLIPSE:
        metrics.ellipse_fits = 1
    elif selected.kind is PrimitiveFitKind.POLYLINE:
        metrics.polyline_fits = 1
    elif selected.kind is PrimitiveFitKind.BEZIER:
        metrics.bezier_fits = 1
    elif selected.kind is PrimitiveFitKind.CLOSED_FREEFORM:
        metrics.closed_freeform_fallbacks = 1
    return metrics


def _accepted_candidate(
    kind: PrimitiveFitKind,
    primitives: tuple[SemanticGeometry, ...],
    error: float,
    normalized_error: float,
    confidence: float,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> PrimitiveCandidate:
    parameter_count = sum(_parameter_count(item) for item in primitives)
    object_count = _object_count(primitives)
    return PrimitiveCandidate(
        kind=kind,
        status=FitStatus.ACCEPTED,
        primitives=primitives,
        error=float(error),
        normalized_error=float(normalized_error),
        confidence=max(0.0, min(1.0, float(confidence))),
        complexity=_complexity(kind, parameter_count, object_count),
        object_count=object_count,
        parameter_count=parameter_count,
        diagnostics=dict(diagnostics or {}),
    )


def _rejected_candidate(
    kind: PrimitiveFitKind,
    reason: str,
    *,
    error: float = float("inf"),
    normalized_error: float = float("inf"),
    diagnostics: Mapping[str, Any] | None = None,
) -> PrimitiveCandidate:
    return PrimitiveCandidate(
        kind=kind,
        status=FitStatus.REJECTED,
        error=float(error),
        normalized_error=float(normalized_error),
        rejection_reason=reason,
        diagnostics=dict(diagnostics or {}),
    )


def _bezier_rejected_candidate(fit: CubicBezierFitResult) -> PrimitiveCandidate:
    return _rejected_candidate(
        PrimitiveFitKind.BEZIER,
        fit.rejected_reason or "Bezier fitting failed",
        error=fit.max_error,
        normalized_error=float("inf"),
        diagnostics=fit.to_dict(),
    )


def _style_kwargs(
    geometry: _NormalizedGeometry,
    kind: PrimitiveFitKind,
    confidence: float,
    error: float,
    *,
    segment_index: int | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        **dict(geometry.metadata),
        "fit_kind": kind.value,
        "fit_source_type": geometry.source_type,
    }
    if segment_index is not None:
        metadata["fit_segment_index"] = segment_index
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return {
        "stroke": geometry.stroke,
        "fill": geometry.fill,
        "opacity": geometry.opacity,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "error": max(0.0, float(error)) if isfinite(float(error)) else None,
        "metadata": metadata,
    }


def _absolute_tolerance(tolerance: float, scale: float, config: PrimitiveFittingConfig) -> float:
    value = max(0.0, float(tolerance))
    if config.normalize_error_by_scale:
        return value * max(scale, 1.0)
    return value


def _confidence(normalized_error: float, tolerance: float) -> float:
    limit = max(float(tolerance), _EPSILON)
    if not isfinite(float(normalized_error)):
        return 0.0
    return max(0.0, min(1.0, 1.0 - max(0.0, float(normalized_error)) / limit))


def _complexity(kind: PrimitiveFitKind, parameter_count: int, object_count: int) -> float:
    base = {
        PrimitiveFitKind.EXISTING: 0.0,
        PrimitiveFitKind.POINT: 0.0,
        PrimitiveFitKind.LINE: 1.0,
        PrimitiveFitKind.CIRCLE: 2.0,
        PrimitiveFitKind.ELLIPSE: 3.0,
        PrimitiveFitKind.POLYLINE: 4.0,
        PrimitiveFitKind.BEZIER: 5.0,
        PrimitiveFitKind.CLOSED_FREEFORM: 8.0,
        PrimitiveFitKind.GROUP: 10.0,
    }[kind]
    return base + float(parameter_count) + float(object_count) * 0.01


def _kind_order(kind: PrimitiveFitKind) -> int:
    order = [
        PrimitiveFitKind.EXISTING,
        PrimitiveFitKind.POINT,
        PrimitiveFitKind.LINE,
        PrimitiveFitKind.CIRCLE,
        PrimitiveFitKind.ELLIPSE,
        PrimitiveFitKind.POLYLINE,
        PrimitiveFitKind.BEZIER,
        PrimitiveFitKind.CLOSED_FREEFORM,
        PrimitiveFitKind.GROUP,
    ]
    return order.index(kind)


def _parameter_count(primitive: SemanticGeometry) -> int:
    if isinstance(primitive, PointPrimitive):
        return 2
    if isinstance(primitive, LinePrimitive):
        return 4
    if isinstance(primitive, CirclePrimitive):
        return 3
    if isinstance(primitive, EllipsePrimitive):
        return 5
    if isinstance(primitive, BezierPrimitive):
        return 8
    if isinstance(primitive, PolylinePrimitive):
        return 2 * len(primitive.points)
    if isinstance(primitive, ClosedShapePrimitive):
        return 2 * len(primitive.points)
    if isinstance(primitive, PrimitiveGroup):
        return sum(_parameter_count(item) for item in primitive.items)
    return 0


def _primitive_point_count(primitive: SemanticGeometry) -> int:
    if isinstance(primitive, PointPrimitive):
        return 1
    if isinstance(primitive, LinePrimitive):
        return 2
    if isinstance(primitive, (CirclePrimitive, EllipsePrimitive)):
        return 1
    if isinstance(primitive, BezierPrimitive):
        return 4
    if isinstance(primitive, PolylinePrimitive):
        return len(primitive.points)
    if isinstance(primitive, ClosedShapePrimitive):
        return len(primitive.points)
    if isinstance(primitive, PrimitiveGroup):
        return sum(_primitive_point_count(item) for item in primitive.items)
    return 0


def _output_point_count(primitives: Sequence[SemanticGeometry]) -> int:
    return sum(_primitive_point_count(primitive) for primitive in primitives)


def _bezier_count(primitives: Sequence[SemanticGeometry]) -> int:
    count = 0
    for primitive in primitives:
        if isinstance(primitive, BezierPrimitive):
            count += 1
        elif isinstance(primitive, PrimitiveGroup):
            count += _bezier_count(primitive.items)
    return count


def _object_count(primitives: Sequence[SemanticGeometry]) -> int:
    count = 0
    for primitive in primitives:
        if isinstance(primitive, PrimitiveGroup):
            count += _object_count(primitive.items)
        else:
            count += 1
    return count


def _primitive_summary(primitive: SemanticGeometry) -> dict[str, Any]:
    if isinstance(primitive, PrimitiveGroup):
        return {
            "type": "group",
            "name": primitive.name,
            "items": len(primitive.items),
            "flattened_count": len(primitive.flatten()),
            "metadata": _compact_mapping(primitive.metadata),
        }
    summary = {
        "type": _primitive_type(primitive),
        "point_count": _primitive_point_count(primitive),
        "parameter_count": _parameter_count(primitive),
        "confidence": getattr(primitive, "confidence", None),
        "error": getattr(primitive, "error", None),
        "metadata": _compact_mapping(getattr(primitive, "metadata", {})),
    }
    if isinstance(primitive, CirclePrimitive):
        summary["radius"] = primitive.radius
    elif isinstance(primitive, EllipsePrimitive):
        summary["radius_x"] = primitive.radius_x
        summary["radius_y"] = primitive.radius_y
        summary["rotation"] = primitive.rotation
    elif isinstance(primitive, PolylinePrimitive):
        summary["closed"] = primitive.closed
    elif isinstance(primitive, ClosedShapePrimitive):
        summary["closed"] = True
    return summary


def _primitive_type(primitive: SemanticGeometry) -> str:
    return primitive.to_dict().get("type", type(primitive).__name__) if hasattr(primitive, "to_dict") else type(primitive).__name__


def _compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(mapping).items():
        if isinstance(value, Mapping):
            result[key] = _compact_mapping(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result[key] = list(value)[:12]
            if len(value) > 12:
                result[f"{key}_truncated"] = True
        else:
            result[key] = repr(value)
    return result


def _rounded_point(array: np.ndarray, precision: int) -> Point2D:
    return Point2D(round(float(array[0]), precision), round(float(array[1]), precision))


def _round_point(point: Point2D, precision: int) -> Point2D:
    return Point2D(round(point.x, precision), round(point.y, precision))


def _coerce_point(point: Any) -> Point2D:
    if isinstance(point, Point2D):
        return Point2D(point.x, point.y)
    if isinstance(point, Sequence) and len(point) == 2:
        x = float(point[0])
        y = float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise PrimitiveFittingError("point coordinates must be finite")
        return Point2D(x, y)
    raise TypeError("points must be Point2D or x/y pairs")


def _point_distance(first: Point2D, second: Point2D) -> float:
    return hypot(second.x - first.x, second.y - first.y)


def _looks_like_point_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, PrimitiveGroup)):
        return False
    if not isinstance(value, Sequence):
        return False
    if not value:
        return False
    try:
        tuple(_coerce_point(item) for item in value)
    except (TypeError, ValueError, PrimitiveFittingError):
        return False
    return True


def _looks_like_primitive_collection(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if not isinstance(value, Sequence) or not value:
        return False
    return all(isinstance(item, (PointPrimitive, LinePrimitive, PolylinePrimitive, CirclePrimitive, EllipsePrimitive, BezierPrimitive, ClosedShapePrimitive, PrimitiveGroup)) for item in value)


def _looks_like_svg_parse_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "document_info") and hasattr(value, "metrics")


def _looks_like_centerline_path(value: Any) -> bool:
    return hasattr(value, "points") and hasattr(value, "closure") and type(value).__name__ == "CenterlinePath"


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if isfinite(number) else None


def _coerce_closed_handling(value: ClosedPathHandling | str) -> ClosedPathHandling:
    if isinstance(value, ClosedPathHandling):
        return value
    normalized = str(value).strip().lower()
    for item in ClosedPathHandling:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported closed path handling: {value!r}")


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_non_negative_float(name: str, value: float) -> None:
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_ratio(name: str, value: float) -> None:
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _log_result(result: PrimitiveFitResult) -> None:
    log_event("PrimitiveFit", f"source={result.source_type}")
    log_event("PrimitiveFit", f"points={result.input_point_count}")
    log_event("PrimitiveFit", f"candidates={','.join(candidate.kind.value for candidate in result.candidates)}")
    log_event("PrimitiveFit", f"selected={result.selected_kind.value}")
    log_event("PrimitiveFit", f"error={result.normalized_error:.4f}")
    log_event("PrimitiveFit", f"confidence={result.confidence:.2f}")
    log_event("PrimitiveFit", f"input_points={result.input_point_count}")
    if result.metrics.bezier_segment_count:
        log_event("PrimitiveFit", f"output_beziers={result.metrics.bezier_segment_count}")


__all__ = [
    "ClosedPathHandling",
    "FitStatus",
    "PrimitiveCandidate",
    "PrimitiveFitKind",
    "PrimitiveFitMetrics",
    "PrimitiveFitResult",
    "PrimitiveFitWarning",
    "PrimitiveFitter",
    "PrimitiveFittingConfig",
    "PrimitiveFittingError",
    "fit_primitive",
    "fit_primitives",
]
