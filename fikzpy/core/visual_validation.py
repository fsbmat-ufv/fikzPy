"""High-level visual validation for isolated semantic Classic outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite, pi
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from fikzpy.core.complexity_metrics import ComplexityMetrics, compute_complexity_metrics
from fikzpy.core.fidelity_score import DEFAULT_SCORE_WEIGHTS, FidelityScoreConfig
from fikzpy.core.fidelity_score import FidelityScoreError, FidelityScoreResult, compute_fidelity_score
from fikzpy.core.lineart_diagnostics import LineArtDiagnosticsConfig, LineArtDiagnosticsResult, analyze_line_art_mask
from fikzpy.core.raster_metrics import RasterComparisonMetrics, RasterMetricsConfig
from fikzpy.core.raster_metrics import image_to_rgb_array, raster_summary
from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive, EllipsePrimitive
from fikzpy.core.semantic_geometry import FillStyle, LinePrimitive, Point2D, PointPrimitive, PolylinePrimitive
from fikzpy.core.semantic_geometry import Primitive, PrimitiveGroup, RGBColor
from fikzpy.core.semantic_geometry import SemanticGeometry
from fikzpy.core.semantic_rasterizer import SemanticRasterizationConfig, rasterize_semantic_primitives
from fikzpy.core.semantic_tikz_exporter import TikzExportResult, export_primitives_to_tikz


class VisualValidationError(ValueError):
    """Raised when visual validation cannot continue."""


class ValidationStatus(Enum):
    """Final validation state."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


class FidelityMetricKind(Enum):
    """High-level fidelity score components."""

    VISUAL = "visual"
    FILLED_REGION = "filled_region"
    THIN_STROKE = "thin_stroke"
    SEMANTIC = "semantic"
    READABILITY = "readability"


class RegressionSeverity(Enum):
    """Severity levels for validation warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class VisualValidationConfig:
    """Configuration for semantic rasterization, comparison, and acceptance."""

    target_size: tuple[int, int] | None = None
    preserve_aspect_ratio: bool = True
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
    compute_complexity: bool = True
    compute_tikz_complexity: bool = True
    use_external_tikz_renderer: bool = False
    external_renderer_timeout: float = 10.0
    fail_on_external_renderer_error: bool = False
    minimum_acceptable_score: float = 0.72
    minimum_fidelity_score: float = 0.60
    minimum_filled_region_recall: float = 0.62
    minimum_thin_stroke_recall: float = 0.45
    maximum_complexity_ratio: float = 4.0
    validate_lineart_fill_usage: bool = True
    reject_overfilled_lineart: bool = True
    lineart_source_confidence_threshold: float = 0.55
    max_filled_area_ratio_for_lineart: float = 0.08
    max_white_cutout_ratio_for_lineart: float = 0.20
    max_white_cutout_count_for_lineart: int = 3
    max_dark_mass_growth_for_lineart: float = 1.65
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    strict: bool = False
    deterministic: bool = True
    save_debug_images: bool = False
    debug_output_dir: str | Path | None = None

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
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        for name in (
            "preserve_aspect_ratio",
            "compare_edges",
            "compare_filled_regions",
            "compare_thin_strokes",
            "compute_complexity",
            "compute_tikz_complexity",
            "use_external_tikz_renderer",
            "fail_on_external_renderer_error",
            "validate_lineart_fill_usage",
            "reject_overfilled_lineart",
            "strict",
            "deterministic",
            "save_debug_images",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        for name in (
            "thin_stroke_max_width",
            "external_renderer_timeout",
            "minimum_acceptable_score",
            "minimum_fidelity_score",
            "minimum_filled_region_recall",
            "minimum_thin_stroke_recall",
            "maximum_complexity_ratio",
            "lineart_source_confidence_threshold",
            "max_filled_area_ratio_for_lineart",
            "max_white_cutout_ratio_for_lineart",
            "max_dark_mass_growth_for_lineart",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if int(self.max_white_cutout_count_for_lineart) < 0:
            raise ValueError("max_white_cutout_count_for_lineart must be non-negative.")
        object.__setattr__(self, "max_white_cutout_count_for_lineart", int(self.max_white_cutout_count_for_lineart))
        if self.save_debug_images and self.debug_output_dir is None:
            raise ValueError("debug_output_dir is required when save_debug_images=True.")

    def fidelity_config(self) -> FidelityScoreConfig:
        """Return the fidelity-score configuration equivalent to this validation config."""
        return FidelityScoreConfig(
            foreground_threshold=self.foreground_threshold,
            dark_pixel_threshold=self.dark_pixel_threshold,
            edge_threshold=self.edge_threshold,
            filled_region_min_area=self.filled_region_min_area,
            thin_stroke_max_width=self.thin_stroke_max_width,
            small_detail_max_area=self.small_detail_max_area,
            component_connectivity=self.component_connectivity,
            compare_edges=self.compare_edges,
            compare_filled_regions=self.compare_filled_regions,
            compare_thin_strokes=self.compare_thin_strokes,
            minimum_acceptable_score=self.minimum_acceptable_score,
            minimum_fidelity_score=self.minimum_fidelity_score,
            minimum_filled_region_recall=self.minimum_filled_region_recall,
            minimum_thin_stroke_recall=self.minimum_thin_stroke_recall,
            maximum_complexity_ratio=self.maximum_complexity_ratio,
            weights=self.weights,
            strict=self.strict,
            deterministic=self.deterministic,
        )

    def raster_metrics_config(self) -> RasterMetricsConfig:
        """Return the raster-summary configuration for this validation config."""
        return RasterMetricsConfig(
            target_size=self.target_size,
            background_color=self.background_color,
            foreground_threshold=self.foreground_threshold,
            dark_pixel_threshold=self.dark_pixel_threshold,
            edge_threshold=self.edge_threshold,
            filled_region_min_area=self.filled_region_min_area,
            thin_stroke_max_width=self.thin_stroke_max_width,
            small_detail_max_area=self.small_detail_max_area,
            component_connectivity=self.component_connectivity,
            compare_edges=self.compare_edges,
            compare_filled_regions=self.compare_filled_regions,
            compare_thin_strokes=self.compare_thin_strokes,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        data = dict(self.__dict__)
        data["target_size"] = list(self.target_size) if self.target_size else None
        data["background_color"] = list(self.background_color)
        data["debug_output_dir"] = str(self.debug_output_dir) if self.debug_output_dir is not None else None
        data["weights"] = dict(sorted((str(key), float(value)) for key, value in self.weights.items()))
        return data


@dataclass(frozen=True)
class VisualValidationWarning:
    """Structured warning emitted by semantic visual validation."""

    code: str
    message: str
    severity: RegressionSeverity = RegressionSeverity.WARNING

    def to_dict(self) -> dict[str, str]:
        """Return serializable warning diagnostics."""
        return {"code": self.code, "message": self.message, "severity": self.severity.value}


@dataclass(frozen=True)
class VisualValidationMetrics:
    """Grouped metrics exposed by high-level validation."""

    raster_metrics: RasterComparisonMetrics
    region_metrics: Mapping[str, float]
    edge_metrics: Mapping[str, float]
    filled_region_metrics: Mapping[str, float]
    thin_stroke_metrics: Mapping[str, float]
    complexity_summary: Mapping[str, Any]
    lineart_fill_metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        return {
            "raster_metrics": self.raster_metrics.to_dict(),
            "region_metrics": _round_mapping(self.region_metrics),
            "edge_metrics": _round_mapping(self.edge_metrics),
            "filled_region_metrics": _round_mapping(self.filled_region_metrics),
            "thin_stroke_metrics": _round_mapping(self.thin_stroke_metrics),
            "complexity_summary": dict(self.complexity_summary),
            "lineart_fill_metrics": _round_mapping(self.lineart_fill_metrics),
        }


@dataclass(frozen=True)
class VisualValidationResult:
    """Visual validation output, scores, warnings, and acceptance state."""

    status: ValidationStatus
    metrics: VisualValidationMetrics
    fidelity_score: FidelityScoreResult
    complexity_metrics: ComplexityMetrics | None
    warnings: tuple[VisualValidationWarning, ...]
    source_summary: Mapping[str, Any]
    rendered_summary: Mapping[str, Any]
    regression_flags: tuple[str, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    deterministic_hash: str
    config: VisualValidationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable validation diagnostics."""
        return {
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "fidelity_score": self.fidelity_score.to_dict(),
            "complexity_metrics": self.complexity_metrics.to_dict() if self.complexity_metrics else None,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "source_summary": dict(self.source_summary),
            "rendered_summary": dict(self.rendered_summary),
            "regression_flags": list(self.regression_flags),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "deterministic_hash": self.deterministic_hash,
            "config": self.config.to_dict(),
        }


@dataclass(frozen=True)
class ValidationCase:
    """One serializable validation report case."""

    name: str
    result: VisualValidationResult
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return report-friendly case diagnostics."""
        data = {"name": self.name, "description": self.description}
        data.update(self.result.to_dict())
        return data


@dataclass(frozen=True)
class ValidationReport:
    """Collection of deterministic validation report cases."""

    issue: str
    cases: tuple[ValidationCase, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return report-friendly diagnostics."""
        return {"issue": self.issue, "cases": [case.to_dict() for case in self.cases]}


@dataclass(frozen=True)
class LineArtFillUsageDiagnostics:
    """Diagnostic checks for filled-area overuse on line-art sources."""

    line_art_confidence: float
    source_is_line_art: bool
    filled_area_ratio: float
    black_fill_area_ratio: float
    white_cutout_count: int
    white_cutout_area_ratio: float
    white_cutout_to_black_fill_ratio: float
    dark_mass_growth_ratio: float
    regression_flags: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    diagnostics: LineArtDiagnosticsResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic fill-usage diagnostics."""
        return {
            "line_art_confidence": _rounded(self.line_art_confidence),
            "source_is_line_art": self.source_is_line_art,
            "filled_area_ratio": _rounded(self.filled_area_ratio),
            "black_fill_area_ratio": _rounded(self.black_fill_area_ratio),
            "white_cutout_count": self.white_cutout_count,
            "white_cutout_area_ratio": _rounded(self.white_cutout_area_ratio),
            "white_cutout_to_black_fill_ratio": _rounded(self.white_cutout_to_black_fill_ratio),
            "dark_mass_growth_ratio": _rounded(self.dark_mass_growth_ratio),
            "regression_flags": list(self.regression_flags),
            "rejection_reasons": list(self.rejection_reasons),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics is not None else None,
        }


class VisualValidator:
    """Validate semantic primitives against a source raster."""

    def __init__(self, config: VisualValidationConfig | None = None) -> None:
        self.config = config or VisualValidationConfig()

    def validate(
        self,
        source_image: Any,
        primitives: Any,
        tikz_result: TikzExportResult | None = None,
    ) -> VisualValidationResult:
        """Validate one source image and semantic primitive candidate."""
        warnings: list[VisualValidationWarning] = []
        source_array = image_to_rgb_array(
            source_image,
            target_size=self.config.target_size,
            background_color=self.config.background_color,
        )
        canvas_size = (int(source_array.shape[1]), int(source_array.shape[0]))
        if self.config.use_external_tikz_renderer:
            warning = VisualValidationWarning(
                "external_renderer_unavailable",
                "External TikZ rendering is optional and is not required by this validator.",
                RegressionSeverity.INFO,
            )
            if self.config.fail_on_external_renderer_error:
                if self.config.strict:
                    raise VisualValidationError(warning.message)
                warnings.append(warning)
            else:
                warnings.append(warning)

        raster_result = rasterize_semantic_primitives(
            primitives,
            SemanticRasterizationConfig(
                canvas_size=canvas_size,
                padding=0,
                background_color=self.config.background_color,
                strict=self.config.strict,
            ),
        )
        warnings.extend(
            VisualValidationWarning("rasterizer_warning", warning, RegressionSeverity.WARNING)
            for warning in raster_result.warnings
        )

        code = None
        if tikz_result is not None:
            code = tikz_result.code
        elif self.config.compute_tikz_complexity:
            try:
                tikz_result = export_primitives_to_tikz(primitives)
                code = tikz_result.code
                warnings.extend(
                    VisualValidationWarning("tikz_export_warning", warning.message, RegressionSeverity.WARNING)
                    for warning in tikz_result.warnings
                )
            except Exception as exc:
                if self.config.strict:
                    raise VisualValidationError(str(exc)) from exc
                warnings.append(VisualValidationWarning("tikz_export_unavailable", str(exc), RegressionSeverity.WARNING))

        complexity = None
        if self.config.compute_complexity or self.config.compute_tikz_complexity:
            complexity = compute_complexity_metrics(
                primitives if self.config.compute_complexity else None,
                tikz_code=code if self.config.compute_tikz_complexity else None,
            )

        try:
            fidelity = compute_fidelity_score(source_array, raster_result.image, complexity, self.config.fidelity_config())
        except FidelityScoreError as exc:
            raise VisualValidationError(str(exc)) from exc
        fill_usage = _lineart_fill_usage(source_array, primitives, code or "", fidelity.raster_metrics, self.config)
        regression_flags = tuple(dict.fromkeys((*fidelity.regression_flags, *fill_usage.regression_flags)))
        rejection_reasons = tuple(dict.fromkeys((*fidelity.rejection_reasons, *fill_usage.rejection_reasons)))
        if fill_usage.source_is_line_art and not fill_usage.rejection_reasons:
            regression_flags, rejection_reasons = _suppress_filled_region_rejections_for_lineart(
                regression_flags,
                rejection_reasons,
                fidelity.raster_metrics,
                self.config,
            )
        accepted = fidelity.accepted and not fill_usage.rejection_reasons
        if fill_usage.source_is_line_art and not rejection_reasons and fidelity.raster_metrics.rendered_foreground_pixels > 0:
            accepted = True
        source_summary = raster_summary(source_array, self.config.raster_metrics_config())
        rendered_summary = raster_summary(raster_result.image, self.config.raster_metrics_config())
        metrics = _validation_metrics(fidelity.raster_metrics, complexity, fill_usage)
        status = ValidationStatus.ACCEPTED if accepted else ValidationStatus.REJECTED
        result_warnings = tuple(warnings) + tuple(
            VisualValidationWarning(flag, f"Regression flag: {flag}", RegressionSeverity.ERROR)
            for flag in regression_flags
        )
        payload = {
            "status": status.value,
            "metrics": metrics.to_dict(),
            "fidelity_hash": fidelity.deterministic_hash,
            "complexity": complexity.to_dict() if complexity else None,
            "warnings": [warning.to_dict() for warning in result_warnings],
            "source_summary": source_summary,
            "rendered_summary": rendered_summary,
            "flags": list(regression_flags),
            "rejection_reasons": list(rejection_reasons),
            "config": self.config.to_dict(),
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        result = VisualValidationResult(
            status=status,
            metrics=metrics,
            fidelity_score=fidelity,
            complexity_metrics=complexity,
            warnings=result_warnings,
            source_summary=source_summary,
            rendered_summary=rendered_summary,
            regression_flags=regression_flags,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            deterministic_hash=digest,
            config=self.config,
        )
        if self.config.save_debug_images:
            _save_debug_images(self.config.debug_output_dir, source_array, raster_result.image)
        if self.config.strict and not result.accepted:
            raise VisualValidationError("; ".join(result.rejection_reasons))
        return result


def validate_semantic_output(
    source_image: Any,
    primitives: Any,
    tikz_result: TikzExportResult | None = None,
    config: VisualValidationConfig | None = None,
) -> VisualValidationResult:
    """Validate semantic primitives against a source image without GUI integration."""
    return VisualValidator(config).validate(source_image, primitives, tikz_result)


def _validation_metrics(
    raster: RasterComparisonMetrics,
    complexity: ComplexityMetrics | None,
    lineart_fill_usage: LineArtFillUsageDiagnostics | None = None,
) -> VisualValidationMetrics:
    complexity_summary = {
        "primitive_count": complexity.primitive_count if complexity else 0,
        "tikz_characters": complexity.tikz_characters if complexity else 0,
        "complexity_score": complexity.complexity_score if complexity else 1.0,
        "semantic_compactness_score": complexity.semantic_compactness_score if complexity else 1.0,
        "editability_score": complexity.editability_score if complexity else 1.0,
    }
    return VisualValidationMetrics(
        raster_metrics=raster,
        region_metrics={
            "foreground_iou": raster.foreground_iou,
            "foreground_precision": raster.foreground_precision,
            "foreground_recall": raster.foreground_recall,
            "foreground_f1": raster.foreground_f1,
            "dark_mass_preservation_ratio": raster.dark_mass_preservation_ratio,
            "foreground_fragmentation_delta": raster.foreground_fragmentation_delta,
        },
        edge_metrics={
            "edge_overlap": raster.edge_overlap,
            "edge_recall": raster.edge_recall,
            "edge_precision": raster.edge_precision,
            "edge_f1": raster.edge_f1,
        },
        filled_region_metrics={
            "filled_region_recall": raster.filled_region_recall,
            "large_dark_region_recall": raster.large_dark_region_recall,
            "double_outline_penalty": raster.double_outline_penalty,
            "filled_area_ratio": lineart_fill_usage.filled_area_ratio if lineart_fill_usage else 0.0,
            "white_cutout_count": lineart_fill_usage.white_cutout_count if lineart_fill_usage else 0,
            "white_cutout_area_ratio": lineart_fill_usage.white_cutout_area_ratio if lineart_fill_usage else 0.0,
            "white_cutout_to_black_fill_ratio": lineart_fill_usage.white_cutout_to_black_fill_ratio if lineart_fill_usage else 0.0,
        },
        thin_stroke_metrics={
            "thin_stroke_recall": raster.thin_stroke_recall,
            "small_detail_recall": raster.small_detail_recall,
        },
        complexity_summary=complexity_summary,
        lineart_fill_metrics=lineart_fill_usage.to_dict() if lineart_fill_usage else {},
    )


def _lineart_fill_usage(
    source_array: np.ndarray,
    primitives: Any,
    tikz_code: str,
    raster: RasterComparisonMetrics,
    config: VisualValidationConfig,
) -> LineArtFillUsageDiagnostics:
    if not config.validate_lineart_fill_usage:
        return LineArtFillUsageDiagnostics(0.0, False, 0.0, 0.0, 0, 0.0, 0.0, 1.0, (), ())
    gray = _gray(source_array)
    source_mask = gray <= int(config.dark_pixel_threshold)
    diagnostics = analyze_line_art_mask(
        source_mask,
        LineArtDiagnosticsConfig(
            component_connectivity=config.component_connectivity,
            thin_stroke_max_width=config.thin_stroke_max_width,
            filled_region_min_area=config.filled_region_min_area,
        ),
    )
    source_is_line_art = (
        diagnostics.line_art_confidence >= config.lineart_source_confidence_threshold
        and diagnostics.solid_component_area_ratio <= 0.02
    )
    items = _normalize_validation_primitives(primitives)
    black_area, white_area, primitive_white_count, primitive_black_count = _filled_area_by_color(items)
    code_white_count = _count_white_cutouts_in_tikz(tikz_code)
    code_black_count = _count_black_fills_in_tikz(tikz_code)
    white_count = max(primitive_white_count, code_white_count)
    black_count = max(primitive_black_count, code_black_count)
    canvas_area = max(1.0, float(source_array.shape[0] * source_array.shape[1]))
    black_ratio = black_area / canvas_area
    white_ratio = white_area / canvas_area
    white_to_black = white_area / max(black_area, 1.0)
    dark_growth = raster.rendered_foreground_ratio / max(raster.source_foreground_ratio, 1e-9)
    flags: list[str] = []
    reasons: list[str] = []
    if source_is_line_art:
        excessive_fill = (
            black_ratio > config.max_filled_area_ratio_for_lineart
            or (
                black_count > 0
                and raster.rendered_foreground_ratio > raster.source_foreground_ratio * config.max_dark_mass_growth_for_lineart
            )
        )
        excessive_cutouts = (
            white_count > config.max_white_cutout_count_for_lineart
            or white_to_black > config.max_white_cutout_ratio_for_lineart
        )
        if excessive_fill:
            flags.extend(("excessive_filled_area", "artificial_black_mass", "overfilled_lineart"))
            reasons.append("lineart_has_excessive_filled_area")
        if excessive_cutouts:
            flags.append("excessive_white_cutouts")
            reasons.append("lineart_uses_excessive_white_cutouts")
        if black_count > 0 and dark_growth > config.max_dark_mass_growth_for_lineart:
            flags.append("lineart_converted_to_silhouette")
            reasons.append("lineart_converted_to_silhouette")
    if not config.reject_overfilled_lineart:
        reasons = []
    return LineArtFillUsageDiagnostics(
        line_art_confidence=diagnostics.line_art_confidence,
        source_is_line_art=source_is_line_art,
        filled_area_ratio=black_ratio,
        black_fill_area_ratio=black_ratio,
        white_cutout_count=white_count,
        white_cutout_area_ratio=white_ratio,
        white_cutout_to_black_fill_ratio=white_to_black,
        dark_mass_growth_ratio=dark_growth,
        regression_flags=tuple(dict.fromkeys(flags)),
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        diagnostics=diagnostics,
    )


def _suppress_filled_region_rejections_for_lineart(
    flags: tuple[str, ...],
    reasons: tuple[str, ...],
    raster: RasterComparisonMetrics,
    config: VisualValidationConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if raster.rendered_foreground_pixels <= 0 or raster.thin_stroke_recall < config.minimum_thin_stroke_recall:
        return flags, reasons
    suppressed_flags = {
        "dark_mass_loss",
        "missing_large_dark_regions",
        "low_filled_region_recall",
        "over_simplified",
    }
    suppressed_reasons = {
        "overall_score_below_minimum",
        "fidelity_score_below_minimum",
        "foreground_iou_below_minimum",
        "filled_region_score_below_minimum",
        "dark_regions_lost",
    }
    return (
        tuple(flag for flag in flags if flag not in suppressed_flags),
        tuple(reason for reason in reasons if reason not in suppressed_reasons),
    )


def _normalize_validation_primitives(value: Any) -> tuple[SemanticGeometry, ...]:
    if value is None:
        return ()
    if isinstance(value, PrimitiveGroup):
        return (value,)
    if isinstance(value, _PRIMITIVE_TYPES):
        return (value,)
    if hasattr(value, "primitives"):
        return _normalize_validation_primitives(tuple(value.primitives))
    if hasattr(value, "paths") and type(value).__name__ == "CenterlineResult":
        return tuple(path.to_polyline_primitive() for path in value.paths)
    if isinstance(value, (list, tuple)):
        output: list[SemanticGeometry] = []
        for item in value:
            output.extend(_normalize_validation_primitives(item))
        return tuple(output)
    return ()


def _filled_area_by_color(items: tuple[SemanticGeometry, ...]) -> tuple[float, float, int, int]:
    black_area = 0.0
    white_area = 0.0
    white_count = 0
    black_count = 0
    for item in _flatten_primitives(items):
        fill = getattr(item, "fill", None)
        if not isinstance(fill, FillStyle) or not _visible_fill(fill):
            continue
        area = _primitive_fill_area(item)
        if _is_white(fill.color):
            white_area += area
            white_count += 1
        elif _is_black(fill.color):
            black_area += area
            black_count += 1
    return black_area, white_area, white_count, black_count


def _flatten_primitives(items: tuple[SemanticGeometry, ...]) -> tuple[Primitive, ...]:
    output: list[Primitive] = []
    for item in items:
        if isinstance(item, PrimitiveGroup):
            output.extend(_flatten_primitives(tuple(item.items)))
        elif isinstance(item, _PRIMITIVE_TYPES):
            output.append(item)
    return tuple(output)


def _primitive_fill_area(primitive: Primitive) -> float:
    if isinstance(primitive, ClosedShapePrimitive):
        return abs(_polygon_area(tuple(primitive.points)))
    if isinstance(primitive, PolylinePrimitive) and primitive.closed:
        return abs(_polygon_area(tuple(primitive.points)))
    if isinstance(primitive, CirclePrimitive):
        return pi * primitive.radius * primitive.radius
    if isinstance(primitive, EllipsePrimitive):
        return pi * primitive.radius_x * primitive.radius_y
    return 0.0


def _polygon_area(points: tuple[Point2D, ...]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for first, second in zip(points, (*points[1:], points[0]), strict=False):
        area += first.x * second.y - second.x * first.y
    return area / 2.0


def _visible_fill(fill: FillStyle) -> bool:
    return fill.opacity is None or fill.opacity > 0.0


def _is_white(color: RGBColor) -> bool:
    return color.red >= 240 and color.green >= 240 and color.blue >= 240


def _is_black(color: RGBColor) -> bool:
    return color.red <= 32 and color.green <= 32 and color.blue <= 32


def _count_white_cutouts_in_tikz(code: str) -> int:
    if not code:
        return 0
    direct = len(re.findall(r"fill\s*=\s*white\b", code, flags=re.IGNORECASE))
    rgb = len(re.findall(r"fill\s*=\s*\{rgb,255:red,255;green,255;blue,255\}", code, flags=re.IGNORECASE))
    draw_none_white = len(re.findall(r"draw\s*=\s*none[^\]]*fill\s*=\s*white", code, flags=re.IGNORECASE))
    return max(direct + rgb, draw_none_white)


def _count_black_fills_in_tikz(code: str) -> int:
    if not code:
        return 0
    direct = len(re.findall(r"fill\s*=\s*black\b", code, flags=re.IGNORECASE))
    rgb = len(re.findall(r"fill\s*=\s*\{rgb,255:red,0;green,0;blue,0\}", code, flags=re.IGNORECASE))
    return direct + rgb


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _save_debug_images(debug_output_dir: str | Path | None, source: Any, rendered: Any) -> None:
    if debug_output_dir is None:
        raise VisualValidationError("debug_output_dir is required when saving debug images.")
    directory = Path(debug_output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source).save(directory / "source.png")
    Image.fromarray(rendered).save(directory / "rendered.png")


def _round_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in sorted(mapping.items()):
        if isinstance(value, float):
            output[str(key)] = _rounded(value)
        else:
            output[str(key)] = value
    return output


def _coerce_rgb_tuple(name: str, value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three RGB channels.")
    channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"{name} channels must be between 0 and 255.")
    return channels


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


_PRIMITIVE_TYPES = (
    PointPrimitive,
    LinePrimitive,
    PolylinePrimitive,
    CirclePrimitive,
    EllipsePrimitive,
    BezierPrimitive,
    ClosedShapePrimitive,
)


__all__ = [
    "FidelityMetricKind",
    "LineArtFillUsageDiagnostics",
    "RegressionSeverity",
    "ValidationCase",
    "ValidationReport",
    "ValidationStatus",
    "VisualValidationConfig",
    "VisualValidationError",
    "VisualValidationMetrics",
    "VisualValidationResult",
    "VisualValidationWarning",
    "VisualValidator",
    "validate_semantic_output",
]
