"""High-level visual validation for isolated semantic Classic outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image

from fikzpy.core.complexity_metrics import ComplexityMetrics, compute_complexity_metrics
from fikzpy.core.fidelity_score import DEFAULT_SCORE_WEIGHTS, FidelityScoreConfig
from fikzpy.core.fidelity_score import FidelityScoreError, FidelityScoreResult, compute_fidelity_score
from fikzpy.core.raster_metrics import RasterComparisonMetrics, RasterMetricsConfig
from fikzpy.core.raster_metrics import image_to_rgb_array, raster_summary
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
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
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

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable metrics."""
        return {
            "raster_metrics": self.raster_metrics.to_dict(),
            "region_metrics": _round_mapping(self.region_metrics),
            "edge_metrics": _round_mapping(self.edge_metrics),
            "filled_region_metrics": _round_mapping(self.filled_region_metrics),
            "thin_stroke_metrics": _round_mapping(self.thin_stroke_metrics),
            "complexity_summary": dict(self.complexity_summary),
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
        source_summary = raster_summary(source_array, self.config.raster_metrics_config())
        rendered_summary = raster_summary(raster_result.image, self.config.raster_metrics_config())
        metrics = _validation_metrics(fidelity.raster_metrics, complexity)
        status = ValidationStatus.ACCEPTED if fidelity.accepted else ValidationStatus.REJECTED
        result_warnings = tuple(warnings) + tuple(
            VisualValidationWarning(flag, f"Regression flag: {flag}", RegressionSeverity.ERROR)
            for flag in fidelity.regression_flags
        )
        payload = {
            "status": status.value,
            "metrics": metrics.to_dict(),
            "fidelity_hash": fidelity.deterministic_hash,
            "complexity": complexity.to_dict() if complexity else None,
            "warnings": [warning.to_dict() for warning in result_warnings],
            "source_summary": source_summary,
            "rendered_summary": rendered_summary,
            "flags": list(fidelity.regression_flags),
            "rejection_reasons": list(fidelity.rejection_reasons),
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
            regression_flags=fidelity.regression_flags,
            accepted=fidelity.accepted,
            rejection_reasons=fidelity.rejection_reasons,
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
        },
        thin_stroke_metrics={
            "thin_stroke_recall": raster.thin_stroke_recall,
            "small_detail_recall": raster.small_detail_recall,
        },
        complexity_summary=complexity_summary,
    )


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


__all__ = [
    "FidelityMetricKind",
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
