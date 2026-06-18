"""Semantic Classic pipeline orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fikzpy.core.adaptive_preprocessing import PreprocessingResult, preprocess_image
from fikzpy.core.diagnostics import log_event
from fikzpy.core.filled_region_extraction import FilledRegionExtractionConfig, FilledRegionExtractionError
from fikzpy.core.filled_region_extraction import FilledRegionExtractionResult
from fikzpy.core.filled_region_extraction import extract_filled_regions
from fikzpy.core.geometry_optimization import GeometryOptimizationResult, optimize_fit_results
from fikzpy.core.image_classifier import ImageCategory, ImageClassificationResult, classify_image
from fikzpy.core.mixed_monochrome_pipeline import ForegroundLayerSplit, MixedMonochromeResult
from fikzpy.core.mixed_monochrome_pipeline import extract_mixed_monochrome_primitives, extract_thin_strokes
from fikzpy.core.mixed_monochrome_pipeline import split_foreground_layers
from fikzpy.core.primitive_fitting import PrimitiveFitResult, fit_primitives
from fikzpy.core.semantic_geometry import PrimitiveGroup, SemanticGeometry
from fikzpy.core.semantic_tikz_exporter import TikzExportResult, export_primitives_to_tikz
from fikzpy.core.visual_validation import ValidationStatus, VisualValidationResult, validate_semantic_output
from fikzpy.core.classic_pipeline_config import ClassicFallbackPolicy, ClassicPipelineStatus, ClassicPipelineStrategy
from fikzpy.core.classic_pipeline_config import ClassicSemanticConfig, ClassicValidationPolicy


class ClassicSemanticPipelineError(RuntimeError):
    """Raised when the semantic Classic pipeline cannot produce a result."""


@dataclass(frozen=True)
class ClassicSemanticWarning:
    """Structured warning emitted by the semantic Classic pipeline."""

    code: str
    message: str
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable warning diagnostics."""
        return {"code": self.code, "message": self.message, "stage": self.stage}


@dataclass(frozen=True)
class ClassicPipelineDecision:
    """Strategy decision and diagnostics for Classic extraction."""

    strategy: ClassicPipelineStrategy
    reason: str
    split: ForegroundLayerSplit | None
    category: ImageCategory
    foreground_ratio: float
    dark_pixel_ratio: float
    colored_pixel_ratio: float
    edge_to_foreground_ratio: float
    warnings: tuple[ClassicSemanticWarning, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic decision diagnostics."""
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "split": self.split.to_dict() if self.split is not None else None,
            "category": self.category.value,
            "foreground_ratio": self.foreground_ratio,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "colored_pixel_ratio": self.colored_pixel_ratio,
            "edge_to_foreground_ratio": self.edge_to_foreground_ratio,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class ClassicPipelineStageResult:
    """Diagnostic wrapper for one semantic Classic stage."""

    name: str
    status: ClassicPipelineStatus
    details: Mapping[str, Any]
    warnings: tuple[ClassicSemanticWarning, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return serializable stage diagnostics."""
        return {
            "name": self.name,
            "status": self.status.value,
            "details": _compact_value(self.details),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class ClassicSemanticMetrics:
    """Scalar metrics for semantic Classic output."""

    input_image_size: tuple[int, int]
    image_category: str
    strategy_used: str
    foreground_ratio: float
    dark_pixel_ratio: float
    thin_stroke_primitives: int
    filled_region_primitives: int
    raw_primitive_count: int
    fitted_primitive_count: int
    optimized_primitive_count: int
    tikz_draw_commands: int
    tikz_code_characters: int
    validation_score: float
    filled_region_recall: float
    thin_stroke_recall: float
    dark_mass_preservation_ratio: float
    processing_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable metrics."""
        return {
            "input_image_size": list(self.input_image_size),
            "image_category": self.image_category,
            "strategy_used": self.strategy_used,
            "foreground_ratio": self.foreground_ratio,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "thin_stroke_primitives": self.thin_stroke_primitives,
            "filled_region_primitives": self.filled_region_primitives,
            "raw_primitive_count": self.raw_primitive_count,
            "fitted_primitive_count": self.fitted_primitive_count,
            "optimized_primitive_count": self.optimized_primitive_count,
            "tikz_draw_commands": self.tikz_draw_commands,
            "tikz_code_characters": self.tikz_code_characters,
            "validation_score": self.validation_score,
            "filled_region_recall": self.filled_region_recall,
            "thin_stroke_recall": self.thin_stroke_recall,
            "dark_mass_preservation_ratio": self.dark_mass_preservation_ratio,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass(frozen=True)
class ClassicSemanticResult:
    """Complete semantic Classic pipeline result."""

    tikz_code: str
    strategy_used: ClassicPipelineStrategy
    classification_result: ImageClassificationResult
    preprocessing_result: PreprocessingResult
    raw_primitives: tuple[SemanticGeometry, ...]
    fitted_primitives: tuple[SemanticGeometry, ...]
    optimized_primitives: tuple[SemanticGeometry, ...]
    tikz_export_result: TikzExportResult
    validation_result: VisualValidationResult | None
    metrics: ClassicSemanticMetrics
    warnings: tuple[ClassicSemanticWarning, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    deterministic_hash: str
    status: ClassicPipelineStatus
    decision: ClassicPipelineDecision
    fitting_results: tuple[PrimitiveFitResult, ...]
    optimization_result: GeometryOptimizationResult
    extraction_result: Any | None = None
    stage_results: tuple[ClassicPipelineStageResult, ...] = ()

    @property
    def primitives_raw(self) -> tuple[SemanticGeometry, ...]:
        """Backward-compatible alias for raw primitives."""
        return self.raw_primitives

    @property
    def primitives_fitted(self) -> tuple[SemanticGeometry, ...]:
        """Backward-compatible alias for fitted primitives."""
        return self.fitted_primitives

    @property
    def primitives_optimized(self) -> tuple[SemanticGeometry, ...]:
        """Backward-compatible alias for optimized primitives."""
        return self.optimized_primitives

    @property
    def tikz_result(self) -> TikzExportResult:
        """Backward-compatible alias for the TikZ exporter result."""
        return self.tikz_export_result

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without full image arrays."""
        return {
            "tikz_code": self.tikz_code,
            "strategy_used": self.strategy_used.value,
            "classification_result": self.classification_result.to_dict(),
            "preprocessing_result": self.preprocessing_result.to_dict(),
            "raw_primitives": [primitive.to_dict() for primitive in self.raw_primitives],
            "fitted_primitives": [primitive.to_dict() for primitive in self.fitted_primitives],
            "optimized_primitives": [primitive.to_dict() for primitive in self.optimized_primitives],
            "tikz_export_result": self.tikz_export_result.to_dict(),
            "validation_result": self.validation_result.to_dict() if self.validation_result is not None else None,
            "metrics": self.metrics.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "deterministic_hash": self.deterministic_hash,
            "status": self.status.value,
            "decision": self.decision.to_dict(),
            "fitting_results": [result.to_dict() for result in self.fitting_results],
            "optimization_result": self.optimization_result.to_dict(),
            "extraction_result": _to_dict_if_possible(self.extraction_result),
            "stage_results": [stage.to_dict() for stage in self.stage_results],
        }


class ClassicSemanticPipeline:
    """Run the isolated semantic Classic image-to-TikZ pipeline."""

    def __init__(self, config: ClassicSemanticConfig | None = None) -> None:
        self.config = config or ClassicSemanticConfig()

    def run(self, image: Any) -> ClassicSemanticResult:
        """Run the semantic Classic pipeline for a path, PIL image, or ndarray."""
        if not self.config.enable_semantic_classic:
            raise ClassicSemanticPipelineError("Semantic Classic pipeline is disabled by configuration.")
        source = _normalize_image(image)
        warnings: list[ClassicSemanticWarning] = []
        stages: list[ClassicPipelineStageResult] = []
        try:
            classification = classify_image(source, self.config.image_classifier_config)
            stages.append(_stage("classify", {"category": classification.category.value, "confidence": classification.confidence}))
            preprocessing = preprocess_image(source, self.config.preprocessing_config, category=classification.category)
            stages.append(
                _stage(
                    "preprocess",
                    {
                        "foreground_ratio": preprocessing.metrics.foreground_ratio,
                        "component_count": preprocessing.metrics.component_count,
                        "method": preprocessing.method,
                    },
                )
            )
            decision = self._decide_strategy(classification, preprocessing)
            warnings.extend(decision.warnings)
            stages.append(_stage("decide", {"strategy": decision.strategy.value, "reason": decision.reason}))
            extraction, raw_primitives = self._extract_primitives(preprocessing.binary_mask, decision)
            extraction_warnings = _warnings_from_extraction(extraction)
            warnings.extend(extraction_warnings)
            stages.append(_stage("extract", {"raw_primitives": _object_count(raw_primitives)}, extraction_warnings))

            fitting_results = tuple(fit_primitives(raw_primitives, self.config.fitting_config))
            fitted_primitives = tuple(primitive for result in fitting_results for primitive in result.primitives)
            fitting_warnings = _warnings_from_fitting(fitting_results)
            warnings.extend(fitting_warnings)
            stages.append(_stage("fit", {"fitted_primitives": _object_count(fitted_primitives)}, fitting_warnings))

            optimization_result = optimize_fit_results(fitting_results, self.config.optimization_config)
            optimized_primitives = tuple(optimization_result.primitives)
            optimization_warnings = _warnings_from_optimization(optimization_result)
            warnings.extend(optimization_warnings)
            stages.append(
                _stage(
                    "optimize",
                    {"optimized_primitives": _object_count(optimized_primitives), "success": optimization_result.success},
                    optimization_warnings,
                )
            )

            tikz_result = export_primitives_to_tikz(optimized_primitives, self.config.tikz_export_config)
            tikz_warnings = _warnings_from_tikz(tikz_result)
            warnings.extend(tikz_warnings)
            stages.append(_stage("export", {"draw_commands": tikz_result.metrics.draw_commands}, tikz_warnings))

            validation_result: VisualValidationResult | None = None
            if self.config.validation_policy is not ClassicValidationPolicy.DISABLED:
                validation_result = validate_semantic_output(
                    source,
                    optimized_primitives,
                    tikz_result,
                    self.config.validation_config(),
                )
                validation_warnings = _warnings_from_validation(validation_result)
                warnings.extend(validation_warnings)
                stages.append(
                    _stage(
                        "validate",
                        {
                            "accepted": validation_result.accepted,
                            "score": validation_result.fidelity_score.overall_score,
                        },
                        validation_warnings,
                        status=ClassicPipelineStatus.ACCEPTED if validation_result.accepted else ClassicPipelineStatus.REJECTED,
                    )
                )

            accepted, rejection_reasons = self._acceptance(validation_result, tikz_result)
            status = ClassicPipelineStatus.ACCEPTED if accepted else ClassicPipelineStatus.REJECTED
            if not accepted and self.config.fallback_policy is ClassicFallbackPolicy.LEGACY_EXPLICIT:
                warnings.append(
                    ClassicSemanticWarning(
                        "legacy_fallback_not_invoked",
                        "Legacy fallback is explicit but not invoked by the isolated core pipeline.",
                        "fallback",
                    )
                )

            metrics = _metrics(
                source,
                classification,
                decision,
                raw_primitives,
                fitted_primitives,
                optimized_primitives,
                tikz_result,
                validation_result,
            )
            payload = {
                "strategy": decision.strategy.value,
                "classification": classification.to_dict(),
                "preprocessing": preprocessing.to_dict(),
                "raw": [primitive.to_dict() for primitive in raw_primitives],
                "fitted": [primitive.to_dict() for primitive in fitted_primitives],
                "optimization": optimization_result.to_dict(),
                "tikz": tikz_result.to_dict(),
                "validation": validation_result.to_dict() if validation_result is not None else None,
                "metrics": metrics.to_dict(),
                "warnings": [warning.to_dict() for warning in warnings],
                "accepted": accepted,
                "rejection_reasons": list(rejection_reasons),
            }
            digest = _hash_payload(payload)
            result = ClassicSemanticResult(
                tikz_code=tikz_result.code,
                strategy_used=decision.strategy,
                classification_result=classification,
                preprocessing_result=preprocessing,
                raw_primitives=raw_primitives,
                fitted_primitives=fitted_primitives,
                optimized_primitives=optimized_primitives,
                tikz_export_result=tikz_result,
                validation_result=validation_result,
                metrics=metrics,
                warnings=tuple(warnings),
                accepted=accepted,
                rejection_reasons=rejection_reasons,
                deterministic_hash=digest,
                status=status,
                decision=decision,
                fitting_results=fitting_results,
                optimization_result=optimization_result,
                extraction_result=extraction,
                stage_results=tuple(stages),
            )
            _log_result(result)
            return result
        except Exception as exc:
            if self.config.strict:
                raise ClassicSemanticPipelineError(str(exc)) from exc
            raise

    def _decide_strategy(
        self,
        classification: ImageClassificationResult,
        preprocessing: PreprocessingResult,
    ) -> ClassicPipelineDecision:
        metrics = classification.metrics
        split: ForegroundLayerSplit | None = None
        warnings: list[ClassicSemanticWarning] = []
        if self.config.prefer_external_filled_region_backend and not self.config.allow_external_tracers:
            warnings.append(
                ClassicSemanticWarning(
                    "external_filled_region_backend_disabled",
                    "External filled-region backends are disabled; using internal deterministic extraction.",
                    "decide",
                )
            )
        manual = self.config.strategy if self.config.strategy is not ClassicPipelineStrategy.AUTO else None
        if manual is not None and not self.config.auto_detect_strategy:
            return ClassicPipelineDecision(
                strategy=manual,
                reason="manual strategy selected",
                split=None,
                category=classification.category,
                foreground_ratio=metrics.foreground_ratio,
                dark_pixel_ratio=metrics.dark_pixel_ratio,
                colored_pixel_ratio=metrics.colored_pixel_ratio,
                edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
                warnings=tuple(warnings),
            )
        if manual is not None and manual is not ClassicPipelineStrategy.AUTO:
            return ClassicPipelineDecision(
                strategy=manual,
                reason="manual strategy selected",
                split=None,
                category=classification.category,
                foreground_ratio=metrics.foreground_ratio,
                dark_pixel_ratio=metrics.dark_pixel_ratio,
                colored_pixel_ratio=metrics.colored_pixel_ratio,
                edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
                warnings=tuple(warnings),
            )

        if classification.category is ImageCategory.COLOR_REGIONS and metrics.colored_pixel_ratio > 0.08:
            warnings.append(
                ClassicSemanticWarning(
                    "color_regions_conservative",
                    "Classic semantic output for complex color regions is conservative; Visual remains separate.",
                    "decide",
                )
            )
            return ClassicPipelineDecision(
                strategy=ClassicPipelineStrategy.COLOR_REGIONS,
                reason="color image classified conservatively",
                split=None,
                category=classification.category,
                foreground_ratio=metrics.foreground_ratio,
                dark_pixel_ratio=metrics.dark_pixel_ratio,
                colored_pixel_ratio=metrics.colored_pixel_ratio,
                edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
                warnings=tuple(warnings),
            )

        split = split_foreground_layers(
            preprocessing.binary_mask,
            filled_region_min_area=self.config.filled_region_min_area,
            filled_region_min_ratio=self.config.filled_region_min_ratio,
            thin_stroke_max_width=self.config.thin_stroke_max_width,
            component_connectivity=self.config.component_connectivity,
        )
        if self.config.mixed_monochrome_enabled and split.filled_count > 0 and split.thin_count > 0:
            return ClassicPipelineDecision(
                strategy=ClassicPipelineStrategy.MIXED_MONOCHROME,
                reason="foreground contains filled components and thin strokes",
                split=split,
                category=classification.category,
                foreground_ratio=metrics.foreground_ratio,
                dark_pixel_ratio=metrics.dark_pixel_ratio,
                colored_pixel_ratio=metrics.colored_pixel_ratio,
                edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
                warnings=tuple(warnings),
            )
        if split.filled_count > 0 or classification.category is ImageCategory.BINARY_OUTLINE:
            return ClassicPipelineDecision(
                strategy=ClassicPipelineStrategy.BINARY_OUTLINE,
                reason="foreground dominated by filled or outline regions",
                split=split,
                category=classification.category,
                foreground_ratio=metrics.foreground_ratio,
                dark_pixel_ratio=metrics.dark_pixel_ratio,
                colored_pixel_ratio=metrics.colored_pixel_ratio,
                edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
                warnings=tuple(warnings),
            )
        return ClassicPipelineDecision(
            strategy=ClassicPipelineStrategy.LINE_ART,
            reason="foreground appears to be line art",
            split=split,
            category=classification.category,
            foreground_ratio=metrics.foreground_ratio,
            dark_pixel_ratio=metrics.dark_pixel_ratio,
            colored_pixel_ratio=metrics.colored_pixel_ratio,
            edge_to_foreground_ratio=metrics.edge_to_foreground_ratio,
            warnings=tuple(warnings),
        )

    def _extract_primitives(
        self,
        mask: np.ndarray,
        decision: ClassicPipelineDecision,
    ) -> tuple[Any, tuple[SemanticGeometry, ...]]:
        filled_config = FilledRegionExtractionConfig(
            minimum_area=self.config.filled_region_min_area,
            minimum_fill_ratio=self.config.filled_region_min_ratio,
            component_connectivity=self.config.component_connectivity,
            strict=self.config.strict,
        )
        strategy = decision.strategy
        if strategy is ClassicPipelineStrategy.LINE_ART:
            thin = extract_thin_strokes(mask, self.config.centerline_config)
            group = PrimitiveGroup(
                thin.primitives,
                name="thin_strokes",
                metadata={"source_layer": "thin_stroke", "strategy": strategy.value},
            )
            return thin, (group,) if thin.primitives else ()
        if strategy is ClassicPipelineStrategy.BINARY_OUTLINE:
            filled = extract_filled_regions(mask, filled_config)
            group = PrimitiveGroup(
                filled.primitives,
                name="filled_regions",
                metadata={"source_layer": "filled_region", "strategy": strategy.value},
            )
            return filled, (group,) if filled.primitives else ()
        if strategy is ClassicPipelineStrategy.COLOR_REGIONS:
            filled = extract_filled_regions(mask, filled_config)
            group = PrimitiveGroup(
                filled.primitives,
                name="color_regions_conservative",
                metadata={"source_layer": "filled_region", "strategy": strategy.value, "conservative": True},
            )
            return filled, (group,) if filled.primitives else ()
        if strategy is ClassicPipelineStrategy.MIXED_MONOCHROME:
            mixed = extract_mixed_monochrome_primitives(
                mask,
                filled_config=filled_config,
                centerline_config=self.config.centerline_config,
                filled_region_min_area=self.config.filled_region_min_area,
                filled_region_min_ratio=self.config.filled_region_min_ratio,
                thin_stroke_max_width=self.config.thin_stroke_max_width,
                component_connectivity=self.config.component_connectivity,
            )
            return mixed, tuple(mixed.primitives)
        raise ClassicSemanticPipelineError(f"Unsupported Classic strategy: {strategy.value}")

    def _acceptance(
        self,
        validation_result: VisualValidationResult | None,
        tikz_result: TikzExportResult,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if not tikz_result.code.strip() or tikz_result.metrics.draw_commands <= 0:
            reasons.append("empty_tikz_output")
        if validation_result is None:
            return not reasons, tuple(reasons)
        reasons.extend(validation_result.rejection_reasons)
        raster = validation_result.fidelity_score.raster_metrics
        if raster.filled_region_recall < self.config.minimum_filled_region_recall:
            reasons.append("filled_region_recall_below_classic_minimum")
        if raster.thin_stroke_recall < self.config.minimum_thin_stroke_recall:
            reasons.append("thin_stroke_recall_below_classic_minimum")
        if validation_result.fidelity_score.overall_score < self.config.minimum_acceptance_score:
            reasons.append("score_below_classic_minimum")
        deduped = tuple(dict.fromkeys(reasons))
        return not deduped and validation_result.accepted, deduped


def detect_mixed_monochrome_image(
    image: Any,
    config: ClassicSemanticConfig | None = None,
) -> ClassicPipelineDecision:
    """Detect whether an image should use the mixed monochrome strategy."""
    effective_config = config or ClassicSemanticConfig()
    source = _normalize_image(image)
    classification = classify_image(source, effective_config.image_classifier_config)
    preprocessing = preprocess_image(source, effective_config.preprocessing_config, category=classification.category)
    return ClassicSemanticPipeline(effective_config)._decide_strategy(classification, preprocessing)


def run_classic_semantic_pipeline(
    image: Any,
    config: ClassicSemanticConfig | None = None,
) -> ClassicSemanticResult:
    """Run the isolated semantic Classic pipeline."""
    return ClassicSemanticPipeline(config).run(image)


def _normalize_image(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Image not found: {image}")
        return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("image must not be empty.")
        if array.ndim == 2:
            gray = np.clip(array, 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if array.ndim == 3 and array.shape[2] >= 3:
            return np.clip(array[:, :, :3], 0, 255).astype(np.uint8).copy()
        raise ValueError("image must be grayscale or RGB/BGR-like.")
    if hasattr(image, "convert") and hasattr(image, "mode"):
        converted = image.convert("RGB")
        return np.asarray(converted, dtype=np.uint8).copy()
    raise TypeError(f"Unsupported image input: {type(image).__name__}")


def _stage(
    name: str,
    details: Mapping[str, Any],
    warnings: Iterable[ClassicSemanticWarning] = (),
    *,
    status: ClassicPipelineStatus = ClassicPipelineStatus.ACCEPTED,
) -> ClassicPipelineStageResult:
    return ClassicPipelineStageResult(name=name, status=status, details=dict(details), warnings=tuple(warnings))


def _warnings_from_extraction(extraction: Any) -> list[ClassicSemanticWarning]:
    output: list[ClassicSemanticWarning] = []
    for warning in tuple(getattr(extraction, "warnings", ())):
        output.append(ClassicSemanticWarning(str(warning), str(warning), "extract"))
    if isinstance(extraction, MixedMonochromeResult):
        if extraction.filled_regions.region_count <= 0:
            output.append(ClassicSemanticWarning("no_filled_regions", "No filled regions were extracted.", "extract"))
        if extraction.thin_strokes.path_count <= 0:
            output.append(ClassicSemanticWarning("no_thin_strokes", "No thin strokes were extracted.", "extract"))
    if isinstance(extraction, FilledRegionExtractionResult) and extraction.region_count <= 0:
        output.append(ClassicSemanticWarning("no_filled_regions", "No filled regions were extracted.", "extract"))
    return output


def _warnings_from_fitting(results: Iterable[PrimitiveFitResult]) -> list[ClassicSemanticWarning]:
    output: list[ClassicSemanticWarning] = []
    for result in results:
        for warning in result.warnings:
            output.append(ClassicSemanticWarning(warning.code, warning.message, "fit"))
    return output


def _warnings_from_optimization(result: GeometryOptimizationResult) -> list[ClassicSemanticWarning]:
    return [ClassicSemanticWarning(warning.code, warning.message, "optimize") for warning in result.warnings]


def _warnings_from_tikz(result: TikzExportResult) -> list[ClassicSemanticWarning]:
    return [ClassicSemanticWarning(warning.code, warning.message, "export") for warning in result.warnings]


def _warnings_from_validation(result: VisualValidationResult) -> list[ClassicSemanticWarning]:
    return [ClassicSemanticWarning(warning.code, warning.message, "validate") for warning in result.warnings]


def _metrics(
    source: np.ndarray,
    classification: ImageClassificationResult,
    decision: ClassicPipelineDecision,
    raw: tuple[SemanticGeometry, ...],
    fitted: tuple[SemanticGeometry, ...],
    optimized: tuple[SemanticGeometry, ...],
    tikz_result: TikzExportResult,
    validation_result: VisualValidationResult | None,
) -> ClassicSemanticMetrics:
    raster = validation_result.fidelity_score.raster_metrics if validation_result is not None else None
    return ClassicSemanticMetrics(
        input_image_size=(int(source.shape[1]), int(source.shape[0])),
        image_category=classification.category.value,
        strategy_used=decision.strategy.value,
        foreground_ratio=classification.metrics.foreground_ratio,
        dark_pixel_ratio=classification.metrics.dark_pixel_ratio,
        thin_stroke_primitives=_count_by_source_layer(optimized, "thin_stroke"),
        filled_region_primitives=_count_by_source_layer(optimized, "filled_region"),
        raw_primitive_count=_object_count(raw),
        fitted_primitive_count=_object_count(fitted),
        optimized_primitive_count=_object_count(optimized),
        tikz_draw_commands=tikz_result.metrics.draw_commands,
        tikz_code_characters=len(tikz_result.code),
        validation_score=validation_result.fidelity_score.overall_score if validation_result is not None else 1.0,
        filled_region_recall=raster.filled_region_recall if raster is not None else 1.0,
        thin_stroke_recall=raster.thin_stroke_recall if raster is not None else 1.0,
        dark_mass_preservation_ratio=raster.dark_mass_preservation_ratio if raster is not None else 1.0,
    )


def _count_by_source_layer(items: Iterable[SemanticGeometry], layer: str) -> int:
    count = 0
    for primitive in _flatten(items):
        if str(dict(primitive.metadata).get("source_layer", "")).startswith(layer):
            count += 1
    return count


def _object_count(items: Iterable[SemanticGeometry]) -> int:
    return sum(1 for _ in _flatten(items))


def _flatten(items: Iterable[SemanticGeometry]) -> tuple[Any, ...]:
    output: list[Any] = []
    for item in items:
        if isinstance(item, PrimitiveGroup):
            output.extend(_flatten(item.items))
        else:
            output.append(item)
    return tuple(output)


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _compact_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_compact_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _to_dict_if_possible(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return _compact_value(value)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _log_result(result: ClassicSemanticResult) -> None:
    log_event("ClassicSemantic", f"strategy={result.strategy_used.value}")
    log_event(
        "ClassicSemantic",
        f"classification={result.classification_result.category.value}"
        + (
            f" ambiguous_with={result.classification_result.alternative_category.value}"
            if result.classification_result.alternative_category is not None
            else ""
        ),
    )
    log_event("ClassicSemantic", f"thin_strokes={result.metrics.thin_stroke_primitives}")
    log_event("ClassicSemantic", f"filled_regions={result.metrics.filled_region_primitives}")
    log_event("ClassicSemantic", f"optimized_primitives={result.metrics.optimized_primitive_count}")
    log_event("ClassicSemantic", f"tikz_chars={result.metrics.tikz_code_characters}")
    log_event("ClassicSemantic", f"validation_score={result.metrics.validation_score:.3f}")
    log_event("ClassicSemantic", f"accepted={result.accepted}")


__all__ = [
    "ClassicPipelineDecision",
    "ClassicPipelineStageResult",
    "ClassicSemanticMetrics",
    "ClassicSemanticPipeline",
    "ClassicSemanticPipelineError",
    "ClassicSemanticResult",
    "ClassicSemanticWarning",
    "FilledRegionExtractionError",
    "detect_mixed_monochrome_image",
    "run_classic_semantic_pipeline",
]
