"""Configuration objects for the semantic Classic integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from fikzpy.core.adaptive_preprocessing import PreprocessingConfig
from fikzpy.core.centerline_pipeline import CenterlineConfig
from fikzpy.core.fidelity_score import DEFAULT_SCORE_WEIGHTS
from fikzpy.core.geometry_optimization import GeometryOptimizationConfig
from fikzpy.core.image_classifier import ImageClassifierConfig
from fikzpy.core.lineart_diagnostics import LineArtDiagnosticsConfig
from fikzpy.core.primitive_fitting import PrimitiveFittingConfig
from fikzpy.core.semantic_tikz_exporter import TikzExportConfig
from fikzpy.core.visual_validation import VisualValidationConfig


class ClassicIntegrationError(ValueError):
    """Raised when Classic integration configuration is invalid."""


class ClassicPipelineStatus(Enum):
    """Semantic Classic pipeline status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class ClassicPipelineStrategy(Enum):
    """Available semantic Classic extraction strategies."""

    AUTO = "auto"
    LINE_ART = "line_art"
    BINARY_OUTLINE = "binary_outline"
    COLOR_REGIONS = "color_regions"
    MIXED_MONOCHROME = "mixed_monochrome"


class ClassicFallbackPolicy(Enum):
    """Fallback behavior for rejected semantic Classic output."""

    REJECT_RESULT = "reject_result"
    LEGACY_EXPLICIT = "legacy_explicit"


class ClassicValidationPolicy(Enum):
    """Validation handling for semantic Classic output."""

    VALIDATE_AND_RETURN = "validate_and_return"
    VALIDATE_STRICT = "validate_strict"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ClassicSemanticConfig:
    """Centralized configuration for the semantic Classic pipeline."""

    enable_semantic_classic: bool = True
    strategy: ClassicPipelineStrategy | str = ClassicPipelineStrategy.AUTO
    auto_detect_strategy: bool = True
    fallback_policy: ClassicFallbackPolicy | str = ClassicFallbackPolicy.REJECT_RESULT
    validation_policy: ClassicValidationPolicy | str = ClassicValidationPolicy.VALIDATE_AND_RETURN
    minimum_acceptance_score: float = 0.40
    minimum_filled_region_recall: float = 0.35
    minimum_thin_stroke_recall: float = 0.18
    preserve_filled_regions: bool = True
    preserve_thin_strokes: bool = True
    mixed_monochrome_enabled: bool = True
    filled_region_min_area: int = 48
    filled_region_min_ratio: float = 0.22
    thin_stroke_max_width: float = 3.2
    prefer_lineart_when_ambiguous: bool = True
    lineart_filled_region_strictness: float = 1.0
    max_filled_area_ratio_for_lineart: float = 0.08
    max_white_cutout_ratio_for_lineart: float = 0.20
    max_white_cutout_count_for_lineart: int = 3
    minimum_fill_ratio_for_filled_region: float = 0.24
    minimum_compactness_for_filled_region: float = 0.035
    maximum_skeleton_ratio_for_filled_region: float = 0.24
    line_art_stroke_width: float = 0.4
    mixed_line_art_stroke_width: float = 0.45
    binary_outline_stroke_width: float = 0.6
    filled_region_stroke_width: float = 0.4
    filled_region_draw_outline: bool = False
    reject_overfilled_lineart: bool = True
    validate_lineart_fill_usage: bool = True
    component_connectivity: int = 8
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    image_classifier_config: ImageClassifierConfig = field(default_factory=ImageClassifierConfig)
    lineart_diagnostics_config: LineArtDiagnosticsConfig | None = None
    centerline_config: CenterlineConfig = field(default_factory=CenterlineConfig)
    fitting_config: PrimitiveFittingConfig = field(default_factory=PrimitiveFittingConfig)
    optimization_config: GeometryOptimizationConfig = field(default_factory=GeometryOptimizationConfig)
    tikz_export_config: TikzExportConfig = field(
        default_factory=lambda: TikzExportConfig(
            include_tikzpicture_environment=True,
            include_scope_environment=True,
            coordinate_precision=2,
            style_precision=2,
            preserve_groups=False,
        )
    )
    visual_validation_config: VisualValidationConfig | None = None
    prefer_external_filled_region_backend: bool = False
    allow_external_tracers: bool = False
    strict: bool = False
    debug: bool = False
    save_intermediate_artifacts: bool = False
    debug_output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "enable_semantic_classic",
            "auto_detect_strategy",
            "preserve_filled_regions",
            "preserve_thin_strokes",
            "mixed_monochrome_enabled",
            "prefer_lineart_when_ambiguous",
            "filled_region_draw_outline",
            "reject_overfilled_lineart",
            "validate_lineart_fill_usage",
            "prefer_external_filled_region_backend",
            "allow_external_tracers",
            "strict",
            "debug",
            "save_intermediate_artifacts",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        object.__setattr__(self, "strategy", coerce_classic_strategy(self.strategy))
        object.__setattr__(self, "fallback_policy", coerce_fallback_policy(self.fallback_policy))
        object.__setattr__(self, "validation_policy", coerce_validation_policy(self.validation_policy))
        for name in (
            "minimum_acceptance_score",
            "minimum_filled_region_recall",
            "minimum_thin_stroke_recall",
            "filled_region_min_ratio",
            "max_filled_area_ratio_for_lineart",
            "max_white_cutout_ratio_for_lineart",
            "minimum_fill_ratio_for_filled_region",
            "minimum_compactness_for_filled_region",
            "maximum_skeleton_ratio_for_filled_region",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        strictness = float(self.lineart_filled_region_strictness)
        if not isfinite(strictness) or strictness < 0.0:
            raise ValueError("lineart_filled_region_strictness must be finite and non-negative.")
        object.__setattr__(self, "lineart_filled_region_strictness", strictness)
        if int(self.filled_region_min_area) < 1:
            raise ValueError("filled_region_min_area must be positive.")
        object.__setattr__(self, "filled_region_min_area", int(self.filled_region_min_area))
        width = float(self.thin_stroke_max_width)
        if not isfinite(width) or width <= 0.0:
            raise ValueError("thin_stroke_max_width must be finite and positive.")
        object.__setattr__(self, "thin_stroke_max_width", width)
        for name in (
            "line_art_stroke_width",
            "mixed_line_art_stroke_width",
            "binary_outline_stroke_width",
            "filled_region_stroke_width",
        ):
            stroke_width = float(getattr(self, name))
            if not isfinite(stroke_width) or stroke_width <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, stroke_width)
        if int(self.max_white_cutout_count_for_lineart) < 0:
            raise ValueError("max_white_cutout_count_for_lineart must be non-negative.")
        object.__setattr__(self, "max_white_cutout_count_for_lineart", int(self.max_white_cutout_count_for_lineart))
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        if self.lineart_diagnostics_config is not None and not isinstance(self.lineart_diagnostics_config, LineArtDiagnosticsConfig):
            raise TypeError("lineart_diagnostics_config must be LineArtDiagnosticsConfig or None.")
        if self.save_intermediate_artifacts and self.debug_output_dir is None:
            raise ValueError("debug_output_dir is required when save_intermediate_artifacts=True.")

    def validation_config(self) -> VisualValidationConfig:
        """Return the effective validation configuration."""
        if self.visual_validation_config is not None:
            return self.visual_validation_config
        return VisualValidationConfig(
            filled_region_min_area=self.filled_region_min_area,
            thin_stroke_max_width=self.thin_stroke_max_width,
            component_connectivity=self.component_connectivity,
            minimum_acceptable_score=self.minimum_acceptance_score,
            minimum_filled_region_recall=self.minimum_filled_region_recall,
            minimum_thin_stroke_recall=self.minimum_thin_stroke_recall,
            validate_lineart_fill_usage=self.validate_lineart_fill_usage,
            max_filled_area_ratio_for_lineart=self.max_filled_area_ratio_for_lineart,
            max_white_cutout_ratio_for_lineart=self.max_white_cutout_ratio_for_lineart,
            max_white_cutout_count_for_lineart=self.max_white_cutout_count_for_lineart,
            reject_overfilled_lineart=self.reject_overfilled_lineart,
            weights=DEFAULT_SCORE_WEIGHTS,
            strict=self.validation_policy is ClassicValidationPolicy.VALIDATE_STRICT or self.strict,
            save_debug_images=self.save_intermediate_artifacts,
            debug_output_dir=self.debug_output_dir,
        )

    def lineart_config(self) -> LineArtDiagnosticsConfig:
        """Return the effective line-art diagnostic configuration."""
        if self.lineart_diagnostics_config is not None:
            return self.lineart_diagnostics_config
        strict_fill_ratio = min(
            1.0,
            self.minimum_fill_ratio_for_filled_region * max(1.0, self.lineart_filled_region_strictness),
        )
        return LineArtDiagnosticsConfig(
            component_connectivity=self.component_connectivity,
            thin_stroke_max_width=self.thin_stroke_max_width,
            filled_region_min_area=self.filled_region_min_area,
            minimum_fill_ratio_for_filled_region=strict_fill_ratio,
            minimum_compactness_for_filled_region=self.minimum_compactness_for_filled_region,
            maximum_skeleton_ratio_for_filled_region=self.maximum_skeleton_ratio_for_filled_region,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return {
            "enable_semantic_classic": self.enable_semantic_classic,
            "strategy": self.strategy.value,
            "auto_detect_strategy": self.auto_detect_strategy,
            "fallback_policy": self.fallback_policy.value,
            "validation_policy": self.validation_policy.value,
            "minimum_acceptance_score": self.minimum_acceptance_score,
            "minimum_filled_region_recall": self.minimum_filled_region_recall,
            "minimum_thin_stroke_recall": self.minimum_thin_stroke_recall,
            "preserve_filled_regions": self.preserve_filled_regions,
            "preserve_thin_strokes": self.preserve_thin_strokes,
            "mixed_monochrome_enabled": self.mixed_monochrome_enabled,
            "filled_region_min_area": self.filled_region_min_area,
            "filled_region_min_ratio": self.filled_region_min_ratio,
            "thin_stroke_max_width": self.thin_stroke_max_width,
            "prefer_lineart_when_ambiguous": self.prefer_lineart_when_ambiguous,
            "lineart_filled_region_strictness": self.lineart_filled_region_strictness,
            "max_filled_area_ratio_for_lineart": self.max_filled_area_ratio_for_lineart,
            "max_white_cutout_ratio_for_lineart": self.max_white_cutout_ratio_for_lineart,
            "max_white_cutout_count_for_lineart": self.max_white_cutout_count_for_lineart,
            "minimum_fill_ratio_for_filled_region": self.minimum_fill_ratio_for_filled_region,
            "minimum_compactness_for_filled_region": self.minimum_compactness_for_filled_region,
            "maximum_skeleton_ratio_for_filled_region": self.maximum_skeleton_ratio_for_filled_region,
            "line_art_stroke_width": self.line_art_stroke_width,
            "mixed_line_art_stroke_width": self.mixed_line_art_stroke_width,
            "binary_outline_stroke_width": self.binary_outline_stroke_width,
            "filled_region_stroke_width": self.filled_region_stroke_width,
            "filled_region_draw_outline": self.filled_region_draw_outline,
            "reject_overfilled_lineart": self.reject_overfilled_lineart,
            "validate_lineart_fill_usage": self.validate_lineart_fill_usage,
            "component_connectivity": self.component_connectivity,
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "image_classifier_config": dict(self.image_classifier_config.__dict__),
            "lineart_diagnostics_config": self.lineart_config().to_dict(),
            "centerline_config": self.centerline_config.to_dict(),
            "fitting_config": self.fitting_config.to_dict(),
            "optimization_config": self.optimization_config.to_dict(),
            "tikz_export_config": self.tikz_export_config.to_dict(),
            "visual_validation_config": self.validation_config().to_dict(),
            "prefer_external_filled_region_backend": self.prefer_external_filled_region_backend,
            "allow_external_tracers": self.allow_external_tracers,
            "strict": self.strict,
            "debug": self.debug,
            "save_intermediate_artifacts": self.save_intermediate_artifacts,
            "debug_output_dir": str(self.debug_output_dir) if self.debug_output_dir is not None else None,
        }


def coerce_classic_strategy(value: ClassicPipelineStrategy | str) -> ClassicPipelineStrategy:
    """Coerce strategy values."""
    if isinstance(value, ClassicPipelineStrategy):
        return value
    normalized = str(value).strip().lower()
    aliases = {"mixed": "mixed_monochrome", "binary": "binary_outline", "line": "line_art"}
    normalized = aliases.get(normalized, normalized)
    for item in ClassicPipelineStrategy:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported Classic strategy: {value!r}")


def coerce_fallback_policy(value: ClassicFallbackPolicy | str) -> ClassicFallbackPolicy:
    """Coerce fallback policy values."""
    if isinstance(value, ClassicFallbackPolicy):
        return value
    normalized = str(value).strip().lower()
    for item in ClassicFallbackPolicy:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported Classic fallback policy: {value!r}")


def coerce_validation_policy(value: ClassicValidationPolicy | str) -> ClassicValidationPolicy:
    """Coerce validation policy values."""
    if isinstance(value, ClassicValidationPolicy):
        return value
    normalized = str(value).strip().lower()
    for item in ClassicValidationPolicy:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported Classic validation policy: {value!r}")


__all__ = [
    "ClassicFallbackPolicy",
    "ClassicIntegrationError",
    "ClassicPipelineStatus",
    "ClassicPipelineStrategy",
    "ClassicSemanticConfig",
    "ClassicValidationPolicy",
    "coerce_classic_strategy",
    "coerce_fallback_policy",
    "coerce_validation_policy",
]
