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
    filled_region_max_skeleton_ratio: float = 0.3
    lineart_to_mixed_min_filled_area_ratio: float = 0.06
    component_connectivity: int = 8
    line_art_stroke_width: float = 0.45
    lineart_min_edge_recall: float = 0.35
    lineart_min_foreground_recall: float = 0.55
    lineart_min_contour_coverage: float = 0.55
    lineart_max_fragmentation_ratio: float = 0.6
    enable_lineart_outline_recovery: bool = True
    lineart_outline_recovery_when_centerline_fails: bool = True
    lineart_recovery_stroke_width: float = 0.45
    lineart_preserve_external_contour: bool = True
    reject_underdrawn_lineart: bool = True
    reject_overfilled_lineart: bool = True
    max_filled_area_ratio_for_lineart: float = 0.06
    max_white_cutout_ratio_for_lineart: float = 0.03
    outline_recovery_max_components: int = 24
    outline_recovery_simplification_tolerance: float = 0.01
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    image_classifier_config: ImageClassifierConfig = field(default_factory=ImageClassifierConfig)
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
            "prefer_external_filled_region_backend",
            "allow_external_tracers",
            "strict",
            "debug",
            "save_intermediate_artifacts",
            "enable_lineart_outline_recovery",
            "lineart_outline_recovery_when_centerline_fails",
            "lineart_preserve_external_contour",
            "reject_underdrawn_lineart",
            "reject_overfilled_lineart",
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
            "filled_region_max_skeleton_ratio",
            "lineart_to_mixed_min_filled_area_ratio",
            "lineart_min_edge_recall",
            "lineart_min_foreground_recall",
            "lineart_min_contour_coverage",
            "max_filled_area_ratio_for_lineart",
            "max_white_cutout_ratio_for_lineart",
            "outline_recovery_simplification_tolerance",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        if int(self.filled_region_min_area) < 1:
            raise ValueError("filled_region_min_area must be positive.")
        object.__setattr__(self, "filled_region_min_area", int(self.filled_region_min_area))
        width = float(self.thin_stroke_max_width)
        if not isfinite(width) or width <= 0.0:
            raise ValueError("thin_stroke_max_width must be finite and positive.")
        object.__setattr__(self, "thin_stroke_max_width", width)
        if self.component_connectivity not in {4, 8}:
            raise ValueError("component_connectivity must be 4 or 8.")
        for name in ("line_art_stroke_width", "lineart_recovery_stroke_width"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        fragmentation = float(self.lineart_max_fragmentation_ratio)
        if not isfinite(fragmentation) or fragmentation < 0.0:
            raise ValueError("lineart_max_fragmentation_ratio must be finite and non-negative.")
        object.__setattr__(self, "lineart_max_fragmentation_ratio", fragmentation)
        if int(self.outline_recovery_max_components) < 1:
            raise ValueError("outline_recovery_max_components must be positive.")
        object.__setattr__(self, "outline_recovery_max_components", int(self.outline_recovery_max_components))
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
            weights=DEFAULT_SCORE_WEIGHTS,
            strict=self.validation_policy is ClassicValidationPolicy.VALIDATE_STRICT or self.strict,
            save_debug_images=self.save_intermediate_artifacts,
            debug_output_dir=self.debug_output_dir,
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
            "filled_region_max_skeleton_ratio": self.filled_region_max_skeleton_ratio,
            "lineart_to_mixed_min_filled_area_ratio": self.lineart_to_mixed_min_filled_area_ratio,
            "component_connectivity": self.component_connectivity,
            "line_art_stroke_width": self.line_art_stroke_width,
            "lineart_min_edge_recall": self.lineart_min_edge_recall,
            "lineart_min_foreground_recall": self.lineart_min_foreground_recall,
            "lineart_min_contour_coverage": self.lineart_min_contour_coverage,
            "lineart_max_fragmentation_ratio": self.lineart_max_fragmentation_ratio,
            "enable_lineart_outline_recovery": self.enable_lineart_outline_recovery,
            "lineart_outline_recovery_when_centerline_fails": self.lineart_outline_recovery_when_centerline_fails,
            "lineart_recovery_stroke_width": self.lineart_recovery_stroke_width,
            "lineart_preserve_external_contour": self.lineart_preserve_external_contour,
            "reject_underdrawn_lineart": self.reject_underdrawn_lineart,
            "reject_overfilled_lineart": self.reject_overfilled_lineart,
            "max_filled_area_ratio_for_lineart": self.max_filled_area_ratio_for_lineart,
            "max_white_cutout_ratio_for_lineart": self.max_white_cutout_ratio_for_lineart,
            "outline_recovery_max_components": self.outline_recovery_max_components,
            "outline_recovery_simplification_tolerance": self.outline_recovery_simplification_tolerance,
            "preprocessing_config": self.preprocessing_config.to_dict(),
            "image_classifier_config": dict(self.image_classifier_config.__dict__),
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
