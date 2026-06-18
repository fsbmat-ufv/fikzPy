"""Composite visual fidelity scoring for semantic Classic candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from fikzpy.core.complexity_metrics import ComplexityMetrics
from fikzpy.core.raster_metrics import RasterComparisonMetrics, RasterMetricsConfig, compare_rasters


class FidelityScoreError(ValueError):
    """Raised when a fidelity score cannot be computed."""


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "visual_fidelity": 0.50,
    "filled_region_preservation": 0.15,
    "thin_stroke_preservation": 0.10,
    "semantic_compactness": 0.15,
    "code_readability": 0.10,
}


@dataclass(frozen=True)
class FidelityScoreConfig:
    """Configuration for raster metrics, thresholds, and score weights."""

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
    minimum_acceptable_score: float = 0.72
    minimum_fidelity_score: float = 0.60
    minimum_filled_region_recall: float = 0.62
    minimum_thin_stroke_recall: float = 0.45
    minimum_foreground_iou: float = 0.25
    maximum_complexity_ratio: float = 4.0
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    strict: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
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
        for name in ("compare_edges", "compare_filled_regions", "compare_thin_strokes", "strict", "deterministic"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")
        for name in (
            "thin_stroke_max_width",
            "minimum_acceptable_score",
            "minimum_fidelity_score",
            "minimum_filled_region_recall",
            "minimum_thin_stroke_recall",
            "minimum_foreground_iou",
            "maximum_complexity_ratio",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "weights", _validate_weights(self.weights))

    def raster_config(self) -> RasterMetricsConfig:
        """Return the raster metric configuration equivalent to this score config."""
        return RasterMetricsConfig(
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
        data["weights"] = dict(sorted((str(key), float(value)) for key, value in self.weights.items()))
        return data


@dataclass(frozen=True)
class FidelityScoreResult:
    """Composite score and diagnostics for a rendered semantic output."""

    overall_score: float
    fidelity_score: float
    complexity_score: float
    semantic_score: float
    regression_score: float
    filled_region_score: float
    thin_stroke_score: float
    code_readability_score: float
    raster_metrics: RasterComparisonMetrics
    complexity_metrics: ComplexityMetrics | None
    regression_flags: tuple[str, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    deterministic_hash: str
    config: FidelityScoreConfig

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serializable score diagnostics."""
        return {
            "overall_score": _rounded(self.overall_score),
            "fidelity_score": _rounded(self.fidelity_score),
            "complexity_score": _rounded(self.complexity_score),
            "semantic_score": _rounded(self.semantic_score),
            "regression_score": _rounded(self.regression_score),
            "filled_region_score": _rounded(self.filled_region_score),
            "thin_stroke_score": _rounded(self.thin_stroke_score),
            "code_readability_score": _rounded(self.code_readability_score),
            "raster_metrics": self.raster_metrics.to_dict(),
            "complexity_metrics": self.complexity_metrics.to_dict() if self.complexity_metrics else None,
            "regression_flags": list(self.regression_flags),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
            "config": self.config.to_dict(),
        }


def compute_fidelity_score(
    source_image: Any,
    rendered_image: Any,
    complexity_metrics: ComplexityMetrics | None = None,
    config: FidelityScoreConfig | None = None,
) -> FidelityScoreResult:
    """Compute a 0..1 visual fidelity and semantic complexity score."""
    effective_config = config or FidelityScoreConfig()
    try:
        raster_metrics = compare_rasters(source_image, rendered_image, effective_config.raster_config())
        return _score_metrics(raster_metrics, complexity_metrics, effective_config)
    except Exception as exc:
        if effective_config.strict:
            raise FidelityScoreError(str(exc)) from exc
        raise


def _score_metrics(
    raster_metrics: RasterComparisonMetrics,
    complexity_metrics: ComplexityMetrics | None,
    config: FidelityScoreConfig,
) -> FidelityScoreResult:
    fidelity = _visual_fidelity_score(raster_metrics)
    filled = _filled_region_score(raster_metrics)
    thin = _thin_stroke_score(raster_metrics)
    complexity = complexity_metrics.complexity_score if complexity_metrics else 1.0
    semantic = complexity_metrics.semantic_compactness_score if complexity_metrics else 1.0
    readability = complexity_metrics.editability_score if complexity_metrics else 1.0
    regression_flags = _regression_flags(raster_metrics, complexity_metrics, config)
    regression_score = _regression_score(raster_metrics, regression_flags)

    weights = dict(config.weights)
    total_weight = sum(weights.values())
    weighted = (
        fidelity * weights["visual_fidelity"]
        + filled * weights["filled_region_preservation"]
        + thin * weights["thin_stroke_preservation"]
        + semantic * weights["semantic_compactness"]
        + readability * weights["code_readability"]
    ) / total_weight
    overall = _clamp01(weighted * (0.85 + 0.15 * regression_score))

    rejection_reasons = _rejection_reasons(
        overall=overall,
        fidelity=fidelity,
        filled=filled,
        thin=thin,
        raster_metrics=raster_metrics,
        complexity_metrics=complexity_metrics,
        config=config,
        regression_flags=regression_flags,
    )
    accepted = not rejection_reasons
    warnings = tuple(f"regression:{flag}" for flag in regression_flags)
    payload = {
        "scores": {
            "overall": overall,
            "fidelity": fidelity,
            "filled": filled,
            "thin": thin,
            "complexity": complexity,
            "semantic": semantic,
            "readability": readability,
            "regression": regression_score,
        },
        "raster_metrics": raster_metrics.to_dict(),
        "complexity_metrics": complexity_metrics.to_dict() if complexity_metrics else None,
        "flags": list(regression_flags),
        "rejection_reasons": list(rejection_reasons),
        "config": config.to_dict(),
    }
    deterministic_hash = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    result = FidelityScoreResult(
        overall_score=overall,
        fidelity_score=fidelity,
        complexity_score=complexity,
        semantic_score=semantic,
        regression_score=regression_score,
        filled_region_score=filled,
        thin_stroke_score=thin,
        code_readability_score=readability,
        raster_metrics=raster_metrics,
        complexity_metrics=complexity_metrics,
        regression_flags=regression_flags,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
        deterministic_hash=deterministic_hash,
        config=config,
    )
    if config.strict and not result.accepted:
        raise FidelityScoreError("; ".join(result.rejection_reasons))
    return result


def _visual_fidelity_score(metrics: RasterComparisonMetrics) -> float:
    return _clamp01(
        metrics.foreground_f1 * 0.30
        + metrics.foreground_iou * 0.25
        + (1.0 - metrics.normalized_rmse) * 0.20
        + metrics.edge_f1 * 0.15
        + metrics.structural_proxy * 0.10
    )


def _filled_region_score(metrics: RasterComparisonMetrics) -> float:
    mass = metrics.dark_mass_preservation_ratio
    if mass > 1.0:
        mass = max(0.0, 1.0 - min(1.0, mass - 1.0))
    return _clamp01(metrics.filled_region_recall * 0.70 + metrics.large_dark_region_recall * 0.15 + mass * 0.15)


def _thin_stroke_score(metrics: RasterComparisonMetrics) -> float:
    return _clamp01(metrics.thin_stroke_recall * 0.55 + metrics.small_detail_recall * 0.25 + metrics.edge_recall * 0.20)


def _regression_score(metrics: RasterComparisonMetrics, flags: tuple[str, ...]) -> float:
    flag_penalty = min(0.65, len(flags) * 0.09)
    metric_penalty = (
        (1.0 - metrics.dark_mass_preservation_ratio if metrics.dark_mass_preservation_ratio < 1.0 else 0.0) * 0.25
        + metrics.double_outline_penalty * 0.25
        + metrics.false_negative_rate * 0.20
        + max(0.0, 1.0 - metrics.filled_region_recall) * 0.20
    )
    return _clamp01(1.0 - flag_penalty - metric_penalty)


def _regression_flags(
    metrics: RasterComparisonMetrics,
    complexity_metrics: ComplexityMetrics | None,
    config: FidelityScoreConfig,
) -> tuple[str, ...]:
    flags: list[str] = []
    if metrics.source_foreground_pixels > 0 and metrics.rendered_foreground_pixels == 0:
        flags.append("invisible_output")
    if metrics.source_foreground_ratio > 0.02 and metrics.dark_mass_preservation_ratio < 0.12:
        flags.append("near_empty_output")
    if metrics.source_foreground_ratio > 0.04 and metrics.dark_mass_preservation_ratio < 0.55:
        flags.append("dark_mass_loss")
    if metrics.large_dark_region_recall < config.minimum_filled_region_recall:
        flags.append("missing_large_dark_regions")
    if metrics.filled_region_recall < config.minimum_filled_region_recall:
        flags.append("low_filled_region_recall")
    if metrics.thin_stroke_recall < config.minimum_thin_stroke_recall:
        flags.append("lost_thin_details")
    if metrics.foreground_precision < 0.45 and metrics.rendered_foreground_pixels > metrics.source_foreground_pixels:
        flags.append("excessive_false_foreground")
    if metrics.double_outline_penalty > 0.25:
        flags.append("double_outline_suspected")
    if metrics.foreground_fragmentation_delta > max(3, metrics.source_connected_components * 2):
        flags.append("foreground_fragmented")
    if metrics.foreground_recall < 0.35 and metrics.source_foreground_pixels > 0:
        flags.append("over_simplified")
    if complexity_metrics is not None:
        if (
            metrics.source_foreground_pixels > 0
            and complexity_metrics.point_count / max(metrics.source_foreground_pixels, 1) > config.maximum_complexity_ratio
        ):
            flags.append("excessive_complexity")
        if complexity_metrics.complexity_score < 0.25:
            flags.append("excessive_complexity")
        if complexity_metrics.tikz_raw_path_penalty > 0.50:
            flags.append("non_semantic_tikz")
    return tuple(dict.fromkeys(flags))


def _rejection_reasons(
    *,
    overall: float,
    fidelity: float,
    filled: float,
    thin: float,
    raster_metrics: RasterComparisonMetrics,
    complexity_metrics: ComplexityMetrics | None,
    config: FidelityScoreConfig,
    regression_flags: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if overall < config.minimum_acceptable_score:
        reasons.append("overall_score_below_minimum")
    if fidelity < config.minimum_fidelity_score:
        reasons.append("fidelity_score_below_minimum")
    if raster_metrics.foreground_iou < config.minimum_foreground_iou and raster_metrics.source_foreground_pixels > 0:
        reasons.append("foreground_iou_below_minimum")
    if filled < config.minimum_filled_region_recall:
        reasons.append("filled_region_score_below_minimum")
    if thin < config.minimum_thin_stroke_recall:
        reasons.append("thin_stroke_score_below_minimum")
    if "invisible_output" in regression_flags or "near_empty_output" in regression_flags:
        reasons.append("output_practically_empty")
    if "missing_large_dark_regions" in regression_flags or "dark_mass_loss" in regression_flags:
        reasons.append("dark_regions_lost")
    if "non_semantic_tikz" in regression_flags:
        reasons.append("tikz_looks_raw")
    if complexity_metrics is not None and complexity_metrics.complexity_score <= 0.0:
        reasons.append("complexity_score_invalid")
    return tuple(dict.fromkeys(reasons))


def _validate_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    data = dict(DEFAULT_SCORE_WEIGHTS)
    data.update(dict(weights))
    for key in DEFAULT_SCORE_WEIGHTS:
        if key not in data:
            raise ValueError(f"missing score weight {key!r}.")
        value = float(data[key])
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"score weight {key!r} must be finite and non-negative.")
        data[key] = value
    if sum(data[key] for key in DEFAULT_SCORE_WEIGHTS) <= 0.0:
        raise ValueError("score weights must sum to a positive value.")
    return dict(sorted((key, data[key]) for key in DEFAULT_SCORE_WEIGHTS))


def _rounded(value: float, digits: int = 6) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    rounded = round(number, digits)
    return 0.0 if rounded == 0 else rounded


def _clamp01(value: float) -> float:
    if not isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "FidelityScoreConfig",
    "FidelityScoreError",
    "FidelityScoreResult",
    "compute_fidelity_score",
]
