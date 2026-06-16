"""Deterministic threshold candidate selection for semantic preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from fikzpy.core.diagnostics import log_event


THRESHOLD_METHODS = frozenset(
    {
        "fixed",
        "otsu",
        "adaptive_mean",
        "adaptive_gaussian",
        "global_sweep",
    }
)


@dataclass(frozen=True)
class ThresholdScoreWeights:
    """Weights used to rank threshold masks."""

    continuity: float = 0.22
    edge_coverage: float = 0.26
    foreground_plausibility: float = 0.24
    background_consistency: float = 0.12
    noise: float = 0.10
    fragmentation_penalty: float = 0.08
    tiny_component_penalty: float = 0.10
    extreme_mask_penalty: float = 0.18

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def to_dict(self) -> dict[str, float]:
        """Return a diagnostic dictionary."""
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    """Configuration for threshold candidate generation and ranking."""

    threshold_methods: tuple[str, ...] = (
        "fixed",
        "otsu",
        "adaptive_mean",
        "adaptive_gaussian",
        "global_sweep",
    )
    fixed_threshold: int = 180
    threshold_candidates: tuple[int, ...] = (64, 96, 128, 160, 192, 224)
    adaptive_block_size: int = 35
    adaptive_constant: float = 5.0
    foreground_is_dark: bool = True
    min_foreground_ratio: float = 0.001
    max_foreground_ratio: float = 0.85
    tiny_component_area: int = 4
    edge_threshold: float = 24.0
    edge_dilation: int = 1
    connectivity: int = 8
    ambiguity_margin: float = 0.025
    score_weights: ThresholdScoreWeights = field(default_factory=ThresholdScoreWeights)

    def __post_init__(self) -> None:
        methods = tuple(method.strip().lower() for method in self.threshold_methods)
        if not methods:
            raise ValueError("threshold_methods must not be empty.")
        for method in methods:
            if method not in THRESHOLD_METHODS:
                raise ValueError(f"Unsupported threshold method: {method}")
        object.__setattr__(self, "threshold_methods", methods)
        _validate_byte("fixed_threshold", self.fixed_threshold)
        candidates = tuple(int(value) for value in self.threshold_candidates)
        if not candidates:
            raise ValueError("threshold_candidates must not be empty.")
        for value in candidates:
            _validate_byte("threshold candidate", value)
        object.__setattr__(self, "threshold_candidates", tuple(sorted(set(candidates))))
        object.__setattr__(self, "adaptive_block_size", _odd_at_least(self.adaptive_block_size, 3))
        if not np.isfinite(float(self.adaptive_constant)):
            raise ValueError("adaptive_constant must be finite.")
        if not isinstance(self.foreground_is_dark, bool):
            raise TypeError("foreground_is_dark must be a bool.")
        _validate_ratio("min_foreground_ratio", self.min_foreground_ratio)
        _validate_ratio("max_foreground_ratio", self.max_foreground_ratio)
        if self.min_foreground_ratio > self.max_foreground_ratio:
            raise ValueError("min_foreground_ratio must not exceed max_foreground_ratio.")
        if int(self.tiny_component_area) < 0:
            raise ValueError("tiny_component_area must be non-negative.")
        object.__setattr__(self, "tiny_component_area", int(self.tiny_component_area))
        if not np.isfinite(float(self.edge_threshold)) or float(self.edge_threshold) < 0.0:
            raise ValueError("edge_threshold must be finite and non-negative.")
        if int(self.edge_dilation) < 0:
            raise ValueError("edge_dilation must be non-negative.")
        object.__setattr__(self, "edge_dilation", int(self.edge_dilation))
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8.")
        _validate_ratio("ambiguity_margin", self.ambiguity_margin)
        if not isinstance(self.score_weights, ThresholdScoreWeights):
            raise TypeError("score_weights must be ThresholdScoreWeights.")

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary."""
        return {
            "threshold_methods": list(self.threshold_methods),
            "fixed_threshold": self.fixed_threshold,
            "threshold_candidates": list(self.threshold_candidates),
            "adaptive_block_size": self.adaptive_block_size,
            "adaptive_constant": self.adaptive_constant,
            "foreground_is_dark": self.foreground_is_dark,
            "min_foreground_ratio": self.min_foreground_ratio,
            "max_foreground_ratio": self.max_foreground_ratio,
            "tiny_component_area": self.tiny_component_area,
            "edge_threshold": self.edge_threshold,
            "edge_dilation": self.edge_dilation,
            "connectivity": self.connectivity,
            "ambiguity_margin": self.ambiguity_margin,
            "score_weights": self.score_weights.to_dict(),
        }


@dataclass(frozen=True)
class ThresholdCandidateMetrics:
    """Metrics for one threshold candidate."""

    foreground_ratio: float
    component_count: int
    tiny_component_count: int
    tiny_component_ratio: float
    largest_component_ratio: float
    edge_coverage: float
    continuity_score: float
    noise_score: float
    background_consistency: float
    foreground_plausibility: float
    fragmentation_penalty: float
    extreme_foreground_penalty: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a diagnostic dictionary."""
        return {
            "foreground_ratio": self.foreground_ratio,
            "component_count": self.component_count,
            "tiny_component_count": self.tiny_component_count,
            "tiny_component_ratio": self.tiny_component_ratio,
            "largest_component_ratio": self.largest_component_ratio,
            "edge_coverage": self.edge_coverage,
            "continuity_score": self.continuity_score,
            "noise_score": self.noise_score,
            "background_consistency": self.background_consistency,
            "foreground_plausibility": self.foreground_plausibility,
            "fragmentation_penalty": self.fragmentation_penalty,
            "extreme_foreground_penalty": self.extreme_foreground_penalty,
        }


@dataclass(frozen=True)
class ThresholdCandidate:
    """A threshold candidate, its mask, metrics, and score."""

    method: str
    threshold: float | None
    mask: np.ndarray
    metrics: ThresholdCandidateMetrics
    score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("candidate mask must be 2D.")
        object.__setattr__(self, "mask", _as_binary_mask(self.mask))
        if self.threshold is not None:
            threshold = float(self.threshold)
            if not np.isfinite(threshold):
                raise ValueError("threshold must be finite.")
            object.__setattr__(self, "threshold", threshold)
        score = float(self.score)
        if not np.isfinite(score):
            raise ValueError("score must be finite.")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing the full mask."""
        return {
            "method": self.method,
            "threshold": self.threshold,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "reasons": list(self.reasons),
            "mask_shape": list(self.mask.shape),
            "mask_dtype": str(self.mask.dtype),
        }


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """The selected threshold and ranked candidates."""

    selected: ThresholdCandidate
    candidates: tuple[ThresholdCandidate, ...]
    ambiguous: bool
    alternative: ThresholdCandidate | None = None

    @property
    def best_mask(self) -> np.ndarray:
        """Return a copy of the selected binary mask."""
        return self.selected.mask.copy()

    @property
    def method(self) -> str:
        """Return the selected method."""
        return self.selected.method

    @property
    def threshold(self) -> float | None:
        """Return the selected threshold value when the method has one."""
        return self.selected.threshold

    @property
    def score(self) -> float:
        """Return the selected score."""
        return self.selected.score

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing full masks."""
        return {
            "selected": self.selected.to_dict(),
            "ambiguous": self.ambiguous,
            "alternative": self.alternative.to_dict() if self.alternative is not None else None,
            "ranking": [candidate.to_dict() for candidate in self.candidates],
        }


def select_best_threshold(
    grayscale_image: np.ndarray,
    config: ThresholdSelectionConfig | None = None,
) -> ThresholdSelectionResult:
    """Evaluate threshold candidates and return the highest ranked mask."""
    selection_config = config or ThresholdSelectionConfig()
    gray = _normalize_grayscale(grayscale_image)
    candidates = _generate_candidates(gray, selection_config)
    if not candidates:
        raise ValueError("No threshold candidates were generated.")

    ranked = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.method,
                -1.0 if candidate.threshold is None else candidate.threshold,
            ),
        )
    )
    selected = ranked[0]
    alternative = ranked[1] if len(ranked) > 1 else None
    ambiguous = alternative is not None and selected.score - alternative.score <= selection_config.ambiguity_margin

    log_event("Threshold", f"candidates={len(ranked)}")
    log_event("Threshold", f"selected={selected.method}")
    log_event("Threshold", f"value={selected.threshold}")
    log_event("Threshold", f"score={selected.score:.4f}")
    log_event("Threshold", f"foreground_ratio={selected.metrics.foreground_ratio:.3f}")
    log_event("Threshold", f"components={selected.metrics.component_count}")
    log_event("Threshold", f"tiny_components={selected.metrics.tiny_component_count}")

    return ThresholdSelectionResult(
        selected=selected,
        candidates=ranked,
        ambiguous=ambiguous,
        alternative=alternative if ambiguous else None,
    )


def _generate_candidates(gray: np.ndarray, config: ThresholdSelectionConfig) -> list[ThresholdCandidate]:
    candidates: list[ThresholdCandidate] = []
    if "fixed" in config.threshold_methods:
        candidates.append(_candidate_from_global(gray, config.fixed_threshold, "fixed", config))
    if "otsu" in config.threshold_methods:
        candidates.append(_candidate_from_otsu(gray, config))
    if "global_sweep" in config.threshold_methods:
        for value in config.threshold_candidates:
            candidates.append(_candidate_from_global(gray, value, "global_sweep", config))
    if min(gray.shape) >= 3 and "adaptive_mean" in config.threshold_methods:
        candidates.append(_candidate_from_adaptive(gray, "adaptive_mean", config))
    if min(gray.shape) >= 3 and "adaptive_gaussian" in config.threshold_methods:
        candidates.append(_candidate_from_adaptive(gray, "adaptive_gaussian", config))
    return candidates


def _candidate_from_global(
    gray: np.ndarray,
    threshold: int | float,
    method: str,
    config: ThresholdSelectionConfig,
) -> ThresholdCandidate:
    if config.foreground_is_dark:
        mask = (gray <= float(threshold)).astype(np.uint8) * 255
    else:
        mask = (gray >= float(threshold)).astype(np.uint8) * 255
    return _make_candidate(method, float(threshold), mask, gray, config)


def _candidate_from_otsu(gray: np.ndarray, config: ThresholdSelectionConfig) -> ThresholdCandidate:
    threshold_type = cv2.THRESH_BINARY_INV if config.foreground_is_dark else cv2.THRESH_BINARY
    threshold, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    return _make_candidate("otsu", float(threshold), mask, gray, config)


def _candidate_from_adaptive(gray: np.ndarray, method: str, config: ThresholdSelectionConfig) -> ThresholdCandidate:
    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv2.THRESH_BINARY_INV if config.foreground_is_dark else cv2.THRESH_BINARY
    block_size = _adaptive_block_size(gray, config.adaptive_block_size)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_method,
        threshold_type,
        block_size,
        float(config.adaptive_constant),
    )
    return _make_candidate(method, None, mask, gray, config)


def _make_candidate(
    method: str,
    threshold: float | None,
    mask: np.ndarray,
    gray: np.ndarray,
    config: ThresholdSelectionConfig,
) -> ThresholdCandidate:
    normalized_mask = _as_binary_mask(mask)
    metrics = _candidate_metrics(gray, normalized_mask, config)
    score = _score_metrics(metrics, config)
    reasons = (
        f"foreground={metrics.foreground_ratio:.3f}",
        f"coverage={metrics.edge_coverage:.3f}",
        f"continuity={metrics.continuity_score:.3f}",
    )
    return ThresholdCandidate(
        method=method,
        threshold=threshold,
        mask=normalized_mask,
        metrics=metrics,
        score=score,
        reasons=reasons,
    )


def _candidate_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
    config: ThresholdSelectionConfig,
) -> ThresholdCandidateMetrics:
    foreground = mask > 0
    foreground_pixels = int(np.count_nonzero(foreground))
    total_pixels = int(mask.size)
    foreground_ratio = foreground_pixels / max(total_pixels, 1)

    component_count, tiny_count, largest_ratio = _component_stats(mask, foreground_pixels, config)
    tiny_ratio = tiny_count / max(component_count, 1)
    edge_coverage = _edge_coverage(gray, mask, config)
    fragmentation_penalty = _fragmentation_penalty(component_count, foreground_pixels)
    continuity_score = 1.0 - fragmentation_penalty
    noise_score = 1.0 - tiny_ratio
    background_consistency = _background_consistency(gray, foreground, config.foreground_is_dark)
    foreground_plausibility = _foreground_plausibility(foreground_ratio, config)
    extreme_penalty = _extreme_foreground_penalty(foreground_ratio)

    return ThresholdCandidateMetrics(
        foreground_ratio=_clamp01(foreground_ratio),
        component_count=component_count,
        tiny_component_count=tiny_count,
        tiny_component_ratio=_clamp01(tiny_ratio),
        largest_component_ratio=_clamp01(largest_ratio),
        edge_coverage=_clamp01(edge_coverage),
        continuity_score=_clamp01(continuity_score),
        noise_score=_clamp01(noise_score),
        background_consistency=_clamp01(background_consistency),
        foreground_plausibility=_clamp01(foreground_plausibility),
        fragmentation_penalty=_clamp01(fragmentation_penalty),
        extreme_foreground_penalty=_clamp01(extreme_penalty),
    )


def _score_metrics(metrics: ThresholdCandidateMetrics, config: ThresholdSelectionConfig) -> float:
    weights = config.score_weights
    score = (
        weights.continuity * metrics.continuity_score
        + weights.edge_coverage * metrics.edge_coverage
        + weights.foreground_plausibility * metrics.foreground_plausibility
        + weights.background_consistency * metrics.background_consistency
        + weights.noise * metrics.noise_score
        - weights.fragmentation_penalty * metrics.fragmentation_penalty
        - weights.tiny_component_penalty * metrics.tiny_component_ratio
        - weights.extreme_mask_penalty * metrics.extreme_foreground_penalty
    )
    return float(max(0.0, min(1.0, score)))


def _component_stats(
    mask: np.ndarray,
    foreground_pixels: int,
    config: ThresholdSelectionConfig,
) -> tuple[int, int, float]:
    if foreground_pixels == 0:
        return 0, 0, 0.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=config.connectivity)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    if not areas:
        return 0, 0, 0.0
    tiny_count = sum(1 for area in areas if area <= config.tiny_component_area)
    largest_ratio = max(areas) / max(foreground_pixels, 1)
    return len(areas), tiny_count, largest_ratio


def _edge_coverage(gray: np.ndarray, mask: np.ndarray, config: ThresholdSelectionConfig) -> float:
    edges = _edge_pixels(gray, config.edge_threshold)
    edge_count = int(np.count_nonzero(edges))
    if edge_count == 0:
        return 1.0 if np.count_nonzero(mask) in {0, mask.size} else 0.0
    coverage_mask = mask
    if config.edge_dilation > 0:
        kernel_size = 2 * config.edge_dilation + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        coverage_mask = cv2.dilate(mask, kernel, iterations=1)
    return float(np.count_nonzero((coverage_mask > 0) & edges) / edge_count)


def _edge_pixels(gray: np.ndarray, threshold: float) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return magnitude >= float(threshold)


def _fragmentation_penalty(component_count: int, foreground_pixels: int) -> float:
    if foreground_pixels == 0:
        return 1.0
    expected_scale = np.sqrt(float(foreground_pixels)) + 1.0
    return _clamp01(max(0, component_count - 1) / expected_scale)


def _background_consistency(gray: np.ndarray, foreground: np.ndarray, foreground_is_dark: bool) -> float:
    foreground_count = int(np.count_nonzero(foreground))
    background_count = int(foreground.size - foreground_count)
    if foreground_count == 0 or background_count == 0:
        return 0.5
    foreground_mean = float(np.mean(gray[foreground]))
    background_mean = float(np.mean(gray[~foreground]))
    delta = background_mean - foreground_mean if foreground_is_dark else foreground_mean - background_mean
    return _clamp01(delta / 64.0)


def _foreground_plausibility(ratio: float, config: ThresholdSelectionConfig) -> float:
    if ratio <= 0.0 or ratio >= 1.0:
        return 0.0
    if ratio < config.min_foreground_ratio:
        return _clamp01(ratio / max(config.min_foreground_ratio, 1e-9))
    if ratio > config.max_foreground_ratio:
        return _clamp01((1.0 - ratio) / max(1.0 - config.max_foreground_ratio, 1e-9))
    return 1.0


def _extreme_foreground_penalty(ratio: float) -> float:
    if ratio <= 0.001:
        return 1.0 - _clamp01(ratio / 0.001)
    if ratio >= 0.98:
        return _clamp01((ratio - 0.98) / 0.02)
    return 0.0


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.size == 0:
        raise ValueError("grayscale_image must not be empty.")
    if gray.ndim != 2:
        raise ValueError("grayscale_image must be 2D.")
    if not np.all(np.isfinite(gray)):
        raise ValueError("grayscale_image values must be finite.")
    if np.min(gray) < 0:
        raise ValueError("grayscale_image values must be non-negative.")
    gray_float = gray.astype(np.float64, copy=False)
    if gray_float.max(initial=0.0) <= 1.0 and gray.dtype.kind == "f":
        gray_float = gray_float * 255.0
    if gray_float.max(initial=0.0) > 255.0:
        raise ValueError("grayscale_image values must be in the 0-255 range.")
    return np.rint(np.clip(gray_float, 0.0, 255.0)).astype(np.uint8)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be 2D.")
    return (array > 0).astype(np.uint8) * 255


def _adaptive_block_size(gray: np.ndarray, requested: int) -> int:
    maximum = max(3, min(gray.shape))
    size = min(_odd_at_least(requested, 3), maximum if maximum % 2 == 1 else maximum - 1)
    return max(3, size)


def _validate_byte(name: str, value: int | float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 255.0:
        raise ValueError(f"{name} must be between 0 and 255.")


def _validate_ratio(name: str, value: float) -> None:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _odd_at_least(value: int, minimum: int) -> int:
    size = max(int(value), minimum)
    return size if size % 2 == 1 else size + 1


def _clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "THRESHOLD_METHODS",
    "ThresholdCandidate",
    "ThresholdCandidateMetrics",
    "ThresholdScoreWeights",
    "ThresholdSelectionConfig",
    "ThresholdSelectionResult",
    "select_best_threshold",
]
