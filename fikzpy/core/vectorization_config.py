"""Configuration objects for raster-to-vector conversion modes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Optional grayscale and binary-mask preprocessing settings."""

    use_bilateral_filter: bool = False
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    gaussian_kernel: int = 0
    use_adaptive_threshold: bool = False
    adaptive_block_size: int = 35
    adaptive_offset: int = 9
    morphology: str = "none"
    morphology_kernel: int = 3


@dataclass(frozen=True)
class ContourCleaningConfig:
    """Conservative filtering settings for traced contours."""

    min_length: float = 3.0
    min_points: int = 2
    deduplicate: bool = False
    duplicate_distance: float = 1.0


@dataclass(frozen=True)
class ContourMergingConfig:
    """Endpoint-based contour merging settings."""

    enabled: bool = False
    max_distance: float = 2.5
    max_angle: float = 35.0


@dataclass(frozen=True)
class PathSmoothingConfig:
    """Path smoothing settings applied before TikZ generation."""

    enabled: bool = False
    iterations: int = 0
    simplify_epsilon: float | None = None
    prefer_bezier: bool = False


@dataclass(frozen=True)
class VectorizationConfig:
    """High-level vectorization preset."""

    mode: str = "classic"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cleaning: ContourCleaningConfig = field(default_factory=ContourCleaningConfig)
    merging: ContourMergingConfig = field(default_factory=ContourMergingConfig)
    smoothing: PathSmoothingConfig = field(default_factory=PathSmoothingConfig)

    @classmethod
    def classic(cls) -> "VectorizationConfig":
        """Return the current stable line-art behavior."""
        return cls(mode="classic")

    @classmethod
    def smooth(cls) -> "VectorizationConfig":
        """Return an experimental smoother line-art preset."""
        return cls(
            mode="smooth",
            preprocessing=PreprocessingConfig(
                use_bilateral_filter=True,
                bilateral_diameter=5,
                bilateral_sigma_color=45.0,
                bilateral_sigma_space=45.0,
                morphology="close",
                morphology_kernel=2,
            ),
            cleaning=ContourCleaningConfig(
                min_length=3.0,
                min_points=2,
                deduplicate=False,
                duplicate_distance=1.0,
            ),
            merging=ContourMergingConfig(
                enabled=True,
                max_distance=2.0,
                max_angle=30.0,
            ),
            smoothing=PathSmoothingConfig(
                enabled=True,
                iterations=1,
                simplify_epsilon=0.003,
                prefer_bezier=True,
            ),
        )


def config_for_mode(mode: str) -> VectorizationConfig:
    """Return a vectorization configuration for a public mode name."""
    normalized = mode.strip().lower()
    if normalized in {"classic", "line_art", "line-art"}:
        return VectorizationConfig.classic()
    if normalized == "smooth":
        return VectorizationConfig.smooth()
    if normalized == "contours":
        return VectorizationConfig(mode="contours")
    raise ValueError(f"Unsupported vectorization mode: {mode}")
