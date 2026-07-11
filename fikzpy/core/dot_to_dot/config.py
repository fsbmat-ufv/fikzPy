"""Typed configuration for the independent Ponto a Ponto (dot-to-dot) mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class DotToDotGenerationMode(Enum):
    """Internal generation modes of the dot-to-dot pipeline.

    ``FULL_FIGURE`` replaces the main external contour with numbered points
    automatically. ``SELECTED_SEGMENTS`` replaces only the arcs the user
    selected interactively, numbering them in a single continuous sequence.
    """

    FULL_FIGURE = "full_figure"
    SELECTED_SEGMENTS = "selected_segments"


class NumberingDirection(Enum):
    """Direction in which points are numbered along a path."""

    FORWARD = "forward"
    REVERSE = "reverse"


def coerce_generation_mode(value: DotToDotGenerationMode | str) -> DotToDotGenerationMode:
    """Coerce a generation-mode value from an enum or string."""
    if isinstance(value, DotToDotGenerationMode):
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "full": "full_figure",
        "figura_inteira": "full_figure",
        "figura inteira": "full_figure",
        "segments": "selected_segments",
        "selecionar_trechos": "selected_segments",
        "selecionar trechos": "selected_segments",
    }
    normalized = aliases.get(normalized, normalized)
    for item in DotToDotGenerationMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported dot-to-dot generation mode: {value!r}")


def coerce_direction(value: NumberingDirection | str) -> NumberingDirection:
    """Coerce a numbering-direction value from an enum or string."""
    if isinstance(value, NumberingDirection):
        return value
    normalized = str(value).strip().lower()
    aliases = {"cw": "forward", "ccw": "reverse", "reversed": "reverse"}
    normalized = aliases.get(normalized, normalized)
    for item in NumberingDirection:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported numbering direction: {value!r}")


@dataclass(frozen=True)
class SelectedSegment:
    """One user-selected arc along a subpath, in arc-length ratios.

    ``start_length_ratio`` and ``end_length_ratio`` are normalized positions
    in ``[0, 1]`` along the subpath's arc length. ``reverse`` selects the
    complementary arc on a closed subpath (or reverses numbering on an open
    one). ``point_count`` is how many numbered points this arc receives.
    """

    path_id: int
    subpath_id: int
    start_length_ratio: float
    end_length_ratio: float
    reverse: bool = False
    point_count: int = 10

    def __post_init__(self) -> None:
        for name in ("start_length_ratio", "end_length_ratio"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        if int(self.point_count) < 2:
            raise ValueError("point_count must be at least 2.")
        object.__setattr__(self, "path_id", int(self.path_id))
        object.__setattr__(self, "subpath_id", int(self.subpath_id))
        object.__setattr__(self, "point_count", int(self.point_count))
        object.__setattr__(self, "reverse", bool(self.reverse))


@dataclass(frozen=True)
class DotToDotConfig:
    """User-facing configuration for the dot-to-dot mode."""

    generation_mode: DotToDotGenerationMode | str = DotToDotGenerationMode.FULL_FIGURE
    total_points: int = 20
    start_number: int = 1
    point_radius: float = 2.0
    label_offset: float = 0.35
    include_internal_contours: bool = False
    preserve_internal_details: bool = True
    direction: NumberingDirection | str = NumberingDirection.FORWARD
    selected_segments: tuple[SelectedSegment, ...] = field(default_factory=tuple)
    min_points_per_path: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation_mode", coerce_generation_mode(self.generation_mode))
        object.__setattr__(self, "direction", coerce_direction(self.direction))
        if int(self.total_points) < 2:
            raise ValueError("total_points must be at least 2.")
        object.__setattr__(self, "total_points", int(self.total_points))
        object.__setattr__(self, "start_number", int(self.start_number))
        radius = float(self.point_radius)
        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("point_radius must be finite and positive.")
        object.__setattr__(self, "point_radius", radius)
        offset = float(self.label_offset)
        if not isfinite(offset) or offset < 0.0:
            raise ValueError("label_offset must be finite and non-negative.")
        object.__setattr__(self, "label_offset", offset)
        if int(self.min_points_per_path) < 1:
            raise ValueError("min_points_per_path must be positive.")
        object.__setattr__(self, "min_points_per_path", int(self.min_points_per_path))
        object.__setattr__(self, "include_internal_contours", bool(self.include_internal_contours))
        object.__setattr__(self, "preserve_internal_details", bool(self.preserve_internal_details))
        object.__setattr__(self, "selected_segments", tuple(self.selected_segments))


__all__ = [
    "DotToDotConfig",
    "DotToDotGenerationMode",
    "NumberingDirection",
    "SelectedSegment",
    "coerce_direction",
    "coerce_generation_mode",
]
