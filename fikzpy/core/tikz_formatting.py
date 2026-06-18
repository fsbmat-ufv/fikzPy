"""Formatting helpers for semantic TikZ export."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from fikzpy.core.semantic_geometry import Point2D


class TikzFormattingError(ValueError):
    """Raised when a TikZ value cannot be formatted safely."""


class TikzPathMode(Enum):
    """Line-breaking mode for emitted TikZ paths."""

    AUTO = "auto"
    INLINE = "inline"
    MULTILINE = "multiline"


class TikzYAxisMode(Enum):
    """Coordinate-space interpretation for the y axis."""

    CARTESIAN = "cartesian"
    IMAGE = "image"


@dataclass(frozen=True)
class TikzCoordinateFormatter:
    """Format coordinates and lengths with deterministic rounding."""

    precision: int = 3
    unit: str = ""
    scale: float = 1.0
    invert_y_axis: bool = False
    image_height: float | None = None
    origin: Point2D | tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if int(self.precision) < 0:
            raise ValueError("precision must be non-negative.")
        object.__setattr__(self, "precision", int(self.precision))
        scale = float(self.scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        object.__setattr__(self, "scale", scale)
        if not isinstance(self.invert_y_axis, bool):
            raise TypeError("invert_y_axis must be a bool.")
        if self.image_height is not None:
            height = float(self.image_height)
            if not isfinite(height):
                raise ValueError("image_height must be finite when provided.")
            object.__setattr__(self, "image_height", height)
        if self.unit is None:
            object.__setattr__(self, "unit", "")
        elif not isinstance(self.unit, str):
            raise TypeError("unit must be a string.")
        if self.origin is not None and not isinstance(self.origin, Point2D):
            if not isinstance(self.origin, tuple) or len(self.origin) != 2:
                raise TypeError("origin must be Point2D, an x/y tuple, or None.")
            object.__setattr__(self, "origin", Point2D(float(self.origin[0]), float(self.origin[1])))

    def format_number(self, value: float, *, precision: int | None = None) -> str:
        """Return a compact decimal string with stable negative-zero handling."""
        number = float(value)
        if not isfinite(number):
            raise TikzFormattingError("TikZ numeric values must be finite.")
        digits = self.precision if precision is None else int(precision)
        rounded = round(number, digits)
        if rounded == 0:
            rounded = 0.0
        if digits == 0:
            return str(int(rounded))
        text = f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
        return text if text and text != "-0" else "0"

    def format_length(self, value: float, *, unit: str | None = None, precision: int | None = None) -> str:
        """Return a TikZ length with an optional unit suffix."""
        suffix = self.unit if unit is None else unit
        return f"{self.format_number(value, precision=precision)}{suffix or ''}"

    def transform_point(self, point: Point2D) -> Point2D:
        """Apply configured origin, scale, and y-axis conversion."""
        if not isinstance(point, Point2D):
            raise TypeError("point must be a Point2D.")
        x = float(point.x)
        y = float(point.y)
        if not isfinite(x) or not isfinite(y):
            raise TikzFormattingError("point coordinates must be finite.")
        if self.invert_y_axis:
            if self.image_height is None:
                raise TikzFormattingError("image_height is required when invert_y_axis=True.")
            y = float(self.image_height) - y
        origin = self.origin if isinstance(self.origin, Point2D) else Point2D(0.0, 0.0)
        return Point2D((x - origin.x) * self.scale, (y - origin.y) * self.scale)

    def format_point(self, point: Point2D) -> str:
        """Return a TikZ coordinate tuple."""
        transformed = self.transform_point(point)
        return f"({self.format_length(transformed.x)},{self.format_length(transformed.y)})"

    def to_dict(self) -> dict[str, Any]:
        """Return serializable formatter diagnostics."""
        origin = self.origin if isinstance(self.origin, Point2D) else Point2D(0.0, 0.0)
        return {
            "precision": self.precision,
            "unit": self.unit,
            "scale": self.scale,
            "invert_y_axis": self.invert_y_axis,
            "image_height": self.image_height,
            "origin": origin.to_dict(),
        }


@dataclass(frozen=True)
class TikzPathBuilder:
    """Build readable ``\\draw`` commands from TikZ path fragments."""

    indent: str = "  "
    mode: TikzPathMode = TikzPathMode.AUTO
    max_points_per_line: int = 5
    split_long_paths: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", TikzPathMode(self.mode))
        if int(self.max_points_per_line) < 1:
            raise ValueError("max_points_per_line must be positive.")
        object.__setattr__(self, "max_points_per_line", int(self.max_points_per_line))
        if not isinstance(self.split_long_paths, bool):
            raise TypeError("split_long_paths must be a bool.")

    def draw_lines(
        self,
        path_fragments: list[str],
        *,
        options: str = "",
        level: int = 0,
        force_multiline: bool = False,
    ) -> list[str]:
        """Return one or more lines for a ``\\draw`` command."""
        if not path_fragments:
            raise TikzFormattingError("path_fragments must not be empty.")
        prefix = self.indent * level
        command = "\\draw" + (f"[{options}]" if options else "")
        inline = f"{prefix}{command} {' '.join(path_fragments)};"
        should_split = self._should_split(inline, path_fragments, force_multiline)
        if not should_split:
            return [inline]

        lines = [f"{prefix}{command}"]
        child_prefix = self.indent * (level + 1)
        for index, fragment in enumerate(path_fragments):
            suffix = ";" if index == len(path_fragments) - 1 else ""
            lines.append(f"{child_prefix}{fragment}{suffix}")
        return lines

    def _should_split(self, inline: str, path_fragments: list[str], force_multiline: bool) -> bool:
        if self.mode is TikzPathMode.INLINE:
            return False
        if self.mode is TikzPathMode.MULTILINE:
            return True
        if force_multiline:
            return True
        if not self.split_long_paths:
            return False
        coordinate_like = sum(1 for item in path_fragments if item.startswith("(") or " (" in item)
        return coordinate_like > self.max_points_per_line or len(inline) > 96


def coerce_path_mode(value: TikzPathMode | str) -> TikzPathMode:
    """Coerce user-facing path mode values."""
    if isinstance(value, TikzPathMode):
        return value
    normalized = str(value).strip().lower()
    for item in TikzPathMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported TikZ path mode: {value!r}")


def coerce_y_axis_mode(value: TikzYAxisMode | str) -> TikzYAxisMode:
    """Coerce user-facing y-axis mode values."""
    if isinstance(value, TikzYAxisMode):
        return value
    normalized = str(value).strip().lower()
    for item in TikzYAxisMode:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported TikZ y-axis mode: {value!r}")


__all__ = [
    "TikzCoordinateFormatter",
    "TikzFormattingError",
    "TikzPathBuilder",
    "TikzPathMode",
    "TikzYAxisMode",
    "coerce_path_mode",
    "coerce_y_axis_mode",
]
