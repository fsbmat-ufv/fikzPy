"""Semantic geometry primitives for the future Classic pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeAlias


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _positive_float(name: str, value: float) -> float:
    number = _finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _optional_unit_interval(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    number = _finite_float(name, value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return number


def _non_negative_optional(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    number = _finite_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _color_channel(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0 or value > 255:
        raise ValueError(f"{name} must be between 0 and 255.")
    return value


def _metadata_mapping(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    copied = dict(metadata)
    for key in copied:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings.")
    return MappingProxyType(copied)


def _require_point(name: str, value: "Point2D") -> "Point2D":
    if not isinstance(value, Point2D):
        raise TypeError(f"{name} must be a Point2D.")
    return value


def _point_tuple(name: str, points: Sequence["Point2D"], *, minimum: int) -> tuple["Point2D", ...]:
    items = tuple(points)
    if len(items) < minimum:
        raise ValueError(f"{name} requires at least {minimum} points.")
    for index, point in enumerate(items):
        _require_point(f"{name}[{index}]", point)
    return items


@dataclass(frozen=True)
class Point2D:
    """A finite two-dimensional coordinate."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_float("x", self.x))
        object.__setattr__(self, "y", _finite_float("y", self.y))

    @classmethod
    def from_pair(cls, pair: Sequence[float]) -> "Point2D":
        """Create a point from an ``(x, y)`` sequence."""
        if len(pair) != 2:
            raise ValueError("Point2D input must contain exactly two values.")
        return cls(float(pair[0]), float(pair[1]))

    def as_tuple(self) -> tuple[float, float]:
        """Return the point as an ``(x, y)`` tuple."""
        return self.x, self.y

    def to_dict(self) -> dict[str, float]:
        """Return a diagnostic dictionary for tests and logging."""
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class RGBColor:
    """An RGB color with channels in the inclusive 0-255 range."""

    red: int
    green: int
    blue: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "red", _color_channel("red", self.red))
        object.__setattr__(self, "green", _color_channel("green", self.green))
        object.__setattr__(self, "blue", _color_channel("blue", self.blue))

    @classmethod
    def black(cls) -> "RGBColor":
        """Return the default stroke color."""
        return cls(0, 0, 0)

    def to_dict(self) -> dict[str, int]:
        """Return a diagnostic dictionary for tests and logging."""
        return {"red": self.red, "green": self.green, "blue": self.blue}


@dataclass(frozen=True)
class StrokeStyle:
    """Stroke color, width, opacity, and optional line style hints."""

    color: RGBColor = field(default_factory=RGBColor.black)
    width: float = 1.0
    opacity: float | None = None
    line_cap: str | None = None
    line_join: str | None = None
    dash_pattern: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.color, RGBColor):
            raise TypeError("stroke color must be an RGBColor.")
        object.__setattr__(self, "width", _positive_float("stroke width", self.width))
        object.__setattr__(self, "opacity", _optional_unit_interval("stroke opacity", self.opacity))
        _validate_optional_text("line_cap", self.line_cap)
        _validate_optional_text("line_join", self.line_join)
        if self.dash_pattern is not None:
            dash = tuple(_finite_float("dash pattern value", value) for value in self.dash_pattern)
            if not dash:
                raise ValueError("dash_pattern must not be empty when provided.")
            if any(value < 0.0 for value in dash):
                raise ValueError("dash_pattern values must be non-negative.")
            object.__setattr__(self, "dash_pattern", dash)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "color": self.color.to_dict(),
            "width": self.width,
            "opacity": self.opacity,
            "line_cap": self.line_cap,
            "line_join": self.line_join,
            "dash_pattern": list(self.dash_pattern) if self.dash_pattern is not None else None,
        }


@dataclass(frozen=True)
class FillStyle:
    """Optional fill color and opacity for closed primitives."""

    color: RGBColor
    opacity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.color, RGBColor):
            raise TypeError("fill color must be an RGBColor.")
        object.__setattr__(self, "opacity", _optional_unit_interval("fill opacity", self.opacity))

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {"color": self.color.to_dict(), "opacity": self.opacity}


def _validate_optional_text(name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string when provided.")


@dataclass(frozen=True, kw_only=True)
class PrimitiveBase:
    """Shared non-geometric attributes for semantic primitives."""

    stroke: StrokeStyle = field(default_factory=StrokeStyle)
    fill: FillStyle | None = None
    opacity: float | None = None
    confidence: float | None = None
    error: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stroke, StrokeStyle):
            raise TypeError("stroke must be a StrokeStyle.")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise TypeError("fill must be a FillStyle or None.")
        object.__setattr__(self, "opacity", _optional_unit_interval("opacity", self.opacity))
        object.__setattr__(self, "confidence", _optional_unit_interval("confidence", self.confidence))
        object.__setattr__(self, "error", _non_negative_optional("error", self.error))
        object.__setattr__(self, "metadata", _metadata_mapping(self.metadata))

    def _common_dict(self) -> dict[str, Any]:
        return {
            "stroke": self.stroke.to_dict(),
            "fill": self.fill.to_dict() if self.fill is not None else None,
            "opacity": self.opacity,
            "confidence": self.confidence,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PointPrimitive(PrimitiveBase):
    """A semantic point marker."""

    point: Point2D

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_point("point", self.point)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {"type": "point", "point": self.point.to_dict(), **self._common_dict()}


@dataclass(frozen=True)
class LinePrimitive(PrimitiveBase):
    """A straight segment between two distinct points."""

    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_point("start", self.start)
        _require_point("end", self.end)
        if self.start == self.end:
            raise ValueError("LinePrimitive requires distinct start and end points.")

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "line",
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            **self._common_dict(),
        }


@dataclass(frozen=True)
class PolylinePrimitive(PrimitiveBase):
    """An open or closed path made of straight segments."""

    points: Sequence[Point2D]
    closed: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "points", _point_tuple("points", self.points, minimum=2))
        if not isinstance(self.closed, bool):
            raise TypeError("closed must be a bool.")

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "polyline",
            "points": [point.to_dict() for point in self.points],
            "closed": self.closed,
            **self._common_dict(),
        }


@dataclass(frozen=True)
class CirclePrimitive(PrimitiveBase):
    """A circle represented by center and radius."""

    center: Point2D
    radius: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_point("center", self.center)
        object.__setattr__(self, "radius", _positive_float("radius", self.radius))

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "circle",
            "center": self.center.to_dict(),
            "radius": self.radius,
            **self._common_dict(),
        }


@dataclass(frozen=True)
class EllipsePrimitive(PrimitiveBase):
    """An ellipse represented by center, radii, and rotation angle."""

    center: Point2D
    radius_x: float
    radius_y: float
    rotation: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_point("center", self.center)
        object.__setattr__(self, "radius_x", _positive_float("radius_x", self.radius_x))
        object.__setattr__(self, "radius_y", _positive_float("radius_y", self.radius_y))
        object.__setattr__(self, "rotation", _finite_float("rotation", self.rotation))

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "ellipse",
            "center": self.center.to_dict(),
            "radius_x": self.radius_x,
            "radius_y": self.radius_y,
            "rotation": self.rotation,
            **self._common_dict(),
        }


@dataclass(frozen=True)
class BezierPrimitive(PrimitiveBase):
    """A cubic Bezier segment with two control points."""

    start: Point2D
    control1: Point2D
    control2: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_point("start", self.start)
        _require_point("control1", self.control1)
        _require_point("control2", self.control2)
        _require_point("end", self.end)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "bezier",
            "start": self.start.to_dict(),
            "control1": self.control1.to_dict(),
            "control2": self.control2.to_dict(),
            "end": self.end.to_dict(),
            **self._common_dict(),
        }


@dataclass(frozen=True)
class ClosedShapePrimitive(PrimitiveBase):
    """A closed freeform shape represented by an ordered boundary."""

    points: Sequence[Point2D]
    closed: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "points", _point_tuple("points", self.points, minimum=3))
        if self.closed is not True:
            raise ValueError("ClosedShapePrimitive must be closed.")

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "closed_shape",
            "points": [point.to_dict() for point in self.points],
            "closed": True,
            **self._common_dict(),
        }


Primitive: TypeAlias = (
    PointPrimitive
    | LinePrimitive
    | PolylinePrimitive
    | CirclePrimitive
    | EllipsePrimitive
    | BezierPrimitive
    | ClosedShapePrimitive
)


@dataclass(frozen=True)
class PrimitiveGroup:
    """A diagnostic grouping of semantic primitives."""

    items: Sequence[Primitive | "PrimitiveGroup"]
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        for index, item in enumerate(items):
            if not isinstance(item, _GROUP_ITEM_TYPES):
                raise TypeError(f"items[{index}] must be a semantic primitive or PrimitiveGroup.")
        object.__setattr__(self, "items", items)
        _validate_optional_text("name", self.name)
        object.__setattr__(self, "metadata", _metadata_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic dictionary for tests and logging."""
        return {
            "type": "group",
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }

    def flatten(self) -> tuple[Primitive, ...]:
        """Return all nested primitive items in order."""
        flattened: list[Primitive] = []
        for item in self.items:
            if isinstance(item, PrimitiveGroup):
                flattened.extend(item.flatten())
            else:
                flattened.append(item)
        return tuple(flattened)


SemanticGeometry: TypeAlias = Primitive | PrimitiveGroup

_PRIMITIVE_TYPES = (
    PointPrimitive,
    LinePrimitive,
    PolylinePrimitive,
    CirclePrimitive,
    EllipsePrimitive,
    BezierPrimitive,
    ClosedShapePrimitive,
)
_GROUP_ITEM_TYPES = _PRIMITIVE_TYPES + (PrimitiveGroup,)

__all__ = [
    "BezierPrimitive",
    "CirclePrimitive",
    "ClosedShapePrimitive",
    "EllipsePrimitive",
    "FillStyle",
    "LinePrimitive",
    "Point2D",
    "PointPrimitive",
    "PolylinePrimitive",
    "Primitive",
    "PrimitiveBase",
    "PrimitiveGroup",
    "RGBColor",
    "SemanticGeometry",
    "StrokeStyle",
]
