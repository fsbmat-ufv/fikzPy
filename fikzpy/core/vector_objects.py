"""Internal vector geometry objects used before TikZ export."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


@dataclass(frozen=True)
class Point:
    """A two-dimensional point in vector space."""

    x: float
    y: float

    @classmethod
    def from_pair(cls, pair: tuple[float, float] | list[float]) -> "Point":
        """Create a point from a two-value sequence."""
        if len(pair) != 2:
            raise ValueError("Point input must contain exactly two values.")
        return cls(float(pair[0]), float(pair[1]))

    def as_tuple(self) -> tuple[float, float]:
        """Return the point as an ``(x, y)`` tuple."""
        return self.x, self.y

    def distance_to(self, other: "Point") -> float:
        """Return Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Line:
    """A straight segment between two points."""

    start: Point
    end: Point

    @property
    def length(self) -> float:
        """Return segment length."""
        return self.start.distance_to(self.end)


@dataclass(frozen=True)
class Polyline:
    """A path made of straight segments."""

    points: tuple[Point, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Polyline requires at least two points.")
        object.__setattr__(self, "points", tuple(self.points))

    @property
    def length(self) -> float:
        """Return total polyline length."""
        total = sum(first.distance_to(second) for first, second in zip(self.points, self.points[1:]))
        if self.closed and len(self.points) > 2:
            total += self.points[-1].distance_to(self.points[0])
        return total


@dataclass(frozen=True)
class BezierCurve:
    """A cubic Bezier curve segment."""

    start: Point
    control1: Point
    control2: Point
    end: Point


@dataclass(frozen=True)
class Circle:
    """A circle primitive."""

    center: Point
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Circle radius must be positive.")


@dataclass(frozen=True)
class Ellipse:
    """An ellipse primitive."""

    center: Point
    radius_x: float
    radius_y: float
    rotation: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_x <= 0 or self.radius_y <= 0:
            raise ValueError("Ellipse radii must be positive.")


@dataclass(frozen=True)
class Rectangle:
    """An axis-aligned rectangle primitive."""

    corner1: Point
    corner2: Point


@dataclass(frozen=True)
class Arc:
    """A circular or elliptical arc primitive."""

    center: Point
    radius_x: float
    radius_y: float
    start_angle: float
    end_angle: float
    rotation: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_x <= 0 or self.radius_y <= 0:
            raise ValueError("Arc radii must be positive.")


@dataclass(frozen=True)
class Node:
    """A TikZ node-like text object."""

    position: Point
    text: str
    name: str | None = None


VectorPrimitive: TypeAlias = Line | Polyline | BezierCurve | Circle | Ellipse | Rectangle | Arc | Node


@dataclass(frozen=True)
class PathGroup:
    """A logical group of vector objects."""

    items: tuple[VectorPrimitive | "PathGroup", ...]
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))

    def flatten(self) -> tuple[VectorPrimitive, ...]:
        """Return all primitive objects contained in this group."""
        flattened: list[VectorPrimitive] = []
        for item in self.items:
            if isinstance(item, PathGroup):
                flattened.extend(item.flatten())
            else:
                flattened.append(item)
        return tuple(flattened)


VectorObject: TypeAlias = VectorPrimitive | PathGroup
