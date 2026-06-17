"""SVG affine transform parsing for the semantic SVG parser."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians, sin, tan
import re
from typing import Any

from fikzpy.core.semantic_geometry import Point2D


class SvgTransformError(ValueError):
    """Raised when an SVG transform cannot be parsed safely."""


_TRANSFORM_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_EPSILON = 1e-9


@dataclass(frozen=True)
class SvgTransform:
    """A two-dimensional SVG affine transform.

    The matrix is stored as ``(a, b, c, d, e, f)`` and applied as:

    ``x' = a*x + c*y + e``
    ``y' = b*x + d*y + f``
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d", "e", "f"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise SvgTransformError(f"{name} must be finite.")
            object.__setattr__(self, name, value)

    @classmethod
    def identity(cls) -> "SvgTransform":
        """Return the identity transform."""
        return cls()

    @classmethod
    def translate(cls, tx: float, ty: float = 0.0) -> "SvgTransform":
        """Return a translation transform."""
        return cls(1.0, 0.0, 0.0, 1.0, tx, ty)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> "SvgTransform":
        """Return a scale transform."""
        y_scale = sx if sy is None else sy
        return cls(sx, 0.0, 0.0, y_scale, 0.0, 0.0)

    @classmethod
    def rotate(cls, angle_degrees: float, cx: float | None = None, cy: float | None = None) -> "SvgTransform":
        """Return a rotation transform, optionally around a center."""
        angle = radians(angle_degrees)
        rotation = cls(cos(angle), sin(angle), -sin(angle), cos(angle), 0.0, 0.0)
        if cx is None and cy is None:
            return rotation
        if cx is None or cy is None:
            raise SvgTransformError("rotate center requires both cx and cy.")
        return cls.translate(cx, cy).multiply(rotation).multiply(cls.translate(-cx, -cy))

    @classmethod
    def skew_x(cls, angle_degrees: float) -> "SvgTransform":
        """Return a skewX transform."""
        return cls(1.0, 0.0, tan(radians(angle_degrees)), 1.0, 0.0, 0.0)

    @classmethod
    def skew_y(cls, angle_degrees: float) -> "SvgTransform":
        """Return a skewY transform."""
        return cls(1.0, tan(radians(angle_degrees)), 0.0, 1.0, 0.0, 0.0)

    def multiply(self, other: "SvgTransform") -> "SvgTransform":
        """Return ``self * other`` using SVG affine matrix multiplication."""
        if not isinstance(other, SvgTransform):
            raise TypeError("other must be SvgTransform.")
        return SvgTransform(
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f,
        )

    def apply(self, point: Point2D) -> Point2D:
        """Apply the transform to a point."""
        return Point2D(self.a * point.x + self.c * point.y + self.e, self.b * point.x + self.d * point.y + self.f)

    def apply_xy(self, x: float, y: float) -> Point2D:
        """Apply the transform to raw x/y coordinates."""
        return self.apply(Point2D(x, y))

    @property
    def determinant(self) -> float:
        """Return the determinant of the linear part."""
        return self.a * self.d - self.b * self.c

    def axis_scales(self) -> tuple[float, float]:
        """Return the transformed lengths of unit x and y axes."""
        return hypot(self.a, self.b), hypot(self.c, self.d)

    def has_orthogonal_axes(self, *, tolerance: float = 1e-7) -> bool:
        """Return whether the linear axes are approximately orthogonal."""
        sx, sy = self.axis_scales()
        if sx <= _EPSILON or sy <= _EPSILON:
            return False
        dot = self.a * self.c + self.b * self.d
        return abs(dot) <= tolerance * max(sx * sy, 1.0)

    def is_similarity(self, *, tolerance: float = 1e-7) -> bool:
        """Return whether the transform is rotation/reflection plus uniform scale."""
        sx, sy = self.axis_scales()
        return self.has_orthogonal_axes(tolerance=tolerance) and abs(sx - sy) <= tolerance * max(sx, sy, 1.0)

    def has_general_shear(self, *, tolerance: float = 1e-7) -> bool:
        """Return whether the transform cannot preserve simple axis ellipses."""
        return not self.has_orthogonal_axes(tolerance=tolerance)

    def rotation_degrees(self) -> float:
        """Return the x-axis rotation angle in degrees."""
        from math import atan2, degrees

        return degrees(atan2(self.b, self.a))

    def stroke_scale(self) -> float:
        """Return a conservative scalar to apply to stroke widths."""
        sx, sy = self.axis_scales()
        if sx <= _EPSILON and sy <= _EPSILON:
            return 1.0
        if self.has_orthogonal_axes():
            return max(_EPSILON, (sx + sy) / 2.0)
        determinant_scale = abs(self.determinant) ** 0.5
        return max(_EPSILON, determinant_scale)

    def to_dict(self) -> dict[str, float]:
        """Return a serializable transform dictionary."""
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "e": self.e, "f": self.f}


def parse_transform_list(value: str | None) -> SvgTransform:
    """Parse an SVG transform list into one affine transform."""
    if value is None or not str(value).strip():
        return SvgTransform.identity()
    text = str(value).strip()
    position = 0
    transform = SvgTransform.identity()
    found = False
    for match in _TRANSFORM_RE.finditer(text):
        between = text[position : match.start()].strip()
        if between and between.strip(","):
            raise SvgTransformError(f"Invalid transform syntax near {between!r}.")
        operation = _transform_from_call(match.group(1), _parse_numbers(match.group(2)))
        transform = transform.multiply(operation)
        position = match.end()
        found = True
    trailing = text[position:].strip()
    if trailing and trailing.strip(","):
        raise SvgTransformError(f"Invalid transform syntax near {trailing!r}.")
    if not found:
        raise SvgTransformError(f"Invalid transform list: {value!r}.")
    return transform


def _transform_from_call(name: str, values: tuple[float, ...]) -> SvgTransform:
    normalized = name.strip().lower()
    if normalized == "matrix":
        if len(values) != 6:
            raise SvgTransformError("matrix() requires six numbers.")
        return SvgTransform(*values)
    if normalized == "translate":
        if len(values) not in {1, 2}:
            raise SvgTransformError("translate() requires one or two numbers.")
        return SvgTransform.translate(values[0], values[1] if len(values) == 2 else 0.0)
    if normalized == "scale":
        if len(values) not in {1, 2}:
            raise SvgTransformError("scale() requires one or two numbers.")
        return SvgTransform.scale(values[0], values[1] if len(values) == 2 else None)
    if normalized == "rotate":
        if len(values) == 1:
            return SvgTransform.rotate(values[0])
        if len(values) == 3:
            return SvgTransform.rotate(values[0], values[1], values[2])
        raise SvgTransformError("rotate() requires one or three numbers.")
    if normalized == "skewx":
        if len(values) != 1:
            raise SvgTransformError("skewX() requires one number.")
        return SvgTransform.skew_x(values[0])
    if normalized == "skewy":
        if len(values) != 1:
            raise SvgTransformError("skewY() requires one number.")
        return SvgTransform.skew_y(values[0])
    raise SvgTransformError(f"Unsupported transform function: {name}.")


def _parse_numbers(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    values = tuple(float(match.group(0)) for match in _NUMBER_RE.finditer(text))
    consumed = _NUMBER_RE.sub(" ", text)
    if consumed.replace(",", " ").strip():
        raise SvgTransformError(f"Invalid transform number list: {text!r}.")
    if not all(isfinite(value) for value in values):
        raise SvgTransformError("Transform numbers must be finite.")
    return values


__all__ = ["SvgTransform", "SvgTransformError", "parse_transform_list"]
