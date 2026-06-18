"""Small deterministic rasterizer for semantic primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import cos, isfinite, radians, sin
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fikzpy.core.semantic_geometry import BezierPrimitive, CirclePrimitive, ClosedShapePrimitive
from fikzpy.core.semantic_geometry import EllipsePrimitive, FillStyle, LinePrimitive, Point2D, PointPrimitive
from fikzpy.core.semantic_geometry import PolylinePrimitive, Primitive, PrimitiveGroup, RGBColor
from fikzpy.core.semantic_geometry import SemanticGeometry, StrokeStyle


class RasterizationError(ValueError):
    """Raised when semantic rasterization cannot continue."""


class RasterizationBackend(Enum):
    """Available semantic rasterization backends."""

    PIL_IMAGE_DRAW = "pil_image_draw"


@dataclass(frozen=True)
class SemanticRasterizationConfig:
    """Configuration for deterministic primitive rasterization."""

    canvas_size: tuple[int, int] | None = None
    scale: float = 1.0
    padding: int = 4
    background_color: tuple[int, int, int] = (255, 255, 255)
    invert_y_axis: bool = False
    image_height: float | None = None
    point_radius: float = 2.0
    bezier_samples: int = 48
    ellipse_samples: int = 96
    minimum_stroke_width: int = 1
    strict: bool = False
    backend: RasterizationBackend | str = RasterizationBackend.PIL_IMAGE_DRAW

    def __post_init__(self) -> None:
        if self.canvas_size is not None:
            if len(self.canvas_size) != 2:
                raise ValueError("canvas_size must contain width and height.")
            width, height = int(self.canvas_size[0]), int(self.canvas_size[1])
            if width <= 0 or height <= 0:
                raise ValueError("canvas_size values must be positive.")
            object.__setattr__(self, "canvas_size", (width, height))
        scale = float(self.scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        object.__setattr__(self, "scale", scale)
        padding = int(self.padding)
        if padding < 0:
            raise ValueError("padding must be non-negative.")
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "background_color", _coerce_rgb_tuple("background_color", self.background_color))
        if not isinstance(self.invert_y_axis, bool):
            raise TypeError("invert_y_axis must be a bool.")
        if self.image_height is not None:
            height = float(self.image_height)
            if not isfinite(height):
                raise ValueError("image_height must be finite when provided.")
            object.__setattr__(self, "image_height", height)
        for name in ("point_radius",):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        for name in ("bezier_samples", "ellipse_samples", "minimum_stroke_width"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a bool.")
        object.__setattr__(self, "backend", _coerce_backend(self.backend))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable configuration diagnostics."""
        return {
            "canvas_size": list(self.canvas_size) if self.canvas_size else None,
            "scale": self.scale,
            "padding": self.padding,
            "background_color": list(self.background_color),
            "invert_y_axis": self.invert_y_axis,
            "image_height": self.image_height,
            "point_radius": self.point_radius,
            "bezier_samples": self.bezier_samples,
            "ellipse_samples": self.ellipse_samples,
            "minimum_stroke_width": self.minimum_stroke_width,
            "strict": self.strict,
            "backend": self.backend.value,
        }


@dataclass(frozen=True)
class SemanticRasterizationResult:
    """Raster image and diagnostics produced from semantic primitives."""

    image: np.ndarray
    config: SemanticRasterizationConfig
    backend: RasterizationBackend
    primitive_count: int
    canvas_size: tuple[int, int]
    warnings: tuple[str, ...]
    deterministic_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic diagnostics without embedding image bytes."""
        return {
            "image": {
                "shape": list(self.image.shape),
                "dtype": str(self.image.dtype),
                "sha256": sha256(np.ascontiguousarray(self.image).tobytes()).hexdigest(),
            },
            "config": self.config.to_dict(),
            "backend": self.backend.value,
            "primitive_count": self.primitive_count,
            "canvas_size": list(self.canvas_size),
            "warnings": list(self.warnings),
            "deterministic_hash": self.deterministic_hash,
        }


def rasterize_semantic_primitives(
    primitives: Any,
    config: SemanticRasterizationConfig | None = None,
) -> SemanticRasterizationResult:
    """Rasterize semantic primitives with a lightweight deterministic backend."""
    return _SemanticRasterizer(config or SemanticRasterizationConfig()).rasterize(primitives)


class _SemanticRasterizer:
    def __init__(self, config: SemanticRasterizationConfig) -> None:
        self.config = config
        self.warnings: list[str] = []
        self._transform = _CoordinateTransform(config, Point2D(0.0, 0.0))

    def rasterize(self, primitives: Any) -> SemanticRasterizationResult:
        items = _normalize_input(primitives)
        canvas_size, origin = self._canvas(items)
        self._transform = _CoordinateTransform(self.config, origin)
        base = Image.new("RGBA", canvas_size, (*self.config.background_color, 255))
        for item in items:
            self._draw_item(base, item)
        image = np.asarray(base.convert("RGB"), dtype=np.uint8).copy()
        payload = {
            "config": self.config.to_dict(),
            "primitive_count": _object_count(items),
            "canvas_size": list(canvas_size),
            "warnings": list(self.warnings),
            "image_sha256": sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        }
        digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        return SemanticRasterizationResult(
            image=image,
            config=self.config,
            backend=self.config.backend,
            primitive_count=_object_count(items),
            canvas_size=canvas_size,
            warnings=tuple(self.warnings),
            deterministic_hash=digest,
        )

    def _canvas(self, items: tuple[SemanticGeometry, ...]) -> tuple[tuple[int, int], Point2D]:
        if self.config.canvas_size is not None:
            origin = Point2D(0.0, 0.0)
            return self.config.canvas_size, origin
        bounds = _bounds(items)
        if bounds is None:
            return (1, 1), Point2D(0.0, 0.0)
        min_x, min_y, max_x, max_y = bounds
        padding = self.config.padding
        width = max(1, int(round((max_x - min_x) * self.config.scale)) + padding * 2 + 1)
        height = max(1, int(round((max_y - min_y) * self.config.scale)) + padding * 2 + 1)
        return (width, height), Point2D(min_x - padding / self.config.scale, min_y - padding / self.config.scale)

    def _draw_item(self, base: Image.Image, item: SemanticGeometry) -> None:
        if isinstance(item, PrimitiveGroup):
            if not item.items:
                self._warn("empty_group")
            for child in item.items:
                self._draw_item(base, child)
            return
        try:
            self._draw_primitive(base, item)
        except Exception as exc:
            if self.config.strict:
                raise RasterizationError(str(exc)) from exc
            self._warn(f"skipped_{type(item).__name__}")

    def _draw_primitive(self, base: Image.Image, primitive: Primitive) -> None:
        fill = _fill_rgba(primitive.fill, primitive.opacity)
        stroke = _stroke_rgba(primitive.stroke, primitive.opacity)
        width = max(self.config.minimum_stroke_width, int(round(primitive.stroke.width * self.config.scale)))
        if stroke[3] == 0 and (fill is None or fill[3] == 0):
            self._warn("invisible_primitive")
            return
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if isinstance(primitive, PointPrimitive):
            self._draw_point(draw, primitive, stroke)
        elif isinstance(primitive, LinePrimitive):
            draw.line([self._xy(primitive.start), self._xy(primitive.end)], fill=stroke, width=width)
        elif isinstance(primitive, PolylinePrimitive):
            self._draw_polyline(draw, primitive, stroke, fill, width)
        elif isinstance(primitive, CirclePrimitive):
            self._draw_circle(draw, primitive, stroke, fill, width)
        elif isinstance(primitive, EllipsePrimitive):
            self._draw_ellipse(draw, primitive, stroke, fill, width)
        elif isinstance(primitive, BezierPrimitive):
            draw.line([self._xy(point) for point in _bezier_points(primitive, self.config.bezier_samples)], fill=stroke, width=width)
        elif isinstance(primitive, ClosedShapePrimitive):
            points = [self._xy(point) for point in _without_repeated_close(tuple(primitive.points))]
            if fill is not None and fill[3] > 0:
                draw.polygon(points, fill=fill)
            draw.line(points + [points[0]], fill=stroke, width=width)
        base.alpha_composite(overlay)

    def _draw_point(self, draw: ImageDraw.ImageDraw, primitive: PointPrimitive, stroke: tuple[int, int, int, int]) -> None:
        x, y = self._xy(primitive.point)
        radius = self.config.point_radius * self.config.scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=stroke)

    def _draw_polyline(
        self,
        draw: ImageDraw.ImageDraw,
        primitive: PolylinePrimitive,
        stroke: tuple[int, int, int, int],
        fill: tuple[int, int, int, int] | None,
        width: int,
    ) -> None:
        points = [self._xy(point) for point in _without_repeated_close(tuple(primitive.points))]
        if primitive.closed and fill is not None and fill[3] > 0 and len(points) >= 3:
            draw.polygon(points, fill=fill)
        draw.line(points + ([points[0]] if primitive.closed else []), fill=stroke, width=width)

    def _draw_circle(
        self,
        draw: ImageDraw.ImageDraw,
        primitive: CirclePrimitive,
        stroke: tuple[int, int, int, int],
        fill: tuple[int, int, int, int] | None,
        width: int,
    ) -> None:
        center = self._transform.point(primitive.center)
        radius = primitive.radius * self.config.scale
        box = (center.x - radius, center.y - radius, center.x + radius, center.y + radius)
        draw.ellipse(box, fill=fill if fill and fill[3] > 0 else None, outline=stroke, width=width)

    def _draw_ellipse(
        self,
        draw: ImageDraw.ImageDraw,
        primitive: EllipsePrimitive,
        stroke: tuple[int, int, int, int],
        fill: tuple[int, int, int, int] | None,
        width: int,
    ) -> None:
        center = self._transform.point(primitive.center)
        rx = primitive.radius_x * self.config.scale
        ry = primitive.radius_y * self.config.scale
        if abs(primitive.rotation) <= 1e-12:
            box = (center.x - rx, center.y - ry, center.x + rx, center.y + ry)
            draw.ellipse(box, fill=fill if fill and fill[3] > 0 else None, outline=stroke, width=width)
            return
        points = _ellipse_polygon(center, rx, ry, primitive.rotation, self.config.ellipse_samples)
        if fill is not None and fill[3] > 0:
            draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=stroke, width=width)

    def _xy(self, point: Point2D) -> tuple[float, float]:
        transformed = self._transform.point(point)
        return transformed.x, transformed.y

    def _warn(self, code: str) -> None:
        if self.config.strict:
            raise RasterizationError(code)
        self.warnings.append(code)


@dataclass(frozen=True)
class _CoordinateTransform:
    config: SemanticRasterizationConfig
    origin: Point2D

    def point(self, point: Point2D) -> Point2D:
        x = (point.x - self.origin.x) * self.config.scale
        y_value = point.y
        if self.config.invert_y_axis:
            if self.config.image_height is None:
                raise RasterizationError("image_height is required when invert_y_axis=True.")
            y_value = self.config.image_height - y_value
        y = (y_value - self.origin.y) * self.config.scale
        return Point2D(x, y)


def _bounds(items: tuple[SemanticGeometry, ...]) -> tuple[float, float, float, float] | None:
    values: list[tuple[float, float, float, float]] = []
    for item in items:
        if isinstance(item, PrimitiveGroup):
            child = _bounds(tuple(item.items))
            if child is not None:
                values.append(child)
        elif isinstance(item, PointPrimitive):
            values.append((item.point.x, item.point.y, item.point.x, item.point.y))
        elif isinstance(item, LinePrimitive):
            values.append(_point_bounds((item.start, item.end)))
        elif isinstance(item, PolylinePrimitive):
            values.append(_point_bounds(tuple(item.points)))
        elif isinstance(item, CirclePrimitive):
            values.append((item.center.x - item.radius, item.center.y - item.radius, item.center.x + item.radius, item.center.y + item.radius))
        elif isinstance(item, EllipsePrimitive):
            radius = max(item.radius_x, item.radius_y)
            values.append((item.center.x - radius, item.center.y - radius, item.center.x + radius, item.center.y + radius))
        elif isinstance(item, BezierPrimitive):
            values.append(_point_bounds((item.start, item.control1, item.control2, item.end)))
        elif isinstance(item, ClosedShapePrimitive):
            values.append(_point_bounds(tuple(item.points)))
    if not values:
        return None
    return (
        min(value[0] for value in values),
        min(value[1] for value in values),
        max(value[2] for value in values),
        max(value[3] for value in values),
    )


def _point_bounds(points: tuple[Point2D, ...]) -> tuple[float, float, float, float]:
    return (
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _bezier_points(primitive: BezierPrimitive, samples: int) -> tuple[Point2D, ...]:
    points: list[Point2D] = []
    for index in range(samples + 1):
        t = index / samples
        omt = 1.0 - t
        x = (
            omt**3 * primitive.start.x
            + 3 * omt**2 * t * primitive.control1.x
            + 3 * omt * t**2 * primitive.control2.x
            + t**3 * primitive.end.x
        )
        y = (
            omt**3 * primitive.start.y
            + 3 * omt**2 * t * primitive.control1.y
            + 3 * omt * t**2 * primitive.control2.y
            + t**3 * primitive.end.y
        )
        points.append(Point2D(x, y))
    return tuple(points)


def _ellipse_polygon(center: Point2D, rx: float, ry: float, rotation: float, samples: int) -> list[tuple[float, float]]:
    angle = radians(rotation)
    ca = cos(angle)
    sa = sin(angle)
    points: list[tuple[float, float]] = []
    for index in range(samples):
        theta = 2.0 * np.pi * index / samples
        x = rx * cos(theta)
        y = ry * sin(theta)
        points.append((center.x + x * ca - y * sa, center.y + x * sa + y * ca))
    return points


def _stroke_rgba(stroke: StrokeStyle, opacity: float | None) -> tuple[int, int, int, int]:
    alpha = _alpha((1.0 if stroke.opacity is None else stroke.opacity) * (1.0 if opacity is None else opacity))
    return (*_rgb(stroke.color), alpha)


def _fill_rgba(fill: FillStyle | None, opacity: float | None) -> tuple[int, int, int, int] | None:
    if fill is None:
        return None
    alpha = _alpha((1.0 if fill.opacity is None else fill.opacity) * (1.0 if opacity is None else opacity))
    return (*_rgb(fill.color), alpha)


def _rgb(color: RGBColor) -> tuple[int, int, int]:
    return int(color.red), int(color.green), int(color.blue)


def _alpha(value: float) -> int:
    if not isfinite(float(value)):
        return 0
    return max(0, min(255, int(round(float(value) * 255))))


def _coerce_rgb_tuple(name: str, value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three RGB channels.")
    channels = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"{name} channels must be between 0 and 255.")
    return channels


def _coerce_backend(value: RasterizationBackend | str) -> RasterizationBackend:
    if isinstance(value, RasterizationBackend):
        return value
    normalized = str(value).strip().lower()
    for item in RasterizationBackend:
        if normalized in {item.value, item.name.lower()}:
            return item
    raise ValueError(f"Unsupported rasterization backend: {value!r}")


def _normalize_input(value: Any) -> tuple[SemanticGeometry, ...]:
    if value is None:
        return ()
    if isinstance(value, PrimitiveGroup):
        return (value,)
    if isinstance(value, _PRIMITIVE_TYPES):
        return (value,)
    if _looks_like_geometry_optimization_result(value):
        return tuple(value.primitives)
    if _looks_like_primitive_fit_result(value):
        return tuple(value.primitives)
    if _looks_like_svg_parse_result(value):
        return tuple(value.primitives)
    if _looks_like_centerline_result(value):
        return tuple(path.to_polyline_primitive() for path in value.paths)
    if _looks_like_centerline_path(value):
        return (value.to_polyline_primitive(),)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Path, Mapping)):
        output: list[SemanticGeometry] = []
        for item in value:
            output.extend(_normalize_input(item))
        return tuple(output)
    raise TypeError(f"Unsupported input for semantic rasterization: {type(value).__name__}")


def _object_count(items: tuple[SemanticGeometry, ...]) -> int:
    count = 0
    for item in items:
        if isinstance(item, PrimitiveGroup):
            count += _object_count(tuple(item.items))
        else:
            count += 1
    return count


def _without_repeated_close(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    if len(points) > 1 and points[0] == points[-1]:
        return points[:-1]
    return points


def _looks_like_geometry_optimization_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "operations") and hasattr(value, "deterministic_hash")


def _looks_like_primitive_fit_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "selected_kind") and hasattr(value, "candidates")


def _looks_like_svg_parse_result(value: Any) -> bool:
    return hasattr(value, "primitives") and hasattr(value, "document_info") and hasattr(value, "input_hash")


def _looks_like_centerline_result(value: Any) -> bool:
    return hasattr(value, "paths") and hasattr(value, "metrics") and type(value).__name__ == "CenterlineResult"


def _looks_like_centerline_path(value: Any) -> bool:
    return hasattr(value, "points") and hasattr(value, "closure") and hasattr(value, "to_polyline_primitive")


_PRIMITIVE_TYPES = (
    PointPrimitive,
    LinePrimitive,
    PolylinePrimitive,
    CirclePrimitive,
    EllipsePrimitive,
    BezierPrimitive,
    ClosedShapePrimitive,
)


__all__ = [
    "RasterizationBackend",
    "RasterizationError",
    "SemanticRasterizationConfig",
    "SemanticRasterizationResult",
    "rasterize_semantic_primitives",
]
