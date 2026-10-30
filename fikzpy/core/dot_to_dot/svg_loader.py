"""Parse SVG path data into arc-length aware subpaths for dot-to-dot.

The loader works on an independent copy of whatever SVG string it is given
(strings are immutable, so parsing never mutates the caller's SVG) and reuses
the project's existing :mod:`fikzpy.core.svg_transforms` helpers to apply
element and group transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

from fikzpy.core.svg_transforms import SvgTransform, parse_transform_list

from fikzpy.core.dot_to_dot.models import (
    CubicSegment,
    LineSegment,
    QuadraticSegment,
    Segment,
    SubPath,
    SvgPath,
    Vec2,
)

_TOKEN_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_COMMANDS = set("AaCcHhLlMmQqSsTtVvZz")


class SvgLoaderError(ValueError):
    """Raised when SVG path data cannot be parsed for dot-to-dot."""


@dataclass(frozen=True)
class SvgLoaderConfig:
    """Configuration for parsing SVG into dot-to-dot subpaths."""

    apply_transforms: bool = True
    min_subpath_points: int = 2


def load_svg_paths(svg_source: str, config: SvgLoaderConfig | None = None) -> list[SvgPath]:
    """Parse an SVG document (a copy) into independent :class:`SvgPath` objects."""
    config = config or SvgLoaderConfig()
    try:
        root = ET.fromstring(svg_source)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise SvgLoaderError(f"Invalid SVG document: {exc}") from exc

    paths: list[SvgPath] = []
    counter = 0

    def walk(element: ET.Element, transform: SvgTransform) -> None:
        nonlocal counter
        local = transform
        if config.apply_transforms:
            transform_attr = element.get("transform")
            if transform_attr:
                local = transform.multiply(parse_transform_list(transform_attr))
        if _local_tag(element.tag) == "path":
            data = element.get("d")
            if data:
                subpaths = _parse_path_data(data, local)
                if subpaths:
                    paths.append(SvgPath(tuple(subpaths), counter, element.get("id")))
                    counter += 1
        for child in element:
            walk(child, local)

    walk(root, SvgTransform.identity())
    return paths


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_path_data(data: str, transform: SvgTransform) -> list[SubPath]:
    tokens = _TOKEN_RE.findall(data)
    index = 0

    def read_number() -> float:
        nonlocal index
        while index < len(tokens) and tokens[index] in _COMMANDS:
            index += 1  # pragma: no cover - guards malformed streams
        if index >= len(tokens):
            raise SvgLoaderError("Unexpected end of SVG path data.")
        value = tokens[index]
        index += 1
        return float(value)

    def apply(x: float, y: float) -> Vec2:
        point = transform.apply_xy(x, y)
        return Vec2(float(point.x), float(point.y))

    subpaths: list[SubPath] = []
    segments: list[Segment] = []
    closed = False
    raw_current = (0.0, 0.0)
    raw_start = (0.0, 0.0)
    prev_cubic_control: tuple[float, float] | None = None
    prev_quad_control: tuple[float, float] | None = None

    def flush(is_closed: bool) -> None:
        nonlocal segments
        if len(segments) >= 1:
            subpaths.append(SubPath(segments, closed=is_closed))
        segments = []

    command = ""
    while index < len(tokens):
        token = tokens[index]
        if token in _COMMANDS:
            command = token
            index += 1
        relative = command.islower()
        upper = command.upper()

        if upper == "M":
            if segments:
                flush(closed)
            closed = False
            x = read_number()
            y = read_number()
            if relative:
                x += raw_current[0]
                y += raw_current[1]
            raw_current = (x, y)
            raw_start = (x, y)
            prev_cubic_control = None
            prev_quad_control = None
            command = "l" if relative else "L"
            continue
        if upper == "L":
            x = read_number()
            y = read_number()
            if relative:
                x += raw_current[0]
                y += raw_current[1]
            segments.append(LineSegment(apply(*raw_current), apply(x, y)))
            raw_current = (x, y)
            prev_cubic_control = None
            prev_quad_control = None
            continue
        if upper == "H":
            x = read_number()
            if relative:
                x += raw_current[0]
            target = (x, raw_current[1])
            segments.append(LineSegment(apply(*raw_current), apply(*target)))
            raw_current = target
            prev_cubic_control = None
            prev_quad_control = None
            continue
        if upper == "V":
            y = read_number()
            if relative:
                y += raw_current[1]
            target = (raw_current[0], y)
            segments.append(LineSegment(apply(*raw_current), apply(*target)))
            raw_current = target
            prev_cubic_control = None
            prev_quad_control = None
            continue
        if upper == "C":
            coords = [read_number() for _ in range(6)]
            if relative:
                coords = _offset(coords, raw_current)
            c1 = (coords[0], coords[1])
            c2 = (coords[2], coords[3])
            end = (coords[4], coords[5])
            segments.append(CubicSegment(apply(*raw_current), apply(*c1), apply(*c2), apply(*end)))
            raw_current = end
            prev_cubic_control = c2
            prev_quad_control = None
            continue
        if upper == "S":
            coords = [read_number() for _ in range(4)]
            if relative:
                coords = _offset(coords, raw_current)
            reflected = _reflect(prev_cubic_control, raw_current)
            c2 = (coords[0], coords[1])
            end = (coords[2], coords[3])
            segments.append(CubicSegment(apply(*raw_current), apply(*reflected), apply(*c2), apply(*end)))
            raw_current = end
            prev_cubic_control = c2
            prev_quad_control = None
            continue
        if upper == "Q":
            coords = [read_number() for _ in range(4)]
            if relative:
                coords = _offset(coords, raw_current)
            control = (coords[0], coords[1])
            end = (coords[2], coords[3])
            segments.append(QuadraticSegment(apply(*raw_current), apply(*control), apply(*end)))
            raw_current = end
            prev_quad_control = control
            prev_cubic_control = None
            continue
        if upper == "T":
            coords = [read_number() for _ in range(2)]
            if relative:
                coords = _offset(coords, raw_current)
            control = _reflect(prev_quad_control, raw_current)
            end = (coords[0], coords[1])
            segments.append(QuadraticSegment(apply(*raw_current), apply(*control), apply(*end)))
            raw_current = end
            prev_quad_control = control
            prev_cubic_control = None
            continue
        if upper == "A":
            # Elliptical arcs are not produced by the project's SVG stage; keep
            # the loader robust by approximating them with a straight chord.
            _ = [read_number() for _ in range(5)]
            x = read_number()
            y = read_number()
            if relative:
                x += raw_current[0]
                y += raw_current[1]
            segments.append(LineSegment(apply(*raw_current), apply(x, y)))
            raw_current = (x, y)
            prev_cubic_control = None
            prev_quad_control = None
            continue
        if upper == "Z":
            if raw_current != raw_start:
                segments.append(LineSegment(apply(*raw_current), apply(*raw_start)))
            raw_current = raw_start
            flush(True)
            closed = False
            prev_cubic_control = None
            prev_quad_control = None
            continue
        raise SvgLoaderError(f"Unsupported SVG path command: {command!r}")

    if segments:
        flush(False)
    return [subpath for subpath in subpaths if len(subpath.segments) >= 1]


def _offset(coords: list[float], origin: tuple[float, float]) -> list[float]:
    result: list[float] = []
    for position, value in enumerate(coords):
        result.append(value + (origin[0] if position % 2 == 0 else origin[1]))
    return result


def _reflect(control: tuple[float, float] | None, current: tuple[float, float]) -> tuple[float, float]:
    if control is None:
        return current
    return (2.0 * current[0] - control[0], 2.0 * current[1] - control[1])


__all__ = ["SvgLoaderConfig", "SvgLoaderError", "load_svg_paths"]
