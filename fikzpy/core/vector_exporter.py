"""TikZ exporter for internal vector objects."""

from __future__ import annotations

from dataclasses import dataclass
import re

from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.vector_objects import Arc, BezierCurve, Circle, Ellipse, Line, Node
from fikzpy.core.vector_objects import PathGroup, Point, Polyline, Rectangle, VectorObject


@dataclass(frozen=True)
class VectorObjectStats:
    """Counts of vector object types exported to TikZ."""

    total: int = 0
    lines: int = 0
    polylines: int = 0
    bezier_curves: int = 0
    circles: int = 0
    ellipses: int = 0
    rectangles: int = 0
    arcs: int = 0
    nodes: int = 0


_SAFE_COLOR = re.compile(r"^[A-Za-z][A-Za-z0-9!._-]*$")


def generate_tikz_from_vector_objects(
    objects: tuple[VectorObject, ...] | list[VectorObject],
    *,
    options: TikzOptions | None = None,
    diagnostic_marker: bool = False,
) -> str:
    """Generate a tikzpicture environment from internal vector objects."""
    options = options or TikzOptions()
    flattened = flatten_vector_objects(tuple(objects))
    lines = [
        f"\\begin{{tikzpicture}}[scale={_fmt_number(options.tikz_scale, 3)}]",
    ]
    if diagnostic_marker:
        lines.append("  % FIKZPY VECTOR MODE")

    lines.append("  \\begin{scope}[line cap=round, line join=round]")

    for item in flattened:
        command = _object_to_tikz(item, options)
        if command:
            lines.append(command)

    lines.append("  \\end{scope}")
    if diagnostic_marker:
        lines.append("  \\node[anchor=north west, font=\\tiny] at (current bounding box.north west) {VECTOR MODE};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def count_vector_objects(objects: tuple[VectorObject, ...] | list[VectorObject]) -> VectorObjectStats:
    """Count vector object types, flattening groups."""
    stats = {
        "total": 0,
        "lines": 0,
        "polylines": 0,
        "bezier_curves": 0,
        "circles": 0,
        "ellipses": 0,
        "rectangles": 0,
        "arcs": 0,
        "nodes": 0,
    }
    for item in flatten_vector_objects(tuple(objects)):
        stats["total"] += 1
        if isinstance(item, Line):
            stats["lines"] += 1
        elif isinstance(item, Polyline):
            stats["polylines"] += 1
        elif isinstance(item, BezierCurve):
            stats["bezier_curves"] += 1
        elif isinstance(item, Circle):
            stats["circles"] += 1
        elif isinstance(item, Ellipse):
            stats["ellipses"] += 1
        elif isinstance(item, Rectangle):
            stats["rectangles"] += 1
        elif isinstance(item, Arc):
            stats["arcs"] += 1
        elif isinstance(item, Node):
            stats["nodes"] += 1
    return VectorObjectStats(**stats)


def flatten_vector_objects(objects: tuple[VectorObject, ...]) -> tuple[Line | Polyline | BezierCurve | Circle | Ellipse | Rectangle | Arc | Node, ...]:
    """Flatten vector groups into primitive objects."""
    flattened: list[Line | Polyline | BezierCurve | Circle | Ellipse | Rectangle | Arc | Node] = []
    for item in objects:
        if isinstance(item, PathGroup):
            flattened.extend(item.flatten())
        else:
            flattened.append(item)
    return tuple(flattened)


def _object_to_tikz(item, options: TikzOptions) -> str:
    style = _draw_style(options)
    precision = options.precision
    if isinstance(item, Line):
        return f"  \\draw[{style}] {_format_point(item.start, precision)} -- {_format_point(item.end, precision)};"
    if isinstance(item, Polyline):
        body = " -- ".join(_format_point(point, precision) for point in item.points)
        if item.closed:
            body += " -- cycle"
        return f"  \\draw[{style}] {body};"
    if isinstance(item, BezierCurve):
        return (
            f"  \\draw[{style}] {_format_point(item.start, precision)}"
            f" .. controls {_format_point(item.control1, precision)} and {_format_point(item.control2, precision)}"
            f" .. {_format_point(item.end, precision)};"
        )
    if isinstance(item, Circle):
        return f"  \\draw[{style}] {_format_point(item.center, precision)} circle ({_fmt_number(item.radius, precision)});"
    if isinstance(item, Ellipse):
        rotation = "" if abs(item.rotation) < 1e-9 else f", rotate around={{{_fmt_number(item.rotation, precision)}:{_format_point(item.center, precision)}}}"
        return (
            f"  \\draw[{style}{rotation}] {_format_point(item.center, precision)} "
            f"ellipse ({_fmt_number(item.radius_x, precision)} and {_fmt_number(item.radius_y, precision)});"
        )
    if isinstance(item, Rectangle):
        return f"  \\draw[{style}] {_format_point(item.corner1, precision)} rectangle {_format_point(item.corner2, precision)};"
    if isinstance(item, Arc):
        return (
            f"  \\draw[{style}] {_format_point(item.center, precision)} ++({_fmt_number(item.start_angle, precision)}:{_fmt_number(item.radius_x, precision)} and {_fmt_number(item.radius_y, precision)})"
            f" arc[start angle={_fmt_number(item.start_angle, precision)}, end angle={_fmt_number(item.end_angle, precision)},"
            f" x radius={_fmt_number(item.radius_x, precision)}, y radius={_fmt_number(item.radius_y, precision)}];"
        )
    if isinstance(item, Node):
        name = f" ({item.name})" if item.name else ""
        return f"  \\node{name} at {_format_point(item.position, precision)} {{{item.text}}};"
    return ""


def _draw_style(options: TikzOptions) -> str:
    color = options.line_color.strip()
    if not _SAFE_COLOR.match(color):
        color = "black"
    return f"draw={color}, line width={_fmt_number(options.line_width, 2)}pt"


def _format_point(point: Point, precision: int) -> str:
    return f"({_fmt_number(point.x, precision)},{_fmt_number(point.y, precision)})"


def _fmt_number(value: float, precision: int) -> str:
    text = f"{value:.{max(0, precision)}f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"
