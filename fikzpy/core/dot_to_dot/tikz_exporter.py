"""Emit TikZ for the dot-to-dot mode.

The remaining drawing is converted with a coordinate transform identical to
the rest of the project (``image_point_to_tikz``): pixels map to TikZ units
with an upward y-axis. The numbered points and their labels are emitted as
native TikZ (never routed through svg2tikz), so the numbers stay editable.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from fikzpy.core.dot_to_dot.config import DotToDotConfig
from fikzpy.core.dot_to_dot.models import (
    CubicSegment,
    DotSequence,
    LineSegment,
    QuadraticSegment,
    Segment,
    SubPath,
    Vec2,
)

_SAFE_COLOR = re.compile(r"^[A-Za-z][A-Za-z0-9!._-]*$")


@dataclass(frozen=True)
class CoordinateTransform:
    """Maps SVG/image-space points to TikZ coordinates (y pointing up)."""

    image_height: float
    scale: float

    def to_tikz(self, point: Vec2) -> tuple[float, float]:
        """Return the TikZ ``(x, y)`` for an SVG-space point."""
        return point.x * self.scale, (self.image_height - point.y) * self.scale

    @classmethod
    def from_image(cls, image_width: float, image_height: float, width_units: float) -> "CoordinateTransform":
        """Build a transform from image size and target TikZ width in units."""
        scale = float(width_units) / max(float(image_width), 1.0)
        return cls(image_height=float(image_height), scale=scale)


def _safe_color(value: str) -> str:
    color = value.strip()
    return color if _SAFE_COLOR.match(color) else "black"


def _fmt(value: float, precision: int) -> str:
    text = f"{float(value):.{max(0, precision)}f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _coord(point: Vec2, transform: CoordinateTransform, precision: int) -> str:
    x, y = transform.to_tikz(point)
    return f"({_fmt(x, precision)},{_fmt(y, precision)})"


def _quadratic_as_cubic(segment: QuadraticSegment) -> CubicSegment:
    control1 = segment.start + (segment.control - segment.start).scaled(2.0 / 3.0)
    control2 = segment.end + (segment.control - segment.end).scaled(2.0 / 3.0)
    return CubicSegment(segment.start, control1, control2, segment.end)


def subpath_to_tikz_path(subpath: SubPath, transform: CoordinateTransform, precision: int) -> str:
    """Return a TikZ path body (without the ``\\draw`` prefix) for a subpath."""
    segments = subpath.segments
    if not segments:
        return ""
    parts = [_coord(segments[0].start, transform, precision)]
    for segment in segments:
        parts.append(_segment_commands(segment, transform, precision))
    if subpath.closed:
        parts.append("-- cycle")
    return " ".join(part for part in parts if part)


def _segment_commands(segment: Segment, transform: CoordinateTransform, precision: int) -> str:
    if isinstance(segment, LineSegment):
        return f"-- {_coord(segment.end, transform, precision)}"
    if isinstance(segment, QuadraticSegment):
        segment = _quadratic_as_cubic(segment)
    if isinstance(segment, CubicSegment):
        control1 = _coord(segment.control1, transform, precision)
        control2 = _coord(segment.control2, transform, precision)
        end = _coord(segment.end, transform, precision)
        return f".. controls {control1} and {control2} .. {end}"
    return ""


def styles_block() -> str:
    """Return the reusable ``\\tikzset`` block for points and labels."""
    return (
        "  \\tikzset{\n"
        "    dotToDotPoint/.style={fill=black,draw=black},\n"
        "    dotToDotLabel/.style={font=\\sffamily\\small,inner sep=1pt},\n"
        "  }"
    )


def render_remaining_drawing(
    subpaths: Iterable[SubPath],
    transform: CoordinateTransform,
    *,
    line_color: str,
    line_width: float,
    precision: int,
) -> list[str]:
    """Return ``\\draw`` commands for the drawing that is kept as-is."""
    color = _safe_color(line_color)
    width = _fmt(line_width, 2)
    lines: list[str] = []
    for subpath in subpaths:
        body = subpath_to_tikz_path(subpath, transform, precision)
        if body:
            lines.append(f"  \\draw[draw={color}, line width={width}pt] {body};")
    return lines


def _label_coordinate(dot, transform: CoordinateTransform, label_offset: float, precision: int) -> str:
    x, y = transform.to_tikz(dot.position)
    # Convert the SVG-space outward normal to TikZ orientation (y flips) and
    # offset the label by ``label_offset`` TikZ units.
    outward = Vec2(dot.normal.x, -dot.normal.y).normalized()
    x += outward.x * label_offset
    y += outward.y * label_offset
    return f"({_fmt(x, precision)},{_fmt(y, precision)})"


def render_points(
    sequences: Sequence[DotSequence],
    transform: CoordinateTransform,
    *,
    point_radius: float,
    label_offset: float,
    precision: int,
) -> list[str]:
    """Return native TikZ ``\\fill``/``\\node`` commands for numbered points."""
    radius = _fmt(point_radius, 2)
    lines: list[str] = []
    for sequence in sequences:
        if not sequence.points:
            continue
        lines.append(f"  % dot sequence '{sequence.prefix or 'main'}' ({len(sequence.points)} points)")
        for dot in sequence.points:
            center = _coord(dot.position, transform, precision)
            lines.append(f"  \\fill[dotToDotPoint] {center} circle[radius={radius}pt];")
        for dot in sequence.points:
            label_pos = _label_coordinate(dot, transform, label_offset, precision)
            lines.append(
                f"  \\node[dotToDotLabel,anchor={dot.label_anchor}] at {label_pos} {{{dot.label}}};"
            )
    return lines


def build_tikz_picture(
    config: DotToDotConfig,
    remaining_subpaths: Iterable[SubPath],
    sequences: Sequence[DotSequence],
    transform: CoordinateTransform,
    *,
    tikz_scale: float,
    line_color: str,
    line_width: float,
    precision: int,
) -> str:
    """Assemble the complete ``tikzpicture`` for a dot-to-dot result."""
    lines = [f"\\begin{{tikzpicture}}[scale={_fmt(tikz_scale, 3)}]"]
    lines.append(styles_block())
    lines.append("  \\begin{scope}[line cap=round, line join=round]")
    drawing = render_remaining_drawing(
        remaining_subpaths,
        transform,
        line_color=line_color,
        line_width=line_width,
        precision=precision,
    )
    lines.extend(drawing)
    point_lines = render_points(
        sequences,
        transform,
        point_radius=config.point_radius,
        label_offset=config.label_offset,
        precision=precision,
    )
    lines.extend(point_lines)
    if not drawing and not point_lines:
        lines.append("  % No dot-to-dot content generated.")
    lines.append("  \\end{scope}")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


__all__ = [
    "CoordinateTransform",
    "build_tikz_picture",
    "render_points",
    "render_remaining_drawing",
    "styles_block",
    "subpath_to_tikz_path",
]
