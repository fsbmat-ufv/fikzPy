"""Generate clean TikZ code from detected contours."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from fikzpy.core.bezier_fit import can_use_bezier, catmull_rom_to_bezier
from fikzpy.core.contour_detector import Contour


@dataclass(frozen=True)
class TikzOptions:
    """Formatting and coordinate options for TikZ output."""

    tikz_scale: float = 1.0
    line_width: float = 0.4
    line_color: str = "black"
    use_bezier: bool = False
    width_units: float = 10.0
    precision: int = 2
    max_paths: int | None = None
    bezier_min_points: int = 4
    bezier_tension: float = 1.0


_SAFE_COLOR = re.compile(r"^[A-Za-z][A-Za-z0-9!._-]*$")


def _safe_color(value: str) -> str:
    color = value.strip()
    return color if _SAFE_COLOR.match(color) else "black"


def _fmt_number(value: float, precision: int) -> str:
    text = f"{value:.{max(0, precision)}f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def image_point_to_tikz(
    point: Sequence[float],
    image_shape: tuple[int, ...],
    *,
    width_units: float = 10.0,
) -> np.ndarray:
    """Map image pixels to TikZ coordinates with the y-axis pointing upward."""
    height, width = image_shape[:2]
    scale = float(width_units) / max(float(width), 1.0)
    x, y = float(point[0]), float(point[1])
    return np.array([x * scale, (float(height) - y) * scale], dtype=np.float64)


def _format_point(point: Sequence[float], precision: int) -> str:
    x, y = point
    return f"({_fmt_number(float(x), precision)},{_fmt_number(float(y), precision)})"


def _transformed_points(contour: Contour, image_shape: tuple[int, ...], options: TikzOptions) -> np.ndarray:
    return np.array(
        [
            image_point_to_tikz(point, image_shape, width_units=options.width_units)
            for point in contour.points
        ],
        dtype=np.float64,
    )


def _line_path(points: np.ndarray, *, closed: bool, options: TikzOptions) -> list[str]:
    if len(points) < 2:
        return []

    lines = [f"  \\draw[{_draw_style(options)}] {_format_point(points[0], options.precision)}"]
    for point in points[1:]:
        lines.append(f"    -- {_format_point(point, options.precision)}")

    if closed and len(points) > 2:
        lines[-1] += " -- cycle;"
    else:
        lines[-1] += ";"
    return lines


def _bezier_path(points: np.ndarray, *, closed: bool, options: TikzOptions) -> list[str]:
    if not can_use_bezier(points, min_points=options.bezier_min_points):
        return _line_path(points, closed=closed, options=options)

    segments = catmull_rom_to_bezier(points, closed=closed, tension=options.bezier_tension)
    if not segments:
        return []

    lines = [f"  \\draw[{_draw_style(options)}] {_format_point(segments[0].start, options.precision)}"]
    for segment in segments:
        control1 = _format_point(segment.control1, options.precision)
        control2 = _format_point(segment.control2, options.precision)
        end = _format_point(segment.end, options.precision)
        lines.append(f"    .. controls {control1} and {control2} .. {end}")

    lines[-1] += ";"
    return lines


def _draw_style(options: TikzOptions) -> str:
    return ", ".join(
        [
            f"draw={_safe_color(options.line_color)}",
            f"line width={_fmt_number(options.line_width, 2)}pt",
        ]
    )


def contour_to_tikz(contour: Contour, image_shape: tuple[int, ...], options: TikzOptions) -> str:
    """Convert one contour into a TikZ draw command."""
    points = _transformed_points(contour, image_shape, options)
    if options.use_bezier:
        lines = _bezier_path(points, closed=contour.closed, options=options)
    else:
        lines = _line_path(points, closed=contour.closed, options=options)
    return "\n".join(lines)


def generate_tikz_picture(
    contours: Sequence[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions | None = None,
) -> str:
    """Generate a minimal tikzpicture environment from contours."""
    options = options or TikzOptions()
    selected = list(contours)
    if options.max_paths is not None:
        selected = selected[: max(0, options.max_paths)]

    lines = [
        f"\\begin{{tikzpicture}}[scale={_fmt_number(options.tikz_scale, 3)}]",
        "  \\begin{scope}[line cap=round, line join=round]",
    ]

    body_written = False
    for contour in selected:
        if not contour.is_drawable:
            continue
        command = contour_to_tikz(contour, image_shape, options)
        if command:
            lines.append(command)
            body_written = True

    if not body_written:
        lines.append("  % No contours detected.")

    lines.extend(["  \\end{scope}", "\\end{tikzpicture}"])
    return "\n".join(lines)


def wrap_standalone_document(tikz_picture: str) -> str:
    """Wrap a tikzpicture block in a standalone LaTeX document."""
    return "\n".join(
        [
            "% Generated by fikzPy",
            "\\documentclass[tikz,border=2mm]{standalone}",
            "\\usepackage{tikz}",
            "",
            "\\begin{document}",
            tikz_picture.strip(),
            "\\end{document}",
            "",
        ]
    )
