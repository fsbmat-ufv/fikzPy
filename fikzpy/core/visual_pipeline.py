"""Visual-fidelity raster tracing through SVG-style filled paths."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Sequence

import cv2
import numpy as np

from fikzpy.core.bezier_fit import fit_cubic_beziers
from fikzpy.core.contour_detector import Contour
from fikzpy.core.diagnostics import log_event
from fikzpy.core.tikz_generator import TikzOptions
from fikzpy.core.vector_objects import BezierCurve, Line, Point, Polyline, VectorPrimitive
from fikzpy.core.visual_postprocessor import postprocess_visual_tikz_picture


@dataclass(frozen=True)
class VisualTracingSettings:
    """Settings for visual tracing of ink shapes instead of stroke centerlines."""

    dark_threshold: int = 215
    upsample_factor: int = 3
    adaptive_block_size: int = 21
    adaptive_offset: int = 6
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    denoise_h: float = 5.0
    close_kernel_size: int = 3
    min_component_area: float = 2.0
    contour_simplify_px: float = 0.35
    bezier_error_px: float = 0.7
    ignore_chromatic_annotations: bool = True


@dataclass(frozen=True)
class VisualTraceResult:
    """Detected visual ink contours and preview mask."""

    contours: list[Contour]
    ink_mask: np.ndarray
    input_pixels: int
    ink_pixels: int


@dataclass(frozen=True)
class VisualTikzStats:
    """Small diagnostics for visual TikZ generation."""

    paths: int = 0
    svg_bytes: int = 0
    tikz_bytes: int = 0
    used_svg2tikz: bool = False
    postprocessed: bool = False
    draw_commands: int = 0
    subpaths: int = 0


@dataclass(frozen=True)
class VisualTikzResult:
    """TikZ output generated from visual ink contours."""

    tikz_picture: str
    svg_source: str
    stats: VisualTikzStats


_SAFE_COLOR = re.compile(r"^[A-Za-z][A-Za-z0-9!._-]*$")


def trace_visual_contours(
    image: np.ndarray,
    settings: VisualTracingSettings | None = None,
) -> VisualTraceResult:
    """Trace filled ink shapes for maximum visual fidelity."""
    settings = settings or VisualTracingSettings()
    bgr = _ensure_bgr(image)
    scale = max(1, int(settings.upsample_factor))
    working = _resize_for_trace(bgr, scale)
    mask = _build_high_resolution_mask(working, settings, scale)
    contours = _extract_visual_contours(mask, scale=scale, settings=settings)
    preview_mask = _resize_mask_to_image(mask, image.shape[:2])
    return VisualTraceResult(
        contours=contours,
        ink_mask=preview_mask,
        input_pixels=int(image.shape[0] * image.shape[1]),
        ink_pixels=int(np.count_nonzero(preview_mask)),
    )


def generate_visual_tikz_picture(
    contours: Sequence[Contour],
    image_shape: tuple[int, ...],
    options: TikzOptions | None = None,
    settings: VisualTracingSettings | None = None,
) -> VisualTikzResult:
    """Generate TikZ by converting filled SVG paths with svg2tikz when possible."""
    options = options or TikzOptions()
    settings = settings or VisualTracingSettings()
    svg_source = contours_to_svg(contours, image_shape, options=options, settings=settings)
    tikz_picture, used_svg2tikz = _svg_to_tikz_picture(svg_source, options)
    if not tikz_picture:
        tikz_picture = _generate_direct_visual_tikz(contours, image_shape, options=options, settings=settings)
        used_svg2tikz = False
    postprocess_result = postprocess_visual_tikz_picture(tikz_picture)
    tikz_picture = postprocess_result.tikz_picture
    draw_commands = postprocess_result.stats.output_draw_commands or _count_draw_commands(tikz_picture)
    subpaths = postprocess_result.stats.subpaths or _count_cycle_subpaths(tikz_picture)

    return VisualTikzResult(
        tikz_picture=tikz_picture,
        svg_source=svg_source,
        stats=VisualTikzStats(
            paths=len([contour for contour in contours if contour.is_drawable]),
            svg_bytes=len(svg_source.encode("utf-8")),
            tikz_bytes=len(tikz_picture.encode("utf-8")),
            used_svg2tikz=used_svg2tikz,
            postprocessed=postprocess_result.stats.changed,
            draw_commands=draw_commands,
            subpaths=subpaths,
        ),
    )


def contours_to_svg(
    contours: Sequence[Contour],
    image_shape: tuple[int, ...],
    *,
    options: TikzOptions | None = None,
    settings: VisualTracingSettings | None = None,
) -> str:
    """Serialize visual contours to one SVG path with even-odd filling."""
    options = options or TikzOptions()
    settings = settings or VisualTracingSettings()
    height, width = image_shape[:2]
    width_units = max(float(options.width_units), 1e-6)
    height_units = width_units * (float(height) / max(float(width), 1.0))
    color = _safe_color(options.line_color)

    paths = [
        _contour_to_svg_path(contour, settings)
        for contour in contours
        if contour.is_drawable and len(contour.points) >= 3
    ]
    path_data = " ".join(path for path in paths if path)
    if not path_data:
        path_data = "M 0 0"

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt_number(width_units, 6)}cm" '
            f'height="{_fmt_number(height_units, 6)}cm" viewBox="0 0 {_fmt_number(width, 6)} {_fmt_number(height, 6)}">',
            f'  <path d="{path_data}" fill="{color}" stroke="none" fill-rule="evenodd"/>',
            "</svg>",
        ]
    )


def _build_high_resolution_mask(
    bgr: np.ndarray,
    settings: VisualTracingSettings,
    scale: int,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if settings.denoise_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=float(settings.denoise_h))

    clahe = cv2.createCLAHE(
        clipLimit=max(float(settings.clahe_clip_limit), 0.1),
        tileGridSize=(max(1, int(settings.clahe_tile_grid_size)), max(1, int(settings.clahe_tile_grid_size))),
    )
    enhanced = clahe.apply(gray)

    block_size = _odd(max(15, int(settings.adaptive_block_size) * scale))
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(settings.adaptive_offset),
    )
    dark_limit = min(250, max(0, int(settings.dark_threshold) + 25))
    dark_candidate = (gray <= dark_limit).astype(np.uint8) * 255
    global_dark = (gray <= max(0, int(settings.dark_threshold))).astype(np.uint8) * 255
    mask = cv2.bitwise_and(cv2.bitwise_or(adaptive, global_dark), dark_candidate)

    if settings.ignore_chromatic_annotations and bgr.ndim == 3:
        mask = cv2.bitwise_and(mask, _neutral_ink_mask(bgr))

    close_size = max(1, int(settings.close_kernel_size))
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return _remove_small_components(mask, min_area=max(1.0, settings.min_component_area * scale * scale))


def _extract_visual_contours(
    mask: np.ndarray,
    *,
    scale: int,
    settings: VisualTracingSettings,
) -> list[Contour]:
    found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours: list[Contour] = []
    min_area = max(1.0, settings.min_component_area * scale * scale)
    for raw in found:
        area = float(abs(cv2.contourArea(raw))) / float(scale * scale)
        perimeter = float(cv2.arcLength(raw, True)) / float(scale)
        if area * scale * scale < min_area and perimeter < 2.0:
            continue

        epsilon = max(0.05, float(settings.contour_simplify_px)) * scale
        approx = cv2.approxPolyDP(raw, epsilon, True)
        points = approx.reshape(-1, 2).astype(np.float64) / float(scale)
        if len(points) < 3:
            continue
        contours.append(Contour(points=points, closed=True, area=area, perimeter=perimeter))

    contours.sort(key=lambda item: (item.area, item.perimeter), reverse=True)
    return contours


def _contour_to_svg_path(contour: Contour, settings: VisualTracingSettings) -> str:
    points = tuple(Point(float(x), float(y)) for x, y in contour.points)
    if len(points) < 3:
        return ""

    primitives = fit_cubic_beziers(
        points,
        error_tolerance=max(float(settings.bezier_error_px), 0.05),
        closed=True,
        min_bezier_length=0.9,
        min_points_for_bezier=6,
        straightness_tolerance=0.35,
        control_point_epsilon=1e-4,
        max_depth=18,
    )
    if not primitives:
        return _polyline_svg_path(points)

    start = _primitive_start(primitives[0])
    if start is None:
        return _polyline_svg_path(points)

    commands = [f"M {_fmt_number(start.x, 3)} {_fmt_number(start.y, 3)}"]
    current = start
    for primitive in primitives:
        primitive_start = _primitive_start(primitive)
        if primitive_start is not None and current.distance_to(primitive_start) > 1e-4:
            commands.append(f"L {_fmt_number(primitive_start.x, 3)} {_fmt_number(primitive_start.y, 3)}")
            current = primitive_start

        if isinstance(primitive, BezierCurve):
            commands.append(
                "C "
                f"{_fmt_number(primitive.control1.x, 3)} {_fmt_number(primitive.control1.y, 3)} "
                f"{_fmt_number(primitive.control2.x, 3)} {_fmt_number(primitive.control2.y, 3)} "
                f"{_fmt_number(primitive.end.x, 3)} {_fmt_number(primitive.end.y, 3)}"
            )
            current = primitive.end
        elif isinstance(primitive, Line):
            commands.append(f"L {_fmt_number(primitive.end.x, 3)} {_fmt_number(primitive.end.y, 3)}")
            current = primitive.end
        elif isinstance(primitive, Polyline):
            for point in primitive.points[1:]:
                commands.append(f"L {_fmt_number(point.x, 3)} {_fmt_number(point.y, 3)}")
            current = primitive.points[-1]

    commands.append("Z")
    return " ".join(commands)


def _svg_to_tikz_picture(svg_source: str, options: TikzOptions) -> tuple[str, bool]:
    try:
        import svg2tikz
    except ImportError:
        log_event("Visual", "svg2tikz=unavailable")
        return "", False

    try:
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0] if old_argv else "svg2tikz"]
        converted = svg2tikz.convert_svg(svg_source, no_output=True, returnstring=True)
    except (Exception, SystemExit) as exc:  # pragma: no cover - defensive integration guard
        log_event("Visual", f"svg2tikz_failure={exc!r}")
        return "", False
    finally:
        sys.argv = old_argv

    picture = _extract_tikzpicture(converted)
    if not picture:
        return "", False
    picture = _normalize_svg2tikz_picture(picture, options)
    return picture, True


def _generate_direct_visual_tikz(
    contours: Sequence[Contour],
    image_shape: tuple[int, ...],
    *,
    options: TikzOptions,
    settings: VisualTracingSettings,
) -> str:
    height, width = image_shape[:2]
    scale = float(options.width_units) / max(float(width), 1.0)
    color = _safe_color(options.line_color)
    lines = [f"\\begin{{tikzpicture}}[scale={_fmt_number(options.tikz_scale, 3)}]"]
    parts = []
    for contour in contours:
        path = _contour_to_svg_path(contour, settings)
        if not path:
            continue
        parts.append(_svg_path_to_tikz_coordinates(path, height=height, scale=scale, precision=options.precision))

    if parts:
        lines.append(f"  \\tikzset{{fikzInk/.style={{fill={color}, draw={color}, line width=0pt, even odd rule}}}}")
        lines.append("  \\draw[fikzInk]")
        lines.append("    " + "\n    ".join(parts) + ";")
    else:
        lines.append("  % No visual ink detected.")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _svg_path_to_tikz_coordinates(path: str, *, height: int, scale: float, precision: int) -> str:
    tokens = path.split()
    converted: list[str] = []
    index = 0
    command = ""
    while index < len(tokens):
        token = tokens[index]
        if token in {"M", "L", "C", "Z"}:
            command = token
            index += 1
            if command == "Z":
                converted.append("-- cycle")
            continue
        if command in {"M", "L"}:
            x = float(token) * scale
            y = (float(height) - float(tokens[index + 1])) * scale
            converted.append((" " if command == "M" else " -- ") + _point_text(x, y, precision))
            index += 2
        elif command == "C":
            values = [float(value) for value in tokens[index : index + 6]]
            p1 = _point_text(values[0] * scale, (float(height) - values[1]) * scale, precision)
            p2 = _point_text(values[2] * scale, (float(height) - values[3]) * scale, precision)
            p3 = _point_text(values[4] * scale, (float(height) - values[5]) * scale, precision)
            converted.append(f" .. controls {p1} and {p2} .. {p3}")
            index += 6
        else:
            index += 1
    return "".join(converted).strip()


def _extract_tikzpicture(document: str) -> str:
    begin = document.find("\\begin{tikzpicture}")
    end_marker = "\\end{tikzpicture}"
    end = document.find(end_marker)
    if begin < 0 or end < 0:
        return ""
    return document[begin : end + len(end_marker)]


def _count_draw_commands(tikz_picture: str) -> int:
    return len(re.findall(r"\\draw(?:\[|[ \t\r\n])", tikz_picture))


def _count_cycle_subpaths(tikz_picture: str) -> int:
    return len(re.findall(r"--\s*cycle", tikz_picture))


def _normalize_svg2tikz_picture(picture: str, options: TikzOptions) -> str:
    picture = re.sub(
        r"\\begin\{tikzpicture\}\[[^\]]*\]",
        lambda _: f"\\begin{{tikzpicture}}[scale={_fmt_number(options.tikz_scale, 3)}]",
        picture,
        count=1,
    )
    picture = picture.replace("\x08egin{tikzpicture}", "\\begin{tikzpicture}")
    picture = picture.replace("-- cycle(", "-- cycle (")
    picture = picture.replace("-- cycle\n", "-- cycle\n")
    return picture.strip()


def _neutral_ink_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    red_annotation = (red > green + 35) & (red > blue + 35) & (value > 110)
    neutral = ((saturation <= 65) | (value <= 90)) & ~red_annotation
    return neutral.astype(np.uint8) * 255


def _remove_small_components(mask: np.ndarray, *, min_area: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if float(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _resize_for_trace(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return image.copy()
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def _resize_mask_to_image(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if mask.shape[:2] == (height, width):
        return mask.copy()
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("Expected a grayscale or color image array.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.astype(np.uint8, copy=False)


def _polyline_svg_path(points: Sequence[Point]) -> str:
    first = points[0]
    commands = [f"M {_fmt_number(first.x, 3)} {_fmt_number(first.y, 3)}"]
    commands.extend(f"L {_fmt_number(point.x, 3)} {_fmt_number(point.y, 3)}" for point in points[1:])
    commands.append("Z")
    return " ".join(commands)


def _primitive_start(primitive: VectorPrimitive) -> Point | None:
    if isinstance(primitive, (Line, BezierCurve)):
        return primitive.start
    if isinstance(primitive, Polyline):
        return primitive.points[0]
    return None


def _safe_color(value: str) -> str:
    color = value.strip()
    return color if _SAFE_COLOR.match(color) else "black"


def _fmt_number(value: float, precision: int) -> str:
    text = f"{float(value):.{max(0, precision)}f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _point_text(x: float, y: float, precision: int) -> str:
    return f"({_fmt_number(x, precision)},{_fmt_number(y, precision)})"


def _odd(value: int) -> int:
    size = max(3, int(value))
    return size if size % 2 == 1 else size + 1
