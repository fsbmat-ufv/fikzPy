"""Post-process filled visual TikZ paths into grouped draw commands."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class VisualPostprocessStats:
    """Diagnostics for the visual TikZ post-processing pass."""

    input_path_commands: int = 0
    output_draw_commands: int = 0
    subpaths: int = 0
    groups: int = 0
    changed: bool = False


@dataclass(frozen=True)
class VisualPostprocessResult:
    """TikZ output and diagnostics after post-processing."""

    tikz_picture: str
    stats: VisualPostprocessStats


@dataclass(frozen=True)
class _Subpath:
    text: str
    points: tuple[tuple[float, float], ...]
    area: float
    bbox: tuple[float, float, float, float]
    parent: int | None = None
    depth: int = 0


_PATH_RE = re.compile(
    r"(?P<indent>^[ \t]*)\\path\[(?P<options>[^\]]*)\]\s*(?P<body>.*?);",
    re.MULTILINE | re.DOTALL,
)
_POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def postprocess_visual_tikz_picture(tikz_picture: str) -> VisualPostprocessResult:
    """Convert one monolithic filled path into grouped filled draw commands.

    The visual backend represents ink as filled areas, not stroke centerlines.
    This pass keeps that representation, but replaces the single large
    ``\\path`` command emitted by svg2tikz with smaller ``\\draw`` groups.
    Nested paths stay in the same draw command so TikZ's even-odd fill rule
    keeps holes and islands visually intact.
    """
    match = _PATH_RE.search(tikz_picture)
    if match is None:
        return VisualPostprocessResult(tikz_picture, VisualPostprocessStats())

    options = match.group("options")
    body = match.group("body")
    subpaths = _build_subpaths(_split_subpaths(body))
    if not subpaths:
        return VisualPostprocessResult(tikz_picture, VisualPostprocessStats(input_path_commands=1))

    layers = _layer_nested_subpaths(subpaths)
    color = _extract_fill_color(options)
    replacement = _format_draw_layers(layers, ink_color=color, indent=match.group("indent"))
    processed = tikz_picture[: match.start()] + replacement + tikz_picture[match.end() :]
    return VisualPostprocessResult(
        processed,
        VisualPostprocessStats(
            input_path_commands=1,
            output_draw_commands=len(layers),
            subpaths=len(subpaths),
            groups=len(layers),
            changed=processed != tikz_picture,
        ),
    )


def _split_subpaths(body: str) -> tuple[str, ...]:
    normalized = body.replace("-- cycle(", "-- cycle (")
    subpaths: list[str] = []
    start = 0
    for match in re.finditer(r"--\s*cycle", normalized):
        chunk = normalized[start : match.end()].strip()
        if chunk:
            subpaths.append(_normalize_subpath_text(chunk))
        start = match.end()
    return tuple(subpaths)


def _build_subpaths(path_texts: tuple[str, ...]) -> tuple[_Subpath, ...]:
    subpaths: list[_Subpath] = []
    for text in path_texts:
        points = _extract_points(text)
        if len(points) < 3:
            continue
        subpaths.append(
            _Subpath(
                text=text,
                points=points,
                area=abs(_polygon_area(points)),
                bbox=_bbox(points),
            )
        )
    return tuple(subpaths)


def _layer_nested_subpaths(subpaths: tuple[_Subpath, ...]) -> tuple[tuple[str, tuple[_Subpath, ...]], ...]:
    parents: list[int | None] = []
    for index, subpath in enumerate(subpaths):
        parent = _find_parent(index, subpaths)
        parents.append(parent)

    layers: dict[int, list[_Subpath]] = {}
    for index, subpath in enumerate(subpaths):
        depth = _depth(index, parents)
        layers.setdefault(depth, []).append(_set_parent_and_depth(subpath, parents[index], depth))

    result = []
    for depth in sorted(layers):
        style_name = "fikzInk" if depth % 2 == 0 else "fikzErase"
        result.append((style_name, tuple(layers[depth])))
    return tuple(result)


def _find_parent(index: int, subpaths: tuple[_Subpath, ...]) -> int | None:
    item = subpaths[index]
    point = _representative_point(item.points)
    candidates: list[tuple[float, int]] = []
    for other_index, other in enumerate(subpaths):
        if other_index == index or other.area <= item.area:
            continue
        if not _bbox_contains(other.bbox, point):
            continue
        if _point_in_polygon(point, other.points):
            candidates.append((other.area, other_index))
    if not candidates:
        return None
    return min(candidates)[1]


def _depth(index: int, parents: list[int | None]) -> int:
    depth = 0
    seen: set[int] = set()
    current = index
    while parents[current] is not None and current not in seen:
        seen.add(current)
        parent = parents[current]
        if parent is None:
            break
        current = parent
        depth += 1
    return depth


def _set_parent_and_depth(subpath: _Subpath, parent: int | None, depth: int) -> _Subpath:
    return _Subpath(subpath.text, subpath.points, subpath.area, subpath.bbox, parent, depth)


def _format_draw_layers(
    layers: tuple[tuple[str, tuple[_Subpath, ...]], ...],
    *,
    ink_color: str,
    indent: str,
) -> str:
    style = (
        f"fikzInk/.style={{fill={ink_color}, draw={ink_color}, line width=0pt}}, "
        "fikzErase/.style={fill=white, draw=white, line width=0pt}"
    )
    lines = [f"{indent}\\tikzset{{{style}}}"]
    for style_name, group in layers:
        lines.append(f"{indent}\\draw[{style_name}]")
        for subpath in group:
            lines.append(f"{indent}  {subpath.text}")
        lines[-1] += ";"
    return "\n".join(lines)


def _extract_fill_color(options: str) -> str:
    match = re.search(r"(?:^|,)\s*fill\s*=\s*([^,\]]+)", options)
    if match is None:
        return "black"
    color = match.group(1).strip()
    return color or "black"


def _extract_points(text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in _POINT_RE.findall(text))


def _normalize_subpath_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace("-- cycle", "-- cycle").strip()


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    area = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        area += first[0] * second[1] - second[0] * first[1]
    return area / 2.0


def _bbox(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_contains(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    min_x, min_y, max_x, max_y = bbox
    x, y = point
    return min_x <= x <= max_x and min_y <= y <= max_y


def _representative_point(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = _bbox(points)
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            x_at_y = (previous_x - current_x) * (y - current_y) / (previous_y - current_y + 1e-12) + current_x
            if x < x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside
