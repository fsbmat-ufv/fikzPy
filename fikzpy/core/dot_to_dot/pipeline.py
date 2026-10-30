"""Orchestration for the independent Ponto a Ponto (dot-to-dot) pipeline.

The pipeline consumes a *copy* of an SVG string (typically the intermediate
SVG produced by the project's Visual tracing stage), parses it into its own
geometric model and produces TikZ where the replaced contour is expressed as
numbered native-TikZ points. It never mutates the source SVG and never falls
back to another mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from fikzpy.core.dot_to_dot.config import (
    DotToDotConfig,
    DotToDotGenerationMode,
    NumberingDirection,
    SelectedSegment,
)
from fikzpy.core.dot_to_dot.label_layout import layout_labels
from fikzpy.core.dot_to_dot.label_placement import build_dot_points
from fikzpy.core.dot_to_dot.models import DotPoint, DotSequence, SubPath, SvgPath, Vec2
from fikzpy.core.dot_to_dot.path_sampler import ContourRef, SampledPoint
from fikzpy.core.dot_to_dot.segment_selection import resolve_arc, sample_selected_segment
from fikzpy.core.dot_to_dot.silhouette import SilhouetteError, extract_silhouette, retrace_ink_regions
from fikzpy.core.dot_to_dot.svg_loader import SvgLoaderConfig, load_svg_paths
from fikzpy.core.dot_to_dot.tikz_exporter import CoordinateTransform, build_tikz_picture

_LABEL_HEIGHT_CM = 0.20
_DENSITY_WARNING_FACTOR = 1.6


class DotToDotError(RuntimeError):
    """Raised when the dot-to-dot pipeline cannot produce a result."""


@dataclass(frozen=True)
class DotToDotResult:
    """Result of a dot-to-dot generation run."""

    tikz_code: str
    sequences: tuple[DotSequence, ...]
    dotted_refs: tuple[ContourRef, ...]
    remaining_subpaths: tuple[SubPath, ...]
    paths: tuple[SvgPath, ...]
    total_points: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    silhouette: SubPath | None = None

    @property
    def point_count(self) -> int:
        """Return the total number of generated numbered points."""
        return sum(len(sequence.points) for sequence in self.sequences)


def run_dot_to_dot(
    svg_source: str,
    image_shape: tuple[int, ...],
    config: DotToDotConfig | None = None,
    *,
    tikz_scale: float = 1.0,
    line_color: str = "black",
    line_width: float = 0.4,
    width_units: float = 10.0,
    precision: int = 2,
) -> DotToDotResult:
    """Run the dot-to-dot pipeline over a copy of ``svg_source``."""
    config = config or DotToDotConfig()
    height, width = float(image_shape[0]), float(image_shape[1])
    # ``svg_source`` is a str (immutable); parsing works on an independent copy.
    paths = load_svg_paths(str(svg_source), SvgLoaderConfig())
    transform = CoordinateTransform.from_image(width, height, width_units)

    if config.generation_mode is DotToDotGenerationMode.FULL_FIGURE:
        return _run_full_figure(
            paths,
            config,
            transform,
            canvas_width=width,
            canvas_height=height,
            tikz_scale=tikz_scale,
            line_color=line_color,
            line_width=line_width,
            precision=precision,
        )
    if config.generation_mode is DotToDotGenerationMode.SELECTED_SEGMENTS:
        return _run_selected_segments(
            paths,
            config,
            transform,
            canvas_width=width,
            canvas_height=height,
            tikz_scale=tikz_scale,
            line_color=line_color,
            line_width=line_width,
            precision=precision,
        )
    raise DotToDotError(f"Unsupported generation mode: {config.generation_mode!r}")


def _run_full_figure(
    paths: list[SvgPath],
    config: DotToDotConfig,
    transform: CoordinateTransform,
    *,
    canvas_width: float,
    canvas_height: float,
    tikz_scale: float,
    line_color: str,
    line_width: float,
    precision: int,
) -> DotToDotResult:
    warnings: list[str] = []
    try:
        silhouette_result = extract_silhouette(
            paths,
            canvas_width,
            canvas_height,
            collect_internal_details=config.preserve_internal_details,
        )
    except SilhouetteError as exc:
        raise DotToDotError(str(exc)) from exc
    outline = silhouette_result.outline

    total = outline.total_length
    count = config.total_points
    start_length = _start_length(outline, config.start_point_hint)
    direction = -1.0 if config.direction is NumberingDirection.REVERSE else 1.0
    step = total / count
    positions = [(start_length + direction * index * step) % total for index in range(count)]
    samples = [
        SampledPoint(
            position=outline.point_at_length(length, wrap=True),
            tangent=outline.tangent_at_length(length, wrap=True),
            length=length,
        )
        for length in positions
    ]
    labels = [str(config.start_number + index) for index in range(count)]
    dots = build_dot_points(samples, labels, centroid=outline.centroid(), flip_y=True)
    placements = layout_labels(
        list(dots),
        transform,
        label_offset=config.label_offset,
        point_radius_pt=config.point_radius,
        tikz_scale=tikz_scale,
        silhouette=outline,
    )
    dots = tuple(
        replace(dot, label_xy=(placed.x, placed.y), label_anchor=placed.anchor)
        for dot, placed in zip(dots, placements)
    )
    sequence = DotSequence(prefix="", points=dots, path_id=-1, subpath_id=-1)

    spacing_units = (total * transform.scale) / count
    min_spacing = _DENSITY_WARNING_FACTOR * _LABEL_HEIGHT_CM / max(tikz_scale, 1e-9)
    if spacing_units < min_spacing:
        warnings.append(
            "A quantidade de pontos e alta para o tamanho da figura; "
            "os numeros podem ficar proximos demais."
        )

    # Internal details are re-rendered as filled ink so they keep the Visual
    # look (solid strokes) instead of contour-like stroked ribbon outlines.
    remaining: list[SubPath] = []
    if config.preserve_internal_details and silhouette_result.internal_details:
        remaining = retrace_ink_regions(
            silhouette_result.internal_details, canvas_width, canvas_height
        )

    tikz_code = build_tikz_picture(
        config,
        remaining,
        (sequence,),
        transform,
        tikz_scale=tikz_scale,
        line_color=line_color,
        line_width=line_width,
        precision=precision,
        remaining_render="ink",
    )
    return DotToDotResult(
        tikz_code=tikz_code,
        sequences=(sequence,),
        dotted_refs=(),
        remaining_subpaths=tuple(remaining),
        paths=tuple(paths),
        total_points=count,
        warnings=tuple(warnings),
        silhouette=outline,
    )


def _start_length(outline: SubPath, hint: tuple[float, float] | None) -> float:
    """Return the arc length where point 1 sits.

    With a click hint, the nearest silhouette position wins; otherwise the
    predictable automatic rule applies: topmost point, ties broken leftmost.
    """
    if hint is not None:
        length, _point, _distance = outline.closest_length(Vec2(hint[0], hint[1]))
        return length
    vertices = outline.polyline()
    top = min(vertices, key=lambda point: (point.y, point.x))
    length, _point, _distance = outline.closest_length(top)
    return length


def _run_selected_segments(
    paths: list[SvgPath],
    config: DotToDotConfig,
    transform: CoordinateTransform,
    *,
    canvas_width: float,
    canvas_height: float,
    tikz_scale: float,
    line_color: str,
    line_width: float,
    precision: int,
) -> DotToDotResult:
    if not config.selected_segments:
        raise DotToDotError("Selected-segments generation requires at least one segment.")

    lookup: dict[tuple[int, int], SubPath] = {}
    for path in paths:
        for subpath_id, subpath in enumerate(path.subpaths):
            lookup[(path.path_id, subpath_id)] = subpath

    erase_polylines: list[list[Vec2]] = []
    dotted_keys: dict[tuple[int, int], None] = {}
    dot_points: list[DotPoint] = []
    reverse = config.direction is NumberingDirection.REVERSE
    next_number = config.start_number

    ordered_segments = list(config.selected_segments)
    if reverse:
        ordered_segments = list(reversed(ordered_segments))

    for segment in ordered_segments:
        key = (segment.path_id, segment.subpath_id)
        subpath = lookup.get(key)
        if subpath is None:
            raise DotToDotError(
                f"Selected segment references unknown subpath {key}; re-select the segment."
            )
        samples = sample_selected_segment(subpath, segment)
        labels = [str(next_number + index) for index in range(len(samples))]
        next_number += len(samples)
        centroid = subpath.centroid() if subpath.closed else None
        dot_points.extend(build_dot_points(samples, labels, centroid=centroid, flip_y=True))

        arc = resolve_arc(subpath, segment)
        erase_polylines.append(_arc_polyline(subpath, arc))
        dotted_keys.setdefault(key)

    first_key = (ordered_segments[0].path_id, ordered_segments[0].subpath_id)
    sequences = (
        DotSequence(
            prefix="",
            points=tuple(dot_points),
            path_id=first_key[0],
            subpath_id=first_key[1],
        ),
    )

    # Re-render the remaining drawing as filled ink (the Visual-mode look):
    # the corridors around the selected arcs are erased from the ink itself,
    # so the removed trechos are genuinely absent from the output.
    groups = [list(path.subpaths) for path in paths]
    remaining = retrace_ink_regions(groups, canvas_width, canvas_height, erase_polylines)
    dotted_refs = tuple(
        ContourRef(path_id, subpath_id, lookup[(path_id, subpath_id)])
        for (path_id, subpath_id) in dotted_keys
    )

    tikz_code = build_tikz_picture(
        config,
        remaining,
        sequences,
        transform,
        tikz_scale=tikz_scale,
        line_color=line_color,
        line_width=line_width,
        precision=precision,
        remaining_render="ink",
    )
    return DotToDotResult(
        tikz_code=tikz_code,
        sequences=sequences,
        dotted_refs=dotted_refs,
        remaining_subpaths=tuple(remaining),
        paths=tuple(paths),
        total_points=sum(len(sequence.points) for sequence in sequences),
    )


def _arc_polyline(subpath: SubPath, arc, *, step: float = 2.0) -> list[Vec2]:
    """Sample the removed arc as a dense polyline for corridor erasing."""
    total = subpath.total_length
    if arc.wrap:
        span = (arc.removed_end - arc.removed_start) % total
        if span <= 0.0:
            span = total
    else:
        span = max(arc.removed_end - arc.removed_start, 0.0)
    count = max(2, int(span / max(step, 1e-6)) + 2)
    points: list[Vec2] = []
    for index in range(count):
        raw = arc.removed_start + span * index / (count - 1)
        length = raw % total if arc.wrap else raw
        points.append(subpath.point_at_length(length, wrap=arc.wrap))
    return points


__all__ = ["DotToDotError", "DotToDotResult", "run_dot_to_dot"]
