"""Orchestration for the independent Ponto a Ponto (dot-to-dot) pipeline.

The pipeline consumes a *copy* of an SVG string (typically the intermediate
SVG produced by the project's Visual tracing stage), parses it into its own
geometric model and produces TikZ where the replaced contour is expressed as
numbered native-TikZ points. It never mutates the source SVG and never falls
back to another mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import ascii_uppercase

from fikzpy.core.dot_to_dot.config import (
    DotToDotConfig,
    DotToDotGenerationMode,
    NumberingDirection,
    SelectedSegment,
)
from fikzpy.core.dot_to_dot.label_placement import build_dot_points
from fikzpy.core.dot_to_dot.models import DotPoint, DotSequence, SubPath, SvgPath
from fikzpy.core.dot_to_dot.path_sampler import (
    ContourRef,
    distribute_points,
    relevant_contours,
    sample_subpath,
)
from fikzpy.core.dot_to_dot.segment_selection import (
    remaining_runs_after_removal,
    resolve_arc,
    sample_selected_segment,
)
from fikzpy.core.dot_to_dot.svg_loader import SvgLoaderConfig, load_svg_paths
from fikzpy.core.dot_to_dot.tikz_exporter import CoordinateTransform, build_tikz_picture


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
    contours = relevant_contours(paths, canvas_width, canvas_height)
    if not contours:
        contours = relevant_contours(paths, canvas_width, canvas_height, require_closed=False)
    if not contours:
        raise DotToDotError("No suitable contour was found to convert into points.")

    if config.include_internal_contours:
        selected = _limit_contours(contours, config)
    else:
        selected = contours[:1]

    counts = _allocate_points(selected, config)
    reverse = config.direction is NumberingDirection.REVERSE
    sequences: list[DotSequence] = []
    for index, (contour, count) in enumerate(zip(selected, counts)):
        prefix = "" if index == 0 else _sequence_prefix(index)
        sequence = _build_sequence(contour, count, prefix, config, reverse=reverse)
        sequences.append(sequence)

    dotted_ids = {(ref.path_id, ref.subpath_id) for ref in selected}
    remaining = _remaining_subpaths(paths, dotted_ids, config, warnings)

    tikz_code = build_tikz_picture(
        config,
        remaining,
        sequences,
        transform,
        tikz_scale=tikz_scale,
        line_color=line_color,
        line_width=line_width,
        precision=precision,
    )
    return DotToDotResult(
        tikz_code=tikz_code,
        sequences=tuple(sequences),
        dotted_refs=tuple(selected),
        remaining_subpaths=tuple(remaining),
        paths=tuple(paths),
        total_points=config.total_points,
        warnings=tuple(warnings),
    )


def _run_selected_segments(
    paths: list[SvgPath],
    config: DotToDotConfig,
    transform: CoordinateTransform,
    *,
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

    removed_by_subpath: dict[tuple[int, int], list[tuple[float, float, bool]]] = {}
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
        removed_by_subpath.setdefault(key, []).append((arc.removed_start, arc.removed_end, arc.wrap))

    first_key = (ordered_segments[0].path_id, ordered_segments[0].subpath_id)
    sequences = (
        DotSequence(
            prefix="",
            points=tuple(dot_points),
            path_id=first_key[0],
            subpath_id=first_key[1],
        ),
    )

    remaining = _remaining_after_selection(paths, removed_by_subpath)
    dotted_refs = tuple(
        ContourRef(path_id, subpath_id, lookup[(path_id, subpath_id)])
        for (path_id, subpath_id) in removed_by_subpath
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
    )
    return DotToDotResult(
        tikz_code=tikz_code,
        sequences=sequences,
        dotted_refs=dotted_refs,
        remaining_subpaths=tuple(remaining),
        paths=tuple(paths),
        total_points=sum(len(sequence.points) for sequence in sequences),
    )


def _remaining_after_selection(
    paths: list[SvgPath],
    removed_by_subpath: dict[tuple[int, int], list[tuple[float, float, bool]]],
) -> list[SubPath]:
    remaining: list[SubPath] = []
    for path in paths:
        for subpath_id, subpath in enumerate(path.subpaths):
            key = (path.path_id, subpath_id)
            if key not in removed_by_subpath:
                remaining.append(subpath)
                continue
            for run in remaining_runs_after_removal(subpath, removed_by_subpath[key]):
                remaining.append(SubPath(run, closed=False))
    return remaining


def _limit_contours(contours: list[ContourRef], config: DotToDotConfig) -> list[ContourRef]:
    max_paths = max(1, config.total_points // config.min_points_per_path)
    return contours[:max_paths]


def _allocate_points(selected: list[ContourRef], config: DotToDotConfig) -> list[int]:
    if len(selected) == 1:
        return [config.total_points]
    lengths = [ref.subpath.total_length for ref in selected]
    return distribute_points(lengths, config.total_points, minimum=config.min_points_per_path)


def _build_sequence(
    contour: ContourRef,
    count: int,
    prefix: str,
    config: DotToDotConfig,
    *,
    reverse: bool,
) -> DotSequence:
    samples = sample_subpath(contour.subpath, count, reverse=reverse)
    labels = _labels_for(prefix, count, config.start_number)
    centroid = contour.subpath.centroid() if contour.subpath.closed else None
    dot_points = build_dot_points(samples, labels, centroid=centroid, flip_y=True)
    return DotSequence(
        prefix=prefix,
        points=dot_points,
        path_id=contour.path_id,
        subpath_id=contour.subpath_id,
    )


def _labels_for(prefix: str, count: int, start_number: int) -> list[str]:
    if prefix == "":
        return [str(start_number + index) for index in range(count)]
    return [f"{prefix}{index + 1}" for index in range(count)]


def _sequence_prefix(index: int) -> str:
    # index 1 -> "A", 2 -> "B", ..., 26 -> "Z", 27 -> "AA" (bijective base 26).
    number = index
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = ascii_uppercase[remainder] + letters
    return letters


def _remaining_subpaths(
    paths: list[SvgPath],
    dotted_ids: set[tuple[int, int]],
    config: DotToDotConfig,
    warnings: list[str],
) -> list[SubPath]:
    if not config.preserve_internal_details:
        return []
    remaining: list[SubPath] = []
    for path in paths:
        for subpath_id, subpath in enumerate(path.subpaths):
            if (path.path_id, subpath_id) in dotted_ids:
                continue
            remaining.append(subpath)
    return remaining


__all__ = ["DotToDotError", "DotToDotResult", "run_dot_to_dot"]
