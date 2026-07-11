"""Arc-length sampling, contour selection and point distribution."""

from __future__ import annotations

from dataclasses import dataclass

from fikzpy.core.dot_to_dot.models import SubPath, SvgPath, Vec2

_EPS = 1e-9


@dataclass(frozen=True)
class SampledPoint:
    """A point sampled along a path together with its unit tangent."""

    position: Vec2
    tangent: Vec2
    length: float


@dataclass(frozen=True)
class ContourRef:
    """A relevant contour with identity and geometry."""

    path_id: int
    subpath_id: int
    subpath: SubPath


def sample_positions(total_length: float, count: int, *, closed: bool, reverse: bool = False) -> list[float]:
    """Return arc-length positions for ``count`` equidistant points.

    For a closed path the positions are ``s_i = i * L / count`` for
    ``i = 0 .. count - 1`` so a point is never placed at both ``s = 0`` and
    ``s = L``. For an open path both endpoints are included via
    ``s_i = i * L / (count - 1)``.
    """
    if count < 1:
        return []
    total = max(float(total_length), 0.0)
    if count == 1:
        return [0.0]
    if closed:
        step = total / count
        positions = [index * step for index in range(count)]
    else:
        step = total / (count - 1)
        positions = [index * step for index in range(count)]
    if reverse:
        if closed:
            positions = [(total - value) % total if total > _EPS else 0.0 for value in positions]
        else:
            positions = list(reversed(positions))
    return positions


def sample_subpath(subpath: SubPath, count: int, *, reverse: bool = False) -> list[SampledPoint]:
    """Sample ``count`` equidistant points along a whole subpath."""
    positions = sample_positions(subpath.total_length, count, closed=subpath.closed, reverse=reverse)
    return [
        SampledPoint(
            position=subpath.point_at_length(length, wrap=subpath.closed),
            tangent=subpath.tangent_at_length(length, wrap=subpath.closed),
            length=length,
        )
        for length in positions
    ]


def sample_length_interval(
    subpath: SubPath,
    start_length: float,
    end_length: float,
    count: int,
    *,
    wrap: bool = False,
) -> list[SampledPoint]:
    """Sample ``count`` points along an arc between two arc-length positions.

    The arc is treated as open, so both endpoints are included. When ``wrap``
    is true (a closed subpath and ``end_length`` before ``start_length``) the
    arc continues past the closure point.
    """
    total = subpath.total_length
    span = end_length - start_length
    if wrap and span < 0.0:
        span += total
    if count < 2:
        length = start_length % total if (wrap and total > _EPS) else start_length
        return [
            SampledPoint(
                position=subpath.point_at_length(length, wrap=wrap),
                tangent=subpath.tangent_at_length(length, wrap=wrap),
                length=length,
            )
        ]
    step = span / (count - 1)
    points: list[SampledPoint] = []
    for index in range(count):
        raw = start_length + index * step
        length = raw % total if (wrap and total > _EPS) else raw
        points.append(
            SampledPoint(
                position=subpath.point_at_length(length, wrap=wrap),
                tangent=subpath.tangent_at_length(length, wrap=wrap),
                length=length,
            )
        )
    return points


def _is_background_frame(subpath: SubPath, canvas_width: float, canvas_height: float) -> bool:
    if canvas_width <= _EPS or canvas_height <= _EPS:
        return False
    min_x, min_y, max_x, max_y = subpath.bounding_box()
    box_width = max_x - min_x
    box_height = max_y - min_y
    covers_canvas = box_width >= 0.98 * canvas_width and box_height >= 0.98 * canvas_height
    box_area = box_width * box_height
    fills_box = box_area > _EPS and subpath.area() >= 0.9 * box_area
    return covers_canvas and fills_box


def relevant_contours(
    paths: list[SvgPath],
    canvas_width: float,
    canvas_height: float,
    *,
    min_area_ratio: float = 0.0005,
    require_closed: bool = True,
) -> list[ContourRef]:
    """Return contours worth dotting, main contour first.

    Excludes a full-canvas background/frame rectangle and tiny noise, then
    ranks the survivors by area (falling back to arc length). The first entry
    is treated as the main external contour.
    """
    canvas_area = max(canvas_width * canvas_height, 0.0)
    noise_area = max(min_area_ratio * canvas_area, 1.0)
    candidates: list[tuple[float, float, ContourRef]] = []
    for path in paths:
        for subpath_id, subpath in enumerate(path.subpaths):
            if require_closed and not subpath.closed:
                continue
            if _is_background_frame(subpath, canvas_width, canvas_height):
                continue
            area = subpath.area()
            if area < noise_area and subpath.total_length < max(0.02 * (canvas_width + canvas_height), 4.0):
                continue
            candidates.append((area, subpath.total_length, ContourRef(path.path_id, subpath_id, subpath)))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry[2] for entry in candidates]


def identify_main_contour(
    paths: list[SvgPath],
    canvas_width: float,
    canvas_height: float,
) -> ContourRef | None:
    """Return the main external contour, or ``None`` when nothing qualifies."""
    contours = relevant_contours(paths, canvas_width, canvas_height)
    if not contours:
        # Relax the closed requirement for line-only figures.
        contours = relevant_contours(paths, canvas_width, canvas_height, require_closed=False)
    return contours[0] if contours else None


def distribute_points(lengths: list[float], total: int, *, minimum: int = 1) -> list[int]:
    """Split ``total`` points across paths proportionally to their length.

    Guarantees at least ``minimum`` points per path and that the returned
    counts sum exactly to ``total`` (largest-remainder rounding). Requires
    ``total >= minimum * len(lengths)``.
    """
    count = len(lengths)
    if count == 0:
        return []
    if total < minimum * count:
        raise ValueError(
            f"total ({total}) is too small for {count} paths with minimum {minimum} each."
        )
    remaining = total - minimum * count
    total_length = sum(max(value, 0.0) for value in lengths)
    if total_length <= _EPS:
        base = [remaining // count] * count
        for index in range(remaining - sum(base)):
            base[index % count] += 1
        return [minimum + value for value in base]

    raw = [max(value, 0.0) / total_length * remaining for value in lengths]
    floors = [int(value) for value in raw]
    assigned = sum(floors)
    remainder = remaining - assigned
    order = sorted(range(count), key=lambda index: raw[index] - floors[index], reverse=True)
    for position in range(remainder):
        floors[order[position % count]] += 1
    return [minimum + value for value in floors]


__all__ = [
    "ContourRef",
    "SampledPoint",
    "distribute_points",
    "identify_main_contour",
    "relevant_contours",
    "sample_length_interval",
    "sample_positions",
    "sample_subpath",
]
