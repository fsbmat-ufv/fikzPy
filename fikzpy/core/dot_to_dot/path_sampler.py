"""Arc-length sampling helpers for the dot-to-dot pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from fikzpy.core.dot_to_dot.models import SubPath, Vec2

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


__all__ = [
    "ContourRef",
    "SampledPoint",
    "sample_length_interval",
    "sample_positions",
    "sample_subpath",
]
