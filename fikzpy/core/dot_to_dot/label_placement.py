"""Predictable normal-based placement of dot-to-dot number labels."""

from __future__ import annotations

from typing import Sequence

from fikzpy.core.dot_to_dot.models import DotPoint, Vec2
from fikzpy.core.dot_to_dot.path_sampler import SampledPoint

_EPS = 1e-9

# Node anchors and the direction (TikZ orientation) of each anchor point
# relative to the node centre.
_ANCHOR_DIRECTIONS: tuple[tuple[str, Vec2], ...] = (
    ("east", Vec2(1.0, 0.0)),
    ("north east", Vec2(1.0, 1.0)),
    ("north", Vec2(0.0, 1.0)),
    ("north west", Vec2(-1.0, 1.0)),
    ("west", Vec2(-1.0, 0.0)),
    ("south west", Vec2(-1.0, -1.0)),
    ("south", Vec2(0.0, -1.0)),
    ("south east", Vec2(1.0, -1.0)),
)


def outward_normal(point: Vec2, tangent: Vec2, centroid: Vec2 | None) -> Vec2:
    """Return the unit normal, pointing away from ``centroid`` when given."""
    normal = tangent.perp().normalized()
    if normal.norm() <= _EPS:
        return Vec2(0.0, -1.0)
    if centroid is not None and normal.dot(point - centroid) < 0.0:
        normal = normal.scaled(-1.0)
    return normal


def _anchor_for(outward: Vec2, *, flip_y: bool) -> str:
    """Return the TikZ node anchor that lets the label body sit outward.

    ``flip_y`` accounts for TikZ using an upward y-axis while the geometry is
    in SVG (downward y) space. The chosen anchor is the node edge that faces
    the curve, so the label extends away from the point.
    """
    outward_tikz = Vec2(outward.x, -outward.y if flip_y else outward.y).normalized()
    if outward_tikz.norm() <= _EPS:
        return "center"
    # The anchor should be on the side opposite to ``outward`` so the body
    # extends outward from the coordinate.
    target = outward_tikz.scaled(-1.0)
    best = max(_ANCHOR_DIRECTIONS, key=lambda item: target.dot(item[1].normalized()))
    return best[0]


def build_dot_points(
    samples: Sequence[SampledPoint],
    labels: Sequence[str],
    *,
    centroid: Vec2 | None = None,
    flip_y: bool = True,
) -> tuple[DotPoint, ...]:
    """Attach numbered labels to sampled points with an outward normal.

    The actual label displacement (``label_offset``) is applied later, in TikZ
    units, by the exporter; here we only record the unit outward normal and the
    node anchor so the label sits on the outer side of the curve.
    """
    if len(samples) != len(labels):
        raise ValueError("samples and labels must have the same length.")
    points: list[DotPoint] = []
    for sample, label in zip(samples, labels):
        outward = outward_normal(sample.position, sample.tangent, centroid)
        anchor = _anchor_for(outward, flip_y=flip_y)
        points.append(
            DotPoint(
                position=sample.position,
                tangent=sample.tangent,
                label=label,
                normal=outward,
                label_anchor=anchor,
            )
        )
    return tuple(points)


__all__ = ["build_dot_points", "outward_normal"]
