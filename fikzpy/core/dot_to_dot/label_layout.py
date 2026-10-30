"""Collision-aware label layout for dot-to-dot numbers.

Labels are placed in TikZ coordinates: each number starts on the outward
normal of its dot and is progressively pushed further (or swung sideways)
until it does not overlap other labels, other dots, or the interior of the
silhouette. The layout is greedy and deterministic — simple and predictable,
as the numbers must remain hand-editable TikZ nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import cv2
import numpy as np

from fikzpy.core.dot_to_dot.models import DotPoint, SubPath, Vec2
from fikzpy.core.dot_to_dot.tikz_exporter import CoordinateTransform

_EPS = 1e-9
_CM_PER_PT = 1.0 / 28.4527
# \scriptsize sans digits are roughly 5.5pt tall / 3.6pt wide per character;
# the extents are padded so neighbouring numbers stay clearly separated.
_LABEL_HEIGHT_CM = 0.24
_LABEL_WIDTH_PER_CHAR_CM = 0.16
_OFFSET_FACTORS = (1.0, 1.5, 2.1, 2.9, 3.9, 5.0)
# Odd-numbered dots try slightly larger radii first, staggering consecutive
# labels so neighbours in concave pockets do not land on one ring.
_OFFSET_FACTORS_ODD = (1.3, 1.8, 2.5, 3.4, 4.4, 5.5)
_SWING_ANGLES = (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6)


@dataclass(frozen=True)
class PlacedLabel:
    """A resolved label position (TikZ coordinates) and its node anchor."""

    x: float
    y: float
    anchor: str


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


def _anchor_for_direction(direction: Vec2) -> str:
    if direction.norm() <= _EPS:
        return "center"
    target = direction.scaled(-1.0).normalized()
    best = max(_ANCHOR_DIRECTIONS, key=lambda item: target.dot(item[1].normalized()))
    return best[0]


def _rotate(vector: Vec2, angle: float) -> Vec2:
    return Vec2(
        vector.x * cos(angle) - vector.y * sin(angle),
        vector.x * sin(angle) + vector.y * cos(angle),
    )


def _label_radius(text: str, tikz_scale: float) -> float:
    width = _LABEL_WIDTH_PER_CHAR_CM * max(1, len(text))
    extent = max(width, _LABEL_HEIGHT_CM)
    return extent / (2.0 * max(tikz_scale, _EPS))


def _silhouette_polygon_tikz(silhouette: SubPath | None, transform: CoordinateTransform) -> np.ndarray | None:
    if silhouette is None:
        return None
    points = [transform.to_tikz(point) for point in silhouette.polyline()]
    return np.array(points, np.float32).reshape(-1, 1, 2)


def _is_inside(polygon: np.ndarray | None, x: float, y: float, margin: float) -> bool:
    if polygon is None:
        return False
    distance = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
    return distance > -margin


def layout_labels(
    dots: list[DotPoint],
    transform: CoordinateTransform,
    *,
    label_offset: float,
    point_radius_pt: float,
    tikz_scale: float,
    silhouette: SubPath | None = None,
) -> list[PlacedLabel]:
    """Return one collision-checked label placement per dot, in dot order."""
    polygon = _silhouette_polygon_tikz(silhouette, transform)
    point_radius = point_radius_pt * _CM_PER_PT / max(tikz_scale, _EPS)
    dot_positions = [transform.to_tikz(dot.position) for dot in dots]

    placed: list[tuple[float, float, float]] = []  # (x, y, radius) of accepted labels
    results: list[PlacedLabel] = []
    for index, dot in enumerate(dots):
        base_x, base_y = dot_positions[index]
        outward = Vec2(dot.normal.x, -dot.normal.y).normalized()
        if outward.norm() <= _EPS:
            outward = Vec2(0.0, 1.0)
        radius = _label_radius(dot.label, tikz_scale)
        clearance = point_radius + radius * 0.9

        factors = _OFFSET_FACTORS if index % 2 == 0 else _OFFSET_FACTORS_ODD
        best: tuple[float, PlacedLabel, tuple[float, float]] | None = None
        chosen: tuple[PlacedLabel, tuple[float, float]] | None = None
        for factor in factors:
            for angle in _SWING_ANGLES:
                direction = _rotate(outward, angle)
                distance = max(label_offset, point_radius + radius) * factor
                x = base_x + direction.x * distance
                y = base_y + direction.y * distance
                # The anchor sits on the dot-facing edge of the node, so the
                # real text centre lies one half-extent further out.
                center = (x + direction.x * radius, y + direction.y * radius)
                penalty = 0.0
                if _is_inside(polygon, center[0], center[1], margin=radius * 0.5):
                    penalty += 4.0
                for px, py, pradius in placed:
                    gap = float(np.hypot(center[0] - px, center[1] - py)) - (radius + pradius)
                    if gap < 0.0:
                        # Depth-weighted: deep text-on-text overlap must lose
                        # against any merely-crowded alternative.
                        penalty += 1.0 - 4.0 * gap
                for oindex, (ox, oy) in enumerate(dot_positions):
                    if oindex == index:
                        continue
                    gap = float(np.hypot(center[0] - ox, center[1] - oy)) - clearance
                    if gap < 0.0:
                        penalty += 0.5 - 2.0 * gap
                candidate = PlacedLabel(x, y, _anchor_for_direction(direction))
                if penalty <= _EPS:
                    chosen = (candidate, center)
                    break
                if best is None or penalty < best[0]:
                    best = (penalty, candidate, center)
            if chosen is not None:
                break
        if chosen is None:
            if best is not None:
                chosen = (best[1], best[2])
            else:
                fallback = PlacedLabel(
                    base_x + outward.x * label_offset,
                    base_y + outward.y * label_offset,
                    _anchor_for_direction(outward),
                )
                chosen = (fallback, (fallback.x, fallback.y))
        placed.append((chosen[1][0], chosen[1][1], radius))
        results.append(chosen[0])
    return results


__all__ = ["PlacedLabel", "layout_labels"]
