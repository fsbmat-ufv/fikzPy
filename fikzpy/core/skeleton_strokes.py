"""Trace complete pencil-like strokes from a skeleton.

The plain skeleton tracer in :mod:`fikzpy.core.stroke_tracer` splits a drawing at
*every* skeleton node, so the thinning artefacts of anti-aliased lines (tiny
barbs and micro-junctions) shatter even a single straight line into dozens of
fragments. That produces long, noisy TikZ.

This module rebuilds the skeleton as a small graph and reasons about it the way a
person reads a drawing:

1. split the skeleton into *segments* between nodes (endpoints/junctions);
2. prune short *spurs* -- the thinning barbs that hang off a junction;
3. at each remaining junction, *link* the two segments that continue most
   straight through it, so a stroke that crosses another stroke stays whole;
4. concatenate linked segments into complete open strokes and closed loops.

The result is a handful of long, smooth strokes instead of many fragments, which
the Bezier fitter can then turn into compact ``\\draw`` curves.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from fikzpy.core.contour_detector import Contour, simplify_polyline

Pixel = tuple[int, int]


@dataclass
class _Segment:
    """A skeleton edge between two nodes, as an ordered list of pixels."""

    pixels: list[Pixel]
    node_a: Pixel
    node_b: Pixel
    closed: bool = False

    @property
    def length(self) -> float:
        return _pixel_path_length(self.pixels)


def trace_complete_strokes(
    skeleton: np.ndarray,
    *,
    simplify_epsilon: float = 0.01,
    min_path_length: int = 3,
    smooth_iterations: int = 1,
    spur_length: float = 8.0,
    link_max_angle_deg: float = 60.0,
    min_stroke_length: float = 0.0,
    corner_smoothing: int = 0,
    corner_threshold_cos: float = -0.7,
    clean: bool = True,
) -> list[Contour]:
    """Trace a skeleton into complete strokes with spur pruning and linking."""
    if clean:
        skeleton = clean_skeleton(skeleton)
    pixels = _skeleton_pixels(skeleton)
    if not pixels:
        return []

    neighbors = {pixel: [n for n in _neighbor_pixels(pixel) if n in pixels] for pixel in pixels}
    segments = _build_segments(pixels, neighbors)
    if not segments:
        return []

    segments = _prune_spurs(segments, spur_length=spur_length)
    strokes = _link_segments(segments, link_max_angle_deg=link_max_angle_deg)

    contours: list[Contour] = []
    for pixel_path, closed in strokes:
        if len(pixel_path) < 2:
            continue
        points = np.array([[float(x), float(y)] for y, x in pixel_path], dtype=np.float64)
        points = _smooth_polyline(points, closed=closed, iterations=smooth_iterations)
        simplified = simplify_polyline(points, epsilon_ratio=simplify_epsilon, closed=closed)
        if len(simplified) < 2:
            continue
        if corner_smoothing > 0:
            simplified = _smooth_corners(
                simplified, closed=closed, iterations=corner_smoothing, corner_cos=corner_threshold_cos
            )
        perimeter = _xy_path_length(simplified, closed=closed)
        if len(simplified) < min_path_length and perimeter < max(min_stroke_length, 1.0):
            continue
        if perimeter < min_stroke_length:
            continue
        contours.append(Contour(points=simplified, closed=closed, area=0.0, perimeter=perimeter))

    contours.sort(key=lambda item: item.perimeter, reverse=True)
    return contours


# --------------------------------------------------------------------------- #
# Skeleton cleaning (staircase removal)
# --------------------------------------------------------------------------- #
# Relative positions of the 8 neighbours, bit 0..7. Used both to build the
# simple-point lookup table and to read a pixel's neighbourhood at run time.
_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _build_simple_point_table() -> tuple[bool, ...]:
    """Precompute, for all 256 neighbourhoods, whether the centre is simple.

    A foreground pixel is *simple* when deleting it does not change the topology:
    its foreground 8-neighbours must form exactly one 8-connected component. The
    centre pixel is excluded, so two neighbours only linked through the centre
    count as separate components (a bridge), which correctly keeps the centre.
    """
    adjacency: list[list[int]] = []
    for i, (yi, xi) in enumerate(_NEIGHBOR_OFFSETS):
        neighbours = []
        for j, (yj, xj) in enumerate(_NEIGHBOR_OFFSETS):
            if i != j and abs(yi - yj) <= 1 and abs(xi - xj) <= 1:
                neighbours.append(j)
        adjacency.append(neighbours)

    table: list[bool] = []
    for code in range(256):
        present = [i for i in range(8) if code & (1 << i)]
        if not present:
            table.append(False)
            continue
        seen: set[int] = set()
        stack = [present[0]]
        seen.add(present[0])
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if (code & (1 << neighbour)) and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        table.append(len(seen) == len(present))
    return tuple(table)


_SIMPLE_POINT_TABLE = _build_simple_point_table()


def clean_skeleton(skeleton: np.ndarray, *, max_iterations: int = 8) -> np.ndarray:
    """Remove 8-connected staircase corners while preserving topology.

    skimage's skeleton is 1-pixel thin but, on diagonal strokes, leaves L-shaped
    staircase corners. With 8-connectivity those corners read as degree-3 nodes,
    so a single straight line shatters into hundreds of fragments. Deleting the
    corner pixels (only when they are simple points, i.e. topology-preserving)
    turns the skeleton into clean degree-2 paths with only the real junctions.
    """
    s = (skeleton > 0).astype(np.uint8)
    height, width = s.shape
    for _ in range(max_iterations):
        removed = False
        ys, xs = np.nonzero(s)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if y <= 0 or x <= 0 or y >= height - 1 or x >= width - 1:
                continue
            if not s[y, x]:
                continue
            north, east = s[y - 1, x], s[y, x + 1]
            south, west = s[y + 1, x], s[y, x - 1]
            # Only act on staircase corners: a perpendicular pair of orthogonal
            # neighbours. Collinear neighbours (straight runs) are left alone.
            if not ((north and east) or (east and south) or (south and west) or (west and north)):
                continue
            code = 0
            for bit, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
                if s[y + dy, x + dx]:
                    code |= 1 << bit
            if _SIMPLE_POINT_TABLE[code]:
                s[y, x] = 0
                removed = True
        if not removed:
            break
    return (s * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _build_segments(pixels: set[Pixel], neighbors: dict[Pixel, list[Pixel]]) -> list[_Segment]:
    """Split the skeleton into segments running between nodes (degree != 2)."""
    nodes = {pixel for pixel, items in neighbors.items() if len(items) != 2}
    visited_edges: set[frozenset[Pixel]] = set()
    segments: list[_Segment] = []

    for node in nodes:
        for neighbor in neighbors[node]:
            if frozenset((node, neighbor)) in visited_edges:
                continue
            path = _walk_segment(node, neighbor, neighbors, nodes, visited_edges)
            segments.append(_Segment(pixels=path, node_a=path[0], node_b=path[-1]))

    # Closed loops with no nodes (e.g. a circle): every pixel has degree 2.
    for pixel in pixels:
        for neighbor in neighbors[pixel]:
            if frozenset((pixel, neighbor)) in visited_edges:
                continue
            path = _walk_loop(pixel, neighbor, neighbors, visited_edges)
            if len(path) >= 3:
                segments.append(_Segment(pixels=path, node_a=path[0], node_b=path[0], closed=True))

    return segments


def _walk_segment(
    node: Pixel,
    neighbor: Pixel,
    neighbors: dict[Pixel, list[Pixel]],
    nodes: set[Pixel],
    visited_edges: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [node]
    previous, current = node, neighbor
    while True:
        visited_edges.add(frozenset((previous, current)))
        path.append(current)
        if current in nodes:
            break
        candidates = [
            item
            for item in neighbors[current]
            if item != previous and frozenset((current, item)) not in visited_edges
        ]
        if not candidates:
            break
        previous, current = current, candidates[0]
    return path


def _walk_loop(
    start: Pixel,
    neighbor: Pixel,
    neighbors: dict[Pixel, list[Pixel]],
    visited_edges: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [start]
    previous, current = start, neighbor
    while True:
        visited_edges.add(frozenset((previous, current)))
        path.append(current)
        candidates = [
            item
            for item in neighbors[current]
            if item != previous and frozenset((current, item)) not in visited_edges
        ]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            visited_edges.add(frozenset((current, start)))
            path.append(start)
            break
        previous, current = current, nxt
    return path


# --------------------------------------------------------------------------- #
# Spur pruning
# --------------------------------------------------------------------------- #
def _prune_spurs(segments: list[_Segment], *, spur_length: float) -> list[_Segment]:
    """Drop short barbs hanging off a junction (thinning artefacts)."""
    if spur_length <= 0:
        return segments

    kept = list(segments)
    changed = True
    while changed:
        changed = False
        degree = _node_degrees(kept)
        survivors: list[_Segment] = []
        for segment in kept:
            if segment.closed:
                survivors.append(segment)
                continue
            leaf_a = degree.get(segment.node_a, 0) == 1
            leaf_b = degree.get(segment.node_b, 0) == 1
            junction_a = degree.get(segment.node_a, 0) >= 3
            junction_b = degree.get(segment.node_b, 0) >= 3
            is_spur = (
                segment.length < spur_length
                and ((leaf_a and junction_b) or (leaf_b and junction_a))
            )
            if is_spur:
                changed = True
                continue
            survivors.append(segment)
        kept = survivors
    return kept


def _node_degrees(segments: list[_Segment]) -> dict[Pixel, int]:
    degree: dict[Pixel, int] = {}
    for segment in segments:
        if segment.closed:
            continue
        degree[segment.node_a] = degree.get(segment.node_a, 0) + 1
        degree[segment.node_b] = degree.get(segment.node_b, 0) + 1
    return degree


# --------------------------------------------------------------------------- #
# Junction linking
# --------------------------------------------------------------------------- #
def _link_segments(
    segments: list[_Segment],
    *,
    link_max_angle_deg: float,
) -> list[tuple[list[Pixel], bool]]:
    """Link segments through nodes by straightness, then build complete strokes."""
    open_segments = [segment for segment in segments if not segment.closed]
    closed_strokes = [(segment.pixels, True) for segment in segments if segment.closed]

    # Collect the two ends of every open segment at each node.
    ends_by_node: dict[Pixel, list[tuple[int, int]]] = {}
    for index, segment in enumerate(open_segments):
        ends_by_node.setdefault(segment.node_a, []).append((index, 0))
        ends_by_node.setdefault(segment.node_b, []).append((index, 1))

    pairing: dict[tuple[int, int], tuple[int, int] | None] = {}
    cos_threshold = math.cos(math.radians(180.0 - max(0.0, link_max_angle_deg)))
    for ends in ends_by_node.values():
        _pair_ends_at_node(ends, open_segments, pairing, cos_threshold)

    strokes = _build_strokes_from_pairing(open_segments, pairing)
    return strokes + closed_strokes


def _pair_ends_at_node(
    ends: list[tuple[int, int]],
    segments: list[_Segment],
    pairing: dict[tuple[int, int], tuple[int, int] | None],
    cos_threshold: float,
) -> None:
    for end in ends:
        pairing.setdefault(end, None)

    if len(ends) < 2:
        return

    if len(ends) == 2:
        # A through-node (or a junction left with two segments after pruning).
        pairing[ends[0]] = ends[1]
        pairing[ends[1]] = ends[0]
        return

    # Real junction: greedily pair the straightest continuations.
    directions = {end: _end_direction(segments[end[0]], end[1]) for end in ends}
    scored: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            dot = float(np.dot(directions[ends[i]], directions[ends[j]]))
            scored.append((dot, ends[i], ends[j]))
    scored.sort(key=lambda item: item[0])  # most anti-parallel (straightest) first

    used: set[tuple[int, int]] = set()
    for dot, end_i, end_j in scored:
        if end_i in used or end_j in used:
            continue
        if dot > cos_threshold:  # not straight enough to be the same stroke
            break
        pairing[end_i] = end_j
        pairing[end_j] = end_i
        used.add(end_i)
        used.add(end_j)


def _build_strokes_from_pairing(
    segments: list[_Segment],
    pairing: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[list[Pixel], bool]]:
    used: set[int] = set()
    strokes: list[tuple[list[Pixel], bool]] = []

    def walk(start_seg: int, start_end: int) -> tuple[list[Pixel], bool]:
        path: list[Pixel] = []
        seg_index, enter_end = start_seg, start_end
        closed = False
        while seg_index not in used:
            used.add(seg_index)
            pixels = segments[seg_index].pixels
            oriented = pixels if enter_end == 0 else pixels[::-1]
            if path and oriented and path[-1] == oriented[0]:
                oriented = oriented[1:]
            path.extend(oriented)
            exit_end = 1 - enter_end
            nxt = pairing.get((seg_index, exit_end))
            if nxt is None:
                break
            next_seg, next_end = nxt
            if next_seg in used:
                closed = True
                break
            seg_index, enter_end = next_seg, next_end
        return path, closed

    # Start from free ends (no pairing) so open strokes begin at their tips.
    for index in range(len(segments)):
        if index in used:
            continue
        for end in (0, 1):
            if pairing.get((index, end)) is None:
                path, closed = walk(index, end)
                if path:
                    strokes.append((path, closed))
                break

    # Remaining segments belong to closed loops of linked pieces.
    for index in range(len(segments)):
        if index in used:
            continue
        path, _ = walk(index, 0)
        if path:
            strokes.append((path, True))

    return strokes


def _end_direction(segment: _Segment, end: int) -> np.ndarray:
    """Unit direction pointing from a segment end into the segment."""
    pixels = segment.pixels
    if end == 1:
        pixels = pixels[::-1]
    origin = np.array(pixels[0], dtype=np.float64)
    step = min(len(pixels) - 1, 4)
    far = np.array(pixels[step], dtype=np.float64)
    vector = far - origin
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros(2, dtype=np.float64)
    return vector / norm


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _smooth_polyline(points: np.ndarray, *, closed: bool, iterations: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4 or iterations <= 0:
        return pts
    for _ in range(iterations):
        if closed:
            previous_points = np.roll(pts, 1, axis=0)
            next_points = np.roll(pts, -1, axis=0)
            pts = (previous_points + 2.0 * pts + next_points) / 4.0
        else:
            smoothed = pts.copy()
            smoothed[1:-1] = (pts[:-2] + 2.0 * pts[1:-1] + pts[2:]) / 4.0
            pts = smoothed
    return pts


def _smooth_corners(
    points: np.ndarray,
    *,
    closed: bool,
    iterations: int,
    corner_cos: float,
) -> np.ndarray:
    """Corner-preserving Chaikin smoothing.

    Gentle corners are rounded (so faceted curves become smooth before the Bezier
    fit), while sharp corners -- star tips, box corners -- are kept as-is. A vertex
    is "sharp" when the cosine of the angle between its two edges exceeds
    ``corner_cos`` (i.e. the turn is large).
    """
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(max(0, int(iterations))):
        count = len(pts)
        if count < 4:
            break
        smoothed: list[np.ndarray] = []
        if not closed:
            smoothed.append(pts[0])
        indices = range(count) if closed else range(1, count - 1)
        for i in indices:
            current = pts[i]
            to_prev = pts[(i - 1) % count] - current
            to_next = pts[(i + 1) % count] - current
            norm_prev = float(np.linalg.norm(to_prev))
            norm_next = float(np.linalg.norm(to_next))
            if norm_prev < 1e-9 or norm_next < 1e-9:
                smoothed.append(current)
                continue
            cosine = float(np.dot(to_prev, to_next) / (norm_prev * norm_next))
            if cosine > corner_cos:  # sharp corner: keep it crisp
                smoothed.append(current)
            else:  # gentle corner: cut it to round the curve
                smoothed.append(current + 0.25 * to_prev)
                smoothed.append(current + 0.25 * to_next)
        if not closed:
            smoothed.append(pts[-1])
        pts = np.array(smoothed, dtype=np.float64)
    return pts


def _skeleton_pixels(skeleton: np.ndarray) -> set[Pixel]:
    ys, xs = np.nonzero(skeleton > 0)
    return set(zip(ys.tolist(), xs.tolist()))


def _neighbor_pixels(pixel: Pixel) -> Iterable[Pixel]:
    y, x = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield y + dy, x + dx


def _pixel_path_length(pixels: list[Pixel]) -> float:
    if len(pixels) < 2:
        return 0.0
    total = 0.0
    for (y0, x0), (y1, x1) in zip(pixels, pixels[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _xy_path_length(points: np.ndarray, *, closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if closed and len(points) > 2:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total
