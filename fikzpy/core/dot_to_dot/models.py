"""Geometric model for the dot-to-dot pipeline.

This module is intentionally self-contained: it implements a small, well
tested geometric core (2D points, line/quadratic/cubic Bezier segments and
arc-length aware subpaths) so the dot-to-dot mode never has to reach into the
Visual mode's SVG/svg2tikz flow. Distances are measured along the *real* arc
length of each curve, not along the raw Bezier parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from typing import Sequence

_EPS = 1e-9
_FLATTEN_STEP = 1.0
_MIN_CURVE_SAMPLES = 16
_MAX_CURVE_SAMPLES = 256


@dataclass(frozen=True)
class Vec2:
    """A 2D point / vector with the arithmetic the pipeline needs."""

    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "Vec2":
        """Return this vector multiplied by ``factor``."""
        return Vec2(self.x * factor, self.y * factor)

    def dot(self, other: "Vec2") -> float:
        """Return the dot product with ``other``."""
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        """Return the Euclidean length of this vector."""
        return hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        """Return a unit vector, or the zero vector when degenerate."""
        length = self.norm()
        if length <= _EPS:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def perp(self) -> "Vec2":
        """Return the left-hand perpendicular (rotated 90 degrees)."""
        return Vec2(-self.y, self.x)

    def distance_to(self, other: "Vec2") -> float:
        """Return the Euclidean distance to ``other``."""
        return hypot(self.x - other.x, self.y - other.y)


def _lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


class Segment:
    """Base class for parametric path segments over ``t`` in ``[0, 1]``."""

    start: Vec2
    end: Vec2

    def point(self, t: float) -> Vec2:  # pragma: no cover - abstract
        raise NotImplementedError

    def derivative(self, t: float) -> Vec2:  # pragma: no cover - abstract
        raise NotImplementedError

    def split(self, t: float) -> tuple["Segment", "Segment"]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _sample_count(self) -> int:
        return 2

    def flatten(self) -> list[tuple[float, Vec2]]:
        """Return ``(t, point)`` samples along the segment, endpoints included."""
        count = max(2, self._sample_count())
        return [(i / (count - 1), self.point(i / (count - 1))) for i in range(count)]

    def subsegment(self, t0: float, t1: float) -> "Segment":
        """Return the portion of this segment between ``t0`` and ``t1``.

        Reconstructed exactly using De Casteljau splits, so a cubic sub-arc
        stays a cubic and a line stays a line.
        """
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        lo = min(max(lo, 0.0), 1.0)
        hi = min(max(hi, 0.0), 1.0)
        if hi - lo <= _EPS:
            point = self.point(lo)
            return LineSegment(point, point)
        right = self if lo <= _EPS else self.split(lo)[1]
        remaining = 1.0 - lo
        inner = 1.0 if remaining <= _EPS else (hi - lo) / remaining
        inner = min(max(inner, 0.0), 1.0)
        if inner >= 1.0 - _EPS:
            return right
        return right.split(inner)[0]


@dataclass(frozen=True)
class LineSegment(Segment):
    """A straight line segment."""

    start: Vec2
    end: Vec2

    def point(self, t: float) -> Vec2:
        return _lerp(self.start, self.end, t)

    def derivative(self, t: float) -> Vec2:
        return self.end - self.start

    def split(self, t: float) -> tuple[Segment, Segment]:
        mid = self.point(t)
        return LineSegment(self.start, mid), LineSegment(mid, self.end)

    def _sample_count(self) -> int:
        return 2


@dataclass(frozen=True)
class QuadraticSegment(Segment):
    """A quadratic Bezier segment."""

    start: Vec2
    control: Vec2
    end: Vec2

    def point(self, t: float) -> Vec2:
        u = 1.0 - t
        return self.start.scaled(u * u) + self.control.scaled(2 * u * t) + self.end.scaled(t * t)

    def derivative(self, t: float) -> Vec2:
        u = 1.0 - t
        return (self.control - self.start).scaled(2 * u) + (self.end - self.control).scaled(2 * t)

    def split(self, t: float) -> tuple[Segment, Segment]:
        a = _lerp(self.start, self.control, t)
        b = _lerp(self.control, self.end, t)
        mid = _lerp(a, b, t)
        return QuadraticSegment(self.start, a, mid), QuadraticSegment(mid, b, self.end)

    def _sample_count(self) -> int:
        length = self.start.distance_to(self.control) + self.control.distance_to(self.end)
        return _curve_sample_count(length)


@dataclass(frozen=True)
class CubicSegment(Segment):
    """A cubic Bezier segment."""

    start: Vec2
    control1: Vec2
    control2: Vec2
    end: Vec2

    def point(self, t: float) -> Vec2:
        u = 1.0 - t
        return (
            self.start.scaled(u * u * u)
            + self.control1.scaled(3 * u * u * t)
            + self.control2.scaled(3 * u * t * t)
            + self.end.scaled(t * t * t)
        )

    def derivative(self, t: float) -> Vec2:
        u = 1.0 - t
        return (
            (self.control1 - self.start).scaled(3 * u * u)
            + (self.control2 - self.control1).scaled(6 * u * t)
            + (self.end - self.control2).scaled(3 * t * t)
        )

    def split(self, t: float) -> tuple[Segment, Segment]:
        a = _lerp(self.start, self.control1, t)
        b = _lerp(self.control1, self.control2, t)
        c = _lerp(self.control2, self.end, t)
        d = _lerp(a, b, t)
        e = _lerp(b, c, t)
        mid = _lerp(d, e, t)
        return CubicSegment(self.start, a, d, mid), CubicSegment(mid, e, c, self.end)

    def _sample_count(self) -> int:
        length = (
            self.start.distance_to(self.control1)
            + self.control1.distance_to(self.control2)
            + self.control2.distance_to(self.end)
        )
        return _curve_sample_count(length)


def _curve_sample_count(control_length: float) -> int:
    estimate = int(ceil(max(control_length, 0.0) / _FLATTEN_STEP)) + 1
    return max(_MIN_CURVE_SAMPLES, min(_MAX_CURVE_SAMPLES, estimate))


@dataclass(frozen=True)
class ArcSample:
    """One flattened sample along a subpath, in arc-length order."""

    length: float
    point: Vec2
    segment_index: int
    local_t: float


class SubPath:
    """An ordered chain of segments, optionally closed.

    The subpath precomputes an arc-length table over finely flattened samples
    so callers can address it by real distance (``point_at_length``) rather
    than by raw curve parameter.
    """

    def __init__(self, segments: Sequence[Segment], *, closed: bool) -> None:
        self._segments: tuple[Segment, ...] = tuple(segments)
        if not self._segments:
            raise ValueError("SubPath requires at least one segment.")
        self.closed = bool(closed)
        self._samples: tuple[ArcSample, ...] = self._build_arc_table()

    @property
    def segments(self) -> tuple[Segment, ...]:
        """Return the segment chain."""
        return self._segments

    @property
    def total_length(self) -> float:
        """Return the total arc length of the subpath."""
        return self._samples[-1].length

    @property
    def start(self) -> Vec2:
        """Return the first point of the subpath."""
        return self._segments[0].start

    @property
    def end(self) -> Vec2:
        """Return the last point of the subpath."""
        return self._segments[-1].end

    def _build_arc_table(self) -> tuple[ArcSample, ...]:
        samples: list[ArcSample] = []
        cumulative = 0.0
        previous: Vec2 | None = None
        for index, segment in enumerate(self._segments):
            flat = segment.flatten()
            start_at = 0 if previous is None else 1
            for local_t, point in flat[start_at:]:
                if previous is not None:
                    cumulative += previous.distance_to(point)
                samples.append(ArcSample(cumulative, point, index, local_t))
                previous = point
        if len(samples) == 1:
            only = samples[0]
            samples.append(ArcSample(only.length, only.point, only.segment_index, 1.0))
        return tuple(samples)

    def _sample_index_for_length(self, length: float) -> int:
        low, high = 0, len(self._samples) - 1
        while low < high:
            mid = (low + high) // 2
            if self._samples[mid].length < length:
                low = mid + 1
            else:
                high = mid
        return low

    def point_at_length(self, length: float, *, wrap: bool | None = None) -> Vec2:
        """Return the point at arc-length ``length`` along the subpath."""
        total = self.total_length
        if total <= _EPS:
            return self.start
        wrap_enabled = self.closed if wrap is None else wrap
        if wrap_enabled:
            length = length % total
        length = min(max(length, 0.0), total)
        upper = self._sample_index_for_length(length)
        if upper == 0:
            return self._samples[0].point
        lower = upper - 1
        low_sample = self._samples[lower]
        high_sample = self._samples[upper]
        span = high_sample.length - low_sample.length
        if span <= _EPS:
            return high_sample.point
        frac = (length - low_sample.length) / span
        return _lerp(low_sample.point, high_sample.point, frac)

    def tangent_at_length(self, length: float, *, wrap: bool | None = None) -> Vec2:
        """Return the unit tangent at arc-length ``length``."""
        total = self.total_length
        if total <= _EPS:
            return Vec2(1.0, 0.0)
        wrap_enabled = self.closed if wrap is None else wrap
        if wrap_enabled:
            length = length % total
        length = min(max(length, 0.0), total)
        upper = self._sample_index_for_length(length)
        if upper == 0:
            upper = 1
        lower = upper - 1
        low_sample = self._samples[lower]
        high_sample = self._samples[upper]
        if high_sample.segment_index == low_sample.segment_index:
            span = high_sample.length - low_sample.length
            frac = 0.0 if span <= _EPS else (length - low_sample.length) / span
            local_t = low_sample.local_t + (high_sample.local_t - low_sample.local_t) * frac
            segment = self._segments[low_sample.segment_index]
        else:
            segment = self._segments[high_sample.segment_index]
            local_t = high_sample.local_t
        tangent = segment.derivative(local_t).normalized()
        if tangent.norm() <= _EPS:
            tangent = (high_sample.point - low_sample.point).normalized()
        return tangent

    def _locate(self, length: float) -> tuple[int, float]:
        total = self.total_length
        length = min(max(length, 0.0), total)
        upper = self._sample_index_for_length(length)
        if upper == 0:
            return 0, 0.0
        lower = upper - 1
        low_sample = self._samples[lower]
        high_sample = self._samples[upper]
        if high_sample.segment_index != low_sample.segment_index:
            return high_sample.segment_index, high_sample.local_t
        span = high_sample.length - low_sample.length
        frac = 0.0 if span <= _EPS else (length - low_sample.length) / span
        local_t = low_sample.local_t + (high_sample.local_t - low_sample.local_t) * frac
        return low_sample.segment_index, local_t

    def run_between(self, start_length: float, end_length: float) -> list[Segment]:
        """Return the contiguous forward run of segments from ``start`` to ``end``.

        ``start_length`` must be less than or equal to ``end_length`` and both
        are clamped into ``[0, total_length]``. Boundary segments are split
        exactly with De Casteljau, so a partially selected cubic stays a cubic.
        """
        total = self.total_length
        a = min(max(min(start_length, end_length), 0.0), total)
        b = min(max(max(start_length, end_length), 0.0), total)
        if b - a <= _EPS:
            point = self.point_at_length(a, wrap=False)
            return [LineSegment(point, point)]
        start_index, start_t = self._locate(a)
        end_index, end_t = self._locate(b)
        if start_index == end_index:
            return [self._segments[start_index].subsegment(start_t, end_t)]
        run: list[Segment] = [self._segments[start_index].subsegment(start_t, 1.0)]
        run.extend(self._segments[start_index + 1 : end_index])
        run.append(self._segments[end_index].subsegment(0.0, end_t))
        return [segment for segment in run if not _is_degenerate(segment)]

    def signed_area(self) -> float:
        """Return the signed polygon area of the flattened subpath."""
        points = [sample.point for sample in self._samples]
        area = 0.0
        for index in range(len(points)):
            current = points[index]
            following = points[(index + 1) % len(points)]
            area += current.x * following.y - following.x * current.y
        return area / 2.0

    def area(self) -> float:
        """Return the absolute polygon area of the flattened subpath."""
        return abs(self.signed_area())

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return ``(min_x, min_y, max_x, max_y)`` over flattened samples."""
        xs = [sample.point.x for sample in self._samples]
        ys = [sample.point.y for sample in self._samples]
        return min(xs), min(ys), max(xs), max(ys)

    def polyline(self) -> list[Vec2]:
        """Return the flattened polyline points (for rendering/hit-testing)."""
        return [sample.point for sample in self._samples]

    def centroid(self) -> Vec2:
        """Return the average of the flattened samples."""
        count = len(self._samples)
        sum_x = sum(sample.point.x for sample in self._samples)
        sum_y = sum(sample.point.y for sample in self._samples)
        return Vec2(sum_x / count, sum_y / count)

    def closest_length(self, point: Vec2) -> tuple[float, Vec2, float]:
        """Return ``(arc_length, closest_point, distance)`` nearest to ``point``.

        Used to snap an interactive click onto this subpath.
        """
        best_length = 0.0
        best_point = self._samples[0].point
        best_distance = float("inf")
        samples = self._samples
        for index in range(len(samples) - 1):
            low = samples[index]
            high = samples[index + 1]
            edge = high.point - low.point
            edge_len2 = edge.dot(edge)
            if edge_len2 <= _EPS:
                t = 0.0
                projected = low.point
            else:
                t = min(max((point - low.point).dot(edge) / edge_len2, 0.0), 1.0)
                projected = low.point + edge.scaled(t)
            distance = point.distance_to(projected)
            if distance < best_distance:
                best_distance = distance
                best_point = projected
                best_length = low.length + t * (high.length - low.length)
        return best_length, best_point, best_distance


def _is_degenerate(segment: Segment) -> bool:
    return segment.start.distance_to(segment.end) <= _EPS and isinstance(segment, LineSegment)


@dataclass(frozen=True)
class SvgPath:
    """One SVG ``<path>`` element decomposed into subpaths."""

    subpaths: tuple[SubPath, ...]
    path_id: int
    element_id: str | None = None


@dataclass(frozen=True)
class DotPoint:
    """A single numbered dot sampled along a path.

    ``normal`` is the unit outward normal in SVG/geometry space. When the
    collision-aware layout has run, ``label_xy`` holds the final label
    position in TikZ coordinates; otherwise the exporter falls back to a
    plain normal-direction offset.
    """

    position: Vec2
    tangent: Vec2
    label: str
    normal: Vec2
    label_anchor: str
    label_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class DotSequence:
    """A group of dots forming one independent numbered sequence."""

    prefix: str
    points: tuple[DotPoint, ...]
    path_id: int
    subpath_id: int


__all__ = [
    "ArcSample",
    "CubicSegment",
    "DotPoint",
    "DotSequence",
    "LineSegment",
    "QuadraticSegment",
    "Segment",
    "SubPath",
    "SvgPath",
    "Vec2",
]
