"""Geometry for interactive dot-to-dot segment selection (Stage 3).

Handles snapping clicks onto the nearest subpath, resolving the arc chosen by
two clicks (including the two possible arcs of a closed subpath and the
``reverse`` toggle), sampling points along the removed arc, and mathematically
reconstructing the drawing that remains after one or more arcs are excised.
"""

from __future__ import annotations

from dataclasses import dataclass

from fikzpy.core.dot_to_dot.config import SelectedSegment
from fikzpy.core.dot_to_dot.models import Segment, SubPath, SvgPath, Vec2
from fikzpy.core.dot_to_dot.path_sampler import SampledPoint

_EPS = 1e-9


@dataclass(frozen=True)
class SnapResult:
    """The result of snapping a click onto the nearest subpath."""

    path_id: int
    subpath_id: int
    length: float
    length_ratio: float
    point: Vec2
    distance: float


@dataclass(frozen=True)
class ResolvedArc:
    """A resolved selected arc: its sample positions and removed interval."""

    sample_positions: tuple[float, ...]
    removed_start: float
    removed_end: float
    wrap: bool


def snap_point_to_paths(paths: list[SvgPath], x: float, y: float) -> SnapResult | None:
    """Return the nearest ``(path, subpath, arc-length)`` to a clicked point."""
    click = Vec2(float(x), float(y))
    best: SnapResult | None = None
    for path in paths:
        for subpath_id, subpath in enumerate(path.subpaths):
            length, point, distance = subpath.closest_length(click)
            if best is None or distance < best.distance:
                total = subpath.total_length
                ratio = 0.0 if total <= _EPS else length / total
                best = SnapResult(
                    path_id=path.path_id,
                    subpath_id=subpath_id,
                    length=length,
                    length_ratio=ratio,
                    point=point,
                    distance=distance,
                )
    return best


def _forward_arc_run(subpath: SubPath, start: float, end: float) -> list[Segment]:
    """Return the geometry from ``start`` forward to ``end`` (wrapping if needed)."""
    total = subpath.total_length
    if start <= end + _EPS:
        return subpath.run_between(start, end)
    return subpath.run_between(start, total) + subpath.run_between(0.0, end)


def resolve_arc(subpath: SubPath, segment: SelectedSegment) -> ResolvedArc:
    """Resolve the arc a :class:`SelectedSegment` refers to on ``subpath``."""
    total = subpath.total_length
    start_length = segment.start_length_ratio * total
    end_length = segment.end_length_ratio * total
    count = max(2, segment.point_count)

    if subpath.closed:
        forward_span = (end_length - start_length) % total
        backward_span = (start_length - end_length) % total
        if not segment.reverse:
            span = forward_span
            direction = 1.0
            removed_start, removed_end = start_length, end_length
        else:
            span = backward_span
            direction = -1.0
            removed_start, removed_end = end_length, start_length
        step = span / (count - 1)
        positions = tuple((start_length + direction * index * step) % total for index in range(count))
        return ResolvedArc(positions, removed_start % total, removed_end % total, wrap=True)

    # Open subpath: a single arc between the two clicks, reverse only reorders.
    lo = min(start_length, end_length)
    hi = max(start_length, end_length)
    step = (hi - lo) / (count - 1)
    ascending = tuple(lo + index * step for index in range(count))
    if (start_length <= end_length) == (not segment.reverse):
        positions = ascending
    else:
        positions = tuple(reversed(ascending))
    return ResolvedArc(positions, lo, hi, wrap=False)


def sample_selected_segment(subpath: SubPath, segment: SelectedSegment) -> list[SampledPoint]:
    """Sample the numbered points of a selected arc, in numbering order."""
    arc = resolve_arc(subpath, segment)
    points: list[SampledPoint] = []
    for length in arc.sample_positions:
        points.append(
            SampledPoint(
                position=subpath.point_at_length(length, wrap=subpath.closed),
                tangent=subpath.tangent_at_length(length, wrap=subpath.closed),
                length=length,
            )
        )
    return points


def _normalize_intervals(total: float, intervals: list[tuple[float, float, bool]]) -> list[tuple[float, float]]:
    """Split forward/wrapping arcs into non-wrapping ``[start, end]`` pieces."""
    flats: list[tuple[float, float]] = []
    for start, end, wrap in intervals:
        start = min(max(start, 0.0), total)
        end = min(max(end, 0.0), total)
        if not wrap or start <= end + _EPS:
            if end - start > _EPS:
                flats.append((start, end))
        else:
            if total - start > _EPS:
                flats.append((start, total))
            if end > _EPS:
                flats.append((0.0, end))
    return flats


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + _EPS:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def remaining_runs_after_removal(
    subpath: SubPath,
    removed: list[tuple[float, float, bool]],
) -> list[list[Segment]]:
    """Return the drawing that remains after removing one or more arcs.

    ``removed`` entries are ``(start_length, end_length, wrap)`` forward arcs.
    The complementary runs are reconstructed with exact De Casteljau splits.
    """
    total = subpath.total_length
    flats = _merge_intervals(_normalize_intervals(total, removed))
    if not flats:
        return [list(subpath.segments)]

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in flats:
        if start - cursor > _EPS:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if total - cursor > _EPS:
        gaps.append((cursor, total))

    runs = [subpath.run_between(start, end) for start, end in gaps]

    # On a closed subpath a middle removal leaves a single run that crosses the
    # start/end seam; join the first and last gap when the seam is intact.
    if subpath.closed and len(gaps) >= 2:
        seam_intact = gaps[0][0] <= _EPS and abs(gaps[-1][1] - total) <= _EPS
        if seam_intact:
            joined = runs[-1] + runs[0]
            runs = [joined] + runs[1:-1]
    return [run for run in runs if run]


class SegmentSelectionModel:
    """Ordered, editable collection of confirmed selected segments.

    Holds the interactive selection state (add, undo, delete, clear) separately
    from any GUI so the workflow logic is testable without Qt. Numbering is
    continuous across confirmed segments.
    """

    def __init__(self, start_number: int = 1) -> None:
        self._segments: list[SelectedSegment] = []
        self.start_number = int(start_number)

    @property
    def segments(self) -> tuple[SelectedSegment, ...]:
        """Return the confirmed segments in order."""
        return tuple(self._segments)

    def __len__(self) -> int:
        return len(self._segments)

    def add(self, segment: SelectedSegment) -> None:
        """Append a confirmed segment."""
        self._segments.append(segment)

    def undo(self) -> SelectedSegment | None:
        """Remove and return the most recently added segment, if any."""
        if not self._segments:
            return None
        return self._segments.pop()

    def remove(self, index: int) -> SelectedSegment | None:
        """Remove and return the segment at ``index`` (ignoring bad indices)."""
        if 0 <= index < len(self._segments):
            return self._segments.pop(index)
        return None

    def clear(self) -> None:
        """Remove every confirmed segment."""
        self._segments.clear()

    def numbering_ranges(self) -> list[tuple[int, int]]:
        """Return the ``(first, last)`` label number of each confirmed segment."""
        ranges: list[tuple[int, int]] = []
        current = self.start_number
        for segment in self._segments:
            last = current + max(2, segment.point_count) - 1
            ranges.append((current, last))
            current = last + 1
        return ranges

    def total_points(self) -> int:
        """Return the number of points across all confirmed segments."""
        return sum(max(2, segment.point_count) for segment in self._segments)


class InteractiveSelectionController:
    """Drive the interactive two-click selection workflow without any GUI.

    ``click`` cycles start -> end -> restart. Once both endpoints are on the
    same subpath a pending :class:`SelectedSegment` is available for preview,
    can be inverted (closed subpaths) and confirmed into the model. All heavy
    logic lives here so the Qt canvas stays a thin renderer.
    """

    def __init__(self, start_number: int = 1) -> None:
        self.paths: list[SvgPath] = []
        self.model = SegmentSelectionModel(start_number=start_number)
        self.start: SnapResult | None = None
        self.end: SnapResult | None = None
        self.reverse = False
        self.point_count = 10

    def set_paths(self, paths: list[SvgPath]) -> None:
        """Replace the geometry and reset the pending selection."""
        self.paths = list(paths)
        self.reset_pending()
        self.model.clear()

    def reset_pending(self) -> None:
        """Clear the in-progress start/end clicks."""
        self.start = None
        self.end = None
        self.reverse = False

    def click(self, x: float, y: float) -> SnapResult | None:
        """Register a snapped click, cycling start -> end -> restart."""
        snap = snap_point_to_paths(self.paths, x, y)
        if snap is None:
            return None
        if self.start is None:
            self.start = snap
            self.end = None
            self.reverse = False
        elif self.end is None:
            if (snap.path_id, snap.subpath_id) != (self.start.path_id, self.start.subpath_id):
                # Second click landed on another subpath: restart from here.
                self.start = snap
                self.end = None
            else:
                self.end = snap
        else:
            self.start = snap
            self.end = None
            self.reverse = False
        return snap

    def invert(self) -> None:
        """Toggle which of the two closed-path arcs is selected."""
        self.reverse = not self.reverse

    def set_point_count(self, count: int) -> None:
        """Set how many points the pending segment will receive."""
        self.point_count = max(2, int(count))

    def pending_segment(self) -> SelectedSegment | None:
        """Return the pending :class:`SelectedSegment`, if both clicks are set."""
        if self.start is None or self.end is None:
            return None
        return SelectedSegment(
            path_id=self.start.path_id,
            subpath_id=self.start.subpath_id,
            start_length_ratio=self.start.length_ratio,
            end_length_ratio=self.end.length_ratio,
            reverse=self.reverse,
            point_count=max(2, self.point_count),
        )

    def confirm(self) -> SelectedSegment | None:
        """Confirm the pending segment into the model and reset the pending state."""
        segment = self.pending_segment()
        if segment is None:
            return None
        self.model.add(segment)
        self.reset_pending()
        return segment

    def undo(self) -> SelectedSegment | None:
        """Undo the last confirmed segment (or clear the pending one first)."""
        if self.start is not None or self.end is not None:
            self.reset_pending()
            return None
        return self.model.undo()

    def delete(self, index: int) -> SelectedSegment | None:
        """Delete a confirmed segment by index."""
        return self.model.remove(index)


__all__ = [
    "InteractiveSelectionController",
    "ResolvedArc",
    "SegmentSelectionModel",
    "SnapResult",
    "remaining_runs_after_removal",
    "resolve_arc",
    "sample_selected_segment",
    "snap_point_to_paths",
]
