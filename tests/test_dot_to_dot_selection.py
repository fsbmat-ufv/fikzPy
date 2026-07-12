"""Stage 3 tests: interactive dot-to-dot segment selection geometry."""

from __future__ import annotations

import pytest

from fikzpy.core.dot_to_dot import DotToDotConfig, run_dot_to_dot
from fikzpy.core.dot_to_dot.config import SelectedSegment
from fikzpy.core.dot_to_dot.models import Vec2
from fikzpy.core.dot_to_dot.segment_selection import (
    InteractiveSelectionController,
    SegmentSelectionModel,
    remaining_runs_after_removal,
    resolve_arc,
    sample_selected_segment,
    snap_point_to_paths,
)
from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths


def _svg(paths: str, *, width: float = 100.0, height: float = 100.0) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f"{paths}</svg>"
    )


_SQUARE = _svg('<path d="M 0 0 L 40 0 L 40 40 L 0 40 Z"/>')
_OPEN_LINE = _svg('<path d="M 0 0 L 100 0"/>')
_CUBIC = _svg('<path d="M 0 0 C 0 60 60 60 60 0"/>')
_TWO_SUBPATHS = _svg('<path d="M 0 0 L 20 0 L 20 20 Z M 70 70 L 90 70 L 90 90 Z"/>')


# --- Snapping -----------------------------------------------------------------------


def test_snap_finds_nearest_subpath() -> None:
    paths = load_svg_paths(_TWO_SUBPATHS)
    near_second = snap_point_to_paths(paths, 80.0, 71.0)
    assert near_second is not None
    assert near_second.subpath_id == 1
    assert near_second.distance == pytest.approx(1.0, abs=0.5)


def test_snap_projects_onto_edge() -> None:
    paths = load_svg_paths(_SQUARE)
    snap = snap_point_to_paths(paths, 20.0, 3.0)
    assert snap is not None
    # Snaps onto the top edge (y = 0).
    assert (snap.point.x, snap.point.y) == pytest.approx((20.0, 0.0), abs=1e-6)
    assert snap.length_ratio == pytest.approx(20.0 / 160.0)


# --- Line / curve division ----------------------------------------------------------


def test_line_division_leaves_two_pieces() -> None:
    paths = load_svg_paths(_OPEN_LINE)
    subpath = paths[0].subpaths[0]
    # Remove the middle third of the open line.
    runs = remaining_runs_after_removal(subpath, [(30.0, 60.0, False)])
    assert len(runs) == 2
    left, right = runs
    assert left[0].start.x == pytest.approx(0.0)
    assert left[-1].end.x == pytest.approx(30.0)
    assert right[0].start.x == pytest.approx(60.0)
    assert right[-1].end.x == pytest.approx(100.0)


def test_cubic_division_points_lie_on_original_curve() -> None:
    paths = load_svg_paths(_CUBIC)
    subpath = paths[0].subpaths[0]
    segment = SelectedSegment(0, 0, 0.2, 0.8, point_count=6)
    samples = sample_selected_segment(subpath, segment)
    assert len(samples) == 6
    # Every sampled point must lie on the original curve (snap distance ~ 0).
    for sample in samples:
        _length, _point, distance = subpath.closest_length(sample.position)
        assert distance < 1e-3


# --- Open selection -----------------------------------------------------------------


def test_open_selection_samples_between_clicks() -> None:
    paths = load_svg_paths(_OPEN_LINE)
    subpath = paths[0].subpaths[0]
    segment = SelectedSegment(0, 0, 0.25, 0.75, point_count=3)
    samples = sample_selected_segment(subpath, segment)
    xs = [round(sample.position.x, 3) for sample in samples]
    assert xs == [25.0, 50.0, 75.0]


# --- Closed path: two arcs and inversion --------------------------------------------


def test_closed_path_forward_and_reverse_pick_complementary_arcs() -> None:
    paths = load_svg_paths(_SQUARE)
    subpath = paths[0].subpaths[0]
    forward = SelectedSegment(0, 0, 0.0, 0.25, point_count=5, reverse=False)
    reverse = SelectedSegment(0, 0, 0.0, 0.25, point_count=5, reverse=False | True)

    fwd_arc = resolve_arc(subpath, forward)
    rev_arc = resolve_arc(subpath, reverse)
    total = subpath.total_length
    # Forward arc spans a quarter (40); reverse arc spans the other three quarters.
    fwd_span = (fwd_arc.sample_positions[-1] - fwd_arc.sample_positions[0]) % total
    assert fwd_span == pytest.approx(40.0)
    # Reverse traverses backward from the same start point.
    assert rev_arc.sample_positions[0] == pytest.approx(fwd_arc.sample_positions[0])
    assert rev_arc.sample_positions[1] != pytest.approx(fwd_arc.sample_positions[1])


def test_reverse_removes_the_complementary_arc() -> None:
    paths = load_svg_paths(_SQUARE)
    subpath = paths[0].subpaths[0]
    forward = SelectedSegment(0, 0, 0.0, 0.25, point_count=3, reverse=False)
    reverse = SelectedSegment(0, 0, 0.0, 0.25, point_count=3, reverse=True)

    fwd = resolve_arc(subpath, forward)
    rev = resolve_arc(subpath, reverse)
    # Forward removes [0, 40]; reverse removes the complement [40, 160] (wrapping).
    assert (fwd.removed_start, fwd.removed_end) == pytest.approx((0.0, 40.0))
    assert rev.removed_start == pytest.approx(40.0)
    assert rev.removed_end == pytest.approx(0.0)


def test_closed_middle_removal_leaves_single_wrapping_run() -> None:
    paths = load_svg_paths(_SQUARE)
    subpath = paths[0].subpaths[0]
    # Remove an arc in the middle (does not touch the seam at 0/total).
    runs = remaining_runs_after_removal(subpath, [(40.0, 80.0, False)])
    assert len(runs) == 1  # single open run crossing the start/end seam


# --- Multiple segments & continuous numbering ---------------------------------------


def test_multiple_segments_number_continuously() -> None:
    config = DotToDotConfig(
        generation_mode="selected_segments",
        start_number=1,
        selected_segments=(
            SelectedSegment(0, 0, 0.0, 0.25, point_count=3),
            SelectedSegment(0, 0, 0.5, 0.75, point_count=4),
        ),
    )
    result = run_dot_to_dot(_SQUARE, (100, 100), config, width_units=10.0)
    labels = [dot.label for sequence in result.sequences for dot in sequence.points]
    assert labels == ["1", "2", "3", "4", "5", "6", "7"]
    assert result.point_count == 7


def test_selected_arcs_are_removed_from_the_drawing() -> None:
    config = DotToDotConfig(
        generation_mode="selected_segments",
        selected_segments=(SelectedSegment(0, 0, 0.0, 0.25, point_count=3),),
    )
    result = run_dot_to_dot(_SQUARE, (100, 100), config, width_units=10.0, precision=3)
    # The removed top edge (0,0)->(40,0) becomes tikz (0,10)->(4,10); the drawing
    # must not contain that straight top edge, but must contain the points.
    assert "\\fill[dotToDotPoint]" in result.tikz_code
    # Exactly one remaining open run (the other three edges).
    assert len(result.remaining_subpaths) == 1
    assert result.remaining_subpaths[0].closed is False


def test_unknown_segment_reference_raises() -> None:
    from fikzpy.core.dot_to_dot import DotToDotError

    config = DotToDotConfig(
        generation_mode="selected_segments",
        selected_segments=(SelectedSegment(3, 7, 0.0, 0.5, point_count=3),),
    )
    with pytest.raises(DotToDotError):
        run_dot_to_dot(_SQUARE, (100, 100), config)


# --- Selection model: undo / delete / numbering ranges ------------------------------


def test_selection_model_add_undo_delete() -> None:
    model = SegmentSelectionModel(start_number=1)
    a = SelectedSegment(0, 0, 0.0, 0.25, point_count=3)
    b = SelectedSegment(0, 0, 0.5, 0.75, point_count=4)
    model.add(a)
    model.add(b)
    assert len(model) == 2
    assert model.undo() is b
    assert len(model) == 1
    model.add(b)
    assert model.remove(0) is a
    assert model.segments == (b,)
    model.clear()
    assert len(model) == 0
    assert model.undo() is None


def test_selection_model_numbering_ranges() -> None:
    model = SegmentSelectionModel(start_number=11)
    model.add(SelectedSegment(0, 0, 0.0, 0.25, point_count=10))
    model.add(SelectedSegment(0, 0, 0.5, 0.6, point_count=5))
    assert model.numbering_ranges() == [(11, 20), (21, 25)]
    assert model.total_points() == 15


# --- Interactive controller ---------------------------------------------------------


def test_controller_two_clicks_build_pending_segment() -> None:
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(_SQUARE))
    controller.set_point_count(6)
    controller.click(20.0, 0.0)  # start on top edge
    assert controller.pending_segment() is None  # only start so far
    controller.click(40.0, 20.0)  # end on right edge
    pending = controller.pending_segment()
    assert pending is not None
    assert pending.point_count == 6
    assert pending.start_length_ratio == pytest.approx(20.0 / 160.0)


def test_controller_invert_toggles_reverse() -> None:
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(_SQUARE))
    controller.click(20.0, 0.0)
    controller.click(40.0, 20.0)
    assert controller.pending_segment().reverse is False
    controller.invert()
    assert controller.pending_segment().reverse is True


def test_controller_confirm_undo_and_delete() -> None:
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(_SQUARE))
    controller.click(20.0, 0.0)
    controller.click(40.0, 20.0)
    confirmed = controller.confirm()
    assert confirmed is not None
    assert len(controller.model) == 1
    assert controller.pending_segment() is None  # pending reset after confirm

    # A second selection, then undo removes it.
    controller.click(0.0, 20.0)
    controller.click(20.0, 40.0)
    controller.confirm()
    assert len(controller.model) == 2
    controller.undo()
    assert len(controller.model) == 1
    controller.delete(0)
    assert len(controller.model) == 0


def test_controller_undo_clears_pending_first() -> None:
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(_SQUARE))
    controller.click(20.0, 0.0)
    controller.click(40.0, 20.0)
    controller.confirm()
    controller.click(0.0, 20.0)  # start a new pending selection
    # Undo should clear the pending selection before touching confirmed ones.
    controller.undo()
    assert controller.start is None
    assert len(controller.model) == 1


def test_controller_second_click_prefers_start_subpath() -> None:
    # Two nearby squares: the end click lands slightly nearer the second
    # square, but only marginally, so the arc stays on the start's subpath.
    svg = _svg('<path d="M 0 0 L 40 0 L 40 40 L 0 40 Z M 46 0 L 86 0 L 86 40 L 46 40 Z"/>')
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(svg))
    controller.click(20.0, 0.0)   # start on square 1
    controller.click(43.5, 20.0)  # between the squares, nearer square 2
    pending = controller.pending_segment()
    assert pending is not None
    assert pending.subpath_id == 0


def test_controller_second_click_far_on_other_path_restarts() -> None:
    svg = _svg('<path d="M 0 0 L 40 0 L 40 40 L 0 40 Z M 80 80 L 95 80 L 95 95 L 80 95 Z"/>')
    controller = InteractiveSelectionController()
    controller.set_paths(load_svg_paths(svg))
    controller.click(20.0, 0.0)  # start on square 1
    controller.click(88.0, 80.0)  # clearly on the distant square 2
    assert controller.pending_segment() is None
    assert controller.start is not None
    assert controller.start.subpath_id == 1


def test_controller_confirmed_segments_feed_pipeline() -> None:
    controller = InteractiveSelectionController(start_number=1)
    controller.set_paths(load_svg_paths(_SQUARE))
    controller.set_point_count(3)
    controller.click(20.0, 0.0)
    controller.click(40.0, 0.0)
    controller.confirm()
    config = DotToDotConfig(
        generation_mode="selected_segments",
        selected_segments=controller.model.segments,
        start_number=1,
    )
    result = run_dot_to_dot(_SQUARE, (100, 100), config, width_units=10.0)
    assert result.point_count == 3
