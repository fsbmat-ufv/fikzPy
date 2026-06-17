"""Tests for the complete-stroke skeleton tracer used by the Classic mode."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from fikzpy.core.image_processor import to_gray
from fikzpy.core.skeleton_strokes import (
    clean_skeleton,
    trace_complete_strokes,
    _neighbor_pixels,
    _skeleton_pixels,
    _smooth_corners,
)
from fikzpy.core.stroke_tracer import (
    StrokeTracingSettings,
    extract_ink_mask,
    skeletonize,
    trace_strokes_from_skeleton,
)


def _skeleton_of(draw) -> np.ndarray:
    image = np.full((220, 300, 3), 255, dtype=np.uint8)
    draw(image)
    mask = extract_ink_mask(to_gray(image), StrokeTracingSettings(dark_threshold=215))
    return skeletonize(mask, method="skimage")


def _junction_count(skeleton: np.ndarray) -> int:
    pixels = _skeleton_pixels(skeleton)
    neighbors = {p: [n for n in _neighbor_pixels(p) if n in pixels] for p in pixels}
    degrees = Counter(len(v) for v in neighbors.values())
    return sum(count for degree, count in degrees.items() if degree >= 3)


def test_clean_skeleton_removes_staircase_junctions():
    skeleton = _skeleton_of(lambda im: cv2.line(im, (40, 40), (260, 180), (0, 0, 0), 3))
    before = _junction_count(skeleton)
    cleaned = clean_skeleton(skeleton)
    after = _junction_count(cleaned)

    # A straight line has no real junctions; staircase corners create fake ones.
    assert before > 10
    assert after == 0


def test_clean_skeleton_preserves_connectivity():
    skeleton = _skeleton_of(lambda im: cv2.ellipse(im, (150, 110), (90, 60), 0, 0, 360, (0, 0, 0), 3))
    before = cv2.connectedComponents(skeleton, connectivity=8)[0]
    after = cv2.connectedComponents(clean_skeleton(skeleton), connectivity=8)[0]
    assert after == before


def test_complete_strokes_recovers_a_closed_loop():
    skeleton = _skeleton_of(lambda im: cv2.circle(im, (150, 110), 80, (0, 0, 0), 3))
    contours = trace_complete_strokes(
        skeleton, simplify_epsilon=0.006, spur_length=8.0, min_stroke_length=8.0
    )
    assert len(contours) <= 2
    assert any(contour.closed for contour in contours)


def test_complete_strokes_beats_plain_tracer_on_a_line():
    skeleton = _skeleton_of(lambda im: cv2.line(im, (40, 40), (260, 180), (0, 0, 0), 3))
    plain = trace_strokes_from_skeleton(skeleton, simplify_epsilon=0.006, min_path_length=3)
    complete = trace_complete_strokes(
        skeleton, simplify_epsilon=0.006, spur_length=8.0, min_stroke_length=8.0
    )
    # The plain tracer shatters the line into many fragments; the new one keeps it whole.
    assert len(plain) > len(complete)
    assert len(complete) <= 2


def test_complete_strokes_links_crossing_strokes():
    def draw(im):
        cv2.line(im, (40, 40), (260, 180), (0, 0, 0), 3)
        cv2.line(im, (40, 180), (260, 40), (0, 0, 0), 3)

    skeleton = _skeleton_of(draw)
    contours = trace_complete_strokes(
        skeleton, simplify_epsilon=0.006, spur_length=8.0, link_max_angle_deg=60.0, min_stroke_length=8.0
    )
    # An X keeps its two straight strokes whole instead of four stubs.
    assert len(contours) <= 3


def test_empty_skeleton_returns_no_contours():
    assert trace_complete_strokes(np.zeros((50, 50), dtype=np.uint8)) == []


def test_smooth_corners_keeps_sharp_corners_and_rounds_gentle_ones():
    # A square: four 90-degree corners must survive corner-preserving smoothing.
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    smoothed = _smooth_corners(square, closed=True, iterations=3, corner_cos=-0.7)
    for corner in square:
        assert np.min(np.linalg.norm(smoothed - corner, axis=1)) < 1e-6

    # A many-sided polygon approximating a circle has only gentle corners, which
    # get cut (the point set changes, rounding the shape).
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)]) * 5.0
    rounded = _smooth_corners(circle, closed=True, iterations=2, corner_cos=-0.7)
    assert len(rounded) > len(circle)


def test_short_spurs_are_pruned():
    # A long horizontal stroke with a tiny vertical barb hanging off the middle.
    skeleton = np.zeros((60, 160), dtype=np.uint8)
    skeleton[30, 10:150] = 255
    skeleton[24:30, 80] = 255  # short spur
    contours = trace_complete_strokes(
        skeleton, simplify_epsilon=0.0, spur_length=14.0, min_stroke_length=10.0
    )
    # The barb is dropped and the straight stroke is linked back into one piece.
    assert len(contours) == 1
    assert contours[0].perimeter > 120
