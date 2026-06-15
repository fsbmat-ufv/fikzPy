from __future__ import annotations

import cv2
import numpy as np

from fikzpy.core.image_processor import ProcessingSettings, process_image
from fikzpy.core.stroke_tracer import StrokeTracingSettings, extract_ink_mask, skeletonize
from fikzpy.core.stroke_tracer import trace_strokes_from_skeleton


def test_extract_ink_mask_keeps_dark_line_art() -> None:
    gray = np.full((40, 40), 255, dtype=np.uint8)
    cv2.line(gray, (5, 20), (35, 20), 0, 2)

    mask = extract_ink_mask(gray)

    assert mask[20, 20] == 255
    assert mask[0, 0] == 0


def test_extract_ink_mask_can_keep_faint_gray_lines() -> None:
    gray = np.full((40, 40), 255, dtype=np.uint8)
    cv2.line(gray, (5, 20), (35, 20), 210, 2)

    mask = extract_ink_mask(gray, StrokeTracingSettings(dark_threshold=215))

    assert mask[20, 20] == 255


def test_faint_stroke_recovery_is_opt_in() -> None:
    gray = np.full((50, 70), 255, dtype=np.uint8)
    cv2.line(gray, (8, 25), (62, 25), 244, 2)

    base_mask = extract_ink_mask(gray, StrokeTracingSettings(dark_threshold=215))
    recovered_mask = extract_ink_mask(
        gray,
        StrokeTracingSettings(dark_threshold=215, recover_faint_strokes=True),
    )

    assert base_mask[25, 35] == 0
    assert recovered_mask[25, 35] == 255


def test_faint_stroke_recovery_keeps_flat_background_clean() -> None:
    gray = np.full((50, 70), 232, dtype=np.uint8)

    mask = extract_ink_mask(
        gray,
        StrokeTracingSettings(dark_threshold=215, recover_faint_strokes=True),
    )

    assert np.count_nonzero(mask) == 0


def test_blackhat_recovery_can_keep_low_contrast_strokes() -> None:
    gray = np.full((60, 80), 255, dtype=np.uint8)
    cv2.line(gray, (10, 30), (70, 30), 244, 2)

    mask = extract_ink_mask(
        gray,
        StrokeTracingSettings(
            dark_threshold=215,
            recover_blackhat_strokes=True,
            blackhat_threshold=10,
        ),
    )

    assert mask[30, 40] == 255


def test_trace_strokes_from_skeleton_returns_open_path() -> None:
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    cv2.line(skeleton, (5, 15), (25, 15), 255, 1)

    strokes = trace_strokes_from_skeleton(skeleton, min_path_length=3, smooth_iterations=1)

    assert len(strokes) == 1
    assert not strokes[0].closed
    assert len(strokes[0].points) >= 2


def test_trace_strokes_can_snap_branch_endpoints_to_junction_center() -> None:
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    for y, x in [(9, 9), (9, 10), (10, 9), (10, 10)]:
        skeleton[y, x] = 255
    cv2.line(skeleton, (10, 2), (10, 9), 255, 1)
    cv2.line(skeleton, (10, 10), (10, 17), 255, 1)
    cv2.line(skeleton, (2, 10), (9, 10), 255, 1)
    cv2.line(skeleton, (10, 10), (17, 17), 255, 1)

    strokes = trace_strokes_from_skeleton(
        skeleton,
        min_path_length=3,
        smooth_iterations=0,
        simplify_epsilon=0.0,
        snap_junction_endpoints=True,
    )
    endpoint_counts: dict[tuple[float, float], int] = {}
    for stroke in strokes:
        for point in (stroke.points[0], stroke.points[-1]):
            key = tuple(np.round(point, 2))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

    assert max(endpoint_counts.values()) >= 3


def test_process_image_line_art_mode_finds_internal_strokes() -> None:
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (70, 50), (0, 0, 0), 2)
    cv2.line(image, (20, 30), (60, 30), (0, 0, 0), 2)

    result = process_image(
        image,
        ProcessingSettings(vectorization_mode="line_art", simplify_epsilon=0.006, min_path_length=3),
    )

    assert result.skeleton is not None
    assert result.contours
    assert any(not contour.closed for contour in result.contours)


def test_process_image_enables_faint_recovery_only_for_vector_mode() -> None:
    image = np.full((50, 70, 3), 255, dtype=np.uint8)
    cv2.line(image, (8, 25), (62, 25), (244, 244, 244), 2)

    classic = process_image(
        image,
        ProcessingSettings(vectorization_mode="classic", line_art_threshold=215, min_path_length=3),
    )
    vector = process_image(
        image,
        ProcessingSettings(vectorization_mode="vector", line_art_threshold=215, min_path_length=3),
    )

    assert classic.ink_mask is not None
    assert vector.ink_mask is not None
    assert np.count_nonzero(classic.ink_mask) == 0
    assert np.count_nonzero(vector.ink_mask) > 0
    assert len(vector.contours) >= 1
