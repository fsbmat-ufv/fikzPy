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


def test_trace_strokes_from_skeleton_returns_open_path() -> None:
    skeleton = np.zeros((30, 30), dtype=np.uint8)
    cv2.line(skeleton, (5, 15), (25, 15), 255, 1)

    strokes = trace_strokes_from_skeleton(skeleton, min_path_length=3, smooth_iterations=1)

    assert len(strokes) == 1
    assert not strokes[0].closed
    assert len(strokes[0].points) >= 2


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
