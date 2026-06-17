from __future__ import annotations

import cv2
import numpy as np

from fikzpy.core.image_processor import ProcessingSettings, process_image, visual_settings_from_processing
from fikzpy.core.visual_pipeline import VisualTracingSettings, trace_visual_contours


def test_visual_tracing_ignores_red_annotations() -> None:
    image = np.full((40, 60, 3), 255, dtype=np.uint8)
    cv2.line(image, (5, 20), (55, 20), (0, 0, 0), 2)
    cv2.line(image, (5, 10), (55, 10), (0, 0, 255), 2)

    result = trace_visual_contours(
        image,
        VisualTracingSettings(
            upsample_factor=2,
            close_kernel_size=1,
            min_component_area=1.0,
            ignore_chromatic_annotations=True,
        ),
    )

    black_line_pixels = np.count_nonzero(result.ink_mask[18:23, :])
    red_line_pixels = np.count_nonzero(result.ink_mask[8:13, :])

    assert black_line_pixels > 0
    assert red_line_pixels < black_line_pixels * 0.2


def test_visual_tracing_finds_closed_ink_shapes() -> None:
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.circle(image, (25, 25), 12, (0, 0, 0), 2)

    result = trace_visual_contours(
        image,
        VisualTracingSettings(upsample_factor=2, close_kernel_size=2, min_component_area=1.0),
    )

    assert result.contours
    assert all(contour.closed for contour in result.contours)
    assert result.ink_pixels > 0


def test_visual_reconstruction_uses_filled_ink_preview() -> None:
    image = np.full((50, 60, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 12), (40, 30), (0, 0, 0), -1)

    result = process_image(image, ProcessingSettings(vectorization_mode="visual"))

    assert result.reconstruction_bgr[20, 20].tolist() == [0, 0, 0]


def test_visual_settings_are_mapped_from_existing_controls() -> None:
    settings = visual_settings_from_processing(
        ProcessingSettings(
            smoothing=7,
            simplify_epsilon=0.01,
            line_art_threshold=205,
            stroke_smoothing=2,
        )
    )

    assert settings.dark_threshold == 205
    assert settings.denoise_h == 7
    assert settings.close_kernel_size == 5
    assert settings.contour_simplify_px == 0.6
    assert settings.bezier_error_px == 1.2
