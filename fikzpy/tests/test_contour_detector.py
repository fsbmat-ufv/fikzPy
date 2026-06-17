from __future__ import annotations

import cv2
import numpy as np

from fikzpy.core.contour_detector import detect_contours_from_edges, simplify_contour
from fikzpy.core.image_processor import ProcessingSettings, process_image


def test_detect_contours_from_edges_finds_rectangle() -> None:
    edges = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(edges, (20, 20), (60, 60), 255, 1)

    contours = detect_contours_from_edges(edges, simplify_epsilon=0.02, min_area=1)

    assert contours
    assert contours[0].is_drawable
    assert len(contours[0].points) >= 4


def test_simplify_contour_reduces_noisy_polyline() -> None:
    points = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [2, 2], [1, 2], [0, 2]],
        dtype=np.float32,
    )

    simplified = simplify_contour(points, epsilon_ratio=0.1)

    assert len(simplified) < len(points)
    assert simplified.shape[1] == 2


def test_process_image_returns_overlay_and_reconstruction() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.circle(image, (32, 32), 18, (0, 0, 0), 2)

    result = process_image(
        image,
        ProcessingSettings(
            smoothing=3,
            canny_low=30,
            canny_high=120,
            simplify_epsilon=0.02,
            min_contour_area=1,
        ),
    )

    assert result.edges.shape == image.shape[:2]
    assert result.overlay_bgr.shape == image.shape
    assert result.reconstruction_bgr.shape == image.shape
    assert result.contours


def test_process_image_line_art_alias_matches_classic_mode() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 30), (54, 30), (0, 0, 0), 2)

    classic = process_image(image, ProcessingSettings(vectorization_mode="classic"))
    alias = process_image(image, ProcessingSettings(vectorization_mode="line_art"))

    assert len(classic.contours) == len(alias.contours)
    assert sum(len(contour.points) for contour in classic.contours) == sum(
        len(contour.points) for contour in alias.contours
    )
