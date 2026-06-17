from __future__ import annotations

import cv2
import numpy as np

from fikzpy.core.preprocessing import apply_mask_morphology, preprocess_gray, threshold_ink_mask
from fikzpy.core.vectorization_config import PreprocessingConfig


def test_preprocess_gray_default_is_identity_copy() -> None:
    gray = np.full((12, 12), 128, dtype=np.uint8)

    result = preprocess_gray(gray)

    assert np.array_equal(result, gray)
    assert result is not gray


def test_threshold_ink_mask_detects_dark_line() -> None:
    gray = np.full((40, 40), 255, dtype=np.uint8)
    cv2.line(gray, (5, 20), (35, 20), 0, 2)

    mask = threshold_ink_mask(gray)

    assert mask[20, 20] == 255
    assert mask[0, 0] == 0


def test_apply_mask_morphology_can_close_small_gap() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 5:9] = 255
    mask[10, 10:14] = 255

    closed = apply_mask_morphology(mask, PreprocessingConfig(morphology="close", morphology_kernel=3))

    assert closed[10, 9] == 255
