"""Optional preprocessing filters for vectorization."""

from __future__ import annotations

import cv2
import numpy as np

from fikzpy.core.vectorization_config import PreprocessingConfig


def preprocess_gray(gray: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    """Apply optional denoising filters to a grayscale image."""
    config = config or PreprocessingConfig()
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    result = gray.astype(np.uint8, copy=True)

    if config.use_bilateral_filter:
        result = cv2.bilateralFilter(
            result,
            max(1, int(config.bilateral_diameter)),
            float(config.bilateral_sigma_color),
            float(config.bilateral_sigma_space),
        )

    kernel = _odd_kernel(config.gaussian_kernel)
    if kernel > 1:
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)

    return result


def threshold_ink_mask(gray: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    """Create a binary ink mask with optional adaptive thresholding."""
    config = config or PreprocessingConfig()
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    if config.use_adaptive_threshold:
        block_size = _odd_kernel(max(3, config.adaptive_block_size))
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            int(config.adaptive_offset),
        )

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask


def apply_mask_morphology(mask: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    """Apply optional morphology to a binary ink mask."""
    config = config or PreprocessingConfig()
    operation = config.morphology.strip().lower()
    if operation in {"", "none"}:
        return mask.copy()

    kernel_size = max(1, int(config.morphology_kernel))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    operations = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
    }
    if operation not in operations:
        raise ValueError(f"Unsupported morphology operation: {config.morphology}")

    return cv2.morphologyEx(mask, operations[operation], kernel)


def _odd_kernel(value: int) -> int:
    size = max(0, int(value))
    if size == 0:
        return 0
    return size if size % 2 == 1 else size + 1
