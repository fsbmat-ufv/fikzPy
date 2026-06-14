"""Image loading and contour extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fikzpy.core.contour_detector import Contour, detect_contours_from_edges
from fikzpy.core.stroke_tracer import StrokeTracingSettings, trace_line_art_strokes


@dataclass(frozen=True)
class ProcessingSettings:
    """User-adjustable parameters for image to contour conversion."""

    vectorization_mode: str = "line_art"
    smoothing: int = 5
    canny_low: int = 50
    canny_high: int = 150
    simplify_epsilon: float = 0.006
    min_contour_area: float = 8.0
    min_contour_perimeter: float = 8.0
    min_path_length: int = 3


@dataclass(frozen=True)
class ProcessingResult:
    """Intermediate and final products of the image processing pipeline."""

    original_bgr: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    contours: list[Contour]
    overlay_bgr: np.ndarray
    reconstruction_bgr: np.ndarray
    ink_mask: np.ndarray | None = None
    skeleton: np.ndarray | None = None


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR OpenCV array."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image array to grayscale."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim != 3:
        raise ValueError("Expected a grayscale or color image array.")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalized_kernel_size(value: int) -> int:
    size = max(1, int(value))
    return size if size % 2 == 1 else size + 1


def smooth_image(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply light Gaussian smoothing before edge detection."""
    size = _normalized_kernel_size(kernel_size)
    if size <= 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (size, size), 0)


def detect_edges(gray_or_blurred: np.ndarray, low: int, high: int) -> np.ndarray:
    """Detect image edges with Canny thresholds."""
    low_value = max(0, int(low))
    high_value = max(low_value + 1, int(high))
    return cv2.Canny(gray_or_blurred, low_value, high_value)


def make_overlay(
    image_bgr: np.ndarray,
    contours: list[Contour],
    *,
    color_bgr: tuple[int, int, int] = (40, 40, 230),
    thickness: int = 2,
    alpha: float = 0.78,
) -> np.ndarray:
    """Draw contours on top of the original image."""
    overlay = image_bgr.copy()
    _draw_contours(overlay, contours, color_bgr, thickness)
    return cv2.addWeighted(image_bgr, alpha, overlay, 1.0 - alpha, 0)


def make_reconstruction(
    image_shape: tuple[int, ...],
    contours: list[Contour],
    *,
    line_color_bgr: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw detected contours on a white canvas."""
    height, width = image_shape[:2]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_contours(canvas, contours, line_color_bgr, thickness)
    return canvas


def process_image(image: np.ndarray, settings: ProcessingSettings | None = None) -> ProcessingResult:
    """Run the full image processing pipeline."""
    settings = settings or ProcessingSettings()
    gray = to_gray(image)
    blurred = smooth_image(gray, settings.smoothing)

    ink_mask = None
    skeleton = None
    if settings.vectorization_mode == "contours":
        edges = detect_edges(blurred, settings.canny_low, settings.canny_high)
        contours = detect_contours_from_edges(
            edges,
            simplify_epsilon=settings.simplify_epsilon,
            min_area=settings.min_contour_area,
            min_perimeter=settings.min_contour_perimeter,
        )
    elif settings.vectorization_mode == "line_art":
        contours, ink_mask, skeleton = trace_line_art_strokes(
            gray,
            simplify_epsilon=settings.simplify_epsilon,
            settings=StrokeTracingSettings(min_path_length=settings.min_path_length),
        )
        edges = skeleton
    else:
        raise ValueError(f"Unsupported vectorization mode: {settings.vectorization_mode}")

    overlay = make_overlay(image, contours)
    reconstruction = make_reconstruction(image.shape, contours)

    return ProcessingResult(
        original_bgr=image,
        gray=gray,
        blurred=blurred,
        edges=edges,
        contours=contours,
        overlay_bgr=overlay,
        reconstruction_bgr=reconstruction,
        ink_mask=ink_mask,
        skeleton=skeleton,
    )


def process_image_file(path: str | Path, settings: ProcessingSettings | None = None) -> ProcessingResult:
    """Load an image and run the full processing pipeline."""
    return process_image(load_image(path), settings)


def _draw_contours(
    canvas: np.ndarray,
    contours: list[Contour],
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    for contour in contours:
        if not contour.is_drawable:
            continue
        polyline = np.rint(contour.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polyline], contour.closed, color_bgr, thickness, cv2.LINE_AA)
