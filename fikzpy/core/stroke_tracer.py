"""Trace line-art strokes from a binary ink image."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from fikzpy.core.contour_detector import Contour, simplify_polyline
from fikzpy.core.preprocessing import apply_mask_morphology, preprocess_gray
from fikzpy.core.vectorization_config import PreprocessingConfig


Pixel = tuple[int, int]


@dataclass(frozen=True)
class StrokeTracingSettings:
    """Parameters for line-art stroke extraction."""

    threshold_block_size: int = 35
    threshold_offset: int = 9
    dark_threshold: int = 215
    background_margin: int = 12
    min_component_area: int = 8
    min_path_length: int = 3
    smooth_iterations: int = 1
    recover_faint_strokes: bool = False
    faint_stroke_block_size: int = 31
    faint_stroke_min_delta: int = 7
    faint_stroke_max_gray: int = 246


def extract_ink_mask(gray: np.ndarray, settings: StrokeTracingSettings | None = None) -> np.ndarray:
    """Return a binary mask where dark ink pixels are 255."""
    settings = settings or StrokeTracingSettings()
    if gray.ndim != 2:
        raise ValueError("Line-art extraction expects a grayscale image.")

    denoised = cv2.medianBlur(gray, 3)
    otsu_threshold, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshold = _line_art_threshold(denoised, otsu_threshold, settings)

    if threshold < 245:
        _, mask = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        block_size = _odd_at_least(settings.threshold_block_size, 3)
        mask = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            settings.threshold_offset,
        )

    if settings.recover_faint_strokes:
        mask = _recover_faint_strokes(denoised, mask, settings)

    return _remove_small_components(mask, settings.min_component_area)


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask with the Zhang-Suen algorithm."""
    image = (binary_mask > 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            to_remove = _zhang_suen_candidates(image, step)
            if np.any(to_remove):
                image[to_remove] = 0
                changed = True

    return (image * 255).astype(np.uint8)


def trace_strokes_from_skeleton(
    skeleton: np.ndarray,
    *,
    simplify_epsilon: float = 0.01,
    min_path_length: int = 8,
    smooth_iterations: int = 1,
) -> list[Contour]:
    """Trace skeleton pixels into drawable open and closed strokes."""
    pixels = _skeleton_pixels(skeleton)
    if not pixels:
        return []

    neighbors = {pixel: [item for item in _neighbor_pixels(pixel) if item in pixels] for pixel in pixels}
    visited_edges: set[frozenset[Pixel]] = set()
    paths: list[tuple[list[Pixel], bool]] = []

    start_pixels = [pixel for pixel, items in neighbors.items() if len(items) != 2]
    for start in start_pixels:
        for neighbor in neighbors[start]:
            edge = frozenset((start, neighbor))
            if edge in visited_edges:
                continue
            path = _walk_path(start, neighbor, neighbors, visited_edges)
            paths.append((path, False))

    for pixel in pixels:
        for neighbor in neighbors[pixel]:
            edge = frozenset((pixel, neighbor))
            if edge in visited_edges:
                continue
            path = _walk_loop(pixel, neighbor, neighbors, visited_edges)
            paths.append((path, True))

    contours: list[Contour] = []
    for path, closed in paths:
        if len(path) < min_path_length:
            continue

        points = np.array([[x, y] for y, x in path], dtype=np.float64)
        points = _smooth_polyline(points, closed=closed, iterations=smooth_iterations)
        simplified = simplify_polyline(points, epsilon_ratio=simplify_epsilon, closed=closed)
        if len(simplified) < 2:
            continue

        perimeter = _path_length(simplified, closed=closed)
        contours.append(Contour(points=simplified, closed=closed, area=0.0, perimeter=perimeter))

    contours.sort(key=lambda item: item.perimeter, reverse=True)
    return contours


def trace_line_art_strokes(
    gray: np.ndarray,
    *,
    simplify_epsilon: float = 0.01,
    settings: StrokeTracingSettings | None = None,
    preprocessing: PreprocessingConfig | None = None,
) -> tuple[list[Contour], np.ndarray, np.ndarray]:
    """Extract line-art strokes and return contours, ink mask, and skeleton."""
    settings = settings or StrokeTracingSettings()
    source = preprocess_gray(gray, preprocessing) if preprocessing is not None else gray
    ink_mask = extract_ink_mask(source, settings)
    if preprocessing is not None:
        ink_mask = apply_mask_morphology(ink_mask, preprocessing)
    skeleton = skeletonize(ink_mask)
    contours = trace_strokes_from_skeleton(
        skeleton,
        simplify_epsilon=simplify_epsilon,
        min_path_length=settings.min_path_length,
        smooth_iterations=settings.smooth_iterations,
    )
    return contours, ink_mask, skeleton


def _line_art_threshold(
    denoised: np.ndarray,
    otsu_threshold: float,
    settings: StrokeTracingSettings,
) -> float:
    """Choose a threshold that keeps faint ink on bright backgrounds."""
    background = float(np.percentile(denoised, 95))
    threshold = float(otsu_threshold)

    if background >= 230:
        threshold = max(threshold, float(settings.dark_threshold))
        threshold = min(threshold, background - float(settings.background_margin))

    return max(0.0, min(255.0, threshold))


def _recover_faint_strokes(
    gray: np.ndarray,
    base_mask: np.ndarray,
    settings: StrokeTracingSettings,
) -> np.ndarray:
    """Add faint local-contrast strokes without lowering the global threshold."""
    block_size = _odd_at_least(settings.faint_stroke_block_size, 3)
    background = cv2.GaussianBlur(gray, (block_size, block_size), 0)
    contrast = background.astype(np.int16) - gray.astype(np.int16)

    candidates = (
        (contrast >= int(settings.faint_stroke_min_delta))
        & (gray <= int(settings.faint_stroke_max_gray))
        & (background >= int(settings.faint_stroke_max_gray))
    )
    faint_mask = (candidates.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    faint_mask = cv2.morphologyEx(faint_mask, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_or(base_mask, faint_mask)


def _smooth_polyline(points: np.ndarray, *, closed: bool, iterations: int) -> np.ndarray:
    """Reduce pixel stair-stepping while preserving endpoints of open strokes."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4 or iterations <= 0:
        return pts

    for _ in range(iterations):
        if closed:
            previous_points = np.roll(pts, 1, axis=0)
            next_points = np.roll(pts, -1, axis=0)
            pts = (previous_points + 2.0 * pts + next_points) / 4.0
        else:
            smoothed = pts.copy()
            smoothed[1:-1] = (pts[:-2] + 2.0 * pts[1:-1] + pts[2:]) / 4.0
            pts = smoothed

    return pts


def _zhang_suen_candidates(image: np.ndarray, step: int) -> np.ndarray:
    padded = np.pad(image, 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]

    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
    neighbor_count = sum(neighbors)
    transitions = sum((neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1) for index in range(8))

    if step == 0:
        condition_a = (p2 * p4 * p6) == 0
        condition_b = (p4 * p6 * p8) == 0
    else:
        condition_a = (p2 * p4 * p8) == 0
        condition_b = (p2 * p6 * p8) == 0

    return (image == 1) & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & condition_a & condition_b


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _skeleton_pixels(skeleton: np.ndarray) -> set[Pixel]:
    ys, xs = np.nonzero(skeleton > 0)
    return set(zip(ys.tolist(), xs.tolist()))


def _neighbor_pixels(pixel: Pixel) -> Iterable[Pixel]:
    y, x = pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield y + dy, x + dx


def _walk_path(
    start: Pixel,
    next_pixel: Pixel,
    neighbors: dict[Pixel, list[Pixel]],
    visited_edges: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [start]
    previous = start
    current = next_pixel

    while True:
        visited_edges.add(frozenset((previous, current)))
        path.append(current)

        if current != start and len(neighbors[current]) != 2:
            break

        candidates = [item for item in neighbors[current] if item != previous]
        unvisited = [item for item in candidates if frozenset((current, item)) not in visited_edges]
        if not unvisited:
            break

        previous, current = current, unvisited[0]

    return path


def _walk_loop(
    start: Pixel,
    next_pixel: Pixel,
    neighbors: dict[Pixel, list[Pixel]],
    visited_edges: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [start]
    previous = start
    current = next_pixel

    while True:
        visited_edges.add(frozenset((previous, current)))
        path.append(current)

        candidates = [item for item in neighbors[current] if item != previous]
        unvisited = [item for item in candidates if frozenset((current, item)) not in visited_edges]
        if not unvisited:
            break

        next_item = unvisited[0]
        if next_item == start:
            visited_edges.add(frozenset((current, start)))
            break

        previous, current = current, next_item

    return path


def _path_length(points: np.ndarray, *, closed: bool) -> float:
    if len(points) < 2:
        return 0.0

    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if closed and len(points) > 2:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total


def _odd_at_least(value: int, minimum: int) -> int:
    size = max(int(value), minimum)
    return size if size % 2 == 1 else size + 1
