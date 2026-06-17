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
FloatPixel = tuple[float, float]


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
    snap_junction_endpoints: bool = False
    recover_blackhat_strokes: bool = False
    blackhat_kernel_size: int = 9
    blackhat_threshold: int = 12
    blackhat_min_gray: int = 80
    blackhat_max_gray: int = 250
    denoise_method: str = "median"
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    use_adaptive_threshold: bool = False
    adaptive_min_delta: int = 2
    adaptive_max_gray: int = 252
    close_gaps: bool = False
    closing_kernel_size: int = 3
    skeleton_method: str = "zhang-suen"
    multiscale_skeleton: bool = False
    multiscale_closing_kernel_size: int = 3


def extract_ink_mask(gray: np.ndarray, settings: StrokeTracingSettings | None = None) -> np.ndarray:
    """Return a binary mask where dark ink pixels are 255."""
    settings = settings or StrokeTracingSettings()
    if gray.ndim != 2:
        raise ValueError("Line-art extraction expects a grayscale image.")

    denoised = _denoise_gray(gray, settings)
    threshold_source = _apply_clahe(denoised, settings) if settings.use_clahe else denoised
    otsu_threshold, _ = cv2.threshold(threshold_source, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshold = _line_art_threshold(threshold_source, otsu_threshold, settings)

    if settings.use_adaptive_threshold:
        mask = _adaptive_ink_mask(threshold_source, denoised, threshold, settings)
    elif threshold < 245:
        _, mask = cv2.threshold(threshold_source, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        block_size = _odd_at_least(settings.threshold_block_size, 3)
        mask = cv2.adaptiveThreshold(
            threshold_source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            settings.threshold_offset,
        )

    if settings.recover_faint_strokes:
        mask = _recover_faint_strokes(denoised, mask, settings)
    if settings.recover_blackhat_strokes:
        mask = _recover_blackhat_strokes(denoised, mask, settings)
    if settings.close_gaps:
        mask = _close_mask_gaps(mask, settings.closing_kernel_size)

    return _remove_small_components(mask, settings.min_component_area)


def skeletonize(binary_mask: np.ndarray, *, method: str = "zhang-suen") -> np.ndarray:
    """Thin a binary mask with skimage when available, otherwise Zhang-Suen."""
    normalized = method.strip().lower()
    if normalized in {"skimage", "auto"}:
        skimage_result = _skeletonize_with_skimage(binary_mask)
        if skimage_result is not None:
            return skimage_result

    return _skeletonize_zhang_suen(binary_mask)


def skeletonize_multiscale(
    binary_mask: np.ndarray,
    *,
    method: str = "zhang-suen",
    closing_kernel_size: int = 3,
) -> np.ndarray:
    """Skeletonize the original and lightly closed masks, then thin the union."""
    base = skeletonize(binary_mask, method=method)
    closed = _close_mask_gaps(binary_mask, closing_kernel_size)
    if np.array_equal(closed, binary_mask):
        return base

    closed_skeleton = skeletonize(closed, method=method)
    combined = cv2.bitwise_or(base, closed_skeleton)
    return skeletonize(combined, method=method)


def _skeletonize_zhang_suen(binary_mask: np.ndarray) -> np.ndarray:
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
    snap_junction_endpoints: bool = False,
) -> list[Contour]:
    """Trace skeleton pixels into drawable open and closed strokes."""
    pixels = _skeleton_pixels(skeleton)
    if not pixels:
        return []

    neighbors = {pixel: [item for item in _neighbor_pixels(pixel) if item in pixels] for pixel in pixels}
    junction_centers = _junction_centers(neighbors) if snap_junction_endpoints else {}
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

        points = _path_to_xy_points(path, junction_centers)
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
    if settings.multiscale_skeleton:
        skeleton = skeletonize_multiscale(
            ink_mask,
            method=settings.skeleton_method,
            closing_kernel_size=settings.multiscale_closing_kernel_size,
        )
    else:
        skeleton = skeletonize(ink_mask, method=settings.skeleton_method)
    contours = trace_strokes_from_skeleton(
        skeleton,
        simplify_epsilon=simplify_epsilon,
        min_path_length=settings.min_path_length,
        smooth_iterations=settings.smooth_iterations,
        snap_junction_endpoints=settings.snap_junction_endpoints,
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


def _denoise_gray(gray: np.ndarray, settings: StrokeTracingSettings) -> np.ndarray:
    """Apply a small denoising pass before thresholding."""
    method = settings.denoise_method.strip().lower()
    if method in {"", "none"}:
        return gray.astype(np.uint8, copy=True)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, 5, 35.0, 35.0)
    if method == "nlmeans":
        return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.medianBlur(gray, 3)


def _apply_clahe(gray: np.ndarray, settings: StrokeTracingSettings) -> np.ndarray:
    """Enhance local contrast so faint pencil-like strokes survive thresholding."""
    tile_size = max(2, int(settings.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(settings.clahe_clip_limit)),
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def _adaptive_ink_mask(
    threshold_source: np.ndarray,
    gray: np.ndarray,
    global_threshold: float,
    settings: StrokeTracingSettings,
) -> np.ndarray:
    """Combine Gaussian adaptive thresholding with conservative local-contrast gates."""
    block_size = _odd_at_least(settings.threshold_block_size, 3)
    adaptive = cv2.adaptiveThreshold(
        threshold_source,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(settings.threshold_offset),
    )
    _, global_mask = cv2.threshold(threshold_source, global_threshold, 255, cv2.THRESH_BINARY_INV)

    background = cv2.GaussianBlur(gray, (block_size, block_size), 0)
    contrast = background.astype(np.int16) - gray.astype(np.int16)
    plausible_ink = (gray <= int(settings.adaptive_max_gray)) | (contrast >= int(settings.adaptive_min_delta))
    adaptive = cv2.bitwise_and(adaptive, (plausible_ink.astype(np.uint8) * 255))
    return cv2.bitwise_or(global_mask, adaptive)


def _close_mask_gaps(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Reconnect one- or two-pixel gaps in a binary ink mask."""
    size = _odd_at_least(kernel_size, 1)
    if size <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _skeletonize_with_skimage(binary_mask: np.ndarray) -> np.ndarray | None:
    """Return a skimage skeleton when scikit-image is installed."""
    try:
        from skimage.morphology import skeletonize as skimage_skeletonize
    except ImportError:
        return None

    skeleton = skimage_skeletonize(binary_mask > 0)
    return (skeleton.astype(np.uint8) * 255)


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


def _recover_blackhat_strokes(
    gray: np.ndarray,
    base_mask: np.ndarray,
    settings: StrokeTracingSettings,
) -> np.ndarray:
    """Recover weak dark strokes with a conservative morphological black-hat pass."""
    kernel_size = _odd_at_least(settings.blackhat_kernel_size, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    candidates = (
        (blackhat >= int(settings.blackhat_threshold))
        & (gray >= int(settings.blackhat_min_gray))
        & (gray <= int(settings.blackhat_max_gray))
    )
    blackhat_mask = (candidates.astype(np.uint8) * 255)
    return cv2.bitwise_or(base_mask, blackhat_mask)


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


def _junction_centers(neighbors: dict[Pixel, list[Pixel]]) -> dict[Pixel, FloatPixel]:
    """Return a center point for each connected cluster of skeleton junction pixels."""
    junction_pixels = {pixel for pixel, items in neighbors.items() if len(items) >= 3}
    centers: dict[Pixel, FloatPixel] = {}
    visited: set[Pixel] = set()

    for pixel in junction_pixels:
        if pixel in visited:
            continue
        stack = [pixel]
        component: list[Pixel] = []
        visited.add(pixel)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in _neighbor_pixels(current):
                if neighbor in junction_pixels and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        center_y = float(sum(item[0] for item in component) / len(component))
        center_x = float(sum(item[1] for item in component) / len(component))
        for item in component:
            centers[item] = (center_y, center_x)

    return centers


def _path_to_xy_points(path: list[Pixel], junction_centers: dict[Pixel, FloatPixel]) -> np.ndarray:
    """Convert a skeleton path to x/y points and snap endpoints to junction centers."""
    points = np.array([[float(x), float(y)] for y, x in path], dtype=np.float64)
    if len(points) == 0 or not junction_centers:
        return points

    start_center = junction_centers.get(path[0])
    if start_center is not None:
        center_y, center_x = start_center
        points[0] = (center_x, center_y)

    end_center = junction_centers.get(path[-1])
    if end_center is not None:
        center_y, center_x = end_center
        points[-1] = (center_x, center_y)

    return points


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
