"""Consolidated single-silhouette extraction for the dot-to-dot mode.

The Visual intermediate SVG of a real figure contains many paths, filled
ribbons and white cutouts. Picking the largest isolated subpath is not enough
to represent "the figure": instead, this module rasterizes a copy of the
parsed SVG geometry into a monochrome mask, merges the visible elements with a
morphological closing proportional to the image size, keeps the main connected
region, and extracts its single external contour. The result is one closed
:class:`SubPath` that the pipeline replaces with numbered points.

Everything here works on the dot-to-dot copy of the geometry; the SVG used by
the Visual mode is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from fikzpy.core.bezier_fit import fit_cubic_beziers
from fikzpy.core.dot_to_dot.models import CubicSegment, LineSegment, Segment, SubPath, SvgPath, Vec2
from fikzpy.core.vector_objects import BezierCurve, Line, Point, Polyline

_EPS = 1e-9
_TARGET_RENDER_DIM = 1200
_CLOSING_RATIO = 0.02
_EROSION_RADIUS_RATIO = 0.0125
_SMOOTH_EPSILON_RATIO = 0.0018
_MIN_DETAIL_LENGTH_RATIO = 0.01
_DETAIL_INSIDE_FRACTION = 0.85


class SilhouetteError(RuntimeError):
    """Raised when no usable silhouette can be extracted from the figure."""


@dataclass(frozen=True)
class SilhouetteResult:
    """A consolidated external silhouette plus classified internal details.

    ``internal_details`` groups the interior subpaths by their original
    ``<path>`` element so even-odd parity (holes) survives later ink
    rendering.
    """

    outline: SubPath
    internal_details: tuple[tuple[SubPath, ...], ...]
    render_scale: float
    closing_kernel: int


def _is_background_frame(subpath: SubPath, canvas_width: float, canvas_height: float) -> bool:
    """Return whether a subpath is a page/canvas frame rather than the figure."""
    if canvas_width <= _EPS or canvas_height <= _EPS:
        return False
    min_x, min_y, max_x, max_y = subpath.bounding_box()
    box_width = max_x - min_x
    box_height = max_y - min_y
    covers_canvas = box_width >= 0.98 * canvas_width and box_height >= 0.98 * canvas_height
    box_area = box_width * box_height
    fills_box = box_area > _EPS and subpath.area() >= 0.9 * box_area
    return covers_canvas and fills_box


def _odd(value: int) -> int:
    size = max(3, int(value))
    return size if size % 2 == 1 else size + 1


def _render_ink_mask(
    paths: list[SvgPath],
    canvas_width: float,
    canvas_height: float,
    scale: float,
) -> np.ndarray:
    """Union-fill every visible subpath into a monochrome raster mask.

    Union filling (instead of even-odd) intentionally fills white cutouts:
    only the *external* contour of the merged figure matters here.
    """
    width = max(2, int(round(canvas_width * scale)))
    height = max(2, int(round(canvas_height * scale)))
    mask = np.zeros((height, width), np.uint8)
    stroke = max(1, int(round(scale)))
    for path in paths:
        for subpath in path.subpaths:
            if _is_background_frame(subpath, canvas_width, canvas_height):
                continue
            points = np.array(
                [[point.x * scale, point.y * scale] for point in subpath.polyline()],
                np.int32,
            ).reshape(-1, 1, 2)
            if len(points) < 2:
                continue
            if subpath.closed and len(points) >= 3:
                cv2.fillPoly(mask, [points], 255)
            else:
                # Open strokes have no fill; give them a visible thickness.
                cv2.polylines(mask, [points], False, 255, thickness=stroke)
    return mask


def _polyline_orientation_area(points: np.ndarray) -> float:
    """Shoelace signed area in image coordinates (positive = visually clockwise)."""
    x = points[:, 0]
    y = points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2.0)


def _subpath_from_polyline(points: np.ndarray, *, closed: bool) -> SubPath:
    segments = [
        LineSegment(Vec2(float(points[i][0]), float(points[i][1])),
                    Vec2(float(points[i + 1][0]), float(points[i + 1][1])))
        for i in range(len(points) - 1)
    ]
    if closed:
        first = Vec2(float(points[0][0]), float(points[0][1]))
        last = Vec2(float(points[-1][0]), float(points[-1][1]))
        if last.distance_to(first) > _EPS:
            segments.append(LineSegment(last, first))
    if not segments:
        raise SilhouetteError("Silhouette contour is degenerate.")
    return SubPath(segments, closed=closed)


def extract_silhouette(
    paths: list[SvgPath],
    canvas_width: float,
    canvas_height: float,
    *,
    collect_internal_details: bool = False,
) -> SilhouetteResult:
    """Extract the single consolidated external silhouette of the figure.

    The returned outline is a closed subpath in SVG coordinates, oriented so
    that forward traversal is visually clockwise.
    """
    if not paths:
        raise SilhouetteError("No SVG paths available for silhouette extraction.")
    max_dim = max(canvas_width, canvas_height, 1.0)
    scale = max(1.0, _TARGET_RENDER_DIM / max_dim)

    mask = _render_ink_mask(paths, canvas_width, canvas_height, scale)
    if int(np.count_nonzero(mask)) == 0:
        raise SilhouetteError("The figure rendered to an empty mask.")

    kernel_size = _odd(int(round(_CLOSING_RATIO * max_dim * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    if count <= 1:
        raise SilhouetteError("No connected figure region was found.")
    main_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_mask = np.where(labels == main_label, np.uint8(255), np.uint8(0))

    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SilhouetteError("The main region has no external contour.")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, _SMOOTH_EPSILON_RATIO * perimeter)
    smoothed = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)
    if len(smoothed) < 3:
        raise SilhouetteError("The silhouette contour is too small.")

    # Enforce visually clockwise traversal (positive shoelace in image coords).
    if _polyline_orientation_area(smoothed) < 0.0:
        smoothed = smoothed[::-1]

    outline = _subpath_from_polyline(smoothed / scale, closed=True)

    details: tuple[SubPath, ...] = ()
    if collect_internal_details:
        details = _internal_details(paths, main_mask, canvas_width, canvas_height, scale)

    return SilhouetteResult(
        outline=outline,
        internal_details=details,
        render_scale=scale,
        closing_kernel=kernel_size,
    )


def _internal_details(
    paths: list[SvgPath],
    main_mask: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    scale: float,
) -> tuple[tuple[SubPath, ...], ...]:
    """Return the original subpaths strictly inside the silhouette, per path.

    Boundary ribbons (the strokes forming the outer line itself) sit on the
    silhouette edge, so eroding the region by a size-proportional radius keeps
    only genuinely interior details (eyes, mouth, spots) and never restores
    the replaced external outline.
    """
    max_dim = max(canvas_width, canvas_height, 1.0)
    # Kernel size is a diameter: erosion depth equals the radius below.
    radius = max(1, int(round(_EROSION_RADIUS_RATIO * max_dim * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    interior = cv2.erode(main_mask, kernel)
    height, width = interior.shape
    min_length = _MIN_DETAIL_LENGTH_RATIO * max_dim

    groups: list[tuple[SubPath, ...]] = []
    for path in paths:
        details: list[SubPath] = []
        for subpath in path.subpaths:
            if _is_background_frame(subpath, canvas_width, canvas_height):
                continue
            if subpath.total_length < min_length:
                continue
            points = subpath.polyline()
            inside = 0
            for point in points:
                px = min(max(int(round(point.x * scale)), 0), width - 1)
                py = min(max(int(round(point.y * scale)), 0), height - 1)
                if interior[py, px]:
                    inside += 1
            if inside >= _DETAIL_INSIDE_FRACTION * len(points):
                details.append(subpath)
        if details:
            groups.append(tuple(details))
    return tuple(groups)


_INK_MIN_CONTOUR_AREA_PX = 4.0
_INK_SIMPLIFY_PX = 0.35
_INK_BEZIER_ERROR_PX = 0.7
_ERASE_MIN_RADIUS_RATIO = 0.008
_ERASE_MAX_RADIUS_RATIO = 0.03


def estimate_stroke_width(mask: np.ndarray) -> float:
    """Estimate the typical ink stroke width (in mask pixels).

    For a ribbon of width ``w`` the interior distance transform ranges over
    ``0..w/2`` with median around ``w/4``, so four times the median recovers
    the stroke width.
    """
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    inked = distances[mask > 0]
    if inked.size == 0:
        return 1.0
    return max(1.0, 4.0 * float(np.median(inked)))


def retrace_ink_regions(
    groups: Sequence[Sequence[SubPath]],
    canvas_width: float,
    canvas_height: float,
    erase_polylines: Sequence[Sequence[Vec2]] = (),
) -> list[SubPath]:
    """Re-derive the remaining *filled ink* after erasing the selected arcs.

    The Visual intermediate SVG encodes ink as even-odd filled ribbons, so
    stroking its subpaths produces contour-like double lines. Instead, each
    group (one original ``<path>``) is rasterized faithfully with even-odd
    fill, the corridors around the selected arcs are erased with a width
    proportional to the estimated stroke width, and the surviving ink is
    re-vectorized into smooth closed subpaths ready for even-odd filling in
    TikZ. The output therefore looks like the Visual mode with the selected
    trechos genuinely absent — nothing is hidden under white bands.
    """
    max_dim = max(canvas_width, canvas_height, 1.0)
    scale = max(1.0, _TARGET_RENDER_DIM / max_dim)
    width = max(2, int(round(canvas_width * scale)))
    height = max(2, int(round(canvas_height * scale)))
    mask = np.zeros((height, width), np.uint8)
    stroke = max(1, int(round(scale)))

    for group in groups:
        polygons = []
        for subpath in group:
            if _is_background_frame(subpath, canvas_width, canvas_height):
                continue
            points = np.array(
                [[point.x * scale, point.y * scale] for point in subpath.polyline()],
                np.int32,
            ).reshape(-1, 1, 2)
            if len(points) < 2:
                continue
            if subpath.closed and len(points) >= 3:
                polygons.append(points)
            else:
                cv2.polylines(mask, [points], False, 255, thickness=stroke)
        if polygons:
            # fillPoly applies the even-odd rule across the polygons, so the
            # white cutouts of each original path are preserved as holes.
            group_mask = np.zeros_like(mask)
            cv2.fillPoly(group_mask, polygons, 255)
            mask |= group_mask

    if erase_polylines and int(np.count_nonzero(mask)) > 0:
        stroke_width = estimate_stroke_width(mask)
        radius = 1.5 * stroke_width
        radius = min(max(radius, _ERASE_MIN_RADIUS_RATIO * max_dim * scale),
                     _ERASE_MAX_RADIUS_RATIO * max_dim * scale)
        thickness = max(1, int(round(2.0 * radius)))
        for polyline in erase_polylines:
            points = np.array(
                [[point.x * scale, point.y * scale] for point in polyline],
                np.int32,
            ).reshape(-1, 1, 2)
            if len(points) >= 2:
                cv2.polylines(mask, [points], False, 0, thickness=thickness)
            elif len(points) == 1:
                cv2.circle(mask, tuple(points[0][0]), int(round(radius)), 0, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_area = _INK_MIN_CONTOUR_AREA_PX * scale * scale
    subpaths: list[SubPath] = []
    for contour in contours:
        if abs(cv2.contourArea(contour)) < min_area:
            continue
        epsilon = max(0.05, _INK_SIMPLIFY_PX * scale)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)
        if len(approx) < 3:
            continue
        subpath = _fit_ink_subpath(approx / scale, error_px=_INK_BEZIER_ERROR_PX)
        if subpath is not None:
            subpaths.append(subpath)
    return subpaths


def _fit_ink_subpath(points: np.ndarray, *, error_px: float) -> SubPath | None:
    """Fit a smooth closed subpath through re-traced ink contour points."""
    vertices = tuple(Point(float(x), float(y)) for x, y in points)
    primitives = fit_cubic_beziers(
        vertices,
        error_tolerance=max(error_px, 0.05),
        closed=True,
        min_bezier_length=0.9,
        min_points_for_bezier=6,
        straightness_tolerance=0.35,
        control_point_epsilon=1e-4,
        max_depth=18,
    )
    segments: list[Segment] = []
    if primitives:
        current: Point | None = None
        for primitive in primitives:
            start = _primitive_start(primitive)
            if start is None:
                continue
            if current is not None and current.distance_to(start) > 1e-6:
                segments.append(LineSegment(_vec(current), _vec(start)))
            if isinstance(primitive, BezierCurve):
                segments.append(
                    CubicSegment(
                        _vec(primitive.start),
                        _vec(primitive.control1),
                        _vec(primitive.control2),
                        _vec(primitive.end),
                    )
                )
                current = primitive.end
            elif isinstance(primitive, Line):
                segments.append(LineSegment(_vec(primitive.start), _vec(primitive.end)))
                current = primitive.end
            elif isinstance(primitive, Polyline):
                for index in range(len(primitive.points) - 1):
                    segments.append(
                        LineSegment(_vec(primitive.points[index]), _vec(primitive.points[index + 1]))
                    )
                current = primitive.points[-1]
    if not segments:
        # Fall back to the raw polygon.
        for index in range(len(points) - 1):
            segments.append(
                LineSegment(
                    Vec2(float(points[index][0]), float(points[index][1])),
                    Vec2(float(points[index + 1][0]), float(points[index + 1][1])),
                )
            )
    if not segments:
        return None
    return SubPath(segments, closed=True)


def _primitive_start(primitive) -> Point | None:
    if isinstance(primitive, (Line, BezierCurve)):
        return primitive.start
    if isinstance(primitive, Polyline):
        return primitive.points[0]
    return None


def _vec(point: Point) -> Vec2:
    return Vec2(float(point.x), float(point.y))


__all__ = [
    "SilhouetteError",
    "SilhouetteResult",
    "estimate_stroke_width",
    "extract_silhouette",
    "retrace_ink_regions",
]
