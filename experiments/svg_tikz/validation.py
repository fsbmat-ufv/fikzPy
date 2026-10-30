"""Independent SVG rendering and finite-resolution comparison for the experiment."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def viewport(svg: str) -> tuple[float, float]:
    """Return the viewBox extent in SVG user units, requiring an explicit frame."""
    values = ET.fromstring(svg).get("viewBox", "").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError("The experiment requires an explicit finite viewBox.")
    x, y, width, height = map(float, values)
    if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Invalid viewBox.")
    return width, height


def render_svg(svg: str, scale: float = 4.0) -> np.ndarray:
    """Render with QtSvg, independently of semantic geometry and TikZ export."""
    width, height = viewport(svg)
    w, h = int(np.ceil(width * scale)), int(np.ceil(height * scale))
    if w * h > 40_000_000:
        raise ValueError("SVG validation canvas exceeds the experiment's pixel limit.")
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    if not renderer.isValid():
        raise ValueError("QtSvg rejected the SVG.")
    canvas = QImage(w, h, QImage.Format.Format_RGBA8888)
    canvas.fill(0)
    painter = QPainter(canvas)
    try:
        renderer.render(painter, QRectF(0, 0, w, h))
    finally:
        painter.end()
    return np.frombuffer(canvas.bits(), np.uint8).reshape(h, canvas.bytesPerLine())[:, :w * 4].reshape(h, w, 4).copy()


def topology(mask: np.ndarray) -> dict[str, int]:
    """Count foreground components and enclosed background components (8/4 connectivity)."""
    binary = mask.astype(np.uint8)
    components = cv2.connectedComponents(binary, connectivity=8)[0] - 1
    background = np.pad(1 - binary, 1, constant_values=1)
    holes = cv2.connectedComponents(background, connectivity=4)[0] - 2
    return {"components": int(components), "holes": int(max(holes, 0))}


def white_rgb(rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA image on white without losing alpha during validation."""
    alpha = rgba[:, :, 3:4].astype(float) / 255
    return np.rint(rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def compare_images(before: np.ndarray, after: np.ndarray, *, rgba: bool = False) -> dict:
    """Measure image error, edge displacement and topology without registration tricks."""
    if before.shape != after.shape:
        raise ValueError(f"Comparison frames differ: {before.shape} != {after.shape}")
    a, b = (white_rgb(before), white_rgb(after)) if rgba else (before, after)
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    masks = (gray_a < 245, gray_b < 245)
    union = np.count_nonzero(masks[0] | masks[1])
    edges = [cv2.Canny(gray, 80, 160) > 0 for gray in (gray_a, gray_b)]
    distances = []
    for source, target in (edges, edges[::-1]):
        if source.any() and target.any():
            distance = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distances.extend(distance[source].tolist())
        elif source.any():
            distances.append(float(np.hypot(*source.shape)))
    result = {
        "foreground_iou": float(np.count_nonzero(masks[0] & masks[1]) / union) if union else 1.0,
        "rgb_mae_255": float(np.abs(a.astype(float) - b.astype(float)).mean()),
        "edge_p95_pixels": float(np.percentile(distances, 95)) if distances else 0.0,
        "edge_max_pixels": float(max(distances, default=0.0)),
        "before_topology": topology(masks[0]),
        "after_topology": topology(masks[1]),
    }
    result["topology_equal"] = result["before_topology"] == result["after_topology"]
    if rgba:
        result["alpha_topology_equal"] = topology(before[:, :, 3] >= 128) == topology(after[:, :, 3] >= 128)
    return result


def passes_visual_gate(metrics: dict, minimum_iou: float = 0.99, maximum_mae: float = 0.5) -> bool:
    """Apply fidelity constraints before considering code size."""
    return bool(metrics["topology_equal"] and metrics.get("alpha_topology_equal", True)
                and metrics["foreground_iou"] >= minimum_iou and metrics["rgb_mae_255"] <= maximum_mae)
