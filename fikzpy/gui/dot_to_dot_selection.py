"""Dot-to-dot segment selection on the main preview image.

Instead of a secondary canvas, this module attaches an event filter to the
main :class:`ImageViewer` while the Ponto a Ponto "Selecionar trechos"
submode is active. Clicks on the displayed image are mapped back to image/SVG
coordinates (accounting for proportional scaling and letterboxing) and fed to
the pure-Python :class:`InteractiveSelectionController`; the current state is
painted as a color overlay on a copy of the preview image. Overlay colors are
interface-only and never reach the exported TikZ.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtWidgets import QLabel

from fikzpy.core.dot_to_dot.config import SelectedSegment
from fikzpy.core.dot_to_dot.segment_selection import (
    InteractiveSelectionController,
    sample_selected_segment,
)
from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths

_PATH_COLOR = (170, 170, 170)  # BGR gray: traced paths available for selection
_CONFIRMED_COLOR = (30, 130, 235)  # BGR orange: confirmed arcs
_PENDING_COLOR = (235, 120, 30)  # BGR blue: arc still being edited
_MARKER_START_COLOR = (60, 175, 60)  # BGR green: first click
_MARKER_END_COLOR = (60, 60, 220)  # BGR red: second click


def widget_to_image_coords(
    click_x: float,
    click_y: float,
    label_size: tuple[int, int],
    pixmap_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Map a click on the viewer label to original-image pixel coordinates.

    The pixmap is scaled proportionally and centered inside the label, so the
    click must be corrected for the letterbox offsets and the scale factor.
    Returns ``None`` when the click lands outside the drawn pixmap.
    """
    label_w, label_h = label_size
    pix_w, pix_h = pixmap_size
    img_w, img_h = image_size
    if pix_w <= 0 or pix_h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    offset_x = (label_w - pix_w) / 2.0
    offset_y = (label_h - pix_h) / 2.0
    local_x = click_x - offset_x
    local_y = click_y - offset_y
    if local_x < 0.0 or local_y < 0.0 or local_x > pix_w or local_y > pix_h:
        return None
    return local_x * (img_w / pix_w), local_y * (img_h / pix_h)


def render_selection_overlay(
    base_bgr: np.ndarray,
    controller: InteractiveSelectionController,
) -> np.ndarray:
    """Paint the traced paths and the selection state over a faded copy."""
    white = np.full_like(base_bgr, 255)
    overlay = cv2.addWeighted(base_bgr, 0.30, white, 0.70, 0.0)

    for path in controller.paths:
        for subpath in path.subpaths:
            points = _polyline_i32(subpath.polyline())
            if len(points) >= 2:
                cv2.polylines(overlay, [points], subpath.closed, _PATH_COLOR, 1, cv2.LINE_AA)

    for segment in controller.model.segments:
        _draw_arc(overlay, controller, segment, _CONFIRMED_COLOR, thickness=2)

    pending = controller.pending_segment()
    if pending is not None:
        _draw_arc(overlay, controller, pending, _PENDING_COLOR, thickness=2)
        subpath = _subpath_for(controller, pending)
        if subpath is not None:
            for sample in sample_selected_segment(subpath, pending):
                center = (int(round(sample.position.x)), int(round(sample.position.y)))
                cv2.circle(overlay, center, 4, _PENDING_COLOR, -1, cv2.LINE_AA)

    if controller.start is not None:
        _draw_marker(overlay, controller.start.point, _MARKER_START_COLOR, square=False)
    if controller.end is not None:
        _draw_marker(overlay, controller.end.point, _MARKER_END_COLOR, square=True)
    return overlay


def _polyline_i32(points) -> np.ndarray:
    return np.array([[p.x, p.y] for p in points], np.int32).reshape(-1, 1, 2)


def _subpath_for(controller: InteractiveSelectionController, segment: SelectedSegment):
    for path in controller.paths:
        if path.path_id == segment.path_id and 0 <= segment.subpath_id < len(path.subpaths):
            return path.subpaths[segment.subpath_id]
    return None


def _draw_arc(
    overlay: np.ndarray,
    controller: InteractiveSelectionController,
    segment: SelectedSegment,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    subpath = _subpath_for(controller, segment)
    if subpath is None:
        return
    dense = replace(segment, point_count=max(48, segment.point_count))
    samples = sample_selected_segment(subpath, dense)
    points = _polyline_i32([sample.position for sample in samples])
    if len(points) >= 2:
        cv2.polylines(overlay, [points], False, color, thickness, cv2.LINE_AA)


def _draw_marker(overlay: np.ndarray, point, color: tuple[int, int, int], *, square: bool) -> None:
    x, y = int(round(point.x)), int(round(point.y))
    if square:
        cv2.rectangle(overlay, (x - 5, y - 5), (x + 5, y + 5), color, -1)
        cv2.rectangle(overlay, (x - 7, y - 7), (x + 7, y + 7), color, 1, cv2.LINE_AA)
    else:
        cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 8, color, 1, cv2.LINE_AA)


class DotToDotSelectionLayer(QObject):
    """Interaction layer that turns the main preview into a selection surface.

    While active, an event filter on the viewer's display label intercepts
    mouse presses; everything else about the main viewer stays untouched, and
    deactivating removes the filter and lets the window restore its normal
    preview rendering.
    """

    selectionChanged = Signal()

    def __init__(self, viewer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self.controller = InteractiveSelectionController()
        self._base_image: np.ndarray | None = None
        self._active = False

    # --- geometry ---------------------------------------------------------------

    def set_svg(self, svg_source: str, base_image: np.ndarray) -> None:
        """Load a fresh SVG copy and the preview image; resets selections."""
        self.controller.set_paths(load_svg_paths(str(svg_source)))
        self._base_image = base_image.copy()
        self._refresh()

    def has_geometry(self) -> bool:
        """Return whether path geometry is loaded."""
        return bool(self.controller.paths)

    def segments(self) -> tuple[SelectedSegment, ...]:
        """Return the confirmed selected segments."""
        return self.controller.model.segments

    def set_start_number(self, start_number: int) -> None:
        """Update the starting number used for continuous numbering."""
        self.controller.model.start_number = int(start_number)

    def set_point_count(self, count: int) -> None:
        """Set how many points the pending segment will receive."""
        self.controller.set_point_count(count)

    # --- activation -------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Return whether the interaction layer is currently installed."""
        return self._active

    def activate(self) -> None:
        """Install the event filter on the viewer's display label."""
        if self._active:
            return
        target = self._display_label()
        if target is not None:
            target.installEventFilter(self)
            self._active = True
            self._refresh()

    def deactivate(self) -> None:
        """Remove the event filter and stop painting overlays."""
        if not self._active:
            return
        target = self._display_label()
        if target is not None:
            target.removeEventFilter(self)
        self._active = False

    def _display_label(self) -> QLabel | None:
        widget = self._viewer.widget()
        return widget if isinstance(widget, QLabel) else None

    # --- interaction ------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        if self._active and event.type() == QEvent.Type.MouseButtonPress:
            label = self._display_label()
            if label is not None and self._base_image is not None:
                pixmap = label.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    position = event.position()
                    mapped = widget_to_image_coords(
                        position.x(),
                        position.y(),
                        (label.width(), label.height()),
                        (pixmap.width(), pixmap.height()),
                        (self._base_image.shape[1], self._base_image.shape[0]),
                    )
                    if mapped is not None:
                        self.controller.click(mapped[0], mapped[1])
                        self._refresh()
                        return True
        return super().eventFilter(obj, event)

    def _refresh(self) -> None:
        if self._active and self._base_image is not None:
            self._viewer.set_image(render_selection_overlay(self._base_image, self.controller))
        self.selectionChanged.emit()

    def refresh_overlay(self) -> None:
        """Repaint the selection overlay on the main preview."""
        self._refresh()

    # --- actions ----------------------------------------------------------------

    def new_segment(self) -> None:
        """Start a new pending selection."""
        self.controller.reset_pending()
        self._refresh()

    def invert(self) -> None:
        """Invert the pending arc (closed subpaths)."""
        self.controller.invert()
        self._refresh()

    def confirm(self) -> None:
        """Confirm the pending selection into the model."""
        self.controller.confirm()
        self._refresh()

    def cancel(self) -> None:
        """Discard the pending selection without touching confirmed ones."""
        self.controller.reset_pending()
        self._refresh()

    def undo(self) -> None:
        """Undo the pending selection or the last confirmed one."""
        self.controller.undo()
        self._refresh()

    def delete_last(self) -> None:
        """Delete the most recently confirmed selection."""
        self.controller.delete(len(self.controller.model) - 1)
        self._refresh()

    def clear(self) -> None:
        """Remove all confirmed selections and any pending one."""
        self.controller.model.clear()
        self.controller.reset_pending()
        self._refresh()

    def numbering_summary(self) -> str:
        """Return a short human summary of confirmed numbering ranges."""
        ranges = self.controller.model.numbering_ranges()
        if not ranges:
            return "Nenhum trecho confirmado"
        parts = [f"pontos {first}-{last}" for first, last in ranges]
        return "; ".join(parts)


__all__ = [
    "DotToDotSelectionLayer",
    "render_selection_overlay",
    "widget_to_image_coords",
]
