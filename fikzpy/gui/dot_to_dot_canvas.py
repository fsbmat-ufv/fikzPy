"""Interactive canvas for dot-to-dot segment selection.

The widget is a thin renderer: all selection logic lives in
:class:`fikzpy.core.dot_to_dot.segment_selection.InteractiveSelectionController`.
The highlight colours here are preview-only and never reach the exported TikZ.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from fikzpy.core.dot_to_dot.config import SelectedSegment
from fikzpy.core.dot_to_dot.models import Vec2
from fikzpy.core.dot_to_dot.segment_selection import (
    InteractiveSelectionController,
    sample_selected_segment,
)
from fikzpy.core.dot_to_dot.svg_loader import load_svg_paths

_MARGIN = 16.0


class DotToDotCanvas(QWidget):
    """Render traced paths and let the user pick arcs to convert into points."""

    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = InteractiveSelectionController()
        self._image_size = (1.0, 1.0)  # (width, height) in SVG user units
        self.setMinimumSize(260, 260)

    # --- data -----------------------------------------------------------------------

    def set_svg(self, svg_source: str, image_shape: tuple[int, ...]) -> None:
        """Load a fresh SVG (copy) into the canvas and reset the selection."""
        paths = load_svg_paths(str(svg_source))
        self.controller.set_paths(paths)
        self._image_size = (float(image_shape[1]), float(image_shape[0]))
        self.update()
        self.selectionChanged.emit()

    def has_geometry(self) -> bool:
        """Return whether any path geometry is loaded."""
        return bool(self.controller.paths)

    def segments(self) -> tuple[SelectedSegment, ...]:
        """Return the confirmed selected segments."""
        return self.controller.model.segments

    def set_start_number(self, start_number: int) -> None:
        """Update the starting number used for continuous numbering."""
        self.controller.model.start_number = int(start_number)
        self.update()

    def set_point_count(self, count: int) -> None:
        """Set how many points the pending segment will receive."""
        self.controller.set_point_count(count)
        self.update()

    # --- interactive actions --------------------------------------------------------

    def new_segment(self) -> None:
        """Start a new pending selection."""
        self.controller.reset_pending()
        self._changed()

    def invert(self) -> None:
        """Invert the pending arc (closed subpaths)."""
        self.controller.invert()
        self._changed()

    def confirm(self) -> None:
        """Confirm the pending selection into the model."""
        self.controller.confirm()
        self._changed()

    def undo(self) -> None:
        """Undo the pending selection or the last confirmed one."""
        self.controller.undo()
        self._changed()

    def delete_last(self) -> None:
        """Delete the most recently confirmed selection."""
        self.controller.delete(len(self.controller.model) - 1)
        self._changed()

    def clear(self) -> None:
        """Remove all confirmed selections and any pending one."""
        self.controller.model.clear()
        self.controller.reset_pending()
        self._changed()

    def numbering_summary(self) -> str:
        """Return a short human summary of confirmed numbering ranges."""
        ranges = self.controller.model.numbering_ranges()
        if not ranges:
            return "Nenhum trecho confirmado"
        parts = [f"pontos {first}-{last}" for first, last in ranges]
        return "; ".join(parts)

    def _changed(self) -> None:
        self.update()
        self.selectionChanged.emit()

    # --- coordinate mapping ---------------------------------------------------------

    def _view_scale_offset(self) -> tuple[float, float, float]:
        width, height = self._image_size
        avail_w = max(self.width() - 2 * _MARGIN, 1.0)
        avail_h = max(self.height() - 2 * _MARGIN, 1.0)
        scale = min(avail_w / max(width, 1e-6), avail_h / max(height, 1e-6))
        offset_x = _MARGIN + (avail_w - width * scale) / 2.0
        offset_y = _MARGIN + (avail_h - height * scale) / 2.0
        return scale, offset_x, offset_y

    def _to_widget(self, point: Vec2) -> QPointF:
        scale, offset_x, offset_y = self._view_scale_offset()
        return QPointF(point.x * scale + offset_x, point.y * scale + offset_y)

    def _to_svg(self, wx: float, wy: float) -> tuple[float, float]:
        scale, offset_x, offset_y = self._view_scale_offset()
        return (wx - offset_x) / scale, (wy - offset_y) / scale

    # --- events ---------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # pragma: no cover - requires Qt events
        position = event.position()
        sx, sy = self._to_svg(position.x(), position.y())
        self.controller.click(sx, sy)
        self._changed()

    def paintEvent(self, event) -> None:  # pragma: no cover - requires Qt paint
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        self._paint_paths(painter)
        self._paint_confirmed(painter)
        self._paint_pending(painter)
        painter.end()

    def _paint_paths(self, painter: QPainter) -> None:
        pen = QPen(QColor(150, 150, 150))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        for path in self.controller.paths:
            for subpath in path.subpaths:
                polygon = QPolygonF([self._to_widget(point) for point in subpath.polyline()])
                painter.drawPolyline(polygon)

    def _paint_confirmed(self, painter: QPainter) -> None:
        pen = QPen(QColor(210, 120, 20))
        pen.setWidthF(2.4)
        painter.setPen(pen)
        for segment in self.controller.model.segments:
            self._paint_arc(painter, segment)

    def _paint_pending(self, painter: QPainter) -> None:
        pending = self.controller.pending_segment()
        if pending is None:
            if self.controller.start is not None:
                self._paint_marker(painter, self.controller.start.point, QColor(30, 120, 220))
            return
        pen = QPen(QColor(30, 120, 220))
        pen.setWidthF(2.6)
        painter.setPen(pen)
        self._paint_arc(painter, pending)
        for sample in sample_selected_segment(self._subpath_for(pending), pending):
            self._paint_marker(painter, sample.position, QColor(30, 120, 220), radius=3.5)

    def _paint_arc(self, painter: QPainter, segment: SelectedSegment) -> None:
        subpath = self._subpath_for(segment)
        if subpath is None:
            return
        dense = replace(segment, point_count=max(24, segment.point_count))
        samples = sample_selected_segment(subpath, dense)
        polygon = QPolygonF([self._to_widget(sample.position) for sample in samples])
        painter.drawPolyline(polygon)

    def _paint_marker(self, painter: QPainter, point: Vec2, color: QColor, radius: float = 4.0) -> None:
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._to_widget(point), radius, radius)

    def _subpath_for(self, segment: SelectedSegment):
        for path in self.controller.paths:
            if path.path_id == segment.path_id and 0 <= segment.subpath_id < len(path.subpaths):
                return path.subpaths[segment.subpath_id]
        return None


__all__ = ["DotToDotCanvas"]
