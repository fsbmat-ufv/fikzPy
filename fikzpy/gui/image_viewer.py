"""Image display widget for NumPy/OpenCV arrays."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea


class ImageViewer(QScrollArea):
    """A small responsive image viewer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel("Nenhuma imagem carregada")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(360, 280)
        self._label.setStyleSheet("background: #f6f6f6; border: 1px solid #d0d0d0;")
        self._pixmap: QPixmap | None = None

        self.setWidgetResizable(True)
        self.setWidget(self._label)

    def clear(self) -> None:
        """Clear the current image."""
        self._pixmap = None
        self._label.setText("Nenhuma imagem carregada")
        self._label.setPixmap(QPixmap())

    def set_image(self, image: np.ndarray) -> None:
        """Set the current image from an OpenCV-style array."""
        qimage = numpy_to_qimage(image)
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        viewport_size = self.viewport().size()
        scaled = self._pixmap.scaled(
            viewport_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setText("")
        self._label.setPixmap(scaled)


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """Convert grayscale/BGR/BGRA arrays to a detached QImage."""
    if image.ndim == 2:
        contiguous = np.ascontiguousarray(image)
        height, width = contiguous.shape
        bytes_per_line = contiguous.strides[0]
        return QImage(
            contiguous.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        ).copy()

    if image.ndim != 3:
        raise ValueError("Expected a 2D grayscale or 3D color image.")

    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        fmt = QImage.Format.Format_RGBA8888
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fmt = QImage.Format.Format_RGB888

    contiguous = np.ascontiguousarray(rgb)
    height, width, channels = contiguous.shape
    bytes_per_line = channels * width
    return QImage(contiguous.data, width, height, bytes_per_line, fmt).copy()
