from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.primitive_detection import detect_ellipse, detect_line, detect_rectangle


def test_detect_line_returns_line_primitive() -> None:
    contour = Contour(points=np.array([[0, 0], [5, 0], [10, 0]], dtype=float), closed=False)

    primitive = detect_line(contour)

    assert primitive is not None
    assert primitive.kind == "line"


def test_detect_rectangle_returns_rectangle_primitive() -> None:
    contour = Contour(points=np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float), closed=True)

    primitive = detect_rectangle(contour)

    assert primitive is not None
    assert primitive.kind == "rectangle"


def test_detect_ellipse_returns_circle_for_round_contour() -> None:
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    points = np.column_stack([10 + 5 * np.cos(angles), 10 + 5 * np.sin(angles)])
    contour = Contour(points=points, closed=True)

    primitive = detect_ellipse(contour)

    assert primitive is not None
    assert primitive.kind == "circle"
