from __future__ import annotations

import numpy as np

from fikzpy.core.contour_detector import Contour
from fikzpy.core.tikz_generator import TikzOptions, generate_tikz_picture, wrap_standalone_document


def test_generate_tikz_picture_uses_draw_and_cycle() -> None:
    contour = Contour(
        points=np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float),
        closed=True,
    )

    code = generate_tikz_picture(
        [contour],
        (40, 40, 3),
        TikzOptions(width_units=4, line_color="black", line_width=0.5),
    )

    assert "\\begin{tikzpicture}[scale=1]" in code
    assert "\\draw[draw=black, line width=0.5pt]" in code
    assert "-- cycle;" in code


def test_generate_tikz_picture_can_emit_bezier_controls() -> None:
    contour = Contour(
        points=np.array([[0, 0], [10, 15], [20, 0], [30, 10]], dtype=float),
        closed=False,
    )

    code = generate_tikz_picture([contour], (40, 40, 3), TikzOptions(use_bezier=True))

    assert ".. controls" in code


def test_wrap_standalone_document_contains_tikz_package() -> None:
    document = wrap_standalone_document("\\begin{tikzpicture}\n\\end{tikzpicture}")

    assert "\\documentclass[tikz,border=2mm]{standalone}" in document
    assert "\\usepackage{tikz}" in document
