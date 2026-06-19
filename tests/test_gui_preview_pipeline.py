from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from fikzpy.gui.main_window import MainWindow


def test_preview_compile_loads_pdf_matching_fresh_tex(monkeypatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Visual"))
    window.code_editor.set_code("\\begin{tikzpicture}\n  % FIKZPY VECTOR MODE\n\\end{tikzpicture}")

    opened: list[QUrl] = []
    compiled_tex: list[Path] = []

    def fake_compile(tex_path, **kwargs):
        tex = Path(tex_path)
        compiled_tex.append(tex)
        pdf = tex.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 fake")
        return SimpleNamespace(returncode=0, output="", tex_path=tex, pdf_path=pdf, command=("pdflatex", tex.name))

    monkeypatch.setattr("fikzpy.core.latex_compiler.compile_latex_document", fake_compile)
    monkeypatch.setattr("fikzpy.gui.main_window.QDesktopServices.openUrl", lambda url: opened.append(url) or True)

    window.compile_latex()

    assert compiled_tex
    assert "fikzpy_visual_" in compiled_tex[0].name
    assert opened
    assert Path(opened[0].toLocalFile()) == compiled_tex[0].with_suffix(".pdf")


def test_preview_temp_names_are_mode_specific() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.code_editor.set_code("\\begin{tikzpicture}\n\\end{tikzpicture}")

    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Classic Auto"))
    classic_path = window._write_temporary_tex()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Visual"))
    visual_path = window._write_temporary_tex()

    assert "fikzpy_classic_" in classic_path.name
    assert "fikzpy_visual_" in visual_path.name
    assert classic_path != visual_path


def test_gui_exposes_classic_strategies_and_primary_modes() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    labels = [
        window.vectorization_mode_combo.itemText(index)
        for index in range(window.vectorization_mode_combo.count())
    ]
    assert labels == ["Classic Auto", "Classic Line Art", "Classic Filled", "Visual", "Contornos"]

    modes_and_strategies = [
        window.vectorization_mode_combo.itemData(index)
        for index in range(window.vectorization_mode_combo.count())
    ]
    assert modes_and_strategies == [
        ("classic", "auto"),
        ("classic", "line_art"),
        ("classic", "filled"),
        ("visual", "auto"),
        ("contours", "auto"),
    ]


def test_classic_line_art_selection_maps_to_classic_mode_and_line_art_strategy() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Classic Line Art"))

    settings = window.processing_settings()
    options = window.tikz_options()

    assert settings.vectorization_mode == "classic"
    assert options.classic_strategy == "line_art"


def test_classic_filled_selection_maps_to_classic_mode_and_filled_strategy() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Classic Filled"))

    settings = window.processing_settings()
    options = window.tikz_options()

    assert settings.vectorization_mode == "classic"
    assert options.classic_strategy == "filled"


def test_classic_auto_selection_maps_to_classic_mode_and_auto_strategy() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Classic Auto"))

    settings = window.processing_settings()
    options = window.tikz_options()

    assert settings.vectorization_mode == "classic"
    assert options.classic_strategy == "auto"


def test_visual_selection_keeps_visual_flow_unaffected_by_classic_strategy() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Visual"))

    settings = window.processing_settings()
    options = window.tikz_options()

    assert settings.vectorization_mode == "visual"
    assert options.use_bezier is True


def test_contornos_selection_keeps_contours_flow_unaffected_by_classic_strategy() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Contornos"))

    settings = window.processing_settings()

    assert settings.vectorization_mode == "contours"
