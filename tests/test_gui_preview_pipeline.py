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


def test_gui_exposes_exactly_the_five_primary_modes() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    labels = [
        window.vectorization_mode_combo.itemText(index)
        for index in range(window.vectorization_mode_combo.count())
    ]
    assert labels == ["Classic Auto", "Classic Filled", "Visual", "Contornos", "Ponto a Ponto"]

    modes_and_strategies = [
        window.vectorization_mode_combo.itemData(index)
        for index in range(window.vectorization_mode_combo.count())
    ]
    assert modes_and_strategies == [
        ("classic", "auto"),
        ("classic", "filled"),
        ("visual", "auto"),
        ("contours", "auto"),
        ("dot_to_dot", "full_figure"),
    ]


def test_classic_line_art_is_not_selectable() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.vectorization_mode_combo.findText("Classic Line Art") == -1


def test_ponto_a_ponto_selection_maps_to_dot_to_dot_mode() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))

    settings = window.processing_settings()

    assert settings.vectorization_mode == "dot_to_dot"


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


def test_dot_to_dot_panel_visible_only_for_ponto_a_ponto() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Visual"))
    assert window.dot_to_dot_group.isHidden()

    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))
    assert not window.dot_to_dot_group.isHidden()


def test_dot_to_dot_config_reflects_panel_values() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))

    window.dtd_total_points_spin.setValue(24)
    window.dtd_start_number_spin.setValue(5)
    window.dtd_point_radius_spin.setValue(3.0)
    window.dtd_label_offset_spin.setValue(0.5)
    window.dtd_preserve_details_check.setChecked(True)
    window.dtd_direction_combo.setCurrentIndex(window.dtd_direction_combo.findText("Anti-horario"))

    config = window.dot_to_dot_config()

    assert config.generation_mode.value == "full_figure"
    assert config.total_points == 24
    assert config.start_number == 5
    assert config.point_radius == 3.0
    assert config.label_offset == 0.5
    assert config.preserve_internal_details is True
    assert config.direction.value == "reverse"


def test_dot_to_dot_defaults_keep_details_off_and_no_internal_contours_option() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.dtd_preserve_details_check.isChecked() is False
    assert not hasattr(window, "dtd_include_internal_check")
    config = window.dot_to_dot_config()
    assert config.preserve_internal_details is False


def test_selected_segments_panel_visible_only_for_that_submode() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))

    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Figura inteira"))
    assert window.dtd_selection_group.isHidden()

    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Selecionar trechos"))
    assert not window.dtd_selection_group.isHidden()


def _selection_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M 0 0 L 40 0 L 40 40 L 0 40 Z"/></svg>'
    )


def test_selected_segments_config_includes_confirmed_segments() -> None:
    import numpy as np

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))
    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Selecionar trechos"))

    base = np.full((100, 100, 3), 255, np.uint8)
    window.dot_to_dot_layer.set_svg(_selection_svg(), base)
    controller = window.dot_to_dot_layer.controller
    controller.set_point_count(4)
    controller.click(20.0, 0.0)
    controller.click(40.0, 20.0)
    controller.confirm()

    config = window.dot_to_dot_config()
    assert config.generation_mode.value == "selected_segments"
    assert len(config.selected_segments) == 1
    assert config.selected_segments[0].point_count == 4


def test_no_secondary_canvas_in_side_panel() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    # The old DotToDotCanvas widget is gone; selection happens on the main
    # preview through the interaction layer.
    assert not hasattr(window, "dot_to_dot_canvas")
    assert window.dot_to_dot_layer is not None
    # No widget inside the selection group renders an image.
    from PySide6.QtWidgets import QLabel, QPushButton

    for child in window.dtd_selection_group.findChildren(object):
        assert not type(child).__name__ == "DotToDotCanvas"


def test_selection_layer_activates_only_in_selected_segments_submode() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Visual"))
    assert window.dot_to_dot_layer.active is False

    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Ponto a Ponto"))
    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Selecionar trechos"))
    assert window.dot_to_dot_layer.active is True

    # Leaving the submode deactivates the layer and clears selections.
    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Figura inteira"))
    assert window.dot_to_dot_layer.active is False
    assert window.dot_to_dot_layer.segments() == ()

    # Leaving the mode entirely also keeps it inactive.
    window.dtd_submode_combo.setCurrentIndex(window.dtd_submode_combo.findText("Selecionar trechos"))
    assert window.dot_to_dot_layer.active is True
    window.vectorization_mode_combo.setCurrentIndex(window.vectorization_mode_combo.findText("Classic Auto"))
    assert window.dot_to_dot_layer.active is False
