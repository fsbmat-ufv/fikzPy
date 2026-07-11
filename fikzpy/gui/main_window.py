"""Main PySide6 window for fikzPy."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fikzpy.core.diagnostics import log_event, sha256_file
from fikzpy.core.dot_to_dot import DotToDotConfig
from fikzpy.core.image_processor import ProcessingResult, ProcessingSettings, load_image
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.tikz_pipeline import TikzBuildResult, build_tikz_from_image
from fikzpy.gui.code_editor import CodeEditor
from fikzpy.gui.image_viewer import ImageViewer
from fikzpy.templates import get_template_groups


class MainWindow(QMainWindow):
    """Desktop MVP for converting images into TikZ paths."""

    VIEW_ORDER = ("original", "overlay", "reconstruction")
    VIEW_LABELS = {
        "original": "Original",
        "overlay": "Com tracos",
        "reconstruction": "Somente desenho",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("fikzPy - Image to TikZ")
        self.resize(1180, 760)

        self.image_path: Path | None = None
        self.original_image = None
        self.result: ProcessingResult | None = None
        self.build_result: TikzBuildResult | None = None
        self.current_view_index = 0
        self.latex_engine = "pdflatex"
        self.manual_latex_path: Path | None = None
        self.preview_counter = 0

        self.image_viewer = ImageViewer()
        self.code_editor = CodeEditor()
        self.status_label = QLabel("Pronto")

        self._build_actions()
        self._build_central_widget()
        self._build_settings_dock()
        self._build_menus()
        self.statusBar().addPermanentWidget(self.status_label)

    def _build_actions(self) -> None:
        self.open_action = QAction("Abrir imagem...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image)

        self.generate_action = QAction("Gerar TikZ", self)
        self.generate_action.triggered.connect(self.generate_tikz)

        self.copy_action = QAction("Copiar codigo", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.triggered.connect(self.copy_code)

        self.export_action = QAction("Exportar .tex...", self)
        self.export_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.export_action.triggered.connect(self.export_tex)

        self.compile_action = QAction("Compilar e visualizar PDF", self)
        self.compile_action.triggered.connect(self.compile_latex)

        self.toggle_view_action = QAction("Alternar visualizacao", self)
        self.toggle_view_action.triggered.connect(self.toggle_view)

        self.exit_action = QAction("Sair", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

    def _build_central_widget(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        toolbar = QToolBar("Acoes")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.generate_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_view_action)
        toolbar.addSeparator()
        toolbar.addAction(self.copy_action)
        toolbar.addAction(self.export_action)
        toolbar.addAction(self.compile_action)
        self.addToolBar(toolbar)

        button_row = QHBoxLayout()
        self.view_button = QPushButton("Visualizacao: Original")
        self.view_button.clicked.connect(self.toggle_view)
        button_row.addWidget(self.view_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_viewer)
        splitter.addWidget(self.code_editor)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        self.setCentralWidget(root)

    def _build_settings_dock(self) -> None:
        dock = QDockWidget("Parametros", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        panel = QWidget()
        form = QFormLayout(panel)

        self.vectorization_mode_combo = QComboBox()
        self.vectorization_mode_combo.addItem("Classic Auto", ("classic", "auto"))
        self.vectorization_mode_combo.addItem("Classic Filled", ("classic", "filled"))
        self.vectorization_mode_combo.addItem("Visual", ("visual", "auto"))
        self.vectorization_mode_combo.addItem("Contornos", ("contours", "auto"))
        self.vectorization_mode_combo.addItem("Ponto a Ponto", ("dot_to_dot", "full_figure"))
        form.addRow("Modo", self.vectorization_mode_combo)

        self.ink_threshold_spin = QSpinBox()
        self.ink_threshold_spin.setRange(120, 245)
        self.ink_threshold_spin.setValue(215)
        form.addRow("Limiar tinta", self.ink_threshold_spin)

        self.stroke_smoothing_spin = QSpinBox()
        self.stroke_smoothing_spin.setRange(0, 3)
        self.stroke_smoothing_spin.setValue(1)
        form.addRow("Suav. traco", self.stroke_smoothing_spin)

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 31)
        self.smoothing_spin.setSingleStep(2)
        self.smoothing_spin.setValue(5)
        form.addRow("Suavizacao", self.smoothing_spin)

        self.canny_low_spin = QSpinBox()
        self.canny_low_spin.setRange(0, 500)
        self.canny_low_spin.setValue(50)
        form.addRow("Canny baixo", self.canny_low_spin)

        self.canny_high_spin = QSpinBox()
        self.canny_high_spin.setRange(1, 500)
        self.canny_high_spin.setValue(150)
        form.addRow("Canny alto", self.canny_high_spin)

        self.simplify_spin = QDoubleSpinBox()
        self.simplify_spin.setRange(0.001, 0.2)
        self.simplify_spin.setDecimals(3)
        self.simplify_spin.setSingleStep(0.005)
        self.simplify_spin.setValue(0.006)
        form.addRow("Simplificacao", self.simplify_spin)

        self.tikz_scale_spin = QDoubleSpinBox()
        self.tikz_scale_spin.setRange(0.1, 20.0)
        self.tikz_scale_spin.setDecimals(2)
        self.tikz_scale_spin.setValue(1.0)
        form.addRow("Escala TikZ", self.tikz_scale_spin)

        self.width_units_spin = QDoubleSpinBox()
        self.width_units_spin.setRange(1.0, 100.0)
        self.width_units_spin.setDecimals(1)
        self.width_units_spin.setValue(10.0)
        form.addRow("Largura em unidades", self.width_units_spin)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(0.4)
        form.addRow("Espessura", self.line_width_spin)

        self.line_color_edit = QLineEdit("black")
        form.addRow("Cor TikZ", self.line_color_edit)

        self.bezier_check = QCheckBox("Usar curvas Bezier")
        form.addRow("", self.bezier_check)

        regenerate_button = QPushButton("Atualizar TikZ")
        regenerate_button.clicked.connect(self.generate_tikz)
        form.addRow("", regenerate_button)

        self._build_dot_to_dot_group(form)

        for widget in (
            self.smoothing_spin,
            self.canny_low_spin,
            self.canny_high_spin,
            self.simplify_spin,
            self.tikz_scale_spin,
            self.width_units_spin,
            self.line_width_spin,
            self.ink_threshold_spin,
            self.stroke_smoothing_spin,
        ):
            widget.valueChanged.connect(self._settings_changed)
        self.bezier_check.stateChanged.connect(self._settings_changed)
        self.vectorization_mode_combo.currentIndexChanged.connect(self._settings_changed)
        self.vectorization_mode_combo.currentIndexChanged.connect(self._update_dot_to_dot_visibility)
        self.line_color_edit.editingFinished.connect(self._settings_changed)

        self._update_dot_to_dot_visibility()

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_dot_to_dot_group(self, form: QFormLayout) -> None:
        """Build the Ponto a Ponto options panel (shown only for that mode)."""
        self.dot_to_dot_group = QGroupBox("Ponto a Ponto")
        group_form = QFormLayout(self.dot_to_dot_group)

        self.dtd_submode_combo = QComboBox()
        self.dtd_submode_combo.addItem("Figura inteira", "full_figure")
        group_form.addRow("Submodo", self.dtd_submode_combo)

        self.dtd_total_points_spin = QSpinBox()
        self.dtd_total_points_spin.setRange(2, 500)
        self.dtd_total_points_spin.setValue(20)
        group_form.addRow("Numero de pontos", self.dtd_total_points_spin)

        self.dtd_start_number_spin = QSpinBox()
        self.dtd_start_number_spin.setRange(0, 9999)
        self.dtd_start_number_spin.setValue(1)
        group_form.addRow("Numero inicial", self.dtd_start_number_spin)

        self.dtd_point_radius_spin = QDoubleSpinBox()
        self.dtd_point_radius_spin.setRange(0.2, 20.0)
        self.dtd_point_radius_spin.setDecimals(2)
        self.dtd_point_radius_spin.setSingleStep(0.5)
        self.dtd_point_radius_spin.setValue(2.0)
        group_form.addRow("Tamanho dos pontos (pt)", self.dtd_point_radius_spin)

        self.dtd_label_offset_spin = QDoubleSpinBox()
        self.dtd_label_offset_spin.setRange(0.0, 5.0)
        self.dtd_label_offset_spin.setDecimals(2)
        self.dtd_label_offset_spin.setSingleStep(0.05)
        self.dtd_label_offset_spin.setValue(0.35)
        group_form.addRow("Distancia dos numeros", self.dtd_label_offset_spin)

        self.dtd_direction_combo = QComboBox()
        self.dtd_direction_combo.addItem("Direta", "forward")
        self.dtd_direction_combo.addItem("Invertida", "reverse")
        group_form.addRow("Direcao", self.dtd_direction_combo)

        self.dtd_preserve_details_check = QCheckBox("Manter detalhes internos")
        self.dtd_preserve_details_check.setChecked(True)
        group_form.addRow("", self.dtd_preserve_details_check)

        self.dtd_include_internal_check = QCheckBox("Incluir contornos internos")
        self.dtd_include_internal_check.setChecked(False)
        group_form.addRow("", self.dtd_include_internal_check)

        self.dtd_preview_button = QPushButton("Gerar/atualizar previsualizacao")
        self.dtd_preview_button.clicked.connect(self.generate_tikz)
        group_form.addRow("", self.dtd_preview_button)

        for widget in (
            self.dtd_total_points_spin,
            self.dtd_start_number_spin,
            self.dtd_point_radius_spin,
            self.dtd_label_offset_spin,
        ):
            widget.valueChanged.connect(self._settings_changed)
        for combo in (self.dtd_submode_combo, self.dtd_direction_combo):
            combo.currentIndexChanged.connect(self._settings_changed)
        self.dtd_preserve_details_check.stateChanged.connect(self._settings_changed)
        self.dtd_include_internal_check.stateChanged.connect(self._settings_changed)

        form.addRow(self.dot_to_dot_group)

    def _update_dot_to_dot_visibility(self, *args) -> None:
        mode, _strategy = self._selected_mode_and_strategy()
        self.dot_to_dot_group.setVisible(mode == "dot_to_dot")

    def _build_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("Arquivo")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menu.addMenu("Editar")
        edit_menu.addAction(self.copy_action)

        generate_menu = menu.addMenu("Gerar TikZ")
        generate_menu.addAction(self.generate_action)
        templates_menu = generate_menu.addMenu("Templates TikZ")
        self._add_template_actions(templates_menu)

        view_menu = menu.addMenu("Visualizacao")
        view_menu.addAction(self.toggle_view_action)

        settings_menu = menu.addMenu("Configuracoes")
        latex_menu = settings_menu.addMenu("Distribuicao LaTeX")
        for label in ("MiKTeX", "TeX Live", "MacTeX"):
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, value=label: self.detect_latex(value))
            latex_menu.addAction(action)
        latex_menu.addSeparator()
        latex_menu.addAction("Detectar automaticamente", lambda: self.detect_latex())
        latex_menu.addAction("Caminho manual...", self.choose_manual_latex_path)

        engine_menu = settings_menu.addMenu("Engine")
        for engine in ("pdflatex", "xelatex", "lualatex"):
            action = QAction(engine, self)
            action.setCheckable(True)
            action.setChecked(engine == self.latex_engine)
            action.triggered.connect(lambda checked=False, value=engine: self.set_latex_engine(value))
            engine_menu.addAction(action)

        help_menu = menu.addMenu("Ajuda")
        help_menu.addAction("Sobre", self.show_about)

    def _add_template_actions(self, menu) -> None:
        for group_name, templates in get_template_groups().items():
            group_menu = menu.addMenu(group_name)
            for template_name, code in templates.items():
                action = QAction(template_name, self)
                action.triggered.connect(
                    lambda checked=False, snippet=code, name=template_name: self.load_template(name, snippet)
                )
                group_menu.addAction(action)

    def _selected_mode_and_strategy(self) -> tuple[str, str]:
        """Return (vectorization_mode, classic_strategy) for the active main-mode selection."""
        data = self.vectorization_mode_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[0]), str(data[1])
        return str(data or "classic"), "auto"

    def processing_settings(self) -> ProcessingSettings:
        """Read image-processing settings from the UI."""
        mode, _strategy = self._selected_mode_and_strategy()
        return ProcessingSettings(
            vectorization_mode=mode,
            smoothing=self.smoothing_spin.value(),
            canny_low=self.canny_low_spin.value(),
            canny_high=self.canny_high_spin.value(),
            simplify_epsilon=self.simplify_spin.value(),
            line_art_threshold=self.ink_threshold_spin.value(),
            stroke_smoothing=self.stroke_smoothing_spin.value(),
        )

    def tikz_options(self) -> TikzOptions:
        """Read TikZ generation options from the UI."""
        mode, strategy = self._selected_mode_and_strategy()
        return TikzOptions(
            tikz_scale=self.tikz_scale_spin.value(),
            line_width=self.line_width_spin.value(),
            line_color=self.line_color_edit.text(),
            use_bezier=self.bezier_check.isChecked() or mode in {"visual"},
            width_units=self.width_units_spin.value(),
            classic_strategy=strategy if mode == "classic" else "auto",
        )

    def dot_to_dot_config(self) -> DotToDotConfig:
        """Build a dot-to-dot configuration from the Ponto a Ponto panel."""
        return DotToDotConfig(
            generation_mode=self.dtd_submode_combo.currentData() or "full_figure",
            total_points=self.dtd_total_points_spin.value(),
            start_number=self.dtd_start_number_spin.value(),
            point_radius=self.dtd_point_radius_spin.value(),
            label_offset=self.dtd_label_offset_spin.value(),
            include_internal_contours=self.dtd_include_internal_check.isChecked(),
            preserve_internal_details=self.dtd_preserve_details_check.isChecked(),
            direction=self.dtd_direction_combo.currentData() or "forward",
        )

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir imagem",
            str(Path.home()),
            "Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Todos os arquivos (*.*)",
        )
        if not path:
            return

        try:
            self.image_path = Path(path)
            self.original_image = load_image(self.image_path)
            self.current_view_index = 0
            self.generate_tikz()
        except Exception as exc:  # pragma: no cover - user-facing guard
            QMessageBox.critical(self, "Erro ao abrir imagem", str(exc))

    def generate_tikz(self) -> None:
        if self.original_image is None:
            self.status_label.setText("Abra uma imagem para gerar TikZ.")
            return

        try:
            mode, _strategy = self._selected_mode_and_strategy()
            dot_config = self.dot_to_dot_config() if mode == "dot_to_dot" else None
            self.build_result = build_tikz_from_image(
                self.original_image,
                self.processing_settings(),
                self.tikz_options(),
                dot_to_dot_config=dot_config,
            )
            self.result = self.build_result.processing_result
            self.code_editor.set_code(self.build_result.tikz_code)
            self._update_view()
            self.status_label.setText(f"{len(self.result.contours)} contornos detectados")
        except Exception as exc:  # pragma: no cover - user-facing guard
            log_event("Vectorization", f"failure={exc!r}")
            QMessageBox.critical(self, "Erro ao gerar TikZ", str(exc))

    def _settings_changed(self, *args) -> None:
        if self.original_image is not None:
            self.generate_tikz()

    def toggle_view(self) -> None:
        self.current_view_index = (self.current_view_index + 1) % len(self.VIEW_ORDER)
        self._update_view()

    def _update_view(self) -> None:
        if self.result is None:
            return

        key = self.VIEW_ORDER[self.current_view_index]
        if key == "overlay":
            image = self.result.overlay_bgr
        elif key == "reconstruction":
            image = self.result.reconstruction_bgr
        else:
            image = self.result.original_bgr

        self.image_viewer.set_image(image)
        label = self.VIEW_LABELS[key]
        self.view_button.setText(f"Visualizacao: {label}")
        self.status_label.setText(f"Visualizacao: {label}")

    def copy_code(self) -> None:
        QApplication.clipboard().setText(self.code_editor.code())
        self.status_label.setText("Codigo TikZ copiado")

    def export_tex(self) -> Path | None:
        if not self.code_editor.code().strip():
            QMessageBox.information(self, "Exportar .tex", "Nao ha codigo TikZ para exportar.")
            return None

        default_name = "fikzpy_output.tex"
        if self.image_path is not None:
            default_name = f"{self.image_path.stem}_fikzpy.tex"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar arquivo TeX",
            str(Path.cwd() / default_name),
            "TeX (*.tex)",
        )
        if not path:
            return None

        output_path = Path(path)
        output_path.write_text(wrap_standalone_document(self.code_editor.code()), encoding="utf-8")
        log_event("TikZ", f"output={output_path}")
        log_event("TikZ", f"sha256={sha256_file(output_path)}")
        self.status_label.setText(f"Exportado: {output_path.name}")
        return output_path

    def compile_latex(self) -> None:
        if not self.code_editor.code().strip():
            QMessageBox.information(self, "Compilar LaTeX", "Nao ha codigo TikZ para compilar.")
            return

        try:
            from fikzpy.core.latex_compiler import compile_latex_document
        except ImportError:
            QMessageBox.warning(self, "LaTeX", "O suporte de compilacao LaTeX ainda nao esta disponivel.")
            return

        tex_path = self._write_temporary_tex()
        try:
            result = compile_latex_document(
                tex_path,
                engine=self.latex_engine,
                manual_path=self.manual_latex_path,
            )
        except Exception as exc:  # pragma: no cover - user-facing guard
            log_event("LaTeX", f"failure={exc!r}")
            QMessageBox.critical(self, "Erro ao compilar", str(exc))
            return

        log_event("LaTeX", f"command={' '.join(result.command)}")
        log_event("LaTeX", f"returncode={result.returncode}")
        log_event("LaTeX", f"pdf={result.pdf_path}")
        if result.returncode != 0:
            QMessageBox.warning(self, "Erro LaTeX", result.output[-3000:])
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.pdf_path)))
        log_event("Preview", f"loaded={result.pdf_path}")
        self.status_label.setText(f"PDF gerado: {result.pdf_path.name}")

    def _write_temporary_tex(self) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "fikzpy"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.preview_counter += 1
        mode = self._current_output_mode()
        tex_path = temp_dir / f"fikzpy_{mode}_{self.preview_counter:03d}.tex"
        tex_path.write_text(wrap_standalone_document(self.code_editor.code()), encoding="utf-8")
        log_event("TikZ", f"output={tex_path}")
        log_event("TikZ", f"sha256={sha256_file(tex_path)}")
        log_event("LaTeX", f"pdf={tex_path.with_suffix('.pdf')}")
        return tex_path

    def _current_output_mode(self) -> str:
        if self.build_result is not None:
            return self.build_result.effective_mode
        mode, _strategy = self._selected_mode_and_strategy()
        return mode.lower().replace(" ", "_")

    def detect_latex(self, distribution: str | None = None) -> None:
        try:
            from fikzpy.core.latex_compiler import detect_latex_tools
        except ImportError:
            QMessageBox.warning(self, "LaTeX", "O suporte de deteccao LaTeX ainda nao esta disponivel.")
            return

        tools = detect_latex_tools(distribution=distribution)
        if not tools:
            QMessageBox.information(self, "LaTeX", "Nenhuma instalacao LaTeX foi encontrada automaticamente.")
            return

        chosen = tools[0]
        self.latex_engine = chosen.engine
        self.manual_latex_path = chosen.path
        QMessageBox.information(self, "LaTeX", f"Encontrado: {chosen.engine}\n{chosen.path}")

    def choose_manual_latex_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar pdflatex, xelatex ou lualatex",
            str(Path.home()),
            "Executaveis (*.exe);;Todos os arquivos (*.*)",
        )
        if path:
            self.manual_latex_path = Path(path)
            self.status_label.setText(f"LaTeX manual: {self.manual_latex_path.name}")

    def set_latex_engine(self, engine: str) -> None:
        self.latex_engine = engine
        self.status_label.setText(f"Engine LaTeX: {engine}")

    def load_template(self, name: str, code: str) -> None:
        if self.code_editor.code().strip():
            answer = QMessageBox.question(
                self,
                "Carregar template",
                "Substituir o codigo atual pelo template selecionado?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.code_editor.set_code(code.strip())
        self.status_label.setText(f"Template carregado: {name}")

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "Sobre o fikzPy",
            "fikzPy converte contornos de imagens em codigo TikZ editavel para LaTeX.",
        )


def run_app() -> int:
    """Run the Qt application."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
