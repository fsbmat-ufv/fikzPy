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

from fikzpy.core.image_processor import ProcessingResult, ProcessingSettings, load_image, process_image
from fikzpy.core.tikz_generator import TikzOptions, generate_tikz_picture, wrap_standalone_document
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
        self.current_view_index = 0
        self.latex_engine = "pdflatex"
        self.manual_latex_path: Path | None = None

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
        self.vectorization_mode_combo.addItem("Line art", "line_art")
        self.vectorization_mode_combo.addItem("Contornos", "contours")
        form.addRow("Modo", self.vectorization_mode_combo)

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

        for widget in (
            self.smoothing_spin,
            self.canny_low_spin,
            self.canny_high_spin,
            self.simplify_spin,
            self.tikz_scale_spin,
            self.width_units_spin,
            self.line_width_spin,
        ):
            widget.valueChanged.connect(self._settings_changed)
        self.bezier_check.stateChanged.connect(self._settings_changed)
        self.vectorization_mode_combo.currentIndexChanged.connect(self._settings_changed)
        self.line_color_edit.editingFinished.connect(self._settings_changed)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

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

    def processing_settings(self) -> ProcessingSettings:
        """Read image-processing settings from the UI."""
        return ProcessingSettings(
            vectorization_mode=self.vectorization_mode_combo.currentData(),
            smoothing=self.smoothing_spin.value(),
            canny_low=self.canny_low_spin.value(),
            canny_high=self.canny_high_spin.value(),
            simplify_epsilon=self.simplify_spin.value(),
        )

    def tikz_options(self) -> TikzOptions:
        """Read TikZ generation options from the UI."""
        return TikzOptions(
            tikz_scale=self.tikz_scale_spin.value(),
            line_width=self.line_width_spin.value(),
            line_color=self.line_color_edit.text(),
            use_bezier=self.bezier_check.isChecked(),
            width_units=self.width_units_spin.value(),
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
            self.result = process_image(self.original_image, self.processing_settings())
            code = generate_tikz_picture(self.result.contours, self.result.original_bgr.shape, self.tikz_options())
            self.code_editor.set_code(code)
            self._update_view()
            self.status_label.setText(f"{len(self.result.contours)} contornos detectados")
        except Exception as exc:  # pragma: no cover - user-facing guard
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
            QMessageBox.critical(self, "Erro ao compilar", str(exc))
            return

        if result.returncode != 0:
            QMessageBox.warning(self, "Erro LaTeX", result.output[-3000:])
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.pdf_path)))
        self.status_label.setText(f"PDF gerado: {result.pdf_path.name}")

    def _write_temporary_tex(self) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "fikzpy"
        temp_dir.mkdir(parents=True, exist_ok=True)
        tex_path = temp_dir / "fikzpy_preview.tex"
        tex_path.write_text(wrap_standalone_document(self.code_editor.code()), encoding="utf-8")
        return tex_path

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
