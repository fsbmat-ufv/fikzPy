"""Code editor widget for TikZ output."""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit


class CodeEditor(QPlainTextEdit):
    """A lightweight monospaced TikZ editor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setPlaceholderText("O codigo TikZ gerado aparecera aqui.")

    def set_code(self, code: str) -> None:
        """Replace the editor contents."""
        self.setPlainText(code)

    def code(self) -> str:
        """Return the current editor contents."""
        return self.toPlainText()
