"""Frozen application launcher; source-mode startup remains unchanged."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


def configure_runtime() -> Path:
    """Expose the private TeX distribution only to this application process."""
    app_dir = Path(sys.executable).resolve().parent
    latex_bin = app_dir / "latex" / "bin" / "windows"
    if latex_bin.is_dir():
        os.environ["PATH"] = str(latex_bin) + os.pathsep + os.environ.get("PATH", "")
    return app_dir


def self_test(report_path: Path) -> int:
    """Exercise the installed GUI, all public modes, and the bundled PDF engine."""
    report: dict = {"ok": False, "modes": []}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
        from PySide6.QtWidgets import QApplication

        from fikzpy.core.latex_compiler import compile_tikz_to_pdf, resolve_latex_executable
        from fikzpy.core.tikz_generator import TikzOptions
        from fikzpy.core.tikz_pipeline import build_tikz_from_image
        from fikzpy.gui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        image = np.full((160, 160, 3), 255, dtype=np.uint8)
        cv2.circle(image, (80, 80), 55, (0, 0, 0), 3)
        expected = ["Classic Auto", "Classic Filled", "Visual", "Contornos", "Ponto a Ponto"]
        combo = window.vectorization_mode_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        if labels != expected:
            raise RuntimeError(f"Unexpected public modes: {labels}")
        engine = resolve_latex_executable()
        private_engine = Path(sys.executable).resolve().parent / "latex" / "bin" / "windows" / "pdflatex.exe"
        if engine != private_engine:
            raise RuntimeError(f"Expected private LaTeX engine, got {engine}")
        report["latex"] = str(engine)
        for index, label in enumerate(labels):
            combo.setCurrentIndex(index)
            settings = window.processing_settings()
            options = TikzOptions(classic_strategy=combo.itemData(index)[1])
            result = build_tikz_from_image(image, settings, options)
            if not result.tikz_code.strip():
                raise RuntimeError(f"Empty TikZ in {label}")
            if label == "Visual" and not result.visual_stats.used_svg2tikz:
                raise RuntimeError("Visual unexpectedly used the svg2tikz fallback")
            compiled = compile_tikz_to_pdf(result.tikz_code, report_path.parent / f"mode_{index}.tex")
            (report_path.parent / f"mode_{index}_compile.txt").write_text(compiled.output, encoding="utf-8")
            if compiled.returncode != 0 or not compiled.pdf_path.is_file() or compiled.pdf_path.stat().st_size == 0:
                raise RuntimeError(f"PDF compilation failed in {label}: {compiled.output}")
            report["modes"].append({"name": label, "pdf_bytes": compiled.pdf_path.stat().st_size})
        window.close()
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["ok"] else 1


def main() -> int:
    """Run the regular application or the explicit packaging validation command."""
    configure_runtime()
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return self_test(Path(sys.argv[2]).resolve())
    from fikzpy.main import main as run_application

    return run_application()


if __name__ == "__main__":
    raise SystemExit(main())
