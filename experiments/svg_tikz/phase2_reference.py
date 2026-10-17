"""Additional fixed-grid renderer/reference controls; never changes acceptance thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

from fikzpy.core.latex_compiler import resolve_latex_executable
from .phase2 import BASELINE, difference_panel, evaluate, snapshot
from .phase2_geometry import refit_svg
from .run import _image
from .validation import compare_images, render_svg, white_rgb


def pdf_at_grid(pdf: Path, output: Path, width: int, height: int, poppler: str) -> np.ndarray:
    """Rasterize a fixed PDF frame onto the SVG grid without image registration."""
    completed = subprocess.run([poppler, "-f", "1", "-l", "1", "-singlefile", "-png", "-scale-to-x", str(width),
                                "-scale-to-y", str(height), str(pdf), str(output)],
                               capture_output=True, text=True, timeout=90)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    return _image(output.with_suffix(".png"))


def svg_at_fractional_frame(svg: str, canvas_size: tuple[int, int], target_size: tuple[float, float]) -> np.ndarray:
    """Use the PDF MediaBox/DPI target extent; do not stretch to the rounded PNG canvas."""
    width, height = canvas_size
    tw, th = target_size
    if not 0 < tw <= width or not 0 < th <= height:
        raise ValueError("Target frame must fit the positive canvas.")
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    try:
        QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, tw, th))
    finally:
        painter.end()
    return np.frombuffer(image.bits(), np.uint8).reshape(height, width, 4).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / ".gitattributes").write_text("*.pdf binary\n*.png binary\n*compile*.txt -whitespace\n", encoding="utf-8")
    before_hashes = snapshot(BASELINE)
    poppler = shutil.which("pdftoppm")
    if poppler is None:
        raise RuntimeError("Poppler required for the reference controls.")
    results = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fixed_grid": {}, "stricter_curve_sweep": {}}
    for name in ("circle", "ellipse", "smooth_curve"):
        folder = destination / name
        folder.mkdir()
        source = (BASELINE / name / "before.svg").read_text(encoding="utf-8")
        candidate = (BASELINE / name / "candidate.svg").read_text(encoding="utf-8")
        controls = []
        for scale in (4, 16):
            a, b = white_rgb(render_svg(source, scale)), white_rgb(render_svg(candidate, scale))
            height, width = a.shape[:2]
            aa = pdf_at_grid(BASELINE / name / "before.pdf", folder / f"pdf_before_{scale}x", width, height, poppler)
            bb = pdf_at_grid(BASELINE / name / "candidate.pdf", folder / f"pdf_candidate_{scale}x", width, height, poppler)
            controls.append({"scale": scale, "width": width, "height": height,
                "qt_before_vs_pdf_before": compare_images(a, aa), "qt_candidate_vs_pdf_candidate": compare_images(b, bb),
                "pdf_before_vs_pdf_candidate": compare_images(aa, bb), "qt_before_vs_qt_candidate": compare_images(a, b)})
            if scale == 4:
                difference_panel(b, bb, folder / "qt_vs_pdf_candidate.png", name + ": same candidate SVG/Qt vs TikZ/PDF, fixed 4x grid")
        pdf = _image(BASELINE / name / "before.png")
        legacy = cv2.resize(_image(BASELINE / name / "source.png"), (pdf.shape[1], pdf.shape[0]), interpolation=cv2.INTER_AREA)
        direct = white_rgb(render_svg(source, (pdf.shape[1] - .01) / 200))
        if direct.shape != pdf.shape:
            raise RuntimeError("Direct SVG reference grid mismatch.")
        results["fixed_grid"][name] = {"scale_controls": controls,
            "legacy_resampled_reference_vs_baseline_pdf": compare_images(legacy, pdf),
            "direct_svg_reference_vs_baseline_pdf": compare_images(direct, pdf),
            "reference_note": "These reference controls do not replace the original acceptance protocol."}
        pdfinfo = shutil.which("pdfinfo")
        if pdfinfo:
            metadata = subprocess.check_output([pdfinfo, str(BASELINE / name / "before.pdf")], text=True)
            match = re.search(r"Page size:\s*([\d.]+) x ([\d.]+) pts", metadata)
            if match is None:
                raise RuntimeError("Cannot read PDF physical frame.")
            target = (float(match[1]) * 180 / 72, float(match[2]) * 180 / 72)
            fractional = white_rgb(svg_at_fractional_frame(source, (pdf.shape[1], pdf.shape[0]), target))
            results["fixed_grid"][name]["pdf_media_box_target_pixels"] = list(target)
            results["fixed_grid"][name]["png_canvas_pixels"] = [pdf.shape[1], pdf.shape[0]]
            results["fixed_grid"][name]["physical_frame_svg_reference_vs_pdf"] = compare_images(fractional, pdf)
            difference_panel(fractional, pdf, folder / "physical_frame_qt_vs_pdf.png", name + ": same physical MediaBox frame / 180 dpi")
        difference_panel(legacy, direct, folder / "legacy_reference_vs_direct.png", name + ": resampled legacy SVG raster vs direct same-grid SVG render")
    curve = destination / "curve_sweep"
    curve.mkdir()
    for name in ("before.svg", "before.pdf", "before.png"):
        shutil.copyfile(BASELINE / "smooth_curve" / name, curve / name)
    source = (curve / "before.svg").read_text(encoding="utf-8")
    reference = _image(BASELINE / "smooth_curve" / "source.png")
    try:
        engine = str(resolve_latex_executable())
    except FileNotFoundError:
        engine = None
    for epsilon in (.005, .002, .001):
        candidate, diagnostics = refit_svg(source, epsilon, include_curves=False)
        results["stricter_curve_sweep"][str(epsilon)] = evaluate(source, candidate, curve / f"epsilon_{epsilon}",
            diagnostics, reference, engine, poppler)
    assert before_hashes == snapshot(BASELINE), "Phase-one artifacts changed"
    (destination / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Reference controls:", destination / "results.json")


if __name__ == "__main__":
    main()
