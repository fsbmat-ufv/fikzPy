"""Reproduce phase-two renderer controls, tracing diagnostics and opt-in candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen

from fikzpy.core.image_processor import ProcessingSettings, visual_settings_from_processing
from fikzpy.core.latex_compiler import compile_latex_document, resolve_latex_executable
from fikzpy.core.visual_pipeline import _build_high_resolution_mask, _resize_for_trace
from fikzpy.core.visual_pipeline import contours_to_svg, trace_visual_contours
from fikzpy.core.tikz_generator import TikzOptions

from .optimizer import OptimizationConfig, _dense_polyline, _serialize, _from_points, bounded_error, flatten, parse_paths
from .phase2_geometry import inventory, minimum_points_probe, recognition_reasons, refit_svg
from .phase2_geometry import subdivide_svg, svg_error_bound, tracing_provenance
from .run import ROOT, _image, _render_pdf, convert, framed_document, svg_stats, tikz_stats
from .validation import compare_images, passes_visual_gate, render_svg, viewport, white_rgb

BASELINE = ROOT / "experiments/svg_tikz/runs/final-v2"
CASES = ("circle", "ellipse", "smooth_curve", "original_real", "visual_noisy_grayscale",
         "cara_existing_svg", "visual_line_art_bw", "visual_geometric_diagram", "example_input_real")


def snapshot(directory: Path) -> dict:
    """Hash immutable input artifacts so the run can prove it did not overwrite them."""
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def coverage(rgba: np.ndarray, scale: float) -> dict:
    """Measure integrated alpha coverage, not only a thresholded support mask."""
    alpha = rgba[:, :, 3].astype(float) / 255
    yy, xx = np.indices(alpha.shape)
    mass = max(alpha.sum(), 1e-12)
    mask = alpha >= .5
    y, x = np.where(mask)
    return {"area_user_units2": float(mass / scale ** 2),
            "centroid_user_units": [float((xx * alpha).sum() / mass / scale), float((yy * alpha).sum() / mass / scale)],
            "bbox_pixels": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None}


def stroker_circle(threshold: float | None, prescale: float = 1.0) -> np.ndarray:
    """Controlled QPainter reproduction of QtSvg's curved-stroke area discrepancy."""
    path = QPainterPath()
    path.addEllipse(QRectF(55 * prescale, 25 * prescale, 90 * prescale, 90 * prescale))
    image = QImage(1600, 1120, QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(8 / prescale, 8 / prescale)
    try:
        if threshold is None:
            painter.setPen(QPen(QColor("black"), 2 * prescale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        else:
            stroker = QPainterPathStroker()
            stroker.setWidth(2 * prescale)
            stroker.setCurveThreshold(threshold)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("black"))
            painter.drawPath(stroker.createStroke(path))
    finally:
        painter.end()
    return np.frombuffer(image.bits(), np.uint8).reshape(1120, 1600, 4).copy()


def difference_panel(before: np.ndarray, after: np.ndarray, destination: Path, title: str):
    """Save original frames, a signed difference map and the same enlarged ROI."""
    if before.shape != after.shape:
        raise ValueError("Difference-map frames must agree.")
    delta = np.max(np.abs(before.astype(float) - after.astype(float)), axis=2)
    a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(float)
    b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(float)
    signed = np.full_like(before, 255)
    signed[(a < b) & (delta > 0)] = [210, 25, 45]
    signed[(a > b) & (delta > 0)] = [0, 130, 180]
    Image.fromarray(signed).save(destination.with_name(destination.stem + "_difference.png"))
    heat = cv2.applyColorMap(np.clip(delta * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(destination.with_name(destination.stem + "_heatmap.png"))
    box = cv2.boxFilter(delta, -1, (64, 64), normalize=False)
    cy, cx = np.unravel_index(np.argmax(box), box.shape)
    x = int(np.clip(cx - 48, 0, max(0, before.shape[1] - 96)))
    y = int(np.clip(cy - 48, 0, max(0, before.shape[0] - 96)))
    canvas = Image.new("RGB", (1560, 820), "white")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((15, 8), title + " | red=baseline darker, blue=candidate darker; heatmap gain x8", fill="black")
    for index, (label, array) in enumerate(zip(("Baseline", "Candidate", "Signed differences"), (before, after, signed))):
        image = Image.fromarray(array)
        image.thumbnail((500, 350))
        canvas.paste(image, (index * 520 + (520 - image.width) // 2, 45))
        drawing.text((index * 520 + 10, 28), label, fill="black")
        roi = Image.fromarray(array[y:y + 96, x:x + 96]).resize((384, 384), Image.Resampling.NEAREST)
        canvas.paste(roi, (index * 520 + 60, 425))
    drawing.text((15, 402), f"Same 96x96 pixel ROI enlarged 4x, top-left=({x},{y}); no feature registration", fill="black")
    canvas.save(destination)


def renderer_diagnosis(name: str, folder: Path) -> dict:
    """Disentangle geometric changes, support-threshold sensitivity and Qt stroking."""
    source = (BASELINE / name / "before.svg").read_text(encoding="utf-8")
    proposed = (BASELINE / name / "candidate.svg").read_text(encoding="utf-8")
    metrics = {"scale_sweep": [], "join_cap_controls": []}
    for scale in (1, 2, 4, 8, 16):
        a, b = render_svg(source, scale), render_svg(proposed, scale)
        row = {"scale": scale, **compare_images(a, b, rgba=True),
               "before_coverage": coverage(a, scale), "candidate_coverage": coverage(b, scale), "threshold_iou": {}}
        aa, bb = white_rgb(a), white_rgb(b)
        for threshold in (64, 128, 200, 245):
            x, y = aa.mean(axis=2) < threshold, bb.mean(axis=2) < threshold
            row["threshold_iou"][str(threshold)] = float(np.count_nonzero(x & y) / max(1, np.count_nonzero(x | y)))
        metrics["scale_sweep"].append(row)
        if scale in (4, 16):
            difference_panel(aa, bb, folder / f"qt_scale_{scale}.png", f"{name}: QtSvg {scale}x")
    a, b = render_svg(source, 8), render_svg(proposed, 8)
    root = ET.fromstring(proposed)
    paths = [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "path"]
    for join in ("miter", "round", "bevel"):
        for cap in ("butt", "round"):
            aa = source.replace('stroke-width="2"', f'stroke-width="2" stroke-linejoin="{join}" stroke-linecap="{cap}"')
            bb = proposed.replace('stroke-width="2"', f'stroke-width="2" stroke-linejoin="{join}" stroke-linecap="{cap}"')
            metrics["join_cap_controls"].append({"join": join, "cap": cap,
                **compare_images(render_svg(aa, 4), render_svg(bb, 4), rgba=True)})
    # Hold the parsed cubic geometry fixed and change only its representation.
    flat_root = ET.fromstring(proposed)
    for path in flat_root.iter():
        if path.tag.rsplit("}", 1)[-1] == "path":
            path.set("d", " ".join(_serialize(sp, _from_points(flatten(sp.segments, .001)))
                                      for sp in parse_paths(path.get("d"))))
    flat = ET.tostring(flat_root, encoding="unicode")
    (folder / "candidate_flattened_control.svg").write_text(flat, encoding="utf-8")
    metrics["candidate_vs_its_flattened_geometry"] = compare_images(b, render_svg(flat, 8), rgba=True)
    metrics["baseline_vs_flattened_candidate"] = compare_images(a, render_svg(flat, 8), rgba=True)
    metrics["candidate_flattening_bound_user_units"] = .001
    metrics["centerline_error_bound"] = max(bounded_error(flatten(x.segments, .0005), flatten(y.segments, .0005),
        OptimizationConfig(tolerance=.25, sample_step=.005), .0005)
        for x, y in zip(parse_paths(ET.fromstring(source)[0].get("d")), parse_paths(paths[0].get("d")), strict=True))
    if name in {"circle", "ellipse"}:
        rx, ry = (45, 45) if name == "circle" else (65, 30)
        native = source[:source.index("<path")] + f'<ellipse cx="100" cy="70" rx="{rx}" ry="{ry}" fill="none" stroke="black" stroke-width="2"/></svg>'
        metrics["native_ellipse_vs_arc"] = compare_images(render_svg(native, 8), b, rgba=True)
        metrics["polygon_to_analytic_ellipse_bound"] = rx * (2 * np.pi / 240) ** 2 / 8
        metrics["native_ellipse_coverage"] = coverage(render_svg(native, 8), 8)
        # Bypass stroke generation with an analytic filled annulus (circle only).
        if name == "circle":
            band = source[:source.index("<path")] + ('<path fill-rule="evenodd" d="M146 70A46 46 0 1 1 54 70A46 46 0 1 1 146 70Z '
                'M144 70A44 44 0 1 1 56 70A44 44 0 1 1 144 70Z"/></svg>')
            metrics["analytic_stroke_area"] = float(2 * np.pi * 45 * 2)
            metrics["filled_annulus_coverage"] = coverage(render_svg(band, 8), 8)
            metrics["stroker_threshold_controls"] = [{"threshold": t, **coverage(stroker_circle(t), 8)} for t in (.25, .1, .01, .001)]
            metrics["painter_coordinate_scale_controls"] = [{"prescale": k, **coverage(stroker_circle(None, k), 8)} for k in (1, 10, 100)]
            difference_panel(white_rgb(stroker_circle(.25)), white_rgb(stroker_circle(.001)),
                             folder / "stroker_threshold.png", "Same cubic circle: Qt stroke threshold 0.25 vs 0.001")
    return metrics


def _topology_structure(source: str, candidate: str) -> dict:
    def groups(svg):
        return [s for e in ET.fromstring(svg).iter() if e.tag.rsplit("}", 1)[-1] == "path" for s in parse_paths(e.get("d", ""))]
    old, new = groups(source), groups(candidate)
    return {"subpath_count_equal": len(old) == len(new), "closed_flags_equal": [s.closed for s in old] == [s.closed for s in new]}


def evaluate(source: str, candidate: str, folder: Path, details: dict, original_rgb: np.ndarray | None,
             engine: str | None, poppler: str | None) -> dict:
    """Retain every candidate and apply the original thresholds without relaxing them."""
    folder.mkdir()
    started = perf_counter()
    (folder / "candidate.svg").write_text(candidate, encoding="utf-8")
    before_code, _ = convert(source)
    code, conversion_time = convert(candidate)
    record = {"before": {**tikz_stats(before_code), **svg_stats(source)},
              "candidate": {**tikz_stats(code), **svg_stats(candidate)},
              "details": details, "conversion_seconds": conversion_time,
              "svg_metrics": compare_images(render_svg(source), render_svg(candidate), rgba=True),
              "structure": _topology_structure(source, candidate), "status": "unverified_pdf", "rejection_reasons": []}
    (folder / "candidate.tex").write_text(framed_document(code, source), encoding="utf-8")
    reasons = record["rejection_reasons"]
    if not passes_visual_gate(record["svg_metrics"]):
        reasons.append("svg_visual_or_topology")
    if not all(record["structure"].values()):
        reasons.append("subpath_structure")
    if details.get("geometric_bound", details.get("maximum_error_bound", 0.0)) > .25:
        reasons.append("geometric_error_bound")
    smaller = (record["candidate"]["coordinate_occurrences"] < record["before"]["coordinate_occurrences"]
               and record["candidate"]["tikz_bytes"] < record["before"]["tikz_bytes"])
    if not smaller:
        reasons.append("not_smaller_in_both_metrics")
    if engine and poppler:
        compile_started = perf_counter()
        compiled = compile_latex_document(folder / "candidate.tex", manual_path=engine)
        (folder / "compile.txt").write_text(compiled.output, encoding="utf-8")
        if compiled.returncode != 0 or not compiled.pdf_path.exists():
            record["status"] = "rejected"
            reasons.append("latex_compilation")
        else:
            png = _render_pdf(compiled.pdf_path, poppler)
            record["compile_and_render_seconds"] = perf_counter() - compile_started
            baseline_png = folder.parent / "before.png"
            aa, bb = _image(baseline_png), _image(png)
            record["pdf_metrics"] = compare_images(aa, bb)
            if not passes_visual_gate(record["pdf_metrics"]):
                reasons.append("pdf_visual_or_topology")
            if original_rgb is not None:
                reference = cv2.resize(original_rgb, (bb.shape[1], bb.shape[0]), interpolation=cv2.INTER_AREA)
                record["original_to_baseline"] = compare_images(reference, aa)
                record["original_to_candidate"] = compare_images(reference, bb)
                if (record["original_to_candidate"]["foreground_iou"] < record["original_to_baseline"]["foreground_iou"] - .001
                    or record["original_to_candidate"]["rgb_mae_255"] > record["original_to_baseline"]["rgb_mae_255"] + .1):
                    reasons.append("original_raster_regression")
            record["status"] = "accepted" if not reasons else "rejected"
            difference_panel(aa, bb, folder / "comparison.png", folder.parent.name + " / " + folder.name + ": PDF 180 dpi")
    record["coordinate_reduction_percent"] = 100 * (1 - record["candidate"]["coordinate_occurrences"] / max(1, record["before"]["coordinate_occurrences"]))
    record["byte_reduction_percent"] = 100 * (1 - record["candidate"]["tikz_bytes"] / max(1, record["before"]["tikz_bytes"]))
    record["seconds"] = perf_counter() - started
    (folder / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    # Attributes apply only inside this new evidence bundle.
    (destination / ".gitattributes").write_text("*.pdf binary\n*.png binary\n*compile*.txt -whitespace\n", encoding="utf-8")
    inputs = snapshot(BASELINE)
    environment = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "python": sys.version, "platform": platform.platform(), "baseline_hashes": inputs,
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("phase2*.py")},
        "criteria": {"svg_scale": 4, "pdf_dpi": 180, "minimum_iou": .99, "maximum_rgb_mae_255": .5,
                     "maximum_original_iou_drop": .001, "maximum_original_mae_increase": .1}}
    (destination / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    try:
        engine = str(resolve_latex_executable())
    except FileNotFoundError:
        engine = None
    poppler = shutil.which("pdftoppm")
    results = {}
    for name in CASES:
        print("Phase 2:", name, flush=True)
        folder = destination / name
        folder.mkdir()
        for filename in ("before.svg", "before.pdf", "before.png"):
            shutil.copyfile(BASELINE / name / filename, folder / filename)
        source = (folder / "before.svg").read_text(encoding="utf-8")
        original_rgb = _image(BASELINE / name / "source.png")
        record = {"inventory": inventory(source), "recognition": recognition_reasons(source), "candidates": {}}
        if name in {"circle", "ellipse", "smooth_curve"}:
            record["renderer_diagnosis"] = renderer_diagnosis(name, folder)
            previous = (BASELINE / name / "candidate.svg").read_text(encoding="utf-8")
            for level in ((1, 2) if name != "smooth_curve" else (3,)):
                candidate = subdivide_svg(previous, level)
                record["candidates"][f"subdivide_{level}"] = evaluate(source, candidate, folder / f"subdivide_{level}",
                    {"exact_cubic_subdivision_levels": level, "arc_to_cubic_is_approximate": name != "smooth_curve",
                     "geometric_bound": record["renderer_diagnosis"]["centerline_error_bound"]}, original_rgb, engine, poppler)
            if name == "smooth_curve":
                for epsilon in (.02, .01):
                    variant = f"tight_rdp_{epsilon}"
                    candidate, diagnostics = refit_svg(source, epsilon, include_curves=False)
                    record["candidates"][variant] = evaluate(source, candidate, folder / variant, diagnostics, original_rgb, engine, poppler)
        else:
            settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
            if name == "cara_existing_svg":
                image = cv2.imread(str(ROOT / "tests/Cara.png"))
                original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
            traced = trace_visual_contours(image, settings)
            regenerated = contours_to_svg(traced.contours, image.shape, options=TikzOptions(), settings=settings)
            expected = [e.get("d") for e in ET.fromstring(source).iter() if e.tag.rsplit("}", 1)[-1] == "path"]
            actual = [e.get("d") for e in ET.fromstring(regenerated).iter() if e.tag.rsplit("}", 1)[-1] == "path"]
            record["tracing_path_data_matches_baseline"] = expected == actual
            if expected != actual:
                raise RuntimeError(f"Tracing provenance does not reproduce {name}; refuse causal attribution.")
            mask = _build_high_resolution_mask(_resize_for_trace(image, settings.upsample_factor), settings, settings.upsample_factor)
            record["tracing"] = tracing_provenance(image, traced, settings, mask)
            cv2.imwrite(str(folder / "tracing_mask.png"), traced.ink_mask)
            cv2.imwrite(str(folder / "original_raster.png"), image)
            if name == "visual_noisy_grayscale":
                from scripts.generate_classic_baseline import _noisy_grayscale
                class ZeroNoise:
                    def normal(self, loc, scale, size):
                        return np.zeros(size)
                # Ablate only generated noise; all original drawing commands and
                # tracing parameters remain the same. This is NOT an optimizer output.
                with patch("scripts.generate_classic_baseline.np.random.default_rng", return_value=ZeroNoise()):
                    noiseless = _noisy_grayscale()
                clean_trace = trace_visual_contours(noiseless, settings)
                clean_mask = _build_high_resolution_mask(_resize_for_trace(noiseless, settings.upsample_factor), settings, settings.upsample_factor)
                record["noise_ablation_control"] = tracing_provenance(noiseless, clean_trace, settings, clean_mask)
                cv2.imwrite(str(folder / "noiseless_control.png"), noiseless)
                cv2.imwrite(str(folder / "noiseless_tracing_mask.png"), clean_trace.ink_mask)
            candidate = minimum_points_probe(traced.contours, image.shape, settings)
            details = {"minimum_points_for_bezier": 4, "same_preprocessed_contours": True,
                       "geometric_bound": svg_error_bound(source, candidate)}
            record["candidates"]["tracer_minimum4"] = evaluate(source, candidate, folder / "tracer_minimum4", details, original_rgb, engine, poppler)
            # Micro-components are counted as candidates for inspection, NEVER deleted.
            for epsilon in (.02, .05):
                name_variant = f"mixed_refit_{epsilon}"
                started = perf_counter()
                candidate, diagnostics = refit_svg(source, epsilon, include_curves=True)
                diagnostics["optimization_seconds"] = perf_counter() - started
                record["candidates"][name_variant] = evaluate(source, candidate, folder / name_variant, diagnostics, original_rgb, engine, poppler)
        accepted = {k: v for k, v in record["candidates"].items() if v["status"] == "accepted"}
        record["best_accepted"] = min(accepted, key=lambda k: accepted[k]["candidate"]["tikz_bytes"]) if accepted else None
        (folder / "diagnostics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        results[name] = record
        (destination / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if snapshot(BASELINE) != inputs:
        raise RuntimeError("Phase-one baseline was modified.")
    lines = ["# Phase 2 reproducible diagnostics", "", "Original phase-one criteria retained; no official pipeline changes.", "",
        "| Case / candidate | Status | Coordinates | Reduction | TikZ bytes | Qt IoU | PDF IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in results.items():
        for variant, value in record["candidates"].items():
            lines.append(f"| [{name}/{variant}]({name}/{variant}/comparison.png) | {value['status']} | "
                f"{value['before']['coordinate_occurrences']} → {value['candidate']['coordinate_occurrences']} | "
                f"{value['coordinate_reduction_percent']:.2f}% | {value['before']['tikz_bytes']} → {value['candidate']['tikz_bytes']} | "
                f"{value['svg_metrics']['foreground_iou']:.6f} | {value.get('pdf_metrics', {}).get('foreground_iou', float('nan')):.6f} |")
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Report:", destination / "README.md")


if __name__ == "__main__":
    main()
