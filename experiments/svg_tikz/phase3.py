"""Reproducible pre-SVG reconstruction experiment, with unchanged official tracing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

from fikzpy.core.image_processor import ProcessingSettings, visual_settings_from_processing
from fikzpy.core.latex_compiler import compile_latex_document, resolve_latex_executable
from fikzpy.core.visual_pipeline import contours_to_svg
from .phase2 import BASELINE, difference_panel, snapshot
from .phase3_metrics import geometry_metrics, physical_frame, raster_at_frame, region_identity, rings, svg_at_frame
from .reconstruction import capture_contours, reconstruct_svg, ReconstructionConfig
from .run import ROOT, _image, _render_pdf, convert, framed_document, svg_stats, tikz_stats
from .validation import compare_images, passes_visual_gate, render_svg, white_rgb

CASES = {"original_real": "tests/25.jpg", "cara_existing_svg": "tests/Cara.png",
         "visual_line_art_bw": None, "visual_noisy_grayscale": None}


def compile_svg(svg, folder, stem, engine, poppler):
    code, seconds = convert(svg)
    (folder / f"{stem}.svg").write_text(svg, encoding="utf-8")
    (folder / f"{stem}.tex").write_text(framed_document(code, svg), encoding="utf-8")
    started = perf_counter()
    compiled = compile_latex_document(folder / f"{stem}.tex", manual_path=engine)
    (folder / f"{stem}_compile.txt").write_text(compiled.output, encoding="utf-8")
    if compiled.returncode != 0 or not compiled.pdf_path.is_file():
        raise RuntimeError(f"Compilation failed: {folder / stem}")
    png = _render_pdf(compiled.pdf_path, poppler)
    return {**svg_stats(svg), **tikz_stats(code), "conversion_seconds": seconds,
            "compile_render_seconds": perf_counter() - started}, _image(png)


def evaluate(source, candidate, raw, geometry, folder, before_pdf, original, canvas, target, engine, poppler):
    folder.mkdir()
    candidate_stats, candidate_pdf = compile_svg(candidate, folder, "candidate", engine, poppler)
    svg_a, svg_b = render_svg(source), render_svg(candidate)
    pdf_metrics = compare_images(before_pdf, candidate_pdf)
    svg_metrics = compare_images(svg_a, svg_b, rgba=True)
    pdf_identity = region_identity(before_pdf, candidate_pdf)
    svg_identity = region_identity(white_rgb(svg_a), white_rgb(svg_b))
    ref_before, ref_after = compare_images(original, before_pdf), compare_images(original, candidate_pdf)
    reasons = []
    for name, metrics in (("svg", svg_metrics), ("pdf", pdf_metrics)):
        if not passes_visual_gate(metrics):
            reasons.append(name + "_fidelity_or_topology")
    if not all([*pdf_identity.values(), *svg_identity.values(), geometry["winding_equal"], geometry["closed_equal"]]):
        reasons.append("region_identity_or_structure")
    if geometry["maximum_error_bound"] > .25:
        reasons.append("continuous_geometric_bound")
    if (ref_after["foreground_iou"] < ref_before["foreground_iou"] - .001
            or ref_after["rgb_mae_255"] > ref_before["rgb_mae_255"] + .1):
        reasons.append("original_raster_regression")
    source_stats = {**svg_stats(source), **tikz_stats(convert(source)[0])}
    if (candidate_stats["coordinate_occurrences"] >= source_stats["coordinate_occurrences"]
            or candidate_stats["tikz_bytes"] >= source_stats["tikz_bytes"]):
        reasons.append("not_smaller_in_both_metrics")
    status = "rejected" if reasons else ("accepted_geometry" if geometry["changed_rings"] else "serialization_only")
    result = {"status": status, "rejection_reasons": reasons, "candidate": candidate_stats,
        "svg_metrics": svg_metrics, "pdf_metrics": pdf_metrics,
        "svg_region_identity": svg_identity, "pdf_region_identity": pdf_identity,
        "original_to_baseline": ref_before, "original_to_candidate": ref_after,
        "calibrated_qt_to_pdf_candidate": compare_images(svg_at_frame(candidate, canvas, target), candidate_pdf),
        "geometry": geometry,
        "coordinate_reduction_percent": 100 * (1 - candidate_stats["coordinate_occurrences"] / source_stats["coordinate_occurrences"]),
        "byte_reduction_percent": 100 * (1 - candidate_stats["tikz_bytes"] / source_stats["tikz_bytes"]),
        "segment_reduction_percent": 100 * (1 - candidate_stats["segments"] / source_stats["segments"])}
    difference_panel(before_pdf, candidate_pdf, folder / "comparison.png", folder.parent.name + " / " + folder.name + " / PDF 180 dpi")
    difference_panel(original, candidate_pdf, folder / "raster_comparison.png", "Original raster / candidate PDF / physical frame")
    (folder / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitattributes").write_text("*.pdf binary\n*.png binary\n*.npz binary\n*compile*.txt -whitespace\n", encoding="utf-8")
    settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
    baseline_hashes = snapshot(BASELINE)
    phase2_hashes = snapshot(Path(__file__).parent / "phase2_results")
    engine = str(resolve_latex_executable())
    poppler, pdfinfo = shutil.which("pdftoppm"), shutil.which("pdfinfo")
    if not poppler or not pdfinfo:
        raise RuntimeError("Poppler is required for calibrated PDF validation.")
    environment = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "settings": asdict(settings),
        "latex": engine, "pdftoppm": poppler, "pdfinfo": pdfinfo,
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "baseline_hashes": baseline_hashes, "phase2_hashes": phase2_hashes,
        "criteria": {"iou": .99, "mae_255": .5, "original_iou_drop": .001, "original_mae_increase": .1,
                     "geometric_error": .25, "svg_scale": 4, "pdf_dpi": 180,
                     "raster_mapping": "pixel edges / fractional MediaBox frame / linear interpolation"}}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    records = {}
    for name, relative in CASES.items():
        print(name, flush=True)
        folder = output / name
        folder.mkdir()
        image = cv2.imread(str(ROOT / relative if relative else BASELINE / name / "source.png"))
        if image is None:
            raise RuntimeError("Missing input: " + name)
        started = perf_counter()
        contours, raw, mask = capture_contours(image, settings)
        tracing_seconds = perf_counter() - started
        source = contours_to_svg(contours, image.shape, settings=settings)
        historical = (BASELINE / name / "before.svg").read_text(encoding="utf-8")
        if source != historical:
            # Formatting is not evidence of changed geometry.
            assert next(e for e in ET.fromstring(source).iter() if e.tag.endswith("path")).get("d") == next(
                e for e in ET.fromstring(historical).iter() if e.tag.endswith("path")).get("d")
        record = {"raw_points": sum(map(len, raw)), "rdp_points": sum(len(c.points) for c in contours),
                  "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(), "candidates": {},
                  "source_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "tracing_seconds": tracing_seconds}
        cv2.imwrite(str(folder / "source.png"), image)
        cv2.imwrite(str(folder / "mask.png"), mask)
        np.savez_compressed(folder / "contours.npz", **{f"raw_{i}": p for i, p in enumerate(raw)},
                            **{f"rdp_{i}": c.points for i, c in enumerate(contours)})
        before, before_pdf = compile_svg(source, folder, "before", engine, poppler)
        record["before"] = before
        canvas, target = physical_frame(folder / "before.pdf", before_pdf, pdfinfo)
        original = raster_at_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), canvas, target)
        Image.fromarray(original).save(folder / "original_at_frame.png")
        record["physical_frame"] = {"canvas": canvas, "target": target}
        reference_geometry = geometry_metrics(source, source, raw)
        raw_bounds = [x["raw_to_baseline_bound"] for x in reference_geometry["rings"]]
        record["baseline_geometry"] = reference_geometry
        record["calibrated_qt_to_pdf_baseline"] = compare_images(svg_at_frame(source, canvas, target), before_pdf)
        for strategy in ("A", "B"):
            config = ReconstructionConfig(strategy=strategy)
            started = perf_counter()
            candidate, details, proposal = reconstruct_svg(source, raw, config)
            fitting_seconds = perf_counter() - started
            variants = {}
            for variant, svg in (("proposal", proposal), ("guarded", candidate)):
                geometry = geometry_metrics(source, svg, raw, raw_bounds)
                metrics = evaluate(source, svg, raw, geometry, folder / f"{strategy}_{variant}", before_pdf,
                                   original, canvas, target, engine, poppler)
                variants[variant] = metrics
                print(strategy, variant, metrics["status"], metrics["candidate"]["coordinate_occurrences"],
                      metrics["candidate"]["segments"], metrics["pdf_metrics"]["foreground_iou"], flush=True)
            metrics = {"config": asdict(config), "fitting_seconds": fitting_seconds, "details": details, "variants": variants}
            record["candidates"][strategy] = metrics
        records[name] = record
        (folder / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if snapshot(BASELINE) != baseline_hashes or snapshot(Path(__file__).parent / "phase2_results") != phase2_hashes:
        raise RuntimeError("Historical evidence changed unexpectedly.")
    (output / "historical_hashes_verified.txt").write_text("Phase 1 and Phase 2 files unchanged.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
