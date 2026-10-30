"""Run the opt-in experiment and write a new, non-overwriting comparison bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fikzpy.core.image_processor import ProcessingSettings, visual_settings_from_processing
from fikzpy.core.latex_compiler import compile_latex_document, resolve_latex_executable
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.visual_pipeline import _svg_to_tikz_picture, contours_to_svg, trace_visual_contours
from scripts.generate_classic_baseline import BASELINE_EXAMPLES

from .optimizer import OptimizationConfig, optimize_svg, parse_paths
from .validation import compare_images, passes_visual_gate, render_svg, viewport, white_rgb

ROOT = Path(__file__).resolve().parents[2]


def svg_document(body: str, width: int = 200, height: int = 140) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="12cm" height="{12 * height / width}cm" '
            f'viewBox="0 0 {width} {height}">{body}</svg>')


def _polyline(points) -> str:
    return "M " + " L ".join(f"{x:.8f} {y:.8f}" for x, y in points)


def fixtures():
    """Yield native SVG fixtures and immutable repository input references."""
    line = _polyline([(x, 30 + x * .15) for x in range(10, 190)])
    yield "redundant_line", svg_document(f'<path d="{line}" fill="none" stroke="black" stroke-width="2"/>'), None
    t = np.linspace(0, 2 * np.pi, 241)[:-1]
    for name, rx, ry in (("circle", 45, 45), ("ellipse", 65, 30)):
        points = np.column_stack([100 + rx * np.cos(t), 70 + ry * np.sin(t)])
        yield name, svg_document(f'<path d="{_polyline(points)} Z" fill="none" stroke="black" stroke-width="2"/>'), None
    rings = []
    for rx, ry in ((60, 45), (30, 20), (2, 2)):
        rings.append(_polyline(np.column_stack([100 + rx * np.cos(t), 70 + ry * np.sin(t)])) + " Z")
    yield "nested_holes_small_detail", svg_document(f'<path d="{" ".join(rings)}" fill="#2458a0" fill-rule="evenodd"/>'), None
    points = [(x, 65 + 22 * np.sin(x / 20)) for x in np.linspace(10, 190, 401)]
    curve = _polyline(points)
    yield "smooth_curve", svg_document(f'<path d="{curve}" fill="none" stroke="black" stroke-width="2"/>'), None
    corners = _polyline([(x, 30) for x in range(10, 101)] + [(100, y) for y in range(31, 121)])
    yield "junction_color", svg_document(f'<path d="{corners}" fill="none" stroke="red" stroke-width="2"/>'
        '<path d="M100 30 L180 30 L180 110 L130 110Z" fill="blue" opacity=".7"/>'), None
    for slug, _, factory in BASELINE_EXAMPLES:
        yield "visual_" + slug, factory(), None
    for relative in ("examples/example_input.png", "examples/comparison/original.jpg"):
        image = cv2.imread(str(ROOT / relative))
        if image is not None:
            yield Path(relative).stem + "_real", image, relative
    relative = "examples/comparison/visual_cara_output.svg"
    yield "cara_existing_svg", (ROOT / relative).read_text(encoding="utf-8"), relative


def tikz_stats(code: str) -> dict:
    number = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
    return {"coordinate_occurrences": len(re.findall(rf"\(\s*{number}\s*,\s*{number}\s*\)", code)),
            "line_operators": len(re.findall(r"--(?!\s*cycle)", code)),
            "beziers": code.count(".. controls"),
            "arc_operators": len(re.findall(r"\barc\s*[\[(]", code)),
            "draw_commands": len(re.findall(r"\\(?:draw|path)\[", code)),
            "tikz_bytes": len(code.encode("utf-8"))}


def svg_stats(svg: str) -> dict:
    paths = [e for e in ET.fromstring(svg).iter() if e.tag.rsplit("}", 1)[-1] == "path"]
    subpaths = [s for e in paths for s in parse_paths(e.get("d", ""))]
    return {"svg_bytes": len(svg.encode()), "paths": len(paths), "subpaths": len(subpaths),
            "closed_subpaths": sum(s.closed for s in subpaths),
            "segments": sum(len(s.segments) for s in subpaths),
            "cubic_segments": sum(seg.kind == "bezier" for s in subpaths for seg in s.segments)}


def convert(svg: str) -> tuple[str, float]:
    """Use the existing SVG bridge, retaining one paint operation per SVG path.

    Do not call the Visual white-erase postprocessor: compound fills and color
    must retain the converter's original semantics. Use draw=none when replacing
    a fill-only path command by draw, preserving the original absence of stroke.
    """
    started = perf_counter()
    code, used = _svg_to_tikz_picture(svg, TikzOptions(precision=6))
    if not used or not code:
        raise RuntimeError("svg2tikz bridge unavailable or failed; no silent alternative conversion.")
    # The official bridge extracts tikzpicture, omitting document-level custom
    # color definitions. Recover only those definitions for this experiment.
    if re.search(r"(?:fill|draw)=c[0-9a-fA-F]{6}", code):
        import svg2tikz
        previous_argv = sys.argv[:]
        try:
            sys.argv = ["svg2tikz"]
            document = svg2tikz.convert_svg(svg, no_output=True, returnstring=True)
        finally:
            sys.argv = previous_argv
        definitions = re.findall(r"\\definecolor\{[^}]+\}\{[^}]+\}\{[^}]+\}", document)
        start = code.index("]", code.index(r"\begin{tikzpicture}")) + 1
        code = code[:start] + "\n" + "\n".join(definitions) + "\n" + code[start:]
    code = re.sub(r"\\path\[", lambda _: r"\draw[draw=none,", code)
    return code, perf_counter() - started


def framed_document(code: str, svg: str) -> str:
    """Fix the PDF page frame rather than aligning or rescaling drawings afterwards."""
    root = ET.fromstring(svg)
    width, height = viewport(svg)
    match = re.fullmatch(r"([0-9.]+)cm", root.get("width", ""))
    if match is None:
        raise ValueError("PDF benchmark requires the app's physical width in cm.")
    cm_width = float(match[1])
    cm_height = cm_width * height / width
    start = code.index("]", code.index(r"\begin{tikzpicture}")) + 1
    code = (code[:start] + f"\n  \\begin{{scope}}\n  \\clip (0,0) rectangle ({cm_width},{cm_height});\n" + code[start:])
    code = code.replace(r"\end{tikzpicture}",
        f"\\end{{scope}}\n\\pgfresetboundingbox\n\\path[use as bounding box] (0,0) rectangle ({cm_width},{cm_height});\n" + r"\end{tikzpicture}")
    return wrap_standalone_document(code).replace("border=2mm", "border=0pt")


def _render_pdf(pdf: Path, executable: str) -> Path:
    destination = pdf.with_suffix("")
    result = subprocess.run([executable, "-r", "180", "-f", "1", "-l", "1", "-png", "-singlefile", str(pdf), str(destination)],
                            capture_output=True, text=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"Poppler failed: {result.stderr[-2000:]}")
    with Image.open(destination.with_suffix(".png")) as image:
        if min(image.size) < 10:
            raise RuntimeError("Invalid PDF frame; refusing to score a degenerate page.")
    return destination.with_suffix(".png")


def _image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _montage(folder: Path):
    images = [_image(folder / name) for name in ("source.png", "before.png", "after.png")]
    panels = []
    for label, array in zip(("Source raster / SVG", "Baseline TikZ PDF", "Accepted TikZ PDF"), images):
        image = Image.fromarray(array)
        image.thumbnail((600, 440))
        panel = Image.new("RGB", (620, 480), "white")
        panel.paste(image, ((620 - image.width) // 2, 32))
        ImageDraw.Draw(panel).text((12, 10), label, fill="black")
        panels.append(panel)
    montage = Image.new("RGB", (1860, 480), "#dddddd")
    for i, panel in enumerate(panels):
        montage.paste(panel, (i * 620, 0))
    montage.save(folder / "comparison.png")


def run(output: Path, config: OptimizationConfig, *, compile_pdf: bool = True, input_svg: Path | None = None) -> list[dict]:
    """Build a fresh bundle; refuse to replace even a previous experimental run."""
    output.mkdir(parents=True, exist_ok=False)
    poppler = shutil.which("pdftoppm")
    try:
        engine = str(resolve_latex_executable()) if compile_pdf else None
    except FileNotFoundError:
        engine = None
    environment = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": asdict(config), "latex": engine, "poppler": poppler,
        "python": sys.version, "platform": platform.platform(),
        "experiment_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python", "PySide6", "svg2tikz", "Pillow")}}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    records = []
    cases = fixtures() if input_svg is None else [(input_svg.stem, input_svg.read_text(encoding="utf-8"), str(input_svg.resolve()))]
    for slug, source, reference in cases:
        case_started = perf_counter()
        print(f"Running {slug}", flush=True)
        folder = output / slug
        folder.mkdir()
        tracing_seconds = 0.0
        original_raster = isinstance(source, np.ndarray)
        if original_raster:
            cv2.imwrite(str(folder / "source.png"), source)
            started = perf_counter()
            settings = visual_settings_from_processing(ProcessingSettings(vectorization_mode="visual"))
            traced = trace_visual_contours(source, settings)
            svg = contours_to_svg(traced.contours, source.shape, options=TikzOptions(), settings=settings)
            tracing_seconds = perf_counter() - started
        else:
            svg = source
            Image.fromarray(white_rgb(render_svg(svg))).save(folder / "source.png")
        (folder / "before.svg").write_text(svg, encoding="utf-8")
        result = optimize_svg(svg, config)
        candidate = result.candidate_svg or result.svg
        (folder / "candidate.svg").write_text(candidate, encoding="utf-8")
        before_code, before_time = convert(svg)
        candidate_code, candidate_time = convert(candidate)
        (folder / "before.tex").write_text(framed_document(before_code, svg), encoding="utf-8")
        (folder / "candidate.tex").write_text(framed_document(candidate_code, svg), encoding="utf-8")
        record = {"case": slug, "source_reference": reference,
            "source_sha256": hashlib.sha256((ROOT / reference).read_bytes()).hexdigest() if reference else None,
            "input_svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
            "before": {**tikz_stats(before_code), **svg_stats(svg)},
            "candidate": {**tikz_stats(candidate_code), **svg_stats(candidate)},
            "tracing_seconds": tracing_seconds, "before_conversion_seconds": before_time,
            "candidate_conversion_seconds": candidate_time, "optimization": result.diagnostics,
            "compile_status": "unavailable_or_disabled", "final_gate": "not_render_validated"}
        accepted = result.changed
        reasons = []
        if not result.diagnostics.get("candidate_accepted", False):
            reasons.append("svg_geometry_or_visual_gate")
        if engine and poppler:
            compile_started = perf_counter()
            for name in ("before", "candidate"):
                compiled = compile_latex_document(folder / f"{name}.tex", manual_path=engine)
                (folder / f"{name}_compile.txt").write_text(compiled.output, encoding="utf-8")
                if compiled.returncode != 0 or not compiled.pdf_path.exists() or not compiled.pdf_path.stat().st_size:
                    raise RuntimeError(f"LaTeX failed for {slug}/{name}; inspect the saved log.")
                _render_pdf(compiled.pdf_path, poppler)
            record["compile_and_render_seconds"] = perf_counter() - compile_started
            pdf_metrics = compare_images(_image(folder / "before.png"), _image(folder / "candidate.png"))
            pdf_passed = passes_visual_gate(pdf_metrics, config.minimum_iou, config.maximum_mae)
            accepted = accepted and pdf_passed
            if not pdf_passed:
                reasons.append("pdf_visual_or_topology_gate")
            candidate_pixels = _image(folder / "candidate.png")
            reference_image = cv2.resize(_image(folder / "source.png"), (candidate_pixels.shape[1], candidate_pixels.shape[0]), interpolation=cv2.INTER_AREA)
            original_before = compare_images(reference_image, _image(folder / "before.png"))
            original_candidate = compare_images(reference_image, candidate_pixels)
            source_passed = (original_candidate["foreground_iou"] >= original_before["foreground_iou"] - .001
                              and original_candidate["rgb_mae_255"] <= original_before["rgb_mae_255"] + .1)
            accepted = accepted and source_passed
            if not source_passed:
                reasons.append("original_source_regression")
            record.update(compile_status="success", pdf_candidate_metrics=pdf_metrics,
                          original_to_candidate_pdf=original_candidate,
                          final_gate="accepted" if accepted else "baseline_retained")
        # Code size is secondary to all fidelity checks, and must actually improve.
        smaller = (record["candidate"]["coordinate_occurrences"] < record["before"]["coordinate_occurrences"]
                   and record["candidate"]["tikz_bytes"] < record["before"]["tikz_bytes"])
        accepted = accepted and smaller
        if not smaller:
            reasons.append("no_code_reduction")
        record["accepted_optimization"] = bool(accepted)
        record["rejection_reasons"] = reasons if not accepted else []
        record["final_maximum_error_bound"] = result.diagnostics["maximum_error_bound"] if accepted else 0.0
        if not accepted:
            record["final_gate"] = "baseline_retained"
        final_svg, final_code, source_name = (candidate, candidate_code, "candidate") if accepted else (svg, before_code, "before")
        (folder / "after.svg").write_text(final_svg, encoding="utf-8")
        (folder / "after.tex").write_text(framed_document(final_code, svg), encoding="utf-8")
        for suffix in (".pdf", ".png"):
            path = folder / f"{source_name}{suffix}"
            if path.exists():
                shutil.copyfile(path, folder / f"after{suffix}")
        record["after"] = {**tikz_stats(final_code), **svg_stats(final_svg)}
        record["coordinate_reduction_percent"] = 100 * (1 - record["after"]["coordinate_occurrences"] / max(1, record["before"]["coordinate_occurrences"]))
        record["tikz_reduction_percent"] = 100 * (1 - record["after"]["tikz_bytes"] / max(1, record["before"]["tikz_bytes"]))
        if (folder / "after.png").exists():
            record["pdf_final_metrics"] = compare_images(_image(folder / "before.png"), _image(folder / "after.png"))
            pdf = _image(folder / "after.png")
            # Same viewBox/page frame; resize reference to the physical PDF grid, no feature alignment.
            reference_image = cv2.resize(_image(folder / "source.png"), (pdf.shape[1], pdf.shape[0]), interpolation=cv2.INTER_AREA)
            record["original_to_baseline_pdf"] = compare_images(reference_image, _image(folder / "before.png"))
            record["original_to_final_pdf"] = compare_images(reference_image, pdf)
            _montage(folder)
        record["case_total_seconds"] = perf_counter() - case_started
        (folder / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Experimental SVG → TikZ comparison", "", "Raw SVG bridge baseline; official pipelines unchanged.", "",
             "| Case | Coordinates before → after | Reduction | TikZ bytes before → after | PDF IoU | PDF MAE | Geometric bound | Optimize seconds | Gate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in records:
        m = r.get("pdf_final_metrics", {})
        lines.append(f"| [{r['case']}]({r['case']}/comparison.png) | {r['before']['coordinate_occurrences']} → {r['after']['coordinate_occurrences']} | "
            f"{r['coordinate_reduction_percent']:.1f}% | {r['before']['tikz_bytes']} → {r['after']['tikz_bytes']} | "
            f"{m.get('foreground_iou', float('nan')):.6f} | {m.get('rgb_mae_255', float('nan')):.4f} | "
            f"{r['final_maximum_error_bound']:.4f} | {r['optimization']['seconds']:.3f} | {r['final_gate']} |")
    lines.extend(["", "All numerical errors and rejected candidate measurements: results.json.",
                  "Topology checks are finite-resolution (SVG at 4x, PDFs at 180 dpi), not a proof of isotopy."])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=.25)
    parser.add_argument("--input", type=Path, help="One app-produced SVG instead of the benchmark corpus")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "experiments" / "svg_tikz" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run(output.resolve(), OptimizationConfig(tolerance=args.tolerance, sample_step=min(.1, args.tolerance / 4)),
        compile_pdf=not args.no_compile, input_svg=args.input)
    print(f"Report: {output / 'README.md'}")


if __name__ == "__main__":
    main()
