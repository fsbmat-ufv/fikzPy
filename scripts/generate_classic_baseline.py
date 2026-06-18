"""Generate reproducible Classic-mode baseline artifacts.

The script intentionally uses the public Classic pipeline without modifying
preprocessing, tracing, TikZ generation, compilation, or GUI behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any, Callable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fikzpy.core.diagnostics import sha256_file
from fikzpy.core.classic_semantic_pipeline import run_classic_semantic_pipeline
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.latex_compiler import compile_latex_document
from fikzpy.core.semantic_geometry import ClosedShapePrimitive, FillStyle, LinePrimitive, Point2D, RGBColor, StrokeStyle
from fikzpy.core.semantic_tikz_exporter import export_primitives_to_tikz
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.tikz_pipeline import build_tikz_from_image
from fikzpy.core.visual_validation import validate_semantic_output


ImageFactory = Callable[[], np.ndarray]

DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "classic_semantic_baseline"


BASELINE_EXAMPLES: tuple[tuple[str, str, ImageFactory], ...] = (
    ("line_art_bw", "black-and-white line drawing", lambda: _line_art_bw()),
    ("geometric_diagram", "geometric diagram with straight and curved strokes", lambda: _geometric_diagram()),
    ("silhouette_bw", "black-and-white filled silhouette", lambda: _silhouette_bw()),
    ("simple_color", "simple colored icon", lambda: _simple_color()),
    ("noisy_grayscale", "noisy grayscale drawing", lambda: _noisy_grayscale()),
)


def _line_art_bw() -> np.ndarray:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.line(image, (18, 96), (162, 26), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(image, (62, 52), 26, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(image, (28, 28), (152, 92), (0, 0, 0), 1, cv2.LINE_AA)
    return image


def _geometric_diagram() -> np.ndarray:
    image = np.full((140, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 28), (92, 96), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(image, (146, 66), 32, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(image, (24, 112), (176, 112), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (46, 126), (46, 14), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(image, (92, 96), (146, 66), (0, 0, 0), 1, cv2.LINE_AA)
    return image


def _silhouette_bw() -> np.ndarray:
    image = np.full((140, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (78, 70), (38, 48), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (58, 36), 19, (0, 0, 0), -1, cv2.LINE_AA)
    points = np.array([[96, 42], [136, 24], [116, 64]], dtype=np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0), cv2.LINE_AA)
    cv2.circle(image, (64, 36), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def _simple_color() -> np.ndarray:
    image = np.full((140, 180, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (18, 24), (78, 92), (40, 120, 240), -1, cv2.LINE_AA)
    cv2.rectangle(image, (18, 24), (78, 92), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(image, (126, 58), 34, (210, 80, 45), -1, cv2.LINE_AA)
    cv2.circle(image, (126, 58), 34, (0, 0, 0), 2, cv2.LINE_AA)
    triangle = np.array([[74, 116], [126, 94], [156, 122]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], (80, 180, 80), cv2.LINE_AA)
    cv2.polylines(image, [triangle], True, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def _noisy_grayscale() -> np.ndarray:
    rng = np.random.default_rng(20260616)
    gray = np.full((130, 190), 230, dtype=np.uint8)
    noise = rng.normal(0.0, 8.0, gray.shape)
    gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.line(gray, (18, 108), (168, 24), 92, 2, cv2.LINE_AA)
    cv2.circle(gray, (76, 66), 31, 118, 2, cv2.LINE_AA)
    cv2.line(gray, (36, 34), (152, 102), 150, 1, cv2.LINE_AA)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _dinosaur_lineart_synthetic() -> np.ndarray:
    image = np.full((128, 168, 3), 255, dtype=np.uint8)
    body = np.array(
        [(20, 78), (32, 50), (62, 35), (104, 38), (132, 55), (148, 75), (132, 90), (96, 92), (72, 104), (44, 100)],
        dtype=np.int32,
    )
    cv2.polylines(image, [body], True, (0, 0, 0), 2)
    cv2.circle(image, (117, 55), 3, (0, 0, 0), 1)
    for x in (132, 138, 144):
        cv2.line(image, (x, 68), (x - 3, 75), (0, 0, 0), 1)
    for x in (47, 58, 90):
        cv2.line(image, (x, 96), (x - 5, 116), (0, 0, 0), 2)
        cv2.line(image, (x - 5, 116), (x + 6, 116), (0, 0, 0), 1)
    cv2.line(image, (70, 56), (102, 61), (0, 0, 0), 1)
    cv2.line(image, (67, 70), (102, 75), (0, 0, 0), 1)
    cv2.line(image, (24, 75), (7, 68), (0, 0, 0), 2)
    cv2.line(image, (8, 68), (18, 62), (0, 0, 0), 1)
    return image


def _closed_contour_lineart() -> np.ndarray:
    image = np.full((112, 112, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (56, 56), (36, 26), 0, 0, 360, (0, 0, 0), 2)
    cv2.circle(image, (45, 50), 4, (0, 0, 0), 1)
    cv2.line(image, (30, 66), (82, 68), (0, 0, 0), 1)
    cv2.line(image, (44, 78), (40, 94), (0, 0, 0), 2)
    cv2.line(image, (68, 78), (74, 94), (0, 0, 0), 2)
    return image


def _filled_rectangle_real() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 18), (72, 76), (0, 0, 0), -1)
    cv2.rectangle(image, (40, 34), (56, 52), (255, 255, 255), -1)
    return image


def _mixed_monochrome_synthetic() -> np.ndarray:
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.circle(image, (42, 34), 18, (0, 0, 0), -1)
    cv2.circle(image, (86, 36), 16, (0, 0, 0), -1)
    cv2.rectangle(image, (24, 70), (52, 104), (0, 0, 0), -1)
    cv2.rectangle(image, (76, 68), (105, 105), (0, 0, 0), -1)
    cv2.circle(image, (42, 39), 10, (255, 255, 255), -1)
    cv2.circle(image, (86, 41), 9, (255, 255, 255), -1)
    cv2.line(image, (30, 40), (37, 40), (0, 0, 0), 1)
    cv2.line(image, (46, 40), (52, 40), (0, 0, 0), 1)
    cv2.line(image, (38, 48), (47, 50), (0, 0, 0), 1)
    cv2.line(image, (79, 41), (84, 41), (0, 0, 0), 1)
    cv2.line(image, (90, 41), (96, 41), (0, 0, 0), 1)
    cv2.line(image, (20, 62), (58, 62), (0, 0, 0), 1)
    cv2.line(image, (72, 61), (110, 61), (0, 0, 0), 1)
    cv2.line(image, (60, 76), (72, 90), (0, 0, 0), 1)
    cv2.line(image, (64, 88), (70, 96), (0, 0, 0), 1)
    return image


def generate_baseline(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    compile_pdf: bool = True,
    engine: str = "pdflatex",
) -> list[dict[str, Any]]:
    """Generate baseline images, TeX documents, optional PDFs, and metrics."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for slug, description, factory in BASELINE_EXAMPLES:
        image = factory()
        image_path = destination / f"{slug}.png"
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Could not write image: {image_path}")

        settings = ProcessingSettings(vectorization_mode="classic")
        options = TikzOptions(use_bezier=False)

        started = perf_counter()
        result = build_tikz_from_image(image, settings, options)
        elapsed = perf_counter() - started

        document = wrap_standalone_document(result.tikz_code)
        tex_path = destination / f"{slug}.tex"
        tex_path.write_text(document, encoding="utf-8")

        pdf_path = tex_path.with_suffix(".pdf")
        pdf_status = "skipped"
        compile_returncode: int | None = None
        if compile_pdf:
            compile_result = compile_latex_document(tex_path, engine=engine)
            compile_returncode = compile_result.returncode
            pdf_status = "ok" if compile_result.returncode == 0 else "failed"
            _remove_latex_sidecars(tex_path)
            if compile_result.returncode != 0:
                tail = compile_result.output[-3000:]
                raise RuntimeError(f"LaTeX failed for {tex_path.name}:\n{tail}")
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                raise RuntimeError(f"LaTeX did not create a non-empty PDF for {tex_path.name}")

        records.append(
            _record_metrics(
                slug=slug,
                description=description,
                image=image,
                image_path=image_path,
                tex_path=tex_path,
                pdf_path=pdf_path if compile_pdf else None,
                result_tikz=result.tikz_code,
                effective_mode=result.effective_mode,
                contour_count=len(result.processing_result.contours),
                processing_seconds=elapsed,
                pdf_status=pdf_status,
                compile_returncode=compile_returncode,
            )
        )

    metrics_path = destination / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = destination / "README.md"
    report_path.write_text(_render_markdown_report(records), encoding="utf-8")
    _write_lineart_refinement_report(destination)

    return records


def _write_lineart_refinement_report(destination: Path) -> None:
    cases = [
        _pipeline_lineart_case("dinosaur_lineart_synthetic_good", _dinosaur_lineart_synthetic()),
        _bad_overfilled_case("dinosaur_lineart_bad_overfilled", _dinosaur_lineart_synthetic()),
        _pipeline_lineart_case("line_art_simple", _line_art_bw()),
        _pipeline_lineart_case("closed_contour_lineart", _closed_contour_lineart()),
        _pipeline_lineart_case("filled_rectangle_real", _filled_rectangle_real()),
        _pipeline_lineart_case("silhouette_real", _silhouette_bw()),
        _pipeline_lineart_case("mixed_monochrome_synthetic", _mixed_monochrome_synthetic()),
        _mixed_bad_regression_case("mixed_monochrome_bad_regression", _mixed_monochrome_synthetic()),
    ]
    report = {
        "issue": "Issue 11.5",
        "description": "Classic semantic line-art refinement regression report",
        "cases": cases,
    }
    path = destination / "classic_lineart_refinement_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pipeline_lineart_case(name: str, image: np.ndarray) -> dict[str, Any]:
    result = run_classic_semantic_pipeline(image)
    validation = result.validation_result
    raster = validation.fidelity_score.raster_metrics if validation is not None else None
    return {
        "name": name,
        "strategy_used": result.strategy_used.value,
        "line_art_confidence": round(result.metrics.line_art_confidence, 6),
        "mixed_confidence": round(result.metrics.mixed_monochrome_confidence, 6),
        "filled_region_count": result.metrics.filled_region_primitives,
        "thin_stroke_count": result.metrics.thin_stroke_primitives,
        "filled_area_ratio": round(result.metrics.filled_area_ratio, 6),
        "white_cutout_count": result.metrics.white_cutout_count,
        "white_cutout_area_ratio": round(result.metrics.white_cutout_area_ratio, 6),
        "dark_mass_preservation": round(result.metrics.dark_mass_preservation_ratio, 6),
        "thin_stroke_recall": round(result.metrics.thin_stroke_recall, 6),
        "filled_region_recall": round(result.metrics.filled_region_recall, 6),
        "validation_score": round(result.metrics.validation_score, 6),
        "regression_flags": list(validation.regression_flags) if validation is not None else [],
        "accepted": result.accepted,
        "rejection_reasons": list(result.rejection_reasons),
        "tikz_draw_commands": result.tikz_export_result.metrics.draw_commands,
        "tikz_fill_commands": result.tikz_export_result.metrics.filled_paths_written,
        "tikz_code_characters": len(result.tikz_code),
        "deterministic_hash": result.deterministic_hash,
        "source_foreground_ratio": round(raster.source_foreground_ratio, 6) if raster is not None else 0.0,
        "rendered_foreground_ratio": round(raster.rendered_foreground_ratio, 6) if raster is not None else 0.0,
    }


def _bad_overfilled_case(name: str, image: np.ndarray) -> dict[str, Any]:
    primitives = _bad_overfilled_dinosaur_primitives()
    tikz = export_primitives_to_tikz(primitives)
    validation = validate_semantic_output(image, primitives, tikz)
    return _validation_case_record(name, "bad_overfilled_fixture", primitives, tikz, validation)


def _mixed_bad_regression_case(name: str, image: np.ndarray) -> dict[str, Any]:
    primitives = [
        LinePrimitive(_p(30, 40), _p(37, 40)),
        LinePrimitive(_p(46, 40), _p(52, 40)),
        LinePrimitive(_p(38, 48), _p(47, 50)),
        LinePrimitive(_p(20, 62), _p(58, 62)),
        LinePrimitive(_p(72, 61), _p(110, 61)),
    ]
    tikz = export_primitives_to_tikz(primitives)
    validation = validate_semantic_output(image, primitives, tikz)
    return _validation_case_record(name, "mixed_bad_thin_only_fixture", primitives, tikz, validation)


def _validation_case_record(name: str, strategy: str, primitives: list[Any] | tuple[Any, ...], tikz: Any, validation: Any) -> dict[str, Any]:
    raster = validation.fidelity_score.raster_metrics
    fill_metrics = validation.metrics.filled_region_metrics
    lineart_metrics = validation.metrics.lineart_fill_metrics
    diagnostics = lineart_metrics.get("diagnostics") or {}
    return {
        "name": name,
        "strategy_used": strategy,
        "line_art_confidence": round(float(lineart_metrics.get("line_art_confidence", 0.0)), 6),
        "mixed_confidence": round(float(diagnostics.get("mixed_monochrome_confidence", 0.0)), 6),
        "filled_region_count": sum(1 for primitive in primitives if _has_visible_fill(primitive, RGBColor.black())),
        "thin_stroke_count": sum(1 for primitive in primitives if isinstance(primitive, LinePrimitive)),
        "filled_area_ratio": round(float(fill_metrics.get("filled_area_ratio", 0.0)), 6),
        "white_cutout_count": int(fill_metrics.get("white_cutout_count", 0)),
        "white_cutout_area_ratio": round(float(fill_metrics.get("white_cutout_area_ratio", 0.0)), 6),
        "dark_mass_preservation": round(raster.dark_mass_preservation_ratio, 6),
        "thin_stroke_recall": round(raster.thin_stroke_recall, 6),
        "filled_region_recall": round(raster.filled_region_recall, 6),
        "validation_score": round(validation.fidelity_score.overall_score, 6),
        "regression_flags": list(validation.regression_flags),
        "accepted": validation.accepted,
        "rejection_reasons": list(validation.rejection_reasons),
        "tikz_draw_commands": tikz.metrics.draw_commands,
        "tikz_fill_commands": tikz.metrics.filled_paths_written,
        "tikz_code_characters": len(tikz.code),
        "deterministic_hash": validation.deterministic_hash,
    }


def _bad_overfilled_dinosaur_primitives() -> tuple[ClosedShapePrimitive, ...]:
    white_fill = FillStyle(RGBColor(255, 255, 255))
    white_cutout_stroke = StrokeStyle(RGBColor(255, 255, 255), width=0.1, opacity=0.0)
    return (
        ClosedShapePrimitive(
            (_p(18, 34), _p(150, 34), _p(158, 98), _p(26, 108)),
            stroke=StrokeStyle(RGBColor.black(), width=1.0),
            fill=FillStyle(RGBColor.black()),
            metadata={"source_layer": "filled_region"},
        ),
        ClosedShapePrimitive(
            (_p(38, 52), _p(118, 48), _p(128, 78), _p(42, 84)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (_p(44, 86), _p(78, 84), _p(76, 96), _p(42, 98)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (_p(92, 84), _p(132, 82), _p(130, 94), _p(88, 98)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
        ClosedShapePrimitive(
            (_p(108, 50), _p(124, 50), _p(124, 62), _p(108, 62)),
            stroke=white_cutout_stroke,
            fill=white_fill,
            metadata={"source_layer": "filled_region_hole"},
        ),
    )


def _has_visible_fill(primitive: Any, color: RGBColor) -> bool:
    fill = getattr(primitive, "fill", None)
    if not isinstance(fill, FillStyle):
        return False
    if fill.opacity is not None and fill.opacity <= 0.0:
        return False
    return fill.color == color


def _p(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y))


def _record_metrics(
    *,
    slug: str,
    description: str,
    image: np.ndarray,
    image_path: Path,
    tex_path: Path,
    pdf_path: Path | None,
    result_tikz: str,
    effective_mode: str,
    contour_count: int,
    processing_seconds: float,
    pdf_status: str,
    compile_returncode: int | None,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    document_bytes = tex_path.stat().st_size
    record: dict[str, Any] = {
        "slug": slug,
        "description": description,
        "image": _display_path(image_path),
        "image_size": {"width": int(width), "height": int(height)},
        "image_sha256": sha256_file(image_path),
        "mode": "classic",
        "effective_mode": effective_mode,
        "contours": contour_count,
        "draw_count": result_tikz.count("\\draw"),
        "line_segment_count": result_tikz.count("--"),
        "bezier_count": result_tikz.count(".. controls"),
        "point_count": result_tikz.count("("),
        "tex": _display_path(tex_path),
        "tex_bytes": document_bytes,
        "tex_sha256": sha256_file(tex_path),
        "processing_seconds": round(processing_seconds, 6),
        "pdf_status": pdf_status,
        "compile_returncode": compile_returncode,
    }
    if pdf_path is not None:
        record["pdf"] = _display_path(pdf_path)
        record["pdf_bytes"] = pdf_path.stat().st_size if pdf_path.exists() else 0
        record["pdf_sha256"] = sha256_file(pdf_path) if pdf_path.exists() else None
    else:
        record["pdf"] = None
        record["pdf_bytes"] = 0
        record["pdf_sha256"] = None
    return record


def _render_markdown_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Classic Semantic Baseline",
        "",
        "Generated by `python scripts/generate_classic_baseline.py` using the current Classic pipeline.",
        "",
        "| Example | Size | Draws | `--` | Beziers | TeX bytes | Seconds | PDF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        size = f"{record['image_size']['width']}x{record['image_size']['height']}"
        pdf = record["pdf"] or record["pdf_status"]
        lines.append(
            "| {slug} | {size} | {draw_count} | {line_segment_count} | {bezier_count} | "
            "{tex_bytes} | {processing_seconds:.6f} | {pdf} |".format(
                slug=record["slug"],
                size=size,
                draw_count=record["draw_count"],
                line_segment_count=record["line_segment_count"],
                bezier_count=record["bezier_count"],
                tex_bytes=record["tex_bytes"],
                processing_seconds=float(record["processing_seconds"]),
                pdf=pdf,
            )
        )
    lines.extend(
        [
            "",
            "The committed artifacts are a baseline only. They do not introduce the semantic Classic pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def _remove_latex_sidecars(tex_path: Path) -> None:
    for suffix in (".aux", ".log", ".out"):
        sidecar = tex_path.with_suffix(suffix)
        if sidecar.exists():
            sidecar.unlink()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for images, TeX files, PDFs, and metrics.",
    )
    parser.add_argument("--engine", default="pdflatex", help="LaTeX engine to use for PDF generation.")
    parser.add_argument("--skip-pdf", action="store_true", help="Generate images, TeX, and metrics without compiling PDFs.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    records = generate_baseline(args.output_dir, compile_pdf=not args.skip_pdf, engine=args.engine)
    print(f"Wrote {len(records)} Classic baseline examples to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
