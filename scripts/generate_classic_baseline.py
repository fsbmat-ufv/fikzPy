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
from fikzpy.core.image_processor import ProcessingSettings
from fikzpy.core.latex_compiler import compile_latex_document
from fikzpy.core.tikz_generator import TikzOptions, wrap_standalone_document
from fikzpy.core.tikz_pipeline import build_tikz_from_image


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

    return records


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
