# Vector Pipeline Debug

## Current Flow Found In The Project

1. GUI action: `MainWindow.generate_action` in `fikzpy/gui/main_window.py`
   triggers `MainWindow.generate_tikz()`.
2. Mode selection: `MainWindow.processing_settings()` reads
   `self.vectorization_mode_combo.currentData()`.
3. Image processing: `MainWindow.generate_tikz()` calls
   `process_image(self.original_image, self.processing_settings())`.
4. Contour creation: `fikzpy/core/image_processor.py::process_image()`
   uses `config_for_mode(settings.vectorization_mode)`.
5. Current supported effective modes:
   - `classic`: line-art tracing with `trace_line_art_strokes()`;
   - `smooth`: line-art tracing plus optional preprocessing, filtering,
     merging, and smoothing;
   - `contours`: Canny + OpenCV contour detection.
6. TikZ generation: `MainWindow.generate_tikz()` always calls
   `fikzpy/core/tikz_generator.py::generate_tikz_picture()`, which consumes
   `Contour` objects directly.
7. TeX writing:
   - Export: `MainWindow.export_tex()` writes
     `wrap_standalone_document(self.code_editor.code())`.
   - Preview compile: `MainWindow.compile_latex()` calls
     `MainWindow._write_temporary_tex()`.
8. Temporary preview filename: `MainWindow._write_temporary_tex()` always writes
   `%TEMP%/fikzpy/fikzpy_preview.tex`.
9. LaTeX compilation:
   `fikzpy/core/latex_compiler.py::compile_latex_document()` resolves the TeX
   path, calls `pdflatex`, and reports `LatexCompileResult.pdf_path`.
10. PDF preview: `MainWindow.compile_latex()` opens `result.pdf_path` with
    `QDesktopServices.openUrl(...)`.

## Diagnosis

- `fikzpy/core/vector_objects.py` defines internal vector objects, but no
  current GUI or generation path creates those objects.
- There is no `vector` option in the GUI mode combo.
- `config_for_mode()` does not accept `vector`.
- `MainWindow.generate_tikz()` always uses the contour-based TikZ generator.
- Preview files use the same temporary filename regardless of mode, which can
  hide whether `classic` or `vector` produced the PDF.
- No silent fallback from vector to classic was found because no vector branch
  exists yet in the active generation path.

## Correction Target

Connect only the missing flow:

```text
GUI selected mode=vector
  -> process image as classic contours
  -> convert contours to internal vector objects
  -> export vector objects to TikZ
  -> write mode-specific TeX filename
  -> compile corresponding PDF
  -> open the compiled PDF path
```

The image-processing, smoothing, and Bezier algorithms should not be changed in
this diagnostic task.

## Correction Applied

- `fikzpy/core/vectorization_config.py::config_for_mode()` now accepts
  `vector`.
- `fikzpy/core/tikz_pipeline.py::build_tikz_from_image()` is the controller
  entry point used by the GUI.
- `classic`, `smooth`, and `contours` still use the contour TikZ generator.
- `vector` uses:
  - `process_image()` for contour extraction;
  - `contours_to_vector_objects()` in `fikzpy/core/vector_pipeline.py`;
  - `generate_tikz_from_vector_objects()` in
    `fikzpy/core/vector_exporter.py`.
- `MainWindow.generate_tikz()` now stores a `TikzBuildResult`, preserving the
  processing result and vector diagnostics.
- `MainWindow._write_temporary_tex()` writes mode-specific temporary files,
  such as `fikzpy_classic_001.tex` or `fikzpy_vector_001.tex`.
- `MainWindow.compile_latex()` opens exactly the `LatexCompileResult.pdf_path`
  returned by `compile_latex_document()`.
- The vector exporter adds `% FIKZPY VECTOR MODE` and a small `VECTOR MODE`
  node only for the `vector` path.

## Diagnostic Evidence

The reproducible report is written to:

```text
examples/comparison/pipeline_report.md
```

For the dinosaur input used during this pass:

- `classic_output.tex` and `vector_output.tex` have different SHA-256 hashes.
- `classic` reports zero internal vector objects.
- `vector` reports internal vector objects and Bezier curves.
