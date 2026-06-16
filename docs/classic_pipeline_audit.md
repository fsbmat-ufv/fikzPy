# Classic Pipeline Audit

This audit records the current Classic and Visual/vector flow before any
semantic Classic replacement work. It is intentionally descriptive: no GUI
layout, Classic output, Visual behavior, tracing algorithm, or compilation
logic was changed for this baseline.

## Entry Points

- GUI action: `fikzpy.gui.main_window.MainWindow._build_actions()` creates
  `self.generate_action` with label `Gerar TikZ` and connects it to
  `MainWindow.generate_tikz()`.
- Mode selector: `MainWindow._build_settings_dock()` creates
  `self.vectorization_mode_combo` with public modes `classic`, `vector`,
  `fidelity`, `smooth`, and `contours`.
- Settings bridge: `MainWindow.processing_settings()` reads the selected mode
  and image-processing controls into `fikzpy.core.image_processor.ProcessingSettings`.
- TikZ options bridge: `MainWindow.tikz_options()` builds
  `fikzpy.core.tikz_generator.TikzOptions`. Bezier output is enabled when the
  checkbox is selected or when the public mode is `smooth`, `vector`, or
  `fidelity`.
- Pipeline controller: `MainWindow.generate_tikz()` calls
  `fikzpy.core.tikz_pipeline.build_tikz_from_image()`, stores the returned
  `TikzBuildResult`, updates the code editor, and refreshes the image preview.

## Classic Flow

The current Classic path is the stable line-art path:

```text
MainWindow.generate_tikz()
  -> build_tikz_from_image()
  -> config_for_mode("classic")
  -> process_image()
  -> to_gray()
  -> smooth_image()
  -> trace_line_art_strokes()
  -> generate_tikz_picture()
  -> CodeEditor.set_code()
  -> MainWindow._update_view()
```

Real files and functions:

- `fikzpy/core/tikz_pipeline.py::build_tikz_from_image()` resolves the
  requested mode through `config_for_mode()`, calls `process_image()`, and uses
  `generate_tikz_picture()` for all non-vector effective modes.
- `fikzpy/core/vectorization_config.py::config_for_mode()` maps `classic`,
  `line_art`, and `line-art` to `VectorizationConfig.classic()`.
- `fikzpy/core/image_processor.py::process_image()` converts the image to gray,
  applies light Gaussian blur with `smooth_image()`, and routes `classic` to
  `trace_line_art_strokes()`.
- `fikzpy/core/stroke_tracer.py::trace_line_art_strokes()` calls
  `extract_ink_mask()`, `skeletonize()`, and `trace_strokes_from_skeleton()`.
- `fikzpy/core/stroke_tracer.py::extract_ink_mask()` uses median denoising,
  Otsu/global threshold selection, optional adaptive thresholding when enabled
  by settings, and small-component removal.
- `fikzpy/core/stroke_tracer.py::trace_strokes_from_skeleton()` walks skeleton
  pixels into open or closed `Contour` objects and simplifies them with
  `simplify_polyline()`.
- `fikzpy/core/tikz_generator.py::generate_tikz_picture()` serializes `Contour`
  objects as `\draw` paths. In default Classic UI settings this is a line path,
  because `TikzOptions.use_bezier` is false.

## Classic Preprocessing, Contours, Simplification, TikZ, Preview

- Preprocessing: `to_gray()` and `smooth_image()` in
  `fikzpy/core/image_processor.py`; `extract_ink_mask()` in
  `fikzpy/core/stroke_tracer.py` performs Classic thresholding and component
  cleanup.
- Contour/stroke detection: `trace_line_art_strokes()` returns contours from a
  skeletonized ink mask. The Canny/OpenCV contour detector is not used for
  effective mode `classic`; it is used by effective mode `contours`.
- Simplification: `trace_strokes_from_skeleton()` uses
  `fikzpy.core.contour_detector.simplify_polyline()` with the active
  `simplify_epsilon`.
- TikZ generation: `generate_tikz_picture()` and `contour_to_tikz()` in
  `fikzpy/core/tikz_generator.py` convert image-space points to TikZ-space
  coordinates through `image_point_to_tikz()`.
- TeX export: `MainWindow.export_tex()` writes
  `wrap_standalone_document(self.code_editor.code())`.
- Preview compilation: `MainWindow.compile_latex()` writes a temporary TeX file
  with `MainWindow._write_temporary_tex()` and calls
  `fikzpy.core.latex_compiler.compile_latex_document()`.
- PDF preview: on successful compilation, `MainWindow.compile_latex()` opens
  `LatexCompileResult.pdf_path` with `QDesktopServices.openUrl()`.

## Visual/Vector Flow

There is no separate SVG Visual pipeline in the current source tree. The GUI's
non-Classic visual/vector behavior is represented by public modes `vector` and
`fidelity`:

```text
MainWindow.generate_tikz()
  -> build_tikz_from_image()
  -> config_for_mode("vector" or "fidelity")
  -> process_image()
  -> trace_line_art_strokes()
  -> fit_contours_to_vector_objects()
  -> generate_tikz_from_vector_objects()
```

Real files and functions:

- `fikzpy/core/tikz_pipeline.py::build_tikz_from_image()` routes effective modes
  `vector` and `fidelity` into `fit_contours_to_vector_objects()`.
- `fikzpy/core/vector_pipeline.py::fit_contours_to_vector_objects()` converts
  traced contours into internal vector objects and fitting diagnostics.
- `fikzpy/core/vector_exporter.py::generate_tikz_from_vector_objects()`
  serializes those objects into TikZ.
- `tests/test_tikz_pipeline.py` verifies that Classic and vector outputs remain
  different and that vector failures are not silently reported as Classic.

## Reproducible Baseline

Run the baseline with one command from the repository root:

```bash
python scripts/generate_classic_baseline.py
```

The command writes `examples/classic_semantic_baseline/` with five synthetic
inputs and their Classic-mode outputs:

- `line_art_bw.png`
- `geometric_diagram.png`
- `silhouette_bw.png`
- `simple_color.png`
- `noisy_grayscale.png`

For each input, the script records image size, `\draw` count, `--` count,
`.. controls` count, TeX byte size, processing time, and PDF path in
`baseline_metrics.json`. It also writes a human-readable `README.md` table and
compiles the corresponding PDFs with `pdflatex` by default.

The script calls only the existing public pipeline APIs and uses
`ProcessingSettings(vectorization_mode="classic")` plus
`TikzOptions(use_bezier=False)`, matching the current default Classic output
path.
