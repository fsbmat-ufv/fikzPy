# Development

## Architecture

fikzPy is intentionally modular:

- `fikzpy.core.image_processor` handles image loading, grayscale conversion,
  smoothing, edge detection, overlays, and reconstruction previews.
- `fikzpy.core.contour_detector` detects and simplifies contours.
- `fikzpy.core.stroke_tracer` extracts ink masks, skeletonizes line art, and
  traces open strokes for drawings with internal details.
- `fikzpy.core.bezier_fit` converts simplified polylines into cubic Bezier
  segments when requested.
- `fikzpy.core.tikz_generator` maps image coordinates to TikZ coordinates and
  emits compact code.
- `fikzpy.core.latex_compiler` detects LaTeX engines and compiles `.tex` files.
- `fikzpy.gui` contains the PySide6 interface.
- `fikzpy.templates` contains reusable TikZ snippets.

## Test

```powershell
python -m pytest
```

## Packaging

```powershell
python scripts/build_exe.py
```

## Contribution Notes

- Keep functions small and testable.
- Keep generated TikZ editable by humans.
- Prefer clear parameters over hidden heuristics.
- Do not introduce heavy dependencies without a clear benefit.
- Add tests when changing geometry, contour filtering, or code generation.

## Vectorization Backends

The default backend is `line_art`, which is intended for black-and-white
drawings, sketches, diagrams, and scanned line art. It thresholds dark ink,
skeletonizes strokes, smooths pixel stair-stepping, traces open paths, and then
emits TikZ.

The `contours` backend keeps the first MVP approach based on Canny edges and
OpenCV contours. It is useful for simple closed shapes, but it is usually less
faithful for drawings with teeth, eyes, hatching, labels, and other internal
strokes.

Future optional backends can include:

- Inkscape Trace Bitmap or potrace for raster-to-SVG conversion.
- `svg2tikz` for SVG-to-TikZ conversion after the raster image has already been
  vectorized.
- A dedicated curve-fitting pass for fewer but smoother Bezier paths.
