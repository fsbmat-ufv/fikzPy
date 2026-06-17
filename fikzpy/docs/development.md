# Development

## Architecture

fikzPy is intentionally modular:

- `fikzpy.core.image_processor` handles image loading, grayscale conversion,
  smoothing, edge detection, overlays, and reconstruction previews.
- `fikzpy.core.contour_detector` detects and simplifies contours.
- `fikzpy.core.contour_cleaning` filters tiny paths and approximate duplicates.
- `fikzpy.core.contour_merging` conservatively joins nearby open strokes.
- `fikzpy.core.contour_smoothing` smooths traced path coordinates.
- `fikzpy.core.preprocessing` applies optional denoising and morphology.
- `fikzpy.core.stroke_tracer` extracts ink masks, skeletonizes line art, and
  traces open strokes for drawings with internal details.
- `fikzpy.core.primitive_detection` contains hooks for semantic TikZ shapes.
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

`Classic` preserves the stable line-art behavior. The legacy internal name
`line_art` is still accepted as an alias for `classic`.

`Smooth` is experimental and intentionally modular:

- `preprocessing.py` applies optional bilateral/gaussian filtering and mask
  morphology.
- `contour_cleaning.py` filters tiny paths and can remove approximate
  duplicates.
- `contour_merging.py` conservatively joins nearby open strokes when endpoints
  and directions agree.
- `contour_smoothing.py` smooths path coordinates before TikZ generation.
- `primitive_detection.py` contains hooks for future semantic TikZ primitives.

The `contours` backend keeps the first MVP approach based on Canny edges and
OpenCV contours. It is useful for simple closed shapes, but it is usually less
faithful for drawings with teeth, eyes, hatching, labels, and other internal
strokes.

Future optional backends can include:

- Inkscape Trace Bitmap or potrace for raster-to-SVG conversion.
- `svg2tikz` for SVG-to-TikZ conversion after the raster image has already been
  vectorized.
- A dedicated curve-fitting pass for fewer but smoother Bezier paths.

## Rollback Strategy

The GUI exposes `Classic`, `Smooth`, and `Contornos` in the existing mode combo
box. Select `Classic` to return to the pre-smooth vectorization behavior.

The Git branch `improve-vectorization-v1` keeps the experimental work isolated
from `master`, and each feature is committed separately for easy rollback.
