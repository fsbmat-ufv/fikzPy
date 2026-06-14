# Development

## Architecture

fikzPy is intentionally modular:

- `fikzpy.core.image_processor` handles image loading, grayscale conversion,
  smoothing, edge detection, overlays, and reconstruction previews.
- `fikzpy.core.contour_detector` detects and simplifies contours.
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
