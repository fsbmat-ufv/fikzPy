# Bezier Fitting Report

Input: `examples\comparison\original.jpg`

## Parameters

- `classic`: contour-to-TikZ with straight segments.
- `vector_local`: previous Catmull-Rom local conversion, roughly one Bezier per small segment.
- `vector_fitted`: recursive global cubic fitting over longer contour spans.
- `vector` extraction now enables conservative faint-stroke recovery by local contrast.
- `faint_stroke_block_size`: 31 pixels.
- `faint_stroke_min_delta`: 7 gray levels.
- `faint_stroke_max_gray`: 246.
- `error_tolerance`: proportional per contour, clamped to `[0.01, 0.08]` TikZ units.
- `pre_simplify_tolerance`: proportional per contour, clamped to `[0.001, 0.005]` TikZ units.
- `min_points_for_bezier`: 4.

| Mode | TeX | SHA-256 | Size bytes | `\draw` | `--` | `.. controls` | Vector objects | PDF | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `classic` | `examples\comparison\classic_output.tex` | `fd2f2375aaf4f360feae4fb5983ad851efb69b62e20db6d5dd34af5be86a1e60` | 41723 | 297 | 1321 | 0 | 0 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\classic_output.pdf` | 0 |
| `vector_local` | `examples\comparison\vector_local_output.tex` | `2b85ff738a24c587d67d11f93e163fda68904a2e98a1968e80c2d7b7556ab8ce` | 135678 | 1295 | 129 | 1207 | 1295 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_local_output.pdf` | 0 |
| `vector_fitted` | `examples\comparison\vector_fitted_output.tex` | `4a43db0ee4c302fbfdfc172e44470be4615e5a9482abf793f6f88964c81a727d` | 31207 | 348 | 467 | 107 | 348 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_fitted_output.pdf` | 0 |

## Visual Notes

- `vector_local` is expected to look close to the previous vector output because it replaces each small segment with a local Bezier.
- `vector_fitted` should be visibly different in the code structure: fewer draw commands and far fewer `.. controls` entries.
- The latest vector pass recovers a small set of very light strokes that the classic global threshold leaves out.
- Small teeth, claws, eyes, wrinkles, and spirals remain protected by conservative simplification and fallback to Line/Polyline.
- This pass changes vector-mode stroke extraction only; it does not change image processing for classic mode or GUI layout.
