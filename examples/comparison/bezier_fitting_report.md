# Bezier Fitting Report

Input: `examples\comparison\original.jpg`

## Parameters

- `classic`: contour-to-TikZ with straight segments.
- `vector_local`: previous Catmull-Rom local conversion, roughly one Bezier per small segment.
- `vector_fitted`: recursive global cubic fitting over longer contour spans.
- `error_tolerance`: proportional per contour, clamped to `[0.01, 0.08]` TikZ units.
- `pre_simplify_tolerance`: proportional per contour, clamped to `[0.001, 0.005]` TikZ units.
- `min_points_for_bezier`: 4.

| Mode | TeX | SHA-256 | Size bytes | `\draw` | `--` | `.. controls` | Vector objects | PDF | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `classic` | `examples\comparison\classic_output.tex` | `fd2f2375aaf4f360feae4fb5983ad851efb69b62e20db6d5dd34af5be86a1e60` | 41723 | 297 | 1321 | 0 | 0 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\classic_output.pdf` | 0 |
| `vector_local` | `examples\comparison\vector_local_output.tex` | `223c2319da8f19dbc7cdcea1064a283e1e9c0e81395c30ed8180b580a7efc615` | 134142 | 1280 | 127 | 1194 | 1280 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_local_output.pdf` | 0 |
| `vector_fitted` | `examples\comparison\vector_fitted_output.tex` | `8196dbe2caab7de3ae1e162db65736fbec722935da8ea0f73a3a0033331fb3ce` | 30796 | 343 | 457 | 107 | 343 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_fitted_output.pdf` | 0 |

## Visual Notes

- `vector_local` is expected to look close to the previous vector output because it replaces each small segment with a local Bezier.
- `vector_fitted` should be visibly different in the code structure: fewer draw commands and far fewer `.. controls` entries.
- Small teeth, claws, eyes, wrinkles, and spirals remain protected by conservative simplification and fallback to Line/Polyline.
- This pass changes curve representation only; it does not change image processing or GUI layout.
