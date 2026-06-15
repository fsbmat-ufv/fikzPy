# Bezier Fitting Report

Input: `examples\comparison\original.jpg`

## Parameters

- `classic`: contour-to-TikZ with straight segments.
- `vector_local`: previous Catmull-Rom local conversion, roughly one Bezier per small segment.
- `vector_fitted`: recursive global cubic fitting over longer contour spans.
- `vector` extraction enables conservative faint-stroke recovery by local contrast.
- `vector` extraction also enables black-hat recovery for weak JPEG strokes.
- `vector` snaps skeleton branch endpoints to local junction centers to reduce gaps.
- `vector` exports connected fitted primitives from the same contour as one continuous TikZ path.
- `min_path_length`: 2 in vector mode, after junction snapping collapses artificial microsegments.
- `faint_stroke_block_size`: 31 pixels.
- `faint_stroke_min_delta`: 7 gray levels.
- `faint_stroke_max_gray`: 246.
- `blackhat_kernel_size`: 9 pixels.
- `blackhat_threshold`: 12 gray levels.
- `error_tolerance`: proportional per contour, clamped to `[0.01, 0.08]` TikZ units.
- `pre_simplify_tolerance`: proportional per contour, clamped to `[0.001, 0.005]` TikZ units.
- `min_points_for_bezier`: 4.

| Mode | TeX | SHA-256 | Size bytes | `\draw` | `--` | `.. controls` | Vector objects | PDF | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `classic` | `examples\comparison\classic_output.tex` | `fd2f2375aaf4f360feae4fb5983ad851efb69b62e20db6d5dd34af5be86a1e60` | 41723 | 297 | 1321 | 0 | 0 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\classic_output.pdf` | 0 |
| `vector_local` | `examples\comparison\vector_local_output.tex` | `780ec7fedda615912e23020b4f1bbd74f52482c724746a833398836989a89c28` | 113174 | 450 | 246 | 1446 | 1592 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_local_output.pdf` | 0 |
| `vector_fitted` | `examples\comparison\vector_fitted_output.tex` | `c2178bde8603fd4c137a44fa4e8a75fe2fee56817f77a72f00e5db1519ae1eb6` | 40051 | 450 | 724 | 93 | 476 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_fitted_output.pdf` | 0 |

## Visual Notes

- The diagnostic `VECTOR MODE` node is no longer emitted by the public vector pipeline.
- Black-hat recovery is intended to recover weak JPEG strokes in teeth, mouth, eyes, wrinkles, claws, and spirals.
- Junction snapping reduces small gaps caused by clustered skeleton branch pixels.
- Grouped paths reduce visual joins between consecutive fitted primitives.
- The vector output is intentionally more detailed than the previous fitted version, trading a larger TikZ file for better visual fidelity.
- This pass changes vector-mode stroke extraction and normal vector export only; it does not change the GUI layout.
