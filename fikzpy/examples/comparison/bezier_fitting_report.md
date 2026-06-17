# Bezier Fitting Report

Input: `examples\comparison\original.jpg`

## Parameters

- `classic`: contour-to-TikZ with straight segments.
- `vector_local`: Catmull-Rom local conversion, roughly one Bezier per small segment.
- `vector_fitted`: recursive global cubic fitting over longer contour spans.
- `vector` extraction uses bilateral denoising, CLAHE, Gaussian adaptive thresholding, and conservative local-contrast gates.
- `vector` extraction also enables black-hat recovery for weak JPEG strokes.
- `vector` applies morphological closing before skeletonization to reconnect small gaps.
- `vector` uses `skimage.morphology.skeletonize` when installed, falling back to Zhang-Suen.
- `vector` skeletonization is multi-scale: original mask plus lightly closed mask.
- `vector` snaps skeleton branch endpoints to local junction centers to reduce gaps.
- `vector` exports connected fitted primitives from the same contour as one continuous TikZ path.
- `vector` detects closed circle/ellipse/rectangle primitives when confidence is high.
- `threshold_block_size`: 35 pixels.
- `threshold_offset`: 3 in vector mode.
- `clahe_clip_limit`: 1.8.
- `blackhat_kernel_size`: 9 pixels.
- `blackhat_threshold`: 12 gray levels.
- `error_tolerance`: proportional per contour, clamped to `[0.01, 0.08]` TikZ units.
- `pre_simplify_tolerance`: proportional per contour, clamped to `[0.001, 0.005]` TikZ units.
- `min_points_for_bezier`: 4.

| Mode | TeX | SHA-256 | Size bytes | `\draw` | `--` | `.. controls` | Circles | Ellipses | Vector objects | PDF | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `classic` | `examples\comparison\classic_output.tex` | `fd2f2375aaf4f360feae4fb5983ad851efb69b62e20db6d5dd34af5be86a1e60` | 41723 | 297 | 1321 | 0 | 0 | 0 | 0 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\classic_output.pdf` | 0 |
| `vector_local` | `examples\comparison\vector_local_output.tex` | `eb3815464d1a4613b798867eeb30ed62e42fc1c4f36da71fc3b0babe39993dea` | 111083 | 368 | 128 | 1515 | 0 | 0 | 1600 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_local_output.pdf` | 0 |
| `vector_fitted` | `examples\comparison\vector_fitted_output.tex` | `c3ee3eefd1ff7dc15ac0d8dbdf5e487688f42bee5bab5471827f1447478af432` | 35920 | 368 | 603 | 125 | 0 | 0 | 413 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_fitted_output.pdf` | 0 |

## Visual Notes

- Adaptive preprocessing targets faint gray JPEG strokes that fixed thresholding loses before skeletonization.
- Black-hat recovery is intended for weak teeth, mouth, eyes, wrinkles, claws, and spiral strokes.
- Grouped paths reduce visible joins between consecutive fitted primitives.
- Primitive detection is conservative and appears only when a closed contour is unambiguous.
- This pass changes vector-mode stroke extraction and vector object generation only; it does not change the GUI layout.
