# Pipeline Report

Input: `examples\comparison\original.jpg`

| Mode | TeX | SHA-256 | Draws | `--` | `.. controls` | Circles | Ellipses | Vector objects | PDF | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `classic` | `examples\comparison\classic_output.tex` | `fd2f2375aaf4f360feae4fb5983ad851efb69b62e20db6d5dd34af5be86a1e60` | 297 | 1321 | 0 | 0 | 0 | 0 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\classic_output.pdf` | 0 |
| `vector` | `examples\comparison\vector_output.tex` | `223c2319da8f19dbc7cdcea1064a283e1e9c0e81395c30ed8180b580a7efc615` | 1280 | 127 | 1194 | 0 | 0 | 1280 | `C:\Users\Fernando\Documents\GitHub\fikzPy\examples\comparison\vector_output.pdf` | 0 |

## Vector Object Breakdown

| Mode | Line | Polyline | BezierCurve |
| --- | ---: | ---: | ---: |
| `classic` | 0 | 0 | 0 |
| `vector` | 45 | 41 | 1194 |

## Notes

- `classic` uses the contour-to-TikZ path directly.
- `vector` uses contours -> internal vector objects -> TikZ exporter.
- `vector_output.tex` must contain `% FIKZPY VECTOR MODE` and the visual `VECTOR MODE` node during diagnostics.
- No computer-vision, smoothing, or Bezier-fitting parameters were changed in this diagnostic pass.
