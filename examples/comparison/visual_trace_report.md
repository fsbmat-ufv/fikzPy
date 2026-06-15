# Visual Trace Report

Inputs:

- `tests/25.jpg`
- `tests/Cara.png`

| Input | Mode | TeX | PDF | SHA-256 | Size bytes | Paths | SVG bytes | TikZ bytes | svg2tikz |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tests/25.jpg` | `visual` | `examples/comparison/visual_dinosaur_output.tex` | `examples/comparison/visual_dinosaur_output.pdf` | `e4ad22f88e7bc512ddd0fc670389ca933aaaa060efce309b17e960a17056636b` | 38242 | 89 | 29877 | 38111 | True |
| `tests/Cara.png` | `visual` | `examples/comparison/visual_cara_output.tex` | `examples/comparison/visual_cara_output.pdf` | `0fbed29719be0b97b7eb779367c16589886f2fbbf60d8fc3ae8dcebaf24cfdb4` | 16826 | 55 | 11363 | 16695 | True |

## Notes

- `visual` traces filled ink shapes instead of stroke centerlines.
- Red diagnostic annotations are ignored when possible.
- If a red mark covers an original black stroke, the covered information cannot be recovered from this annotated PNG. Use a clean source image for final output.
- The clean `25.jpg` input is the better fidelity reference.
- This mode favors visual fidelity over short/editable TikZ.
