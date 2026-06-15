# Visual Trace Report

Inputs:

- `tests/25.jpg`
- `tests/Cara.png`

| Input | Mode | TeX | PDF | SHA-256 | Size bytes | Paths | SVG bytes | TikZ bytes | svg2tikz |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tests/25.jpg` | `visual` | `examples/comparison/visual_dinosaur_output.tex` | `examples/comparison/visual_dinosaur_output.pdf` | `98c1ffdb05339a9b4642a2923d9dced90c89e5ef63487db0df55f879a683384c` | 39037 | 89 | 30386 | 38906 | True |
| `tests/Cara.png` | `visual` | `examples/comparison/visual_cara_output.tex` | `examples/comparison/visual_cara_output.pdf` | `27fd8939ad7fe31fe79e8c2ebb57dc7772fba88cbb2e97e4b602fa6d58c056ab` | 17280 | 55 | 11644 | 17149 | True |

## Notes

- `visual` traces filled ink shapes instead of stroke centerlines.
- Red diagnostic annotations are ignored when possible.
- If a red mark covers an original black stroke, the covered information cannot be recovered from this annotated PNG. Use a clean source image for final output.
- The clean `25.jpg` input is the better fidelity reference.
- This mode favors visual fidelity over short/editable TikZ.
