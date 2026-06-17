# Visual Trace Report

Inputs:

- `tests/25.jpg`
- `tests/Cara.png`

| Input | Mode | TeX | PDF | SHA-256 | Size bytes | Subpaths | Draw commands | Postprocessed |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `tests/25.jpg` | `visual` | `examples/comparison/visual_dinosaur_output.tex` | `examples/comparison/visual_dinosaur_output.pdf` | `270d1e882d780fcaf631c5ffda0958dff069a84f508a6465e321cb116f95a227` | 38838 | 89 | 3 | True |
| `tests/Cara.png` | `visual` | `examples/comparison/visual_cara_output.tex` | `examples/comparison/visual_cara_output.pdf` | `5821d52c7a35a2cdf3c83d997eb9c24c06381cf173f645bdaecc89c1e2a70e56` | 17233 | 55 | 2 | True |

## Notes

- `visual` traces filled ink shapes instead of stroke centerlines.
- The svg2tikz monolithic path is post-processed into layered `\draw[fikzInk]` and `\draw[fikzErase]` commands.
- Ink layers preserve black filled shapes; erase layers preserve holes on the white standalone page.
- Cubic Bezier segments emitted by svg2tikz are preserved.
- Red diagnostic annotations are ignored when possible.
- If a red mark covers an original black stroke, the covered information cannot be recovered from this annotated PNG. Use a clean source image for final output.
- The clean `25.jpg` input is the better fidelity reference.
