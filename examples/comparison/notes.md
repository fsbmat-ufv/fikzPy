# fikzPy Vectorization Comparison

Input: `original.jpg`

| Mode | Strokes | Points | TikZ lines | PDF |
| --- | ---: | ---: | ---: | --- |
| `classic` | 297 | 1618 | 1622 | ok |
| `smooth` | 234 | 1231 | 1235 | ok |

## Notes

- `classic` preserves the stable line-art backend used before this branch.
- `smooth` applies optional preprocessing, conservative endpoint merging, path smoothing, and Bezier TikZ generation.
- `smooth` intentionally remains conservative to avoid mixing independent details such as teeth and mouth strokes.
- PDF files are generated artifacts and can be regenerated from the `.tex` files.
