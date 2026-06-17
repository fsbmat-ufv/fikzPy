# Max Fidelity Report

Input: `tests/25.jpg`

| Mode | TeX | SHA-256 | Size bytes | `\draw` | `--` | `.. controls` | Objects | Beziers | Return code |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vector` | `examples\comparison\vector_fitted_output.tex` | `c3ee3eefd1ff7dc15ac0d8dbdf5e487688f42bee5bab5471827f1447478af432` | 35920 | 368 | 603 | 125 | 413 | 125 | 0 |
| `fidelity` | `examples\comparison\fidelity_output.tex` | `a0c860a4bee57e2882b4fb28e6a9e84f495cc907a8ebae88618517de94ccd07c` | 45041 | 372 | 689 | 241 | 528 | 241 | 0 |

## Notes

- `fidelity` uses lower simplification and tighter Bezier fitting than `vector`.
- The goal is maximum visual fidelity, accepting larger TikZ output.
- The GUI layout is unchanged; choose `Fidelidade` in the existing mode selector.
