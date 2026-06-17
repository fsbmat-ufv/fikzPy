# fikzPy

fikzPy is a desktop Python application for converting image contours into clean,
editable TikZ code for LaTeX. The project is designed for academic and teaching
workflows where a raster image can serve as the starting point for a compact
vector drawing.

The current version is an MVP: it opens an image, traces line-art strokes or
detects contours with OpenCV, simplifies the resulting paths, generates TikZ
`\draw` commands, previews the detected strokes, and exports a standalone `.tex`
file.

## Goals

- Load common raster image formats.
- Trace line-art drawings with a small, transparent computer vision pipeline.
- Keep a classic Canny/contour mode for geometric or filled images.
- Generate minimal TikZ using `\draw`, paths, coordinates, scopes, and optional
  cubic Bezier curves.
- Keep generated code concise enough to edit by hand.
- Provide a desktop interface for image preview, overlay preview, code editing,
  copying, exporting, and optional LaTeX compilation.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python -m fikzpy.main
```

After installing the package in editable mode, the console entry point is also
available:

```powershell
python -m pip install -e .
fikzpy
```

## Basic Workflow

1. Open an image with **Arquivo > Abrir imagem**.
2. Choose the vectorization mode. The GUI exposes three primary modes:
   `Classic` for stable editable strokes, `Visual` for maximum visual
   similarity with filled TikZ paths, and `Contornos` for the Canny contour
   pipeline.
3. Adjust ink threshold, stroke smoothing, simplification, TikZ scale, line
   width, line color, and Bezier usage in the parameter panel. In `Visual`
   mode, ink threshold, stroke smoothing, smoothing, and simplification affect
   the filled-path trace. In `Contornos` mode, the Canny thresholds are also
   used.
4. Review the generated TikZ code on the right.
5. Toggle the preview between the original image, contour overlay, and
   reconstructed drawing.
6. Copy the TikZ code or export a standalone `.tex` file.
7. If a LaTeX distribution is installed, compile and open the generated PDF.

## LaTeX Support

fikzPy can use:

- MiKTeX;
- TeX Live;
- MacTeX;
- a manual path to `pdflatex`, `xelatex`, or `lualatex`.

The application first searches the system `PATH`, then common installation
directories. A manual executable path can be configured from the settings menu.

## Project Structure

```text
fikzpy/
  core/        image processing, contour extraction, TikZ generation, LaTeX
  gui/         PySide6 desktop interface
  templates/   reusable TikZ examples inspired by editor snippets
docs/          GitHub Pages-ready documentation
examples/      sample input and output files
scripts/       packaging and documentation helpers
tests/         focused unit tests
```

## Tests

```powershell
python -m pytest
```

## Build a Windows Executable

```powershell
python scripts/build_exe.py
```

The script wraps the PyInstaller command:

```powershell
pyinstaller --onefile --windowed --name fikzPy fikzpy/main.py
```

## Conceptual References

fikzPy uses its own architecture and implementation, but it is conceptually
informed by existing TikZ/Python projects and TikZ editors:

- [tikzpy](https://github.com/narfisaeu/tikzpy/tree/master)
- [Tikz-Python](https://github.com/ltrujello/Tikz-Python)
- [sane_tikz](https://github.com/negrinho/sane_tikz)
- [pytikz](https://github.com/pglammers/pytikz)
- TikzEdt snippets and syntax organization, inspected locally from
  `C:\Users\Fernando\Documents\GitHub\KTikz\TikzEdtBeta0_2_3`

No code was copied from these projects.

## Vectorization Notes

Raster images need a raster-to-vector step before TikZ can be generated. The
GUI exposes the three primary tracing modes:

- `Classic`: stable line-art tracing, kept as the rollback path.
- `Visual`: filled ink-shape tracing through SVG-style paths and `svg2tikz`.
  This is the highest visual-fidelity mode, but the output is less compact and
  less hand-editable than centerline `\draw` paths.
- `Contornos`: classic Canny/contour tracing for geometric or filled images.

The codebase also keeps experimental centerline modes such as `Vector`,
`Fidelidade`, and `Smooth` for tests and future work, but they are no longer
shown in the main GUI selector.

The centerline backends follow this general sequence:

1. threshold dark ink;
2. skeletonize strokes;
3. trace open and closed paths;
4. smooth pixel stair-stepping;
5. simplify paths;
6. emit editable TikZ.

The `Visual` backend follows a different sequence:

1. enhance local contrast and threshold likely black ink;
2. ignore strongly chromatic diagnostic annotations when possible;
3. trace the outer boundary of the ink shapes;
4. fit SVG-style cubic paths;
5. convert the SVG path to TikZ with `svg2tikz`;
6. emit filled `\path[fill=black, even odd rule]` commands.

Use `Visual` when the PDF must look close to the source image. Use `Classic`
when the priority is editable mathematical strokes.

## Comparison Example

The folder `examples/comparison/` contains a reproducible before/after sample:

- `original.jpg`;
- `classic_output.tex` and `classic_output.pdf`;
- `smooth_output.tex` and `smooth_output.pdf`;
- `visual_dinosaur_output.tex`;
- `visual_cara_output.tex`;
- `notes.md` with path and point counts.

To return to the previous behavior in the GUI, select `Classic` in the mode
combo box.

## Roadmap

- Improve Bezier fitting and path reduction.
- Add richer TikZ syntax highlighting in the editor.
- Add PDF preview embedded in the interface.
- Add persistent project settings.
- Add more examples and gallery documentation.
- Add CI for tests and packaging.

## Repository

Prepared for:

```text
https://github.com/fsbmat-ufv/fikzPy.git
```

Push only after local validation and review.
