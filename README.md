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
2. Choose the vectorization mode. `Classic` preserves the stable line-art
   backend, `Smooth` enables the experimental cleanup/smoothing backend, and
   `Contornos` keeps the Canny contour pipeline.
3. Adjust ink threshold, stroke smoothing, simplification, TikZ scale, line
   width, line color, and Bezier usage in the parameter panel. In `Contornos`
   mode, the Canny thresholds are also used. In `Smooth` mode, Bezier output is
   enabled automatically.
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

Raster images need a raster-to-vector step before TikZ can be generated. For
line drawings, fikzPy provides two stroke-tracing modes:

- `Classic`: stable line-art tracing, kept as the rollback path.
- `Smooth`: experimental preprocessing, conservative contour merging, path
  smoothing, and Bezier generation.

The line-art backend follows this general sequence:

1. threshold dark ink;
2. skeletonize strokes;
3. trace open and closed paths;
4. smooth pixel stair-stepping;
5. simplify paths;
6. emit editable TikZ.

`svg2tikz` is a strong candidate for a future optional backend, but it converts
existing SVG paths to TikZ. It does not by itself solve JPEG/PNG recognition, so
it should be paired with an SVG vectorizer such as Inkscape Trace Bitmap or
potrace.

## Comparison Example

The folder `examples/comparison/` contains a reproducible before/after sample:

- `original.jpg`;
- `classic_output.tex` and `classic_output.pdf`;
- `smooth_output.tex` and `smooth_output.pdf`;
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
