# fikzPy

fikzPy is a desktop Python application for converting image contours into clean,
editable TikZ code for LaTeX. The project is designed for academic and teaching
workflows where a raster image can serve as the starting point for a compact
vector drawing.

The current version is an MVP: it opens an image, detects contours with OpenCV,
simplifies the resulting paths, generates TikZ `\draw` commands, previews the
detected strokes, and exports a standalone `.tex` file.

## Goals

- Load common raster image formats.
- Detect edges and contours with a small, transparent computer vision pipeline.
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
2. Adjust smoothing, Canny thresholds, simplification, TikZ scale, line width,
   line color, and Bezier usage in the parameter panel.
3. Review the generated TikZ code on the right.
4. Toggle the preview between the original image, contour overlay, and
   reconstructed drawing.
5. Copy the TikZ code or export a standalone `.tex` file.
6. If a LaTeX distribution is installed, compile and open the generated PDF.

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
