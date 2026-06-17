# Usage

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m fikzpy.main
```

## Generate TikZ

1. Choose **Arquivo > Abrir imagem**.
2. Choose the vectorization mode:
   - `Classic` for the stable line-art behavior;
   - `Visual` for maximum visual fidelity with filled TikZ paths generated
     through SVG-style tracing, `svg2tikz`, and a post-processing pass that
     emits layered `\draw` commands;
   - `Contornos` for the classic Canny contour pipeline.
3. Adjust the parameters:
   - ink threshold for faint strokes;
   - stroke smoothing;
   - smoothing;
   - Canny low/high thresholds;
   - simplification;
   - TikZ scale;
   - line width;
   - line color;
   - Bezier mode.

   In `Visual` mode, ink threshold, stroke smoothing, smoothing, and
   simplification affect the filled-path trace and the generated PDF.
4. Click **Gerar TikZ** if automatic regeneration is not enough.
5. Use **Visualizacao** to compare the original, overlay, and reconstructed
   drawing.
6. Export the result with **Arquivo > Exportar .tex**.

## Compile

Use **Configuracoes > Distribuicao LaTeX** to detect MiKTeX, TeX Live, MacTeX,
or select a manual executable path. Then choose **Compilar e visualizar PDF**.

If LaTeX is not installed, export the `.tex` file and compile it later in a
configured LaTeX environment.

## Compare Modes

Use the files in `examples/comparison/` as a reproducible comparison:

- `original.jpg`;
- `classic_output.tex`;
- `classic_output.pdf`;
- `smooth_output.tex`;
- `smooth_output.pdf`;
- `visual_dinosaur_output.tex`;
- `visual_cara_output.tex`;
- `notes.md`.

`Classic` is the rollback mode. `Visual` should be used when fidelity matters
more than centerline editability; it traces the visible ink as filled paths
rather than as stroke centerlines, then groups the svg2tikz output into
`fikzInk` and `fikzErase` `\draw` layers. The erase layers assume the normal
white standalone page background. `Contornos` is useful for images where Canny
edge contours are a better fit.
