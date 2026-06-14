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
   - `Smooth` for experimental filtering, endpoint merging, smoothing, and
     Bezier paths;
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
4. Click **Gerar TikZ** if automatic regeneration is not enough.
5. Use **Visualizacao** to compare the original, overlay, and reconstructed
   drawing.
6. Export the result with **Arquivo > Exportar .tex**.

## Compile

Use **Configuracoes > Distribuicao LaTeX** to detect MiKTeX, TeX Live, MacTeX,
or select a manual executable path. Then choose **Compilar e visualizar PDF**.

If LaTeX is not installed, export the `.tex` file and compile it later in a
configured LaTeX environment.

## Compare Classic And Smooth

Use the files in `examples/comparison/` as a reproducible comparison:

- `original.jpg`;
- `classic_output.tex`;
- `classic_output.pdf`;
- `smooth_output.tex`;
- `smooth_output.pdf`;
- `notes.md`.

`Classic` is the rollback mode. `Smooth` should reduce fragmented strokes and
angular paths, but it can still lose very faint details if parameters are too
aggressive.
