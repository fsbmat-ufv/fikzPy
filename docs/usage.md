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
   - `Line art` for drawings, sketches, and images with internal strokes;
   - `Contornos` for the classic Canny contour pipeline.
3. Adjust the parameters:
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
