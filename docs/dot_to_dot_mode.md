# Ponto a Ponto (Dot-to-Dot) mode

## Objective

The **Ponto a Ponto** mode turns a figure into a numbered *dot-to-dot* drawing:
a chosen outline is removed and replaced by a sequence of numbered points that a
reader connects in order. It is one of the five public modes exposed by the GUI
(`Classic Auto`, `Classic Filled`, `Visual`, `Contornos`, `Ponto a Ponto`).

The mode has its own self-contained pipeline in
[`fikzpy/core/dot_to_dot/`](../fikzpy/core/dot_to_dot). It reuses the
intermediate SVG produced by the Visual tracing stage, but always works on an
independent copy and never changes the behaviour of the Visual, Contornos,
Classic Auto or Classic Filled modes. There is **no silent fallback** between
modes: if the dot-to-dot pipeline cannot produce a result it raises
`DotToDotError`.

## Submodes: Figura inteira vs Selecionar trechos

- **Figura inteira (`full_figure`)** — automatic. The figure is consolidated
  into a **single external silhouette**, which is removed and replaced by the
  exact number of points requested, numbered `1..N` in one sequence. By default
  nothing else is drawn; the optional **Manter detalhes internos** checkbox
  (off by default) keeps interior details (eyes, mouth, spots, …) as unnumbered
  lines without ever restoring the replaced outline.
- **Selecionar trechos (`selected_segments`)** — interactive. The user clicks a
  start and an end point directly **on the main preview image**; the clicks
  snap to the nearest traced path and the arc between them is highlighted.
  Each confirmed arc is removed and replaced by its own set of points, and
  numbering continues automatically from one arc to the next.

## Flow: SVG → geometry → TikZ

1. The image is traced with the **Visual** tracing stage to obtain ink
   contours, and those contours are serialised to an SVG string
   (`contours_to_svg`). Dot-to-dot receives a *copy* of that string.
2. [`svg_loader`](../fikzpy/core/dot_to_dot/svg_loader.py) parses the SVG into
   an independent geometric model (`SvgPath` → `SubPath` → line/quadratic/cubic
   `Segment`s), applying element and group transforms via the project's existing
   `svg_transforms` helpers.
3. [`models`](../fikzpy/core/dot_to_dot/models.py) provides arc-length aware
   subpaths. Points are distributed **equidistantly along the real arc length**
   of the curve, not along the raw Bezier parameter. For a closed path with `N`
   points the positions are `s_i = i·L/N` for `i = 0 … N-1`, so a point is never
   placed at both `s = 0` and `s = L`.
4. [`silhouette`](../fikzpy/core/dot_to_dot/silhouette.py) rasterizes the copy
   into a monochrome mask, merges the visible elements with a size-proportional
   morphological closing, keeps the main connected region and extracts its
   single external contour (OpenCV `RETR_EXTERNAL` + moderate `approxPolyDP`);
   [`segment_selection`](../fikzpy/core/dot_to_dot/segment_selection.py)
   handles snapping and arc removal for the interactive submode.
5. [`label_placement`](../fikzpy/core/dot_to_dot/label_placement.py) computes
   the outward normal of each point and
   [`label_layout`](../fikzpy/core/dot_to_dot/label_layout.py) places the
   numbers with greedy collision avoidance (against other labels, other dots
   and the silhouette interior), keeping them predominantly outside the figure.
6. [`tikz_exporter`](../fikzpy/core/dot_to_dot/tikz_exporter.py) emits the TikZ:
   the remaining drawing is converted with the same coordinate transform used
   elsewhere in the project (`image_point_to_tikz`: pixels → TikZ units, y-axis
   pointing up), and the points and numbers are emitted as **native TikZ**.

## Isolation from the Visual mode

- Dot-to-dot only *reads* the Visual intermediate SVG, always as a copy of an
  immutable string; it never mutates it and never re-runs the Visual SVG/svg2tikz
  conversion.
- The numbers are **never** routed through `svg2tikz` — they are native
  `\node` commands, so they remain editable.
- The Visual/Contornos/Classic pipelines are untouched; selecting Ponto a Ponto
  runs a separate branch in `build_tikz_from_image`.

## TikZ output

Reusable styles are emitted once per picture:

```latex
\tikzset{
  dotToDotPoint/.style={fill=black,draw=black},
  dotToDotLabel/.style={font=\sffamily\scriptsize,inner sep=1pt},
}
```

Each point and its number are then:

```latex
\fill[dotToDotPoint] (x,y) circle[radius=2pt];
\node[dotToDotLabel,anchor=south] at (xl,yl) {1};
```

## Identifying the silhouette

The silhouette is a *consolidated* view of the figure, not the largest
isolated subpath: every visible subpath (excluding a full-canvas
background/frame rectangle) is union-filled into a raster mask, a
morphological closing proportional to the image size joins strokes that were
drawn separately, the largest connected region is kept, and only its external
contour is used. White cut-outs, sibling ribbon boundaries and small noise
therefore never become "the outline".

Points run **clockwise** by default (invertible), starting at the topmost
point of the silhouette (ties broken to the left) or at the position nearest
an optional `start_point_hint`. Labels are always the pure numeric sequence
`start_number..start_number+N-1` — letter-prefixed sequences no longer exist.

In the interactive submode the numbering continues across the selected arcs
as a single sequence (e.g. arc 1 gets points 1–10, arc 2 gets 11–20).

## Removing selected arcs

Selected arcs are removed *mathematically*, not hidden behind a white band. The
subpath is split with exact De Casteljau splits so that lines, quadratic and
cubic Bezier segments, closed paths, subpaths and multiple arcs per path all
reconstruct the remaining drawing correctly. On a closed path there are two
possible arcs between the two clicks; one is shown initially and **Inverter
trecho** switches to the other.

## Remaining drawing as filled ink

The drawing that is *kept* (internal details in *Figura inteira*, everything
except the selected arcs in *Selecionar trechos*) is re-rendered as **filled
even-odd ink** — the same representation the Visual mode uses — instead of
stroked ribbon outlines, which would look like the Contornos mode. The kept
geometry is rasterized faithfully (even-odd, per original path), the corridors
around the selected arcs are erased with a width proportional to the estimated
stroke thickness, and the surviving ink is re-vectorized into smooth closed
subpaths emitted as one `\draw[dotToDotInk]` command. The removed trechos are
therefore genuinely absent from the output geometry — nothing is hidden under
white bands.

## Known limitations

- The silhouette is the outer boundary of the merged figure: fully disconnected
  secondary figures (e.g. a separate cloud) are dropped in *Figura inteira*;
  use *Selecionar trechos* to dot them.
- The morphological closing may bridge very close but genuinely separate
  strokes; its radius is proportional to the image size, not user-tunable.
- Selected segments are stored as `(path_id, subpath_id, length ratios)`. If the
  tracing parameters are changed between selecting and generating, the trace can
  change and a selection may no longer resolve; the pipeline then raises a clear
  error instead of silently guessing.
- Elliptical-arc (`A`) SVG commands are approximated by a straight chord (the
  project's SVG stage does not emit them).

## Basic usage

From the GUI:

1. Open an image and select **Ponto a Ponto**.
2. For **Figura inteira**: set the number of points, starting number, point
   size, number distance, direction and (optionally) *Manter detalhes
   internos*, then click *Gerar/atualizar previsualizacao*.
3. For **Selecionar trechos**: pick *Selecionar trechos* — the **main preview
   image** becomes interactive. Click a start and an end point on the figure,
   optionally *Inverter*, set the number of points, *Confirmar*, repeat for
   further arcs, then *Gerar TikZ*.

Programmatically:

```python
from fikzpy.core.dot_to_dot import DotToDotConfig, run_dot_to_dot

result = run_dot_to_dot(
    svg_source,            # a copy of the Visual intermediate SVG
    image_shape,           # (height, width)
    DotToDotConfig(total_points=20),
    width_units=10.0,
)
print(result.tikz_code)
```
