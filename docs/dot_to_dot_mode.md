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

- **Figura inteira (`full_figure`)** — automatic. The main external contour of
  the figure is identified, removed, and replaced by the exact number of points
  requested. Internal details (eyes, mouth, spots, …) can be kept drawn.
  Optionally, points can also be distributed across the relevant internal
  contours (see below).
- **Selecionar trechos (`selected_segments`)** — interactive. The user clicks a
  start and an end point on the canvas; the clicks snap to the nearest path and
  the arc between them is highlighted. Each confirmed arc is removed and
  replaced by its own set of points, and numbering continues automatically from
  one arc to the next.

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
4. [`path_sampler`](../fikzpy/core/dot_to_dot/path_sampler.py) samples the
   points, identifies the main contour and distributes points across internal
   contours; [`segment_selection`](../fikzpy/core/dot_to_dot/segment_selection.py)
   handles snapping and arc removal for the interactive submode.
5. [`label_placement`](../fikzpy/core/dot_to_dot/label_placement.py) offsets each
   number along the outward normal of the curve.
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
  dotToDotLabel/.style={font=\sffamily\small,inner sep=1pt},
}
```

Each point and its number are then:

```latex
\fill[dotToDotPoint] (x,y) circle[radius=2pt];
\node[dotToDotLabel,anchor=south] at (xl,yl) {1};
```

## Identifying the main contour

The main contour is chosen conservatively: it is the closed subpath with the
largest area (falling back to the largest arc length), **after** excluding:

- a full-canvas background/frame rectangle (a near-rectangular shape whose
  bounding box covers ≳98% of the canvas);
- tiny noise (area below a small fraction of the canvas and negligible length).

This avoids automatically selecting the page/canvas, an outer frame, small
noise or white cut-out masks as the main outline.

## Handling disconnected contours

When **Incluir contornos internos** is enabled, the requested total number of
points is distributed across the relevant contours proportionally to their
length, with a minimum number of points per contour, and largest-remainder
rounding so the counts always sum **exactly** to the requested total.

Because disconnected contours must not be joined, each contour becomes its own
independent numbered sequence with a distinct prefix:

- main contour: `1, 2, 3, …`
- second contour: `A1, A2, A3, …`
- third contour: `B1, B2, B3, …`

In the interactive submode the numbering instead continues across the selected
arcs as a single sequence (e.g. arc 1 gets points 1–10, arc 2 gets 11–20).

## Removing selected arcs

Selected arcs are removed *mathematically*, not hidden behind a white band. The
subpath is split with exact De Casteljau splits so that lines, quadratic and
cubic Bezier segments, closed paths, subpaths and multiple arcs per path all
reconstruct the remaining drawing correctly. On a closed path there are two
possible arcs between the two clicks; one is shown initially and **Inverter
trecho** switches to the other.

## Known limitations

- Internal details kept by `preserve_internal_details` are rendered as stroked
  outlines, not re-filled, so a solid feature becomes an outline.
- The main-contour heuristic is deliberately conservative; on unusual figures
  the intended outline may need to be selected manually with *Selecionar
  trechos*.
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
   size, number distance, direction and the internal-contour options, then click
   *Gerar/atualizar previsualizacao*.
3. For **Selecionar trechos**: pick *Selecionar trechos*, click a start and an
   end point on the canvas, optionally *Inverter trecho*, set the number of
   points, *Confirmar trecho*, repeat for further arcs, then *Gerar TikZ*.

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
