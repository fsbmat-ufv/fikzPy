# Visual Validation and Fidelity Score

Issue 10 adds an isolated validation layer for the future semantic Classic
pipeline. It measures whether fitted semantic primitives preserve the visual
content of the source image and whether the generated TikZ remains compact,
semantic, and editable.

This layer does not connect the semantic pipeline to the Classic button. It
does not alter Classic, Visual, Contornos, preview, PDF compilation, or the
existing Visual SVG conversion path.

## Purpose

The validator balances:

- visual fidelity;
- preservation of filled dark regions;
- preservation of thin strokes and small details;
- geometric complexity;
- TikZ code size;
- TikZ semantic readability;
- regression flags for known failure modes.

It is designed for Issue 11, where the Classic integration can use these
metrics to decide whether a semantic result is acceptable.

## Semantic Rasterization

`rasterize_semantic_primitives(primitives, config=None)` renders internal
semantic primitives with a small deterministic PIL/ImageDraw backend. It is
used for tests and validation without requiring LaTeX, Poppler, Inkscape,
external tracers, or the GUI.

Supported primitive inputs include:

- `PointPrimitive`;
- `LinePrimitive`;
- `PolylinePrimitive`;
- `CirclePrimitive`;
- `EllipsePrimitive`;
- `BezierPrimitive`, sampled deterministically;
- `ClosedShapePrimitive`;
- `PrimitiveGroup`;
- `GeometryOptimizationResult`;
- `PrimitiveFitResult`;
- sequences of `PrimitiveFitResult`;
- `SvgParseResult`;
- `CenterlineResult` and `CenterlinePath`, converted through their polyline
  representation.

The rasterizer supports stroke color, fill color, approximate line width,
basic opacity, scale, canvas size, background color, and optional y-axis
inversion when configured.

## Raster Metrics

`compare_rasters()` computes deterministic metrics from a source raster and a
rendered candidate:

- mean absolute error;
- root mean squared error;
- normalized RMSE;
- foreground IoU;
- foreground precision, recall, and F1;
- false positive and false negative rates;
- edge overlap, recall, precision, and F1;
- structural proxy score;
- connected component difference;
- bounding box difference;
- centroid shift;
- area ratio.

Foreground is dark-pixel based by default. Thresholds are configurable through
`RasterMetricsConfig`, `FidelityScoreConfig`, or `VisualValidationConfig`.

## Filled Regions

Large connected foreground components are tracked separately from general
foreground pixels. The validator reports:

- `filled_region_recall`;
- `large_dark_region_recall`;
- `dark_mass_preservation_ratio`;
- `double_outline_penalty`;
- `foreground_fragmentation_delta`.

These metrics catch the failure where a filled black region is replaced by a
thin outline or disappears entirely.

## Thin Strokes

Thin-stroke preservation is measured with a distance-transform mask over the
source foreground. The validator reports:

- `thin_stroke_recall`;
- `small_detail_recall`;
- edge recall and precision.

This separates facial features, clothing lines, annotations, and small marks
from larger filled regions.

## Complexity Metrics

`compute_complexity_metrics(primitives=None, tikz_code=None, config=None)`
analyzes both semantic primitives and TikZ code.

Primitive metrics include:

- primitive count;
- primitive counts by type;
- group count and maximum group depth;
- point count;
- linear segment count;
- Bezier segment count;
- closed path count;
- fill count;
- distinct style count;
- optimization operation count when present.

TikZ metrics include:

- line count;
- character count;
- `\draw` command count;
- `\path` command count;
- coordinate count;
- Bezier control count;
- named style count;
- average command length;
- long-line count;
- repeated style count;
- semantic primitive count for constructs such as `circle`, `ellipse`, and
  `.. controls`;
- raw path penalty;
- editability score;
- semantic compactness score;
- complexity score.

The goal is not to minimize characters at any cost. Compact semantic TikZ is
favored over long raw path-like code.

## Composite Score

`compute_fidelity_score()` returns 0..1 scores:

- `overall_score`;
- `fidelity_score`;
- `complexity_score`;
- `semantic_score`;
- `regression_score`;
- `filled_region_score`;
- `thin_stroke_score`;
- `code_readability_score`.

Default weights are explicit:

- visual fidelity: `0.50`;
- filled region preservation: `0.15`;
- thin stroke preservation: `0.10`;
- semantic compactness: `0.15`;
- code readability: `0.10`.

Weights can be overridden through `FidelityScoreConfig.weights` or
`VisualValidationConfig.weights`.

## Acceptance Criteria

The result is rejected when one or more configured criteria fail:

- overall score below `minimum_acceptable_score`;
- fidelity score below `minimum_fidelity_score`;
- filled region score below `minimum_filled_region_recall`;
- thin stroke score below `minimum_thin_stroke_recall`;
- foreground IoU is too low for a non-empty source;
- rendered output is practically empty;
- large dark regions are lost;
- TikZ looks raw and non-semantic;
- complexity score is invalid.

With `strict=True`, critical validation failures raise an exception. With
`strict=False`, the result contains warnings, flags, and `accepted=False`.

## Regression Flags

Regression flags are deterministic strings. Current flags include:

- `invisible_output`;
- `near_empty_output`;
- `dark_mass_loss`;
- `missing_large_dark_regions`;
- `low_filled_region_recall`;
- `lost_thin_details`;
- `excessive_false_foreground`;
- `double_outline_suspected`;
- `foreground_fragmented`;
- `over_simplified`;
- `excessive_complexity`;
- `non_semantic_tikz`.

## Mixed Monochrome Critical Case

The validator includes a regression test for mixed monochrome drawings with:

- thin facial, clothing, and limb strokes;
- large filled black hair or clothing regions;
- small details;
- white interior holes;
- roughly 18 percent dark pixels in the original.

A bad output that keeps only fragmented thin strokes and preserves about one
percent of the dark mass is rejected with flags such as
`missing_large_dark_regions`, `low_filled_region_recall`, and `dark_mass_loss`.
Issue 10 only detects this failure. It does not implement the hybrid strategy
that may address it in Issue 11.

## Programmatic API

```python
from fikzpy.core.visual_validation import validate_semantic_output

result = validate_semantic_output(
    source_image,
    primitives,
    tikz_result=tikz_result,
)

if result.accepted:
    print(result.fidelity_score.overall_score)
else:
    print(result.regression_flags)
```

Lower-level APIs are also available:

```python
from fikzpy.core.semantic_rasterizer import rasterize_semantic_primitives
from fikzpy.core.fidelity_score import compute_fidelity_score
from fikzpy.core.complexity_metrics import compute_complexity_metrics

rendered = rasterize_semantic_primitives(primitives).image
complexity = compute_complexity_metrics(primitives, tikz_code=tikz_code)
score = compute_fidelity_score(source_image, rendered, complexity)
```

## TikZ Rendering

External TikZ/PDF rendering is optional and not required for the core tests.
When `use_external_tikz_renderer=True`, the current validator records a warning
that external rendering is unavailable and continues with the semantic
rasterizer. This keeps the validation deterministic and independent of local
LaTeX installations.

## Debug Images

Debug images are disabled by default. When `save_debug_images=True`, callers
must provide `debug_output_dir`; the validator writes `source.png` and
`rendered.png` there. It does not write into app temporary folders.

## Determinism

For the same input and configuration, the validator returns the same:

- raster metrics;
- complexity metrics;
- scores;
- warnings;
- regression flags;
- acceptance decision;
- deterministic hash;
- `to_dict()` output.

## Limitations

The semantic rasterizer is an approximation. It is intentionally lightweight
and does not replace TeX rendering. It approximates line joins, anti-aliasing,
opacity, and rotated ellipses. The metrics are tuned for deterministic
acceptance checks, not perceptual proof of visual identity.

## Relationship to Issue 11

Issue 10 creates measurement and detection only. Issue 11 will integrate the
semantic Classic path and can use these metrics to decide whether to accept a
semantic output, fall back, or choose a hybrid strategy.

The Visual mode remains separate. This module does not call or modify the
Visual pipeline, its postprocessor, the GUI, preview, or PDF compilation.
