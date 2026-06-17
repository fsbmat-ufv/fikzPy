# Primitive Fitting

## Purpose

Primitive fitting is an isolated Issue 7 layer for the future semantic Classic
pipeline. It consumes ordered geometry that already exists in memory and
returns semantic primitives when a simpler representation fits within a
configured error tolerance.

It does not run image preprocessing, tracers, SVG-to-TikZ bridges, TikZ
generation, PDF compilation, preview code, GUI code, Classic routing, Visual
mode, or Contornos mode.

## Primitive Order

Candidates are built in this order:

1. Existing semantic primitive preservation
2. Point
3. Line
4. Circle
5. Ellipse
6. Simplified polyline
7. Cubic Bezier sequence
8. Closed freeform fallback

The selected candidate must satisfy its fidelity tolerance first. Among valid
candidates, the fitter chooses the simplest representation using parameter
complexity, geometric error, object count, and deterministic kind order.

## Normalization

Before fitting, geometry is copied into `Point2D` values, finite coordinates
are required, consecutive duplicates are removed, explicit closure is
preserved, inferred closure is detected from endpoint distance, and diagnostics
record bounding box, diagonal scale, total path length, duplicate count, and
corner indices.

The original input object is not modified.

## Line Fitting

Lines use orthogonal least squares through PCA:

- centroid calculation;
- principal direction from the largest covariance eigenvector;
- endpoint projection onto the principal axis;
- maximum and RMS orthogonal error;
- path stretch check to reject curved paths that happen to have low average
  deviation;
- confidence from normalized error.

Closed paths are not flattened into lines unless they are point-like.

## Circle Fitting

Circles use deterministic algebraic least squares:

```text
x^2 + y^2 = 2 cx x + 2 cy y + c
```

The fitter records center, radius, radial max/RMS error, angular coverage,
path direction, closure, and confidence. Partial circular arcs are not emitted
as circles because the geometry model does not yet have an arc primitive.

## Ellipse Fitting

Ellipses use a PCA-based full-contour estimate. The covariance axes provide
major/minor directions and radii for complete sampled ellipses. Error is an
approximate normalized radial distance in the ellipse frame, scaled back by
mean axis length.

Circle fitting runs before ellipse fitting, so circles are not needlessly
classified as ellipses.

## Corners

Corner detection uses discrete turn angles between adjacent segments. Strong
turns are preserved by rejecting line and Bezier candidates that would smooth
them away. Circle and ellipse candidates still measure their actual geometric
error, which lets polygonal traced circles be recognized while rectangles and
teeth remain polylines.

## Bezier Fitting

Bezier fitting is implemented in `fikzpy.core.bezier_fitting` with:

- chord-length parameterization;
- endpoint tangents;
- least-squares control distances;
- one deterministic Newton reparameterization pass;
- maximum-error evaluation;
- recursive splitting at the largest error;
- segment-count and recursion-depth limits;
- rejection of line-like, tiny, and unstable curves.

The output uses a small number of cubic `BezierPrimitive` objects rather than
one curve per input segment.

## Closed Shapes

Closed paths preserve closure. The fitter tries circle and ellipse before
Bezier and freeform fallbacks. When no simple primitive is reliable, closed
geometry remains a closed polyline or `ClosedShapePrimitive`; no hole
reconstruction or boolean operation is performed.

## Error Metrics

`fikzpy.core.geometry_error` centralizes reusable metrics:

- point-to-line distance;
- point-to-circle radial error;
- approximate point-to-ellipse error;
- point-to-polyline distance;
- point-to-Bezier sampled distance;
- RMS and max error;
- bounding-box diagonal normalization;
- angular coverage;
- closure error.

## Confidence And Ambiguity

Candidate confidence is derived from normalized error relative to the
candidate tolerance, then adjusted by coverage where appropriate. If another
accepted non-fallback candidate has nearly the same error or confidence, the
result is marked ambiguous and records the alternative kind. Selection remains
deterministic.

## Styles And Metadata

New primitives copy `StrokeStyle`, `FillStyle`, opacity, source metadata, SVG
ids, tracer labels, centerline metadata, and other source fields. When one
input becomes multiple Beziers, each primitive receives the same style plus a
`fit_segment_index`.

Existing `PointPrimitive`, `LinePrimitive`, `CirclePrimitive`,
`EllipsePrimitive`, and default `BezierPrimitive` inputs are preserved instead
of being reconverted.

## API

```python
from fikzpy.core.primitive_fitting import fit_primitive, fit_primitives

result = fit_primitive(geometry)
many = fit_primitives([geometry_a, geometry_b])
```

Supported inputs include:

- `CenterlinePath`;
- `PolylinePrimitive`;
- `ClosedShapePrimitive`;
- `BezierPrimitive`;
- `PrimitiveGroup`;
- `SvgParseResult`;
- ordered `Point2D` or `(x, y)` sequences.

The object-oriented API is also available:

```python
from fikzpy.core.primitive_fitting import PrimitiveFitter

fitter = PrimitiveFitter()
result = fitter.fit(geometry)
```

## Configuration

`PrimitiveFittingConfig` centralizes fit toggles, tolerances, point minimums,
coverage thresholds, corner policy, existing-primitive preservation,
Bezier recursion limits, decimal precision, confidence threshold, and
ambiguity margin. Invalid values are rejected at construction time.

## Limitations

- No `ArcPrimitive` exists yet, so partial arcs remain Beziers or polylines.
- Ellipse fitting is tuned for complete sampled ellipses.
- Corner-aware decomposition into line-plus-curve groups is conservative; full
  semantic merging belongs to later roadmap work.
- No TikZ exporter is implemented here.

## Relationship To Issue 8

Issue 7 only performs local fitting needed to recognize primitives. It does
not implement global simplification, merging compatible objects, style
grouping, or semantic optimization. Those belong to Issue 8.

## Visual Separation

Visual mode remains separate. The new modules are not imported by
`visual_pipeline.py`, `visual_postprocessor.py`, `tikz_pipeline.py`, or
`image_processor.py`, and no GUI route calls primitive fitting in this issue.
