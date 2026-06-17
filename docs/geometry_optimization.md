# Geometry Optimization

## Purpose

Geometry optimization is the isolated Issue 8 layer for the future semantic
Classic pipeline. It consumes semantic primitives that already exist in memory
and returns a smaller, cleaner, and still faithful primitive sequence.

It does not run image preprocessing, tracers, SVG-to-TikZ bridges, TikZ
generation, LaTeX compilation, preview code, GUI code, Classic routing, Visual
mode, or Contornos mode.

## Optimization Flow

The public optimizer uses conservative passes:

1. copy and normalize primitives;
2. optimize nested groups without moving children across groups;
3. remove consecutive duplicate points;
4. remove or convert degenerate primitives;
5. convert trivial polylines and straight Beziers to lines;
6. merge adjacent collinear lines;
7. join compatible adjacent open line/polyline paths;
8. simplify polylines and closed shapes;
9. merge compatible Bezier sequences;
10. remove duplicate primitives;
11. validate the final geometry.

The pass loop stops when no stage changes the sequence or when
`maximum_optimization_passes` is reached.

## Error Budget

Every operation records local max error, RMS error, normalized error, and the
pass number. A change is accepted only when its normalized error stays under
`normalized_error_budget` and the accumulated normalized error stays under
`cumulative_error_budget`.

Zero-error operations such as exact duplicate removal and exact collinear line
merges still produce operation records so the reduction remains auditable.

## Normalization

The optimizer copies input primitives before editing. It accepts:

- a sequence of semantic primitives;
- `PrimitiveGroup`;
- one `PrimitiveFitResult`;
- a list of `PrimitiveFitResult`;
- primitives from `SvgParseResult`;
- `CenterlineResult`;
- a single `CenterlinePath`.

Input order, group hierarchy, styles, fill, opacity, source ids, SVG metadata,
centerline metadata, and existing primitive metadata are preserved unless a
configuration explicitly disables that behavior.

## Collinear Lines

Adjacent `LinePrimitive` objects are merged when:

- styles are equivalent;
- endpoints touch, nearly touch, or overlap within tolerance;
- direction differs by no more than `collinear_angle_tolerance`;
- all endpoints stay within `collinear_distance_tolerance`;
- topology metadata does not mark a junction;
- the error budget remains valid.

The merged line uses the extreme projected endpoints along the shared axis. It
does not merge parallel offset lines or crossing lines.

## Polyline Joining

Open `LinePrimitive` and `PolylinePrimitive` objects can be joined when their
endpoints and local tangents are compatible. Four deterministic orientations
are considered:

- end to start;
- end to end with the second path reversed;
- start to start with the first path reversed;
- start to end.

When `preserve_draw_order=True`, only adjacent paths are joined and the result
is inserted at the first path position.

## Simplification

Polyline simplification uses Ramer-Douglas-Peucker by sections. Endpoints,
closed-path closure, explicit junction indices, metadata corner indices, and
detected sharp corners are fixed before simplifying each section.

For closed shapes, the optimizer preserves orientation and rejects changes
that create a new self-intersection when `preserve_topology=True`.

## Corner Preservation

Corners are detected with the discrete turn-angle helper from the geometric
error module. Points whose turn exceeds `corner_angle_threshold`, points listed
in `junction_indices`, and points listed in `corner_indices` or
`fixed_indices` stay fixed during simplification.

This prevents rectangles, teeth, small triangles, junctions, and intentional
sharp features from being smoothed away as if they were continuous curves.

## Bezier Merging

Consecutive `BezierPrimitive` objects are examined as sequences. A sequence can
be merged only when:

- endpoints have C0 continuity;
- tangent direction is compatible within `bezier_tangent_tolerance`;
- styles match;
- no topology metadata locks the junction;
- the sampled original sequence can be refit with fewer cubic Beziers;
- the combined error stays below `maximum_combined_bezier_error`.

The implementation samples the original Beziers, reuses the Issue 7 cubic
fitter, and accepts the new sequence only when the segment count decreases.

## Closed Shapes

`ClosedShapePrimitive` keeps its closed semantics. The optimizer may remove
duplicate points and simplify the boundary, but it does not run boolean
operations, rebuild holes, join independent rings, or convert a closed freeform
shape into a new circle or ellipse. Existing `CirclePrimitive` and
`EllipsePrimitive` objects remain semantic primitives.

## Duplicates

Duplicate primitive removal is conservative. It only removes practically
identical primitives with compatible styles and no translucent stroke, fill, or
overall opacity that could make overlap visually relevant.

Supported duplicate checks include points, lines in either direction,
polylines in either direction, closed shapes, circles, ellipses, and cubic
Beziers in forward or reversed control order.

## Styles And Metadata

Merges require compatible stroke, fill, opacity, and resolved fill rule when
style preservation is enabled. New primitives copy the style and metadata of
the first source primitive, preserve source ids where practical, and append an
`optimization_history` entry such as:

- `remove_duplicate_point`;
- `convert_to_line`;
- `merge_collinear_lines`;
- `merge_polylines`;
- `simplify_polyline`;
- `merge_beziers`;
- `remove_duplicate_primitive`.

Merged primitives also record `merged_source_ids` without copying large data.

## Groups

With `preserve_groups=True`, each `PrimitiveGroup` is optimized internally and
the hierarchy remains intact. The optimizer does not move primitives across
groups and does not merge across groups by default.

With `preserve_groups=False`, groups are flattened explicitly before the normal
passes run.

## Metrics

`GeometryOptimizationMetrics` records input/output primitive counts, point
counts, reductions, duplicate points, degenerate conversions/removals,
collinear line merges, polyline joins, simplification reductions, Bezier
sequence merges, duplicate primitive removals, groups preserved/removed,
operation counts, maximum/RMS/normalized/cumulative errors, topology/style
rejections, warnings, and pass count.

## API

```python
from fikzpy.core.geometry_optimization import optimize_primitives

result = optimize_primitives(primitives)
print(result.metrics.primitive_reduction_ratio)
```

For Issue 7 results:

```python
from fikzpy.core.geometry_optimization import optimize_fit_results

optimized = optimize_fit_results(fit_results)
```

The object-oriented API is:

```python
from fikzpy.core.geometry_optimization import GeometryOptimizer

optimizer = GeometryOptimizer()
result = optimizer.optimize(primitives)
```

## Configuration

`GeometryOptimizationConfig` centralizes all tolerances and switches:

- duplicate-point cleanup;
- degenerate primitive handling;
- collinear line merging;
- adjacent polyline joining;
- polyline and closed-shape simplification;
- topology preservation;
- Bezier sequence merging;
- duplicate primitive removal;
- group, draw-order, style, and metadata preservation;
- cumulative and normalized error budgets;
- decimal precision;
- strict mode;
- maximum optimization passes.

Invalid parameters are rejected at construction time.

## Limitations

- This issue does not repeat full primitive fitting from Issue 7.
- It does not recognize new circles, ellipses, arcs, or complex symbols.
- Closed-shape holes are preserved only as metadata and grouping; no boolean
  reconstruction is attempted.
- Bezier merging depends on sampled comparison and the existing cubic fitter.
- The optimizer is intentionally conservative when style, opacity, topology, or
  draw order could change the visual result.

## Future Relationship To Issue 9

Issue 8 produces cleaner semantic geometry for a future exporter. It does not
write TikZ. Issue 9 can consume `GeometryOptimizationResult.primitives` and
serialize the optimized semantic objects into human-readable TikZ.

## Visual Separation

Visual mode remains separate. The new optimization modules are not imported by
`visual_pipeline.py`, `visual_postprocessor.py`, `tikz_pipeline.py`, or
`image_processor.py`, and no GUI route calls this optimizer in Issue 8.
