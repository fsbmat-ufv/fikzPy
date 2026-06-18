# Semantic TikZ Exporter

## Purpose

The semantic TikZ exporter is the isolated Issue 9 serialization layer for the
future Classic semantic pipeline. It receives internal primitives that were
already parsed, fitted, simplified, or optimized, and writes compact
human-readable TikZ.

It does not run image preprocessing, tracing, primitive fitting, geometry
optimization, PDF compilation, previews, GUI code, or the current Classic
button.

## Semantic Export Versus SVG Bridging

The exporter writes TikZ from semantic objects such as `LinePrimitive`,
`CirclePrimitive`, `EllipsePrimitive`, `BezierPrimitive`, and
`ClosedShapePrimitive`.

It does not serialize raw SVG paths as final output. A circle remains TikZ
`circle`, an ellipse remains TikZ `ellipse`, and a line remains a `--` segment
when those primitives already exist in memory.

The Visual mode remains separate and keeps its existing high-fidelity path. The
new exporter is not imported by Visual, the GUI, or the current Classic
pipeline.

## API

```python
from fikzpy.core.semantic_tikz_exporter import export_primitives_to_tikz

result = export_primitives_to_tikz(primitives)
print(result.code)
```

Object-oriented use is also available:

```python
from fikzpy.core.semantic_tikz_exporter import SemanticTikzExporter, TikzExportConfig

config = TikzExportConfig(include_tikzpicture_environment=True)
result = SemanticTikzExporter(config).export(primitives)
```

## Supported Inputs

The public API accepts:

- sequences of semantic primitives;
- a single semantic primitive;
- `PrimitiveGroup`;
- `GeometryOptimizationResult`;
- `PrimitiveFitResult`;
- a list of `PrimitiveFitResult`;
- `SvgParseResult`;
- `CenterlineResult`, converted through its centerline polylines;
- simple combinations of those structures.

The exporter only unwraps existing in-memory objects. It does not call earlier
pipeline stages.

## Supported Primitives

- `PointPrimitive` -> small TikZ circle marker;
- `LinePrimitive` -> `\draw (a,b) -- (c,d);`;
- `PolylinePrimitive` -> one `\draw` with consecutive `--` segments;
- `CirclePrimitive` -> TikZ `circle[radius=...]`;
- `EllipsePrimitive` -> TikZ `ellipse[x radius=..., y radius=...]`;
- rotated ellipses -> `rotate around={angle:(center)}`;
- `BezierPrimitive` -> TikZ cubic `.. controls ... and ... ..`;
- contiguous compatible Beziers -> one command with multiple cubic segments;
- `ClosedShapePrimitive` -> `-- cycle`;
- `PrimitiveGroup` -> `scope` when group preservation is enabled.

## Styles

`StrokeStyle` is converted to TikZ options for:

- `draw`;
- `line width`;
- `draw opacity`;
- `line cap`;
- `line join`;
- `dash pattern`.

`FillStyle` is converted to:

- `fill`;
- `fill opacity`.

Overall primitive opacity is emitted as `opacity`. RGB colors use the robust
TikZ form:

```latex
{rgb,255:red,R;green,G;blue,B}
```

Default black stroke and default line width are omitted by default to keep code
short. Set `omit_default_styles=False` to emit them explicitly.

## Fill Handling

When fill is absent, no `fill` option is emitted. When stroke opacity is zero
and fill is visible, the exporter writes `draw=none` plus the fill options. A
primitive with no visible stroke and no visible fill is skipped with a warning,
or raises in strict mode.

Fill rules from metadata are handled conservatively. `evenodd` becomes
`even odd rule`; unknown rules produce a warning.

## Groups

With `preserve_groups=True`, `PrimitiveGroup` becomes:

```latex
\begin{scope}[group-name]
  ...
\end{scope}
```

With `preserve_groups=False`, groups are flattened in drawing order. Empty
groups are skipped with a warning.

## Coordinates

`TikzExportConfig` centralizes coordinate formatting:

- `coordinate_precision`;
- `unit`;
- `scale`;
- `coordinate_origin`;
- `normalize_coordinates`;
- `invert_y_axis`;
- `image_height`.

Numbers are rounded deterministically, trailing zeros are removed, and `-0` is
normalized to `0`. If `invert_y_axis=True`, `image_height` is required. In
tolerant mode, missing height produces a warning and leaves y unchanged; in
strict mode it raises.

## Formatting

Short paths are emitted on one line:

```latex
\draw (0,0) -- (1,0);
```

Long paths are split consistently:

```latex
\draw
  (0,0)
  -- (1,0)
  -- (2,0);
```

The controls are:

- `indent`;
- `indent_size`;
- `max_points_per_line`;
- `split_long_paths`;
- `path_mode`.

## Style Grouping

When `group_styles=True`, consecutive commands with identical style options
can be wrapped in a TikZ `scope` without changing drawing order.

When `define_common_styles=True` or `use_named_styles=True`, styles reused at
least twice can be emitted with `\tikzset`:

```latex
\tikzset{
  fikzStyle0/.style={draw={rgb,255:red,255;green,0;blue,0}, line width=0.4pt}
}
```

Styles used once are not named by default.

## Tikzpicture Versus Figonly

The default output is figonly body code. Set
`include_tikzpicture_environment=True` or
`code_output_mode=TikzCodeOutputMode.TIKZPICTURE` to wrap the body:

```latex
\begin{tikzpicture}[scale=1]
  ...
\end{tikzpicture}
```

`include_scope_environment=True` adds an inner scope for common TikZ defaults.

## Metrics

`TikzExportResult.metrics` records:

- input, exported, and skipped primitive counts;
- draw and path command counts;
- coordinates written;
- Bezier segments, circles, ellipses, closed paths, and fills;
- styles and named styles;
- groups written;
- warning count;
- code line and character counts.

The result also exposes direct summary fields, style definitions,
`deterministic_hash`, and `to_dict()`.

## Warnings

Warnings are deterministic and structured. They cover:

- invisible primitives;
- unsupported or disabled style details;
- missing `image_height` for y inversion;
- empty groups;
- corrupted closed-shape state;
- holes/subpaths preserved only as metadata;
- unknown fill rules;
- invalid geometry in tolerant mode.

With `strict=True`, these warnings raise `TikzExportError`.

## Limitations

- No visual fidelity score is computed in Issue 9.
- No PDF rendering or image comparison is performed.
- No Classic button integration is performed.
- No automatic pipeline selection is performed.
- Holes and multi-subpath filled shapes are not reconstructed as full TikZ
  compound paths yet; metadata is preserved and warnings are emitted.
- No new primitive fitting or geometry optimization is performed.
- Relative coordinate emission is reserved for future work.

## Future Relationship To Issue 10

Issue 10 can render and compare the semantic output against raster or PDF
targets, then compute fidelity and complexity scores. This exporter already
provides deterministic code, metrics, warnings, and hashes that can feed that
future validation layer.
