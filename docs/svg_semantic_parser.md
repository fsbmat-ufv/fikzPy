# SVG Semantic Parser

## Purpose

The SVG semantic parser converts traced SVG into the internal semantic geometry
model created for the future Classic semantic pipeline. It is isolated
infrastructure: it does not generate TikZ, does not call external tracers, does
not compile LaTeX, and is not connected to the current Classic, Visual, or
Contornos modes.

## Architecture

The parser is split into small modules:

- `fikzpy.core.svg_semantic_parser`: XML traversal, SVG element conversion,
  path parsing, `defs`/`use`, diagnostics, and public API.
- `fikzpy.core.svg_transforms`: affine transform parsing and point
  application.
- `fikzpy.core.svg_styles`: style, paint, color, opacity, and inheritance
  resolution.

The public API is:

```python
from fikzpy.core.svg_semantic_parser import parse_svg_to_primitives

result = parse_svg_to_primitives(svg_text)
for primitive in result.primitives:
    print(primitive.to_dict())
```

## Semantic Parser Versus svg2tikz

`svg2tikz` remains part of the existing Visual mode path. This parser does not
import or call `svg2tikz`; it reads SVG as geometry and produces internal
objects such as `LinePrimitive`, `BezierPrimitive`, and `ClosedShapePrimitive`.
A later exporter will decide how to write those objects as TikZ.

## Classic Semantic Versus Visual

Visual mode still uses its existing image -> SVG-style path -> `svg2tikz` ->
post-processing flow. Issue 6 does not reuse this parser inside Visual mode and
does not alter GUI mode names, preview behavior, or Visual rendering.

## Input Sources

`parse_svg_to_primitives()` accepts:

- SVG text strings containing `<svg`;
- UTF-8 bytes;
- `pathlib.Path`;
- string paths to existing SVG files;
- successful `TracerResult` instances from the optional raster tracer adapters.

For `TracerResult`, the parser validates `success=True`, uses `svg_text` when
available, falls back to `svg_path`, checks consistency when both are present,
and records tracer metadata.

## Supported Elements

Supported drawing elements:

- `line`
- `polyline`
- `polygon`
- `rect`
- `circle`
- `ellipse`
- `path`
- `g`
- `defs`
- `use`

Metadata-like elements such as `title`, `desc`, `metadata`, `symbol`, and
`clipPath` are ignored safely in tolerant mode. Unknown elements produce
warnings unless `strict=True`.

## Path Commands

The path parser supports absolute and relative forms:

- `M` / `m`
- `L` / `l`
- `H` / `h`
- `V` / `v`
- `C` / `c`
- `S` / `s`
- `Q` / `q`
- `T` / `t`
- `A` / `a`
- `Z` / `z`

It supports implicit command repetition, multiple coordinate pairs after
`M/m`, scientific notation, comma or whitespace separators, and adjacent
negative numbers.

Line-only paths become `LinePrimitive`, `PolylinePrimitive`, or
`ClosedShapePrimitive`. Curved paths become `BezierPrimitive` objects or a
`PrimitiveGroup` when a path has multiple segments or subpaths.

## Arcs

SVG elliptical arcs are converted deterministically to one or more cubic
Bezier segments. Radii are corrected according to the SVG endpoint-to-center
algorithm. Degenerate arcs with zero radius become line segments. The converter
uses segments of at most 90 degrees, which keeps the number of Beziers small.

When `convert_arcs_to_beziers=False`, the parser warns instead of silently
dropping arc commands.

## Transforms

Supported transform functions:

- `translate(tx[, ty])`
- `scale(sx[, sy])`
- `rotate(angle)`
- `rotate(angle, cx, cy)`
- `matrix(a,b,c,d,e,f)`
- `skewX(angle)`
- `skewY(angle)`

Transforms are accumulated through nested groups, `use`, and the root viewBox
normalization transform. Points are transformed before primitives are returned.
Circles remain circles under uniform scale, become ellipses under non-uniform
orthogonal scale, and are converted to Beziers under general shear/skew.

Stroke width uses a conservative scalar derived from the transform.

## Styles And Colors

Supported style inputs:

- presentation attributes;
- inline `style`;
- inherited group styles;
- `inherit`;
- `none`;
- `currentColor`.

Supported properties include:

- `stroke`
- `stroke-width`
- `stroke-opacity`
- `stroke-linecap`
- `stroke-linejoin`
- `stroke-dasharray`
- `fill`
- `fill-opacity`
- `fill-rule`
- `opacity`
- `color`
- `display`
- `visibility`

Supported colors include short and long hex forms, hex alpha forms,
`rgb(...)`, percentage RGB values, basic named colors, `transparent`, `none`,
and `currentColor`.

## Units, ViewBox, And preserveAspectRatio

Lengths support:

- unitless values;
- `px`;
- `pt`;
- `pc`;
- `mm`;
- `cm`;
- `in`.

The internal default target unit is `px`, using modern SVG/CSS-compatible
`96 dpi`. `viewBox` is parsed and, when enough dimensions are available, a
normalization transform is applied. `preserveAspectRatio` supports `none`,
`meet`, and `slice` with `xMin`, `xMid`, `xMax`, `YMin`, `YMid`, and `YMax`
alignment.

Percentage dimensions are recorded with a warning and fall back to viewBox
dimensions for normalization when necessary.

## Groups, Defs, And Use

When `preserve_groups=True`, SVG groups become `PrimitiveGroup` objects with
child order preserved. When `preserve_groups=False`, groups are flattened while
keeping the drawing order.

Internal `use` references are resolved by id for:

```xml
<use href="#shape"/>
<use xlink:href="#shape"/>
```

The parser applies `x`, `y`, transforms, and inherited styles. External
references are rejected by default. Circular references and excessive reference
depth raise clear errors in strict mode.

## Strict And Tolerant Modes

Default tolerant mode keeps parsing when it is safe, emits structured warnings,
and ignores only the problematic element.

`strict=True` raises parser-specific exceptions for malformed XML, invalid
paths, bad transforms, unsupported elements, invalid styles, unresolved
references, and unsafe reference behavior.

## Metrics And Warnings

`SvgParseResult.metrics` records element counts, parsed primitives, groups,
paths, subpaths, transforms, inherited styles, arcs, converted arcs, ignored
elements, unsupported elements, and warning counts.

`SvgParseWarning` contains a stable code, message, tag, and element id when
available.

`to_dict()` omits the full SVG source, XML objects, and other non-serializable
state.

## Dependencies

Issue 6 uses only the Python standard library plus existing fikzPy modules. It
does not add `svgelements` or `svgpathtools` yet. The path and arc conversion
needed for this issue are implemented locally so the current environment and
Visual mode remain unchanged.

## Limitations

- No semantic fitting is performed. A circle encoded as a `path` remains a path
  represented by lines or Beziers.
- No boolean reconstruction of holes is performed; multiple rings/subpaths are
  preserved for later stages.
- CSS stylesheets, selectors, filters, masks, clipping, gradients, patterns,
  text layout, and external references are not implemented.
- General affine transforms on circles and ellipses are conservatively
  converted to Beziers.
- The parser is not connected to the Classic button or any GUI workflow.

## Future Relationship To Issue 7

Issue 7 can consume the returned primitives and groups to fit simpler geometry
where appropriate. That future stage may recognize circles, ellipses, lines,
and optimized Bezier groups from traced paths. Issue 6 deliberately stops at
safe SVG parsing and semantic primitive construction.
