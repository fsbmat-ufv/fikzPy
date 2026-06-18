# Classic Semantic Integration

## Purpose

Issue 11 connects the isolated semantic pipeline to the existing Classic mode. Classic now targets compact, readable, manually editable TikZ made from internal primitives instead of the older stroke-only contour flow that could lose filled black regions in mixed monochrome artwork.

The integration is intentionally narrow:

- Classic calls the semantic pipeline.
- Visual keeps its independent SVG/svg2tikz-oriented flow.
- Contornos keeps the existing contour-to-TikZ flow.
- The GUI layout, mode names, preview, and PDF path remain unchanged.

## Classic Semantic vs Visual

Classic semantic output is optimized for editability and semantic TikZ primitives:

- centerline strokes become lines, polylines, and fitted primitives;
- filled regions become closed paths with `fill`;
- fitted circles and ellipses can be exported as TikZ `circle` and `ellipse`;
- validation rejects outputs that drop major dark regions or details.

Visual remains the fidelity-first branch. It may emit larger code and continues to use its existing visual SVG/svg2tikz path when that is the selected mode.

## Flow

The Classic route now follows:

```text
source image
  -> classify_image()
  -> preprocess_image()
  -> strategy decision
  -> semantic primitive extraction
  -> fit_primitives()
  -> optimize_fit_results()
  -> export_primitives_to_tikz()
  -> validate_semantic_output()
  -> TikZ returned to the existing editor/preview path
```

Programmatic API:

```python
from fikzpy.core.classic_semantic_pipeline import run_classic_semantic_pipeline

result = run_classic_semantic_pipeline(image)
print(result.tikz_code)
print(result.accepted)
```

The class API is also available:

```python
from fikzpy.core.classic_pipeline_config import ClassicSemanticConfig
from fikzpy.core.classic_semantic_pipeline import ClassicSemanticPipeline

pipeline = ClassicSemanticPipeline(ClassicSemanticConfig())
result = pipeline.run(image)
```

## Strategies

`LINE_ART`

Uses centerline extraction for thin strokes, converts centerline paths to `PolylinePrimitive`, fits and optimizes them, then exports semantic TikZ.

`BINARY_OUTLINE`

Uses internal filled-region extraction over the foreground mask. Large compact or thick components become `ClosedShapePrimitive` objects with black fill, preserving holes with deterministic white fill primitives when possible.

`COLOR_REGIONS`

Uses a conservative Classic semantic route for color-region-like images and emits a warning. It does not reimplement Visual and does not call the Visual/svg2tikz flow.

`MIXED_MONOCHROME`

Uses a hybrid route for black-and-white drawings that contain both thin line detail and large filled regions. Filled components are extracted as closed filled shapes, while thin components are processed with centerlines. Metadata records `source_layer`, `component_id`, bbox, area, and strategy. Diagnostic groups are preserved in `ClassicSemanticResult`, while Classic TikZ output flattens those groups by default so diagnostic names are not emitted as TikZ scope options.

## Automatic Decision

The decision combines the Issue 2 classifier with foreground component metrics:

- color-rich images become `COLOR_REGIONS`;
- thin-only foreground becomes `LINE_ART`;
- large compact or thick dark regions become `BINARY_OUTLINE`;
- simultaneous filled components and thin strokes become `MIXED_MONOCHROME`.

The detection is deterministic and does not use file names.

## Hybrid Preservation

The mixed monochrome route is designed for the observed Classic failure case: images with roughly 18% dark pixels where the legacy output could preserve only thin fragments and drop most filled black mass.

The hybrid route preserves:

- large black hair, clothing, silhouettes, or blocks as filled paths;
- thin facial, hand, clothing, or contour details as centerline paths;
- source-layer metadata for diagnostics and validation.

Ambiguous components are treated conservatively. Avoiding loss of filled dark mass is preferred over maximum compactness.

Issue 11.5 adds a stricter line-art refinement layer for the opposite failure mode: line drawings that should stay as thin strokes but could otherwise become black filled masses with white cutouts. See `docs/classic_lineart_refinement.md` for the line-art diagnostics, filled-region strictness, white-cutout validation, and stroke-width policy.

## Filled Regions

`extract_filled_regions()` uses OpenCV contour extraction over the binary foreground mask. It is internal and deterministic, with no required Potrace, VTracer, AutoTrace, Inkscape, LaTeX, or svg2tikz dependency.

Each extracted filled region records:

- area;
- bounding box;
- centroid;
- fill ratio;
- component id;
- source layer;
- hole metadata when applicable.

## Thin Strokes

Thin strokes use the existing centerline pipeline. Centerline paths become semantic polylines with `source_layer=thin_stroke` and component/path metadata. Fitting may simplify them into lines, polylines, or other supported primitives while preserving style and metadata.

## Validation

The pipeline calls `validate_semantic_output()` from Issue 10 unless validation is disabled by configuration. The result includes:

- overall fidelity score;
- filled-region recall;
- thin-stroke recall;
- dark-mass preservation ratio;
- regression flags;
- rejection reasons.

Rejected outputs are returned as `accepted=False`; the pipeline does not silently switch to Visual.

## Acceptance And Fallback

Default fallback policy is `REJECT_RESULT`. A rejected Classic semantic result remains visible in diagnostics and carries warnings/reasons such as:

- `near_empty_output`;
- `missing_large_dark_regions`;
- `low_filled_region_recall`;
- `dark_mass_loss`;
- `lost_thin_details`;
- `score_below_classic_minimum`.

The explicit legacy fallback setting is documented in configuration but the isolated core pipeline does not invoke the legacy generator by itself.

## GUI Integration

The integration point is the Classic branch in `fikzpy/core/tikz_pipeline.py`. When the selected effective mode is `classic`, it calls:

```python
run_classic_semantic_pipeline(...)
```

The returned TikZ is passed to the same editor, preview, and PDF workflow already used by the app. A semantic rasterization preview is converted into the existing `ProcessingResult` shape so GUI code does not need a layout or API change.

## Configuration

`ClassicSemanticConfig` centralizes:

- strategy and auto-detection;
- fallback and validation policy;
- minimum acceptance score;
- filled-region and thin-stroke thresholds;
- nested preprocessing, classifier, centerline, fitting, optimization, TikZ export, and validation configs;
- external backend preferences, disabled by default.

Defaults use the new semantic Classic, preserve filled regions and thin strokes, validate output, and avoid external tools.

## Result Object

`ClassicSemanticResult` contains:

- `tikz_code`;
- `tikz_export_result`;
- `raw_primitives`;
- `fitted_primitives`;
- `optimized_primitives`;
- classification and preprocessing results;
- validation result;
- metrics;
- warnings;
- strategy used;
- acceptance status and rejection reasons;
- deterministic hash;
- `to_dict()`.

## Visual Preservation

Visual is not modified by this integration. Its files and svg2tikz path remain separate. Selecting Visual in the GUI still routes through the existing visual pipeline; selecting Contornos still routes through the existing contour pipeline.

## Limitations

The internal filled-region extractor is deterministic and lightweight, but not a full replacement for specialized tracing backends. It approximates contours as closed paths and may preserve complex holes using white filled shapes when perfect subpath semantics are unavailable.

Color-region handling is conservative. For complex color fidelity, Visual remains the better user-facing choice.

The semantic Classic route validates raster fidelity with the Issue 10 rasterizer, not mandatory PDF rendering. PDF rendering and broad benchmark documentation are reserved for later work.

## Next Issue

Issue 12 - Benchmark e documentação final.
