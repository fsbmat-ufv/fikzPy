# Centerline Pipeline

## Purpose

The centerline pipeline is isolated infrastructure for future semantic Classic
line-art vectorization. It consumes a binary line mask, or a
`PreprocessingResult`, and extracts ordered centerline paths from the
one-pixel skeleton of the drawing.

This issue does not connect the pipeline to the current Classic button, Vector
mode, Visual mode, preview, PDF compilation, or document export.

## Flow

```text
image or PreprocessingResult
-> binary mask
-> skeletonization
-> skeleton graph
-> conservative spur pruning
-> ordered centerline paths
-> optional endpoint merge
-> optional pixel-path simplification
-> structured diagnostics
```

Raw image inputs are passed through `preprocess_image(...,
category=ImageCategory.LINE_ART)`. `PreprocessingResult` inputs reuse the
existing `binary_mask` and do not run preprocessing again.

## Skeletonization

The configured public method is `skimage_skeletonize`. When `scikit-image` is
available, the implementation uses `skimage.morphology.skeletonize` to reduce
foreground masks to one-pixel centerlines. If the runtime does not have
`scikit-image` installed, the module uses a deterministic Zhang-Suen thinning
fallback so tests and diagnostics remain executable.

The input mask is copied and normalized. The original mask is not modified.

## Graph Model

`fikzpy.core.skeleton_graph` converts the skeleton into a deterministic graph.
Pixels are classified by neighbor count:

- endpoint: one neighbor;
- path pixel: two neighbors;
- junction: three or more neighbors;
- isolated pixel: no neighbors;
- cycle anchor: synthetic anchor for pure loop components.

The fallback graph stores:

- `SkeletonNode`;
- `SkeletonEdge`;
- ordered pixel chains for each edge;
- component ids;
- backend diagnostics;
- topology warnings.

`sknw` is optional. When `CenterlineConfig.use_sknw_if_available=True`, the
pipeline records whether the package is available, but the public result still
uses the internal fallback graph shape. No external graph object is exposed as
required API.

## Path Extraction

Each graph edge becomes a `CenterlinePath` unless it is filtered by
`minimum_path_length`. Components with a cyclic two-core are emitted as closed
paths so square and circular loops remain closed even when pixel-corner
connectivity creates local junction clusters.

Coordinates are stored as `Point2D` in image pixel coordinates with `x=column`
and `y=row`. Path order and identifiers are deterministic.

## Spur Pruning

Spur pruning is conservative and optional. A removable spur must:

- connect an endpoint to a junction;
- be shorter than the configured absolute or relative threshold;
- fit within `maximum_pruning_ratio`;
- not be a cycle.

When `preserve_small_details=True`, pruning is disabled. The result records
removed spur count, removed length, and ratio-limit warnings.

## Endpoint Merge

Endpoint merging is disabled by default. When enabled, two open paths may be
merged only when:

- endpoint distance is within `maximum_endpoint_distance`;
- local endpoint tangents face the connection direction within
  `maximum_merge_angle`;
- the paths are open and distinct.

The merge is represented as a simple linear connection inside the intermediate
path only. No curve fitting is performed.

## Simplification

Optional simplification uses a small Ramer-Douglas-Peucker implementation.
It preserves open-path endpoints and keeps closed paths closed. The tolerance
is intentionally small because this stage only removes redundant skeleton
pixels. Semantic simplification belongs to a later issue.

## Configuration

`CenterlineConfig` centralizes all parameters:

- connectivity;
- skeleton method;
- spur pruning controls;
- detail preservation;
- cycle preservation;
- endpoint merge distance and angle;
- minimum path length;
- simplification tolerance;
- optional `sknw` detection;
- strict topology validation;
- maximum pruning ratio.

Invalid values are rejected at construction time.

## Metrics

`CenterlineMetrics` records:

- skeleton pixel count;
- connected component count;
- node, endpoint, junction, isolated pixel, and edge counts;
- open and closed path counts;
- cycle count;
- pixels before and after pruning;
- spurs removed and removed length;
- paths before and after merging;
- points before and after simplification.

`CenterlineResult.to_dict()` includes array summaries and hashes instead of
full matrices.

## Programmatic Use

```python
from fikzpy.core.adaptive_preprocessing import preprocess_image
from fikzpy.core.centerline_pipeline import extract_centerlines
from fikzpy.core.image_classifier import ImageCategory

preprocessed = preprocess_image("line_art.png", category=ImageCategory.LINE_ART)
result = extract_centerlines(preprocessed)

for path in result.paths:
    print(path.id, path.length, path.closure)
```

To convert paths explicitly to semantic geometry:

```python
from fikzpy.core.centerline_pipeline import centerline_paths_to_polylines

polylines = centerline_paths_to_polylines(result)
```

This conversion produces `PolylinePrimitive` objects only. It does not infer
lines, circles, ellipses, or curves.

## Dependencies

The implementation uses existing project dependencies:

- `numpy`;
- `opencv-python` through the graph component labeling helper;
- `scikit-image` when installed for skeletonization.

Optional:

- `sknw`, detected only for diagnostics in this issue.

## Limitations

- Junction clusters are represented conservatively and can produce several
  short edge fragments around dense intersections.
- Endpoint merging is intentionally off by default.
- The pipeline does not classify paths into geometric primitives.
- It does not parse SVG, run external raster tracers, compute a fidelity score,
  or export documents.
- Color-region and filled-outline routing belong to later issues.

## Future Relationship To Issue 7

Issue 7 can consume `CenterlinePath` or explicit `PolylinePrimitive` output and
fit simpler semantic geometry where appropriate. This issue only produces
ordered centerline trajectories and topology diagnostics.
