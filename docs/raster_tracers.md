# Optional Raster Tracers

## Purpose

Issue 5 adds isolated optional adapters for external raster-to-SVG engines.
They provide a common diagnostic API for producing an intermediate SVG file
from raster input. The adapters are not connected to the current Classic,
Vector, Visual, preview, compilation, or GUI flows.

## Tracers

### Potrace

Potrace is intended for binary artwork:

- black-and-white silhouettes;
- logos;
- filled regions;
- closed outline shapes.

The adapter consumes a binary mask when one is provided. A
`PreprocessingResult` reuses its `binary_mask`; a raw image is passed through
the existing adaptive preprocessing stage for a binary diagnostic mask. The
temporary Potrace input is written as PBM.

Main configuration fields:

- `potrace_turdsize`;
- `potrace_alphamax`;
- `potrace_opticurve`;
- `potrace_opttolerance`;
- `potrace_invert`;
- `potrace_turnpolicy`;
- `potrace_path`.

### AutoTrace

AutoTrace is exposed as an optional external comparison strategy for outline
mode and centerline mode. It does not replace the internal centerline pipeline
from Issue 4.

Main configuration fields:

- `autotrace_centerline`;
- `autotrace_color_count`;
- `autotrace_despeckle_level`;
- `autotrace_corner_threshold`;
- `autotrace_error_threshold`;
- `autotrace_background_color`;
- `autotrace_path`.

### VTracer

VTracer is intended for color regions and filled images with multiple tones.
The adapter checks for a Python module first and then for a command-line
executable. Python VTracer has priority when both are available because it
avoids a subprocess call.

Main configuration fields:

- `vtracer_colormode`;
- `vtracer_hierarchical`;
- `vtracer_mode`;
- `vtracer_filter_speckle`;
- `vtracer_color_precision`;
- `vtracer_layer_difference`;
- `vtracer_corner_threshold`;
- `vtracer_length_threshold`;
- `vtracer_max_iterations`;
- `vtracer_splice_threshold`;
- `vtracer_path_precision`;
- `vtracer_path`.

Only parameters accepted by a detected Python function are passed to that
function.

## Availability Detection

Availability is deterministic and conservative:

- configured executable paths are checked first;
- otherwise `shutil.which()` is used;
- Python VTracer is imported optionally;
- executable versions are queried with `--version` using a short safe
  subprocess call;
- no recursive disk search is performed;
- no dependency is installed automatically;
- `PATH` is not modified.

Manual installation is expected outside fikzPy. Configure executable paths with
`TracerConfig` when tools are not on `PATH`.

## Subprocess Safety

External tracers are executed with:

- argument lists;
- `shell=False`;
- captured stdout and stderr;
- a finite timeout;
- UTF-8 decoding with replacement;
- exclusive temporary directories by default;
- no arbitrary user argument passthrough.

Temporary files are removed after successful execution unless an explicit
output directory or `keep_temporary_files=True` is used. On error, diagnostic
files are preserved when `preserve_diagnostics_on_error=True`.

## SVG Validation

Every produced SVG is checked before a successful `TracerResult` is returned:

- output file exists;
- file is not empty;
- content contains an SVG element;
- XML parses successfully;
- HTML/error pages are rejected;
- root element is SVG;
- SHA-256, byte count, width, height, and viewBox are recorded when present.

This is only basic SVG validation. Semantic SVG parsing belongs to Issue 6.

## Programmatic API

```python
from fikzpy.core.tracers import TracerConfig, TracerKind, trace_image

config = TracerConfig(timeout_seconds=20.0)
result = trace_image("input.png", TracerKind.VTRACER, config=config)

print(result.success)
print(result.svg_sha256)
print(result.to_dict())
```

Availability:

```python
from fikzpy.core.tracers import list_tracer_availability

for item in list_tracer_availability():
    print(item.to_dict())
```

Explicit first-available tracing exists for diagnostics, but no current
application flow calls it. The Classic integration and automatic routing belong
to later issues.

## Local Availability Report

The current worktree records local availability in:

```text
examples/classic_semantic_baseline/tracer_availability.json
```

That file is diagnostic only and does not change existing baseline artifacts.

## Limitations

- No SVG semantic parser is implemented here.
- No primitive recognition or fitting is implemented here.
- No automatic tracer selection based on the classifier is implemented here.
- No GUI controls are added.
- External executable behavior varies by installed version; unsupported
  options may still need adapter tuning in later integration work.
- VTracer Python APIs differ by version, so the adapter filters parameters at
  call time.

## Relationship To Issue 6

Issue 6 can consume validated SVG text or paths from `TracerResult` and parse
them into the semantic geometry model. This issue intentionally stops at
producing and validating an intermediate SVG.
