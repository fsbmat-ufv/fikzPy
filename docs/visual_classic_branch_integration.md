# Visual and Classic Branch Integration

This document records the safe integration between the semantic Classic
development branch and the Visual mode branch before Issue 6.

## Source branches

- Semantic Classic branch: `codex/classic-raster-tracer-adapters`
- Semantic Classic commit: `f42c59e`
- Visual branch: `codex/visual-postprocessor`
- Visual commit: `495bce2`
- Integration branch: `codex/integrate-visual-and-classic`
- Common ancestor referenced by the roadmap task: `d989c4d`

## Scope

The integration only merges the Visual mode line into the semantic Classic
development line. It does not implement Issue 6 and does not connect the
semantic Classic pipeline to the interface.

The merge preserves:

- the current Classic behavior;
- the Visual mode and its filled-shape tracing flow;
- the Contornos mode;
- `fikzpy/core/visual_pipeline.py`;
- `fikzpy/core/visual_postprocessor.py`;
- Visual use of `svg2tikz` when available;
- the semantic Classic infrastructure from Issues 0 through 5.

## Modules preserved

The following modules were checked after the merge:

- `fikzpy/core/semantic_geometry.py`
- `fikzpy/core/image_classifier.py`
- `fikzpy/core/adaptive_preprocessing.py`
- `fikzpy/core/threshold_selector.py`
- `fikzpy/core/centerline_pipeline.py`
- `fikzpy/core/skeleton_graph.py`
- `fikzpy/core/tracers/`
- `fikzpy/core/visual_pipeline.py`
- `fikzpy/core/visual_postprocessor.py`

## Mode routing

The GUI mode selector exposes the primary modes:

- `classic`
- `visual`
- `contours`

Programmatic checks confirmed that:

- `config_for_mode("classic")` resolves to `classic`;
- `config_for_mode("visual")` resolves to `visual`;
- `config_for_mode("contours")` resolves to `contours`;
- `process_image(..., ProcessingSettings(vectorization_mode="visual"))`
  dispatches to the Visual tracing pipeline.

## Conflict resolution

The merge completed automatically with no textual conflicts. No manual code
resolution was required.

## Verification

The following commands were executed after integration:

```powershell
py -m pytest
py -m pytest tests/test_visual_pipeline.py
py -m pytest tests/test_gui_preview_pipeline.py
py -m pytest tests/test_raster_tracers.py
py -m pytest tests/test_centerline_pipeline.py
```

Results:

- Full suite: `241 passed`
- Visual pipeline tests: `4 passed`
- GUI preview pipeline tests: `3 passed`
- Raster tracer adapter tests: `41 passed`
- Centerline pipeline tests: `41 passed`

No baseline artifacts from previous Classic semantic issues were removed or
rewritten by this integration.
