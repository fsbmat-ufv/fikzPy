# AGENTS.md — fikzPy

## Core rule

**Do not change the current GUI layout unless explicitly requested.**

The current interface, menus, panels, buttons, colors, dimensions and interaction flow must be preserved. Improvements should focus on the image-to-TikZ pipeline.

---

## Project vision

**fikzPy** is a Python desktop application for converting raster images into clean, editable LaTeX/TikZ code.

The objective is not merely to trace pixels. The application should reconstruct the figure using semantic geometric primitives whenever possible.

Preferred output:

```latex
\draw (0,0) -- (4,0);
\draw (2,2) circle[radius=1];
\draw (4,3) ellipse[x radius=2, y radius=1];
\draw (0,0) .. controls (1,1) and (2,1) .. (3,0);
```

Avoid output consisting of hundreds of tiny segments or tiny Bézier curves when a simpler representation fits the image.

---

## Mandatory reading

Before changing the project, read:

```text
AGENTS.md
ROADMAP_CLASSIC_SEMANTIC.md
```

When working on the semantic Classic pipeline, execute the roadmap **one issue at a time**.

Do not implement the entire roadmap in a single pass.

---

## Development principles

1. Never break existing functionality.
2. Preserve the current GUI.
3. Preserve the Visual mode.
4. Preserve the old Classic implementation as a fallback until the semantic pipeline is validated.
5. Improve one subsystem at a time.
6. Keep every change reversible through Git.
7. Prefer small, testable changes.
8. Prefer clear code over clever code.
9. Do not silently hide failures.
10. Fidelity has priority over code-size reduction.

---

## Git workflow

Before editing:

```bash
git status
git branch --show-current
```

For substantial work, create a dedicated branch:

```bash
git checkout -b descriptive-branch-name
```

After each roadmap issue:

1. run the relevant tests;
2. inspect `git diff`;
3. generate a comparison example;
4. create one focused commit;
5. stop for review.

Never:

- use `git push --force`;
- discard uncommitted changes without explicit approval;
- combine unrelated changes in one commit;
- perform a massive rewrite in one commit;
- overwrite the stable implementation without a rollback path.

---

## Protected behavior

Do not modify unless explicitly requested:

- GUI layout;
- menu organization;
- button placement;
- Visual-mode behavior;
- PDF compilation subsystem;
- working export behavior;
- existing user settings.

New behavior should use existing controls or internal configuration until a GUI change is explicitly approved.

---

## Modes that must be preserved

### Classic legacy

The current stable raster-to-TikZ implementation.

It must remain available as a fallback until the semantic Classic pipeline passes regression tests.

### Classic semantic

The new experimental pipeline that:

- classifies the image;
- chooses the most appropriate tracing method;
- creates internal primitives;
- simplifies geometry;
- generates compact TikZ.

### Vector

Experimental internal vector-object workflow, if present.

### Visual

SVG-based high-fidelity workflow.

The Visual mode must remain intact while Classic is improved.

---

## Preferred semantic architecture

```text
Image upload
    ↓
Image classification
    ↓
Preprocessing
    ↓
Tracing strategy
    ↓
Internal geometric primitives
    ↓
Primitive fitting
    ↓
Simplification
    ↓
TikZ exporter
    ↓
Compilation and preview
    ↓
Fidelity and complexity scoring
```

Avoid:

```text
Image → raw contours → huge TikZ
```

Avoid using raw `svg2tikz` output as the final Classic result. SVG may be used as an intermediate format, baseline or fallback.

---

## Image classification

The semantic Classic pipeline should classify images into:

```text
LINE_ART
BINARY_OUTLINE
COLOR_REGIONS
```

Suggested routing:

```text
LINE_ART
→ centerline / skeleton / graph

BINARY_OUTLINE
→ Potrace or AutoTrace outline

COLOR_REGIONS
→ VTracer or equivalent
```

Keep a manual override available through configuration.

---

## Internal primitives

Use an intermediate representation before writing TikZ.

Suggested dataclasses:

- `PointPrimitive`
- `LinePrimitive`
- `PolylinePrimitive`
- `CirclePrimitive`
- `EllipsePrimitive`
- `BezierPrimitive`
- `ClosedShapePrimitive`
- `PrimitiveGroup`

A primitive may contain:

- geometry;
- stroke style;
- color;
- line width;
- optional fill;
- confidence;
- estimated fitting error;
- source metadata.

Primitive classes must not generate complete TikZ documents directly. The exporter is responsible for serialization.

---

## Primitive selection order

Try the simplest representation first:

```text
Line
Circle
Ellipse
Polyline
Bezier
Closed freeform shape
```

Choose the simplest primitive whose fitting error is below the configured tolerance.

A line must not become a Bézier when `\draw (a,b) -- (c,d);` is sufficient.

A circle must not become dozens of segments or Béziers when `circle` fits.

---

## Bézier rules

Use cubic Bézier curves only when simpler primitives do not fit.

Prefer fitting one Bézier to several consecutive points.

Do not convert every tiny segment into a tiny Bézier.

Avoid degenerate curves where:

- control points are almost equal to endpoints;
- the curve is extremely short;
- a straight line fits equally well;
- the curve only reproduces pixel noise.

The goals are:

- smoother drawings;
- fewer control points;
- shorter TikZ code;
- easier manual editing;
- stable continuity between adjacent curves.

---

## Centerline tracing

For line drawings, prefer centerline tracing over outline tracing.

Recommended flow:

```text
binary image
→ skeletonize
→ graph
→ prune tiny spurs
→ ordered segments
→ primitive fitting
```

Requirements:

- avoid double borders;
- preserve topology;
- preserve loops;
- preserve meaningful junctions;
- remove only microscopic noise;
- never convert every skeleton pixel into TikZ.

---

## SVG handling

Preferred workflow:

```text
Raster
→ Potrace / AutoTrace / VTracer
→ SVG
→ robust SVG parser
→ internal primitives
→ simplification
→ TikZ
```

Do not use this as the main Classic workflow:

```text
Raster → SVG → raw svg2tikz output → final result
```

`svg2tikz` may be used as:

- fallback;
- baseline;
- debugging aid;
- Visual-mode behavior.

Suggested SVG libraries:

- `svgelements`;
- `svgpathtools`.

---

## Optional tracing engines

The project may integrate:

- Potrace;
- AutoTrace;
- VTracer.

Rules:

1. Treat them as optional dependencies.
2. Detect availability automatically.
3. Fail gracefully.
4. Log the command or API call.
5. Use unique temporary directories.
6. Preserve the last valid preview on failure.
7. Do not crash if a tracer is missing.
8. Document installation and fallback behavior.

---

## Image-processing principles

Use conservative preprocessing.

Possible operations:

- autocontrast;
- grayscale conversion;
- mild denoise;
- Otsu threshold;
- adaptive threshold;
- threshold sweep;
- conservative morphological closing;
- tiny-speck removal.

Avoid destroying:

- eyes;
- teeth;
- claws;
- wrinkles;
- spirals;
- internal contours;
- faint strokes;
- small colored regions.

Do not apply aggressive filtering by default.

---

## Simplification principles

Allowed operations include:

- remove duplicate points;
- remove microscopic segments;
- merge collinear lines;
- merge compatible paths;
- simplify polylines;
- merge compatible Béziers;
- remove degenerate curves;
- reduce unnecessary decimal places.

Every simplification must respect a maximum error.

Never simplify only to reduce file size.

Fidelity has priority over minimalism.

---

## TikZ generation

Generate human-readable TikZ.

### Point

```latex
\draw (x,y) circle[radius=0.5pt];
```

### Line

```latex
\draw (x_1,y_1) -- (x_2,y_2);
```

### Polyline

```latex
\draw (x_1,y_1) -- (x_2,y_2) -- (x_3,y_3);
```

### Circle

```latex
\draw (c_x,c_y) circle[radius=r];
```

### Ellipse

```latex
\draw[rotate around={a:(c_x,c_y)}]
  (c_x,c_y) ellipse[x radius=r_x, y radius=r_y];
```

### Bézier

```latex
\draw (p_0)
  .. controls (c_1) and (c_2) .. (p_1);
```

### Closed colored shape

```latex
\draw[fill={rgb,255:red,R;green,G;blue,B}]
  ... -- cycle;
```

Rules:

- use `\draw` consistently;
- group common styles when safe;
- avoid repeated options;
- keep indentation consistent;
- round coordinates sensibly;
- preserve colors and fills;
- keep the output editable by humans.

---

## Fidelity and complexity

Evaluate both:

### Fidelity

- edge similarity;
- component preservation;
- shape overlap;
- color preservation;
- topology;
- preservation of small details.

### Complexity

- number of primitives;
- number of points;
- number of Béziers;
- number of `\draw` commands;
- `.tex` file size.

A smaller result must not be selected if it removes important elements.

---

## PDF preview safety

The preview system must:

- never overwrite a locked PDF;
- use unique compilation names when necessary;
- load only valid, non-empty PDFs;
- preserve the last valid preview on failure;
- report compilation errors clearly;
- avoid silent fallback.

Do not modify the compilation subsystem unless the task explicitly requires it.

---

## Fallback rules

Fallbacks must never be silent.

If the semantic pipeline fails:

1. log the exception;
2. identify the failed stage;
3. preserve the last valid preview;
4. optionally use Classic legacy;
5. clearly record that fallback occurred.

Never present legacy output as if it came from the semantic pipeline.

---

## Coding style

Requirements:

- Python 3.11+;
- small functions;
- descriptive names;
- English docstrings;
- minimal dependencies;
- modular architecture;
- explicit error handling;
- type hints where useful;
- no unnecessary abstraction.

Prefer:

- clear data flow;
- pure functions for geometry;
- small adapters for external tools;
- isolated exporters;
- reproducible configuration.

---

## Testing

Every new core feature should include automated tests.

Test categories:

1. simple line drawing;
2. geometric sketch with lines and circles;
3. black-and-white silhouette;
4. color icon;
5. noisy grayscale drawing.

Tests should verify:

- pipeline selection;
- primitive types;
- compact output;
- valid TikZ syntax;
- semantic representation of simple shapes;
- stable fallback;
- no GUI changes;
- Visual-mode preservation;
- legacy Classic preservation until final approval.

---

## Baseline and regression

Before changing Classic behavior:

1. create reproducible baseline artifacts;
2. record output metrics;
3. preserve PDFs and `.tex` files;
4. compare every new implementation against the baseline.

Recommended metrics:

- processing time;
- `\draw` count;
- `--` count;
- `.. controls` count;
- number of points;
- file size;
- fidelity score.

---

## Documentation

When adding a feature, document:

- purpose;
- algorithm;
- configuration;
- dependencies;
- limitations;
- fallback;
- before/after examples;
- tests performed.

Keep updated when applicable:

```text
README.md
ROADMAP_CLASSIC_SEMANTIC.md
docs/classic_semantic_mode.md
docs/external_tracers.md
docs/benchmark_classic_semantic.md
```

---

## Roadmap execution rule

When instructed to work on a roadmap issue:

1. read `AGENTS.md`;
2. read `ROADMAP_CLASSIC_SEMANTIC.md`;
3. work only on the requested issue;
4. do not begin later issues;
5. run tests;
6. create one focused commit;
7. stop and report.

Do not implement the full roadmap at once.

---

## Safety rules

Never:

- remove working features without explicit approval;
- change the GUI layout without explicit approval;
- modify Visual mode while working on Classic;
- silently replace Classic output;
- hide failures;
- discard user changes;
- introduce a mandatory heavy dependency without justification;
- generate huge unreadable paths when a semantic primitive fits;
- convert every contour point into a TikZ command;
- convert every small line segment into a tiny Bézier.

---

## Long-term goal

fikzPy should become a high-quality open-source tool for generating clean, editable TikZ from raster images.

Its core advantage should be:

```text
faithful vectorization
+
semantic geometry
+
minimal TikZ
+
human editability
+
reproducibility
```

The emphasis is on:

- mathematical representation;
- clean geometry;
- high-quality TikZ;
- maintainability;
- controlled complexity;
- faithful rendering;
- safe incremental development.
