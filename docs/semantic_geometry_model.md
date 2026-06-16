# Semantic Geometry Model

## Purpose

The semantic geometry model is an isolated intermediate representation for the
future Classic semantic pipeline. It lets later stages describe detected
geometry before any exporter decides how to write the result.

This issue only adds the model. It is not connected to the current Classic,
Vector, or Visual flows.

## Available Primitives

- `PointPrimitive`: a single semantic point marker.
- `LinePrimitive`: a straight segment with distinct start and end points.
- `PolylinePrimitive`: an open or closed sequence of straight segments.
- `CirclePrimitive`: a center point and positive radius.
- `EllipsePrimitive`: a center point, positive x/y radii, and rotation.
- `BezierPrimitive`: one cubic curve with start, two controls, and end.
- `ClosedShapePrimitive`: a closed freeform boundary with at least three
  points.
- `PrimitiveGroup`: a diagnostic grouping for primitives and nested groups.

## Auxiliary Types

- `Point2D`: finite x/y coordinates.
- `RGBColor`: red, green, and blue channels in the 0-255 range.
- `StrokeStyle`: stroke color, width, opacity, and optional cap/join/dash
  hints.
- `FillStyle`: optional fill color and fill opacity.

`Point2D`, `RGBColor`, `StrokeStyle`, and `FillStyle` are frozen dataclasses so
shared geometry and style values remain predictable in tests and future
pipeline stages.

## Shared Primitive Fields

Each primitive can carry:

- stroke style;
- optional fill style;
- optional overall opacity;
- optional fitting confidence;
- optional estimated geometric error;
- optional source metadata.

The metadata field is copied into a read-only mapping at construction time.

## Validations

The model performs conservative validation:

- coordinates must be finite;
- RGB channels must be integers from 0 to 255;
- opacity and confidence must be between 0 and 1;
- geometric error must be non-negative;
- stroke width, circle radius, and ellipse radii must be positive;
- line start and end must be distinct;
- polylines require at least two points;
- closed shapes require at least three points and `closed=True`;
- groups may contain only semantic primitives or nested primitive groups.

The model avoids rejecting future legitimate geometry beyond these basic
structural checks.

## Serialization

Every primitive exposes `to_dict()` for diagnostics and tests. The dictionary
contains the primitive type, geometry, style, fill, confidence, error, and
metadata.

This serialization is internal. It does not produce TikZ and does not generate
complete documents.

## Separation From Export

The model does not:

- process images;
- import OpenCV;
- access the GUI;
- run external tracers;
- compile LaTeX;
- generate TikZ documents.

A future exporter will be responsible for translating these primitives into
human-readable TikZ.

## Current Limitations

- The model is not wired into the Classic pipeline yet.
- It does not classify images or choose tracing strategies.
- It does not fit primitives to contours.
- It does not simplify or merge geometry.
- It does not score fidelity or complexity.

Those responsibilities belong to later roadmap issues.

## Use In Later Issues

Issue 2 can attach image-classification metadata to primitive sources. Later
tracing and fitting issues can produce these objects from centerlines, traced
SVG paths, or optional external tracers. The semantic TikZ exporter can then
serialize this stable representation without coupling geometry detection to
output formatting.
