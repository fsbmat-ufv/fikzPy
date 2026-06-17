# fikzPy

fikzPy converts raster image contours into editable TikZ code for LaTeX.

## What It Does

- Opens an image in a desktop GUI.
- Detects edges and contours with OpenCV.
- Simplifies contours with Douglas-Peucker.
- Generates concise TikZ paths.
- Exports standalone `.tex` files.
- Optionally compiles TikZ to PDF when LaTeX is installed.

## Interface Preview

Placeholder for future screenshots:

```text
+------------------------+-----------------------------+
| Original / overlay     | Generated TikZ code         |
| preview                |                             |
+------------------------+-----------------------------+
| Parameters and actions: open, generate, copy, export |
+------------------------------------------------------+
```

## Example

Input:

```text
examples/example_input.png
```

Output:

```text
examples/example_output.tex
```

## Documentation

- [Usage](usage.md)
- [Development](development.md)
