"""Bezier curve TikZ snippets."""

TEMPLATES = {
    "Bezier curve": r"""\begin{tikzpicture}
  \draw (0,0) .. controls (1,2) and (2,2) .. (3,0);
\end{tikzpicture}
""",
    "Smooth path": r"""\begin{tikzpicture}
  \draw[line width=0.6pt]
    (0,0) .. controls (1,1.5) and (2,-1) .. (3,0.5)
    .. controls (4,2) and (5,1) .. (6,1.5);
\end{tikzpicture}
""",
}
