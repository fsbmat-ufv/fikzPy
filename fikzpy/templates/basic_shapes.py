"""Basic TikZ shape snippets."""

TEMPLATES = {
    "Line": r"""\begin{tikzpicture}
  \draw (0,0) -- (3,1);
\end{tikzpicture}
""",
    "Circle": r"""\begin{tikzpicture}
  \draw (0,0) circle (1);
\end{tikzpicture}
""",
    "Rectangle": r"""\begin{tikzpicture}
  \draw (0,0) rectangle (3,2);
\end{tikzpicture}
""",
    "Grid": r"""\begin{tikzpicture}
  \draw[step=0.5, gray!40, very thin] (-1,-1) grid (4,3);
\end{tikzpicture}
""",
    "Axes": r"""\begin{tikzpicture}
  \draw[->] (-1,0) -- (4,0) node[right] {$x$};
  \draw[->] (0,-1) -- (0,3) node[above] {$y$};
\end{tikzpicture}
""",
    "Fill": r"""\begin{tikzpicture}
  \filldraw[fill=blue!15, draw=blue!70] (0,0) rectangle (3,2);
\end{tikzpicture}
""",
    "Style": r"""\begin{tikzpicture}
  \tikzset{important/.style={draw=red!70, line width=1pt, rounded corners}}
  \draw[important] (0,0) rectangle (3,2);
\end{tikzpicture}
""",
}
