"""Node and label TikZ snippets."""

TEMPLATES = {
    "Named nodes": r"""\begin{tikzpicture}
  \node[draw, circle] (A) at (0,0) {A};
  \node[draw, circle] (B) at (3,1) {B};
  \draw (A) -- (B);
\end{tikzpicture}
""",
    "Text label": r"""\begin{tikzpicture}
  \draw (0,0) circle (1);
  \node[above right] at (0.7,0.7) {label};
\end{tikzpicture}
""",
    "Node style": r"""\begin{tikzpicture}
  \tikzset{box/.style={draw, rounded corners, inner sep=6pt}}
  \node[box] at (0,0) {fikzPy};
\end{tikzpicture}
""",
}
