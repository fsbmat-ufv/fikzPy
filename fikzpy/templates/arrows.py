"""Arrow TikZ snippets."""

TEMPLATES = {
    "Simple arrow": r"""\begin{tikzpicture}
  \draw[->] (0,0) -- (3,0);
\end{tikzpicture}
""",
    "Double arrow": r"""\begin{tikzpicture}
  \draw[<->, thick] (0,0) -- (3,1);
\end{tikzpicture}
""",
    "Bent arrow": r"""\begin{tikzpicture}
  \draw[->] (0,0) to[out=45, in=135] (3,0);
\end{tikzpicture}
""",
}
