"""
tikz_generator.py
-----------------
Geração de código TikZ minimal a partir de Paths extraídos pelo tracer.py.

Objetivo
--------
  Produzir código TikZ legível, conciso e fácil de editar à mão —
  exatamente como você faria no TikzEdt: usando draw com --,
  curvas ..controls.. ou primitivas geométricas (circle, ellipse).

Saída exemplo
-------------
  BEGIN{tikzpicture}
    DRAW (1.20,3.40) -- (2.10,3.40) -- (2.10,2.80);
    DRAW (3.00,1.50) .. controls (3.20,1.80) and (3.50,1.90) .. (3.80,1.60);
  END{tikzpicture}

Primitivas reconhecidas automaticamente
---------------------------------------
  - Segmentos de reta simples  : DRAW p0 -- p1;
  - Curvas de Bézier cúbico    : DRAW p0 .. controls c1 and c2 .. p3;
  - Paths compostos (mistura)  : DRAW p0 -- p1 .. controls c1 and c2 .. p3 ...;
  - Círculos / elipses         → detectados por fit_circle_ellipse() e emitidos
                                  como \\draw (cx,cy) circle (r)  ou
                                        DRAW (cx,cy) ellipse (rx and ry);

Coordenadas
-----------
  A imagem tem origem (0,0) no canto superior esquerdo.
  TikZ tem y crescendo para cima.
  A conversão aplica:
      tikz_x = col * scale
      tikz_y = (img_height - row) * scale
  onde scale é fornecido pelo chamador (padrão: 0.05, mapeando ~500px → 25cm).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .tracer import CubicBezierSeg, LineSeg, Path, Point, Segment


# ---------------------------------------------------------------------------
# Detecção de círculos / elipses
# ---------------------------------------------------------------------------

def _fit_circle(pts: np.ndarray) -> Optional[tuple[float, float, float]]:
    """
    Tenta ajustar um círculo a um conjunto de pontos.

    Usa o método algébrico de mínimos quadrados (Kasa 1976).
    Retorna (cx, cy, r) se o fit for bom; None caso contrário.

    Critério: razão entre desvio-padrão dos resíduos e raio < 0.08.
    """
    n = len(pts)
    if n < 6:
        return None

    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    b = x ** 2 + y ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy = result[0], result[1]
    r = math.sqrt(max(result[2] + cx ** 2 + cy ** 2, 0))
    if r < 1e-3:
        return None

    residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    if residuals.std() / r < 0.08:
        return cx, cy, r
    return None


def _fit_ellipse(pts: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """
    Tenta ajustar uma elipse usando cv2.fitEllipse.

    Retorna (cx, cy, rx, ry) se o fit for bom; None caso contrário.
    """
    try:
        import cv2
    except ImportError:
        return None

    if len(pts) < 6:
        return None

    pts_i = pts.astype(np.float32)
    try:
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(pts_i.reshape(-1, 1, 2))
    except cv2.error:
        return None

    rx, ry = ma / 2.0, mi / 2.0
    if rx < 1e-3 or ry < 1e-3:
        return None

    # Verifica a qualidade do fit: distância média dos pontos à elipse
    # Aproximação: pontos devem estar próximos a (rx, ry) em coordenadas rotacionadas
    theta = math.radians(angle)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    xr = (pts[:, 0] - cx) * cos_a + (pts[:, 1] - cy) * sin_a
    yr = -(pts[:, 0] - cx) * sin_a + (pts[:, 1] - cy) * cos_a
    # Distância normalizada à elipse
    dist = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    if dist.std() < 0.1:
        return cx, cy, rx, ry
    return None


# ---------------------------------------------------------------------------
# Formatação de coordenadas
# ---------------------------------------------------------------------------

def _fmt(v: float, prec: int = 2) -> str:
    """Formata um float removendo zeros desnecessários."""
    return f"{v:.{prec}f}".rstrip("0").rstrip(".")


def _fmt_pt(x: float, y: float, prec: int = 2) -> str:
    return f"({_fmt(x, prec)},{_fmt(y, prec)})"


def _transform(
    pt: Point,
    scale: float,
    img_height: int,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float]:
    """
    Converte (col, row) de pixel para coordenadas TikZ.

    tikz_x = (col - origin_x) * scale
    tikz_y = (img_height - row - origin_y) * scale
    """
    col, row = pt
    tx = (col - origin_x) * scale
    ty = (img_height - row - origin_y) * scale
    return tx, ty


# ---------------------------------------------------------------------------
# Serialização de segmentos individuais
# ---------------------------------------------------------------------------

def _seg_to_tikz(
    seg: Segment,
    scale: float,
    img_height: int,
    origin_x: float,
    origin_y: float,
    prec: int,
    is_first: bool,
) -> str:
    """Serializa um segmento como trecho de path TikZ."""
    T = lambda pt: _transform(pt, scale, img_height, origin_x, origin_y)

    if isinstance(seg, LineSeg):
        p0x, p0y = T(seg.p0)
        p1x, p1y = T(seg.p1)
        if is_first:
            return f"{_fmt_pt(p0x, p0y, prec)} -- {_fmt_pt(p1x, p1y, prec)}"
        return f" -- {_fmt_pt(p1x, p1y, prec)}"

    if isinstance(seg, CubicBezierSeg):
        p0x, p0y = T(seg.p0)
        c1x, c1y = T(seg.c1)
        c2x, c2y = T(seg.c2)
        p3x, p3y = T(seg.p3)
        if is_first:
            return (
                f"{_fmt_pt(p0x, p0y, prec)} "
                f".. controls {_fmt_pt(c1x, c1y, prec)} and {_fmt_pt(c2x, c2y, prec)} .. "
                f"{_fmt_pt(p3x, p3y, prec)}"
            )
        return (
            f" .. controls {_fmt_pt(c1x, c1y, prec)} and {_fmt_pt(c2x, c2y, prec)} .. "
            f"{_fmt_pt(p3x, p3y, prec)}"
        )

    raise TypeError(f"Segmento desconhecido: {type(seg)}")


# ---------------------------------------------------------------------------
# Path completo
# ---------------------------------------------------------------------------

def _path_to_tikz(
    path: Path,
    scale: float,
    img_height: int,
    origin_x: float,
    origin_y: float,
    prec: int,
    draw_options: str,
    indent: str,
) -> str:
    """Serializa um Path como comando \\draw completo."""
    parts = []
    for i, seg in enumerate(path):
        parts.append(_seg_to_tikz(seg, scale, img_height, origin_x, origin_y, prec, i == 0))

    # Detecção heurística de path fechado:
    # se o último ponto do último segmento está muito próximo do primeiro ponto
    # do primeiro segmento, fecha com --cycle
    def _first_pt(s: Segment) -> Point:
        return s.p0

    def _last_pt(s: Segment) -> Point:
        if isinstance(s, LineSeg):
            return s.p1
        return s.p3

    T = lambda pt: _transform(pt, scale, img_height, origin_x, origin_y)
    first_tx, first_ty = T(_first_pt(path[0]))
    last_tx, last_ty = T(_last_pt(path[-1]))
    dist = math.hypot(last_tx - first_tx, last_ty - first_ty)
    close = " -- cycle" if dist < 0.3 * scale * 10 else ""

    body = "".join(parts) + close
    opt = f"[{draw_options}]" if draw_options else ""
    return f"{indent}\\draw{opt} {body};"


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def paths_to_tikz(
    paths: list[Path],
    *,
    img_height: int,
    scale: float = 0.05,
    draw_options: str = "",
    standalone: bool = False,
    preamble_packages: list[str] | None = None,
    coordinate_precision: int = 2,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    detect_circles: bool = True,
    indent: str = "  ",
) -> str:
    """
    Converte uma lista de Paths em código TikZ.

    Parâmetros
    ----------
    paths               : saída de tracer.extract_paths().
    img_height          : altura da imagem original em pixels (para inversão de y).
    scale               : fator de escala pixel→cm (padrão 0.05: 20px = 1cm).
    draw_options        : opcoes de DRAW, ex. "line width=0.4pt, black".
    standalone          : se True, envolve em documentclass{standalone}.
    preamble_packages   : pacotes extras no preâmbulo do standalone.
    coordinate_precision: casas decimais das coordenadas.
    origin_x, origin_y  : deslocamento de origem em pixels (útil para recortes).
    detect_circles      : tenta reconhecer paths circulares/elípticos.
    indent              : string de indentação por nível.

    Retorna
    -------
    str com o código TikZ completo.
    """
    lines: list[str] = []
    prec = coordinate_precision

    for path in paths:
        if not path:
            continue

        # --- Tentativa de detecção de primitiva geométrica ---
        if detect_circles and len(path) >= 4:
            # Coleta todos os pontos do path
            all_pts = []
            for seg in path:
                all_pts.append(seg.p0)
                if isinstance(seg, CubicBezierSeg):
                    all_pts.append(seg.p3)
                elif isinstance(seg, LineSeg):
                    all_pts.append(seg.p1)
            pts_arr = np.array([
                _transform(p, scale, img_height, origin_x, origin_y)
                for p in all_pts
            ], dtype=float)

            circle = _fit_circle(pts_arr)
            if circle is not None:
                cx, cy, r = circle
                opt = f"[{draw_options}]" if draw_options else ""
                lines.append(
                    f"{indent}\\draw{opt} "
                    f"{_fmt_pt(cx, cy, prec)} circle ({_fmt(r, prec)});"
                )
                continue

            ellipse = _fit_ellipse(pts_arr)
            if ellipse is not None:
                cx, cy, rx, ry = ellipse
                opt = f"[{draw_options}]" if draw_options else ""
                lines.append(
                    f"{indent}\\draw{opt} "
                    f"{_fmt_pt(cx, cy, prec)} ellipse "
                    f"({_fmt(rx, prec)} and {_fmt(ry, prec)});"
                )
                continue

        # --- Path genérico ---
        lines.append(
            _path_to_tikz(
                path, scale, img_height, origin_x, origin_y,
                prec, draw_options, indent,
            )
        )

    body = "\n".join(lines)

    if standalone:
        pkgs = preamble_packages or []
        pkg_lines = "\n".join(f"\\usepackage{{{p}}}" for p in pkgs)
        return (
            "\\documentclass[tikz,border=4pt]{standalone}\n"
            + (pkg_lines + "\n" if pkg_lines else "")
            + "\\begin{document}\n"
            "\\begin{tikzpicture}\n"
            + body + "\n"
            "\\end{tikzpicture}\n"
            "\\end{document}\n"
        )

    return "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}"


def paths_to_tikz_with_background(
    paths: list[Path],
    *,
    img_height: int,
    img_width: int,
    image_filename: str,
    scale: float = 0.05,
    draw_options: str = "",
    standalone: bool = False,
    coordinate_precision: int = 2,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    detect_circles: bool = True,
    indent: str = "  ",
) -> str:
    """
    Igual a paths_to_tikz, mas inclui \\includegraphics como fundo ---
    exatamente como se faz no TikzEdt para copiar sobre a imagem.

    O \\node posiciona a imagem com o canto inferior esquerdo em (0,0),
    alinhado com o sistema de coordenadas dos paths gerados.

    Parâmetros adicionais
    ---------------------
    img_width       : largura da imagem em pixels.
    image_filename  : nome/caminho do arquivo de imagem para includegraphics.
    """
    w_cm = _fmt(img_width * scale, coordinate_precision)
    h_cm = _fmt(img_height * scale, coordinate_precision)

    bg = (
        f"{indent}\\node[anchor=south west, inner sep=0] at (0,0) "
        f"{{\\includegraphics[width={w_cm}cm, height={h_cm}cm]{{{image_filename}}}}};"
    )

    paths_code = paths_to_tikz(
        paths,
        img_height=img_height,
        scale=scale,
        draw_options=draw_options,
        standalone=False,
        coordinate_precision=coordinate_precision,
        origin_x=origin_x,
        origin_y=origin_y,
        detect_circles=detect_circles,
        indent=indent,
    )

    # Substitui o begin/end para injetar o nó de background
    inner = paths_code.replace("\\begin{tikzpicture}\n", "").replace("\n\\end{tikzpicture}", "")
    body = bg + "\n" + inner

    if standalone:
        return (
            "\\documentclass[tikz,border=4pt]{standalone}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n"
            "\\begin{tikzpicture}\n"
            + body + "\n"
            "\\end{tikzpicture}\n"
            "\\end{document}\n"
        )

    return "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}"
