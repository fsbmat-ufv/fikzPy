"""
tracer.py
---------
Extração de esqueleto, traçado de paths e fitting de curvas de Bézier cúbico.

Visão geral do pipeline
-----------------------
  máscara binária
    → skeletonize          (reduz traços espessos a linhas de 1 px)
    → graph_from_skeleton  (constrói grafo de adjacência dos pixels ativos)
    → trace_paths          (percorre o grafo e extrai listas de pontos ordenados)
    → fit_bezier_paths     (ajusta curvas de Bézier cúbico a cada sub-path)

Saída
-----
  Lista de Path, onde cada Path é uma lista de segmentos.
  Um segmento é um dos seguintes:
    LineSeg(p0, p1)
    CubicBezierSeg(p0, c1, c2, p3)

  Essas estruturas são consumidas pelo tikz_generator.py.

Algoritmo de Bézier
-------------------
  Implementamos o fitting de Schneider (1990) simplificado:
  dado um conjunto de pontos amostrados, encontramos os pontos de controle
  c1 e c2 que minimizam o erro quadrático médio usando o método de Newton.
  Não depende de scipy.optimize — usa apenas numpy.

Dependências
------------
  numpy, opencv-python, scikit-image (apenas skimage.morphology.skeletonize)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as _ski_skeletonize
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False


# ---------------------------------------------------------------------------
# Tipos de segmento
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


@dataclass
class LineSeg:
    p0: Point
    p1: Point


@dataclass
class CubicBezierSeg:
    p0: Point
    c1: Point
    c2: Point
    p3: Point


Segment = LineSeg | CubicBezierSeg
Path = List[Segment]


# ---------------------------------------------------------------------------
# 1. Esqueletização
# ---------------------------------------------------------------------------

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Reduz traços de espessura arbitrária a linhas de 1 pixel.

    Usa skimage.morphology.skeletonize (Lee 1994) quando disponível,
    com fallback para thinning iterativo via OpenCV.

    Parâmetros
    ----------
    binary : np.ndarray uint8 (H, W), 255 = traço.

    Retorna
    -------
    np.ndarray uint8 (H, W), 255 = pixel de esqueleto.
    """
    if _SKIMAGE_OK:
        # skimage espera bool array
        skel = _ski_skeletonize(binary > 0)
        return (skel * 255).astype(np.uint8)

    # Fallback: Zhang-Suen via OpenCV ximgproc (opcional)
    try:
        thinned = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thinned
    except AttributeError:
        pass

    # Fallback puro numpy – erosão iterativa (mais lento, mas sem deps extras)
    img = (binary > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = np.zeros_like(img)
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(prev, temp)
        img = eroded.copy()
        prev = skeleton.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skeleton * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Grafo de adjacência do esqueleto
# ---------------------------------------------------------------------------

# Direções 8-conectadas
_D8 = [(-1, -1), (-1, 0), (-1, 1),
       (0, -1),           (0, 1),
       (1, -1),  (1, 0),  (1, 1)]


def _build_adjacency(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Constrói dicionário de adjacência para os pixels ativos do esqueleto.

    Retorna {(r, c): [(r1, c1), ...]}  – vizinhos 8-conectados ativos.
    """
    ys, xs = np.where(skel > 0)
    active = set(zip(ys.tolist(), xs.tolist()))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in active:
        neighbors = [(r + dr, c + dc) for dr, dc in _D8 if (r + dr, c + dc) in active]
        adj[(r, c)] = neighbors
    return adj


# ---------------------------------------------------------------------------
# 3. Extração de paths pelo grafo
# ---------------------------------------------------------------------------

def _degree(adj: dict, node: tuple) -> int:
    return len(adj[node])


def trace_paths(
    skel: np.ndarray,
    min_points: int = 4,
) -> list[list[Point]]:
    """
    Percorre o grafo do esqueleto e retorna listas de pontos ordenados.

    Estratégia:
      - Nós terminais (grau 1) e nós de junção (grau >= 3) são tratados como
        pontos de corte.
      - Cada aresta entre dois pontos de corte consecutivos forma um sub-path.
      - Componentes circulares (sem terminais) também são capturados.

    Coordenadas retornadas em (x, y) com y invertido (sistema TikZ).

    Parâmetros
    ----------
    skel       : esqueleto binário uint8.
    min_points : descarta paths com menos de N pontos (remove artefatos isolados).

    Retorna
    -------
    Lista de listas de pontos (x, y) float.
    """
    adj = _build_adjacency(skel)
    if not adj:
        return []

    visited_edges: set[frozenset] = set()
    paths: list[list[Point]] = []

    # Nós terminais e de junção são pontos de partida prioritários
    start_nodes = [n for n in adj if _degree(adj, n) != 2]
    # Se só há ciclos (todos grau 2), pega qualquer nó
    if not start_nodes:
        start_nodes = list(adj.keys())[:1]

    def _walk(start: tuple, came_from: Optional[tuple]) -> list[tuple]:
        path_nodes = [start]
        current = start
        prev = came_from
        while True:
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            # Segue o único vizinho disponível (linha simples)
            nxt = neighbors[0]
            edge = frozenset([current, nxt])
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path_nodes.append(nxt)
            if _degree(adj, nxt) != 2:
                break  # chegou a terminal ou junção
            prev = current
            current = nxt
        return path_nodes

    # Percorre a partir dos nós de corte
    for start in start_nodes:
        for neighbor in adj[start]:
            edge = frozenset([start, neighbor])
            if edge not in visited_edges:
                visited_edges.add(edge)
                pts = _walk(neighbor, start)
                chain = [start] + pts
                if len(chain) >= min_points:
                    # Converte (row, col) → (x, y)  [col=x, row=y]
                    paths.append([(float(c), float(r)) for r, c in chain])

    # Captura arestas restantes (ciclos ou segmentos não visitados)
    unvisited = [(u, v) for u in adj for v in adj[u] if frozenset([u, v]) not in visited_edges]
    seen_unv: set[frozenset] = set()
    for u, v in unvisited:
        edge = frozenset([u, v])
        if edge in seen_unv:
            continue
        seen_unv.add(edge)
        pts = _walk(v, u)
        chain = [u] + pts
        if len(chain) >= min_points:
            paths.append([(float(c), float(r)) for r, c in chain])

    return paths


# ---------------------------------------------------------------------------
# 4. Simplificação Ramer-Douglas-Peucker
# ---------------------------------------------------------------------------

def _rdp(points: list[Point], epsilon: float) -> list[Point]:
    """
    Ramer-Douglas-Peucker recursivo.
    Reduz o número de pontos mantendo a forma geral do path.
    """
    if len(points) < 3:
        return points

    p0 = np.array(points[0])
    p_end = np.array(points[-1])
    seg = p_end - p0
    seg_len = np.linalg.norm(seg)

    if seg_len == 0:
        dists = [np.linalg.norm(np.array(p) - p0) for p in points[1:-1]]
    else:
        seg_unit = seg / seg_len
        dists = [
            abs(np.cross(seg_unit, np.array(p) - p0))
            for p in points[1:-1]
        ]

    idx = int(np.argmax(dists))
    max_dist = dists[idx]

    if max_dist > epsilon:
        left = _rdp(points[:idx + 2], epsilon)
        right = _rdp(points[idx + 1:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def simplify_path(points: list[Point], epsilon: float = 1.5) -> list[Point]:
    """Simplifica uma lista de pontos com RDP. epsilon em pixels."""
    if len(points) < 3:
        return points
    return _rdp(points, epsilon)


# ---------------------------------------------------------------------------
# 5. Fitting de Bézier cúbico (algoritmo de Schneider simplificado)
# ---------------------------------------------------------------------------

def _chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
    """Parametrização por comprimento de corda normalizado [0, 1]."""
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    total = t[-1]
    if total < 1e-10:
        return np.linspace(0.0, 1.0, len(pts))
    return t / total


def _cubic_bezier(t: np.ndarray, p0, c1, c2, p3) -> np.ndarray:
    """Avalia curva de Bézier cúbica em vetor de parâmetros t."""
    u = 1 - t
    return (
        (u ** 3)[:, None] * p0 +
        (3 * u ** 2 * t)[:, None] * c1 +
        (3 * u * t ** 2)[:, None] * c2 +
        (t ** 3)[:, None] * p3
    )


def _fit_cubic(pts: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve o sistema linear para os pontos de controle internos c1, c2
    dados os pontos de ancoragem p0 = pts[0], p3 = pts[-1].

    Minimiza sum ||B(t_i) - pts_i||² em relação a c1 e c2.
    """
    p0 = pts[0]
    p3 = pts[-1]
    n = len(pts)

    # Bases de Bernstein para c1 e c2
    b1 = (3 * (1 - t) ** 2 * t)          # coef de c1
    b2 = (3 * (1 - t) * t ** 2)          # coef de c2

    # Termo constante: pts - p0*(1-t)³ - p3*t³
    u = 1 - t
    rhs = pts - (u ** 3)[:, None] * p0 - (t ** 3)[:, None] * p3

    # Sistema 2×2:  [A11 A12] [c1]   [r1]
    #               [A21 A22] [c2] = [r2]
    A11 = np.dot(b1, b1)
    A12 = np.dot(b1, b2)
    A22 = np.dot(b2, b2)
    r1 = b1 @ rhs          # shape (2,)
    r2 = b2 @ rhs

    det = A11 * A22 - A12 * A12
    if abs(det) < 1e-10:
        # Sistema singular – degenera para interpolação linear
        c1 = p0 + (p3 - p0) / 3
        c2 = p0 + 2 * (p3 - p0) / 3
        return c1, c2

    c1 = (A22 * r1 - A12 * r2) / det
    c2 = (A11 * r2 - A12 * r1) / det
    return c1, c2


def _fit_segment_bezier(
    pts: np.ndarray,
    max_error: float,
) -> list[CubicBezierSeg | LineSeg]:
    """
    Ajusta recursivamente segmentos de Bézier cúbico a `pts`.

    Se o erro máximo exceder `max_error`, divide na pior amostra e recursa.
    Limita a profundidade para evitar explosão em paths muito irregulares.
    """
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        p0 = tuple(pts[0])
        p1 = tuple(pts[1])
        return [LineSeg(p0, p1)]

    t = _chord_length_parameterize(pts)
    c1, c2 = _fit_cubic(pts, t)
    p0 = pts[0]
    p3 = pts[-1]

    evaluated = _cubic_bezier(t, p0, c1, c2, p3)
    errors = np.linalg.norm(evaluated - pts, axis=1)
    max_err = errors.max()

    if max_err <= max_error:
        return [CubicBezierSeg(
            tuple(p0), tuple(c1), tuple(c2), tuple(p3)
        )]

    # Divide no ponto de erro máximo
    split = int(np.argmax(errors))
    split = max(1, min(split, len(pts) - 2))
    left = _fit_segment_bezier(pts[:split + 1], max_error)
    right = _fit_segment_bezier(pts[split:], max_error)
    return left + right


def fit_bezier_paths(
    point_paths: list[list[Point]],
    max_error: float = 2.0,
    use_bezier: bool = True,
) -> list[Path]:
    """
    Converte listas de pontos em listas de segmentos (LineSeg ou CubicBezierSeg).

    Parâmetros
    ----------
    point_paths : saída de trace_paths().
    max_error   : tolerância de fitting em pixels. Menor = mais segmentos, mais fiel.
    use_bezier  : False → força LineSeg em todos os segmentos (modo clássico).

    Retorna
    -------
    Lista de Path (cada Path é list[Segment]).
    """
    result: list[Path] = []
    for pts_raw in point_paths:
        pts = np.array(pts_raw, dtype=float)
        if len(pts) < 2:
            continue
        if use_bezier and len(pts) >= 4:
            segs = _fit_segment_bezier(pts, max_error)
        else:
            # Segmentos de reta entre pontos consecutivos
            segs = [LineSeg(tuple(pts[i]), tuple(pts[i + 1])) for i in range(len(pts) - 1)]
        if segs:
            result.append(segs)
    return result


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def extract_paths(
    binary: np.ndarray,
    *,
    simplify_epsilon: float = 1.5,
    bezier_max_error: float = 2.0,
    use_bezier: bool = True,
    min_path_points: int = 4,
) -> list[Path]:
    """
    Pipeline completo: binário → Path list pronta para o gerador TikZ.

    Parâmetros
    ----------
    binary            : máscara uint8 da preprocessor (255=traço).
    simplify_epsilon  : tolerância RDP em pixels (maior = menos pontos).
    bezier_max_error  : tolerância do fitting de Bézier em pixels.
    use_bezier        : usar curvas de Bézier cúbico; False = só retas.
    min_path_points   : descarta paths com menos de N pontos.

    Retorna
    -------
    Lista de Path.
    """
    skel = skeletonize(binary)
    raw_paths = trace_paths(skel, min_points=min_path_points)
    simplified = [simplify_path(p, epsilon=simplify_epsilon) for p in raw_paths]
    simplified = [p for p in simplified if len(p) >= 2]
    paths = fit_bezier_paths(simplified, max_error=bezier_max_error, use_bezier=use_bezier)
    return paths
