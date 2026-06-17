"""Skeleton graph construction for isolated centerline tracing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import cv2
import numpy as np


Pixel: TypeAlias = tuple[int, int]


class SkeletonNodeType(Enum):
    """Topology role of a skeleton graph node."""

    ENDPOINT = "endpoint"
    JUNCTION = "junction"
    ISOLATED = "isolated"
    CYCLE_ANCHOR = "cycle_anchor"


@dataclass(frozen=True)
class SkeletonNode:
    """A relevant pixel in a one-pixel-wide skeleton graph."""

    id: str
    pixel: Pixel
    node_type: SkeletonNodeType
    component_id: int
    degree: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("node id must not be empty.")
        _validate_pixel("pixel", self.pixel)
        if not isinstance(self.node_type, SkeletonNodeType):
            raise TypeError("node_type must be a SkeletonNodeType.")
        if int(self.component_id) < 0:
            raise ValueError("component_id must be non-negative.")
        if int(self.degree) < 0:
            raise ValueError("degree must be non-negative.")

    @property
    def row(self) -> int:
        """Return the pixel row."""
        return self.pixel[0]

    @property
    def col(self) -> int:
        """Return the pixel column."""
        return self.pixel[1]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostic representation."""
        return {
            "id": self.id,
            "pixel": [self.row, self.col],
            "type": self.node_type.value,
            "component_id": self.component_id,
            "degree": self.degree,
        }


@dataclass(frozen=True)
class SkeletonEdge:
    """An ordered skeleton pixel chain between graph nodes."""

    id: str
    start_node_id: str
    end_node_id: str
    pixels: tuple[Pixel, ...]
    component_id: int
    is_cycle: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("edge id must not be empty.")
        if not self.start_node_id or not self.end_node_id:
            raise ValueError("edge node ids must not be empty.")
        pixels = tuple(self.pixels)
        if not pixels:
            raise ValueError("edge pixels must not be empty.")
        for index, pixel in enumerate(pixels):
            _validate_pixel(f"pixels[{index}]", pixel)
        object.__setattr__(self, "pixels", pixels)
        if int(self.component_id) < 0:
            raise ValueError("component_id must be non-negative.")
        if not isinstance(self.is_cycle, bool):
            raise TypeError("is_cycle must be a bool.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def pixel_length(self) -> int:
        """Return the number of pixels in the ordered chain."""
        return len(self.pixels)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostic representation."""
        return {
            "id": self.id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "component_id": self.component_id,
            "is_cycle": self.is_cycle,
            "pixel_length": self.pixel_length,
            "first_pixel": list(self.pixels[0]),
            "last_pixel": list(self.pixels[-1]),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SkeletonGraph:
    """Deterministic graph representation of a skeleton image."""

    nodes: tuple[SkeletonNode, ...]
    edges: tuple[SkeletonEdge, ...]
    shape: tuple[int, int]
    component_count: int
    backend: str = "fallback"
    sknw_available: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError("shape must be a non-negative 2D shape.")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    @property
    def endpoint_count(self) -> int:
        """Return endpoint node count."""
        return sum(1 for node in self.nodes if node.node_type is SkeletonNodeType.ENDPOINT)

    @property
    def junction_count(self) -> int:
        """Return junction node count."""
        return sum(1 for node in self.nodes if node.node_type is SkeletonNodeType.JUNCTION)

    @property
    def isolated_pixel_count(self) -> int:
        """Return isolated node count."""
        return sum(1 for node in self.nodes if node.node_type is SkeletonNodeType.ISOLATED)

    @property
    def cycle_count(self) -> int:
        """Return cycle edge count."""
        return sum(1 for edge in self.edges if edge.is_cycle)

    def node_by_id(self) -> dict[str, SkeletonNode]:
        """Return nodes keyed by deterministic id."""
        return {node.id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without storing an image matrix."""
        return {
            "shape": list(self.shape),
            "component_count": self.component_count,
            "backend": self.backend,
            "sknw_available": self.sknw_available,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "endpoint_count": self.endpoint_count,
            "junction_count": self.junction_count,
            "isolated_pixel_count": self.isolated_pixel_count,
            "cycle_count": self.cycle_count,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
        }


def build_skeleton_graph(
    skeleton: np.ndarray,
    *,
    connectivity: int = 8,
    use_sknw_if_available: bool = False,
    strict: bool = False,
) -> SkeletonGraph:
    """Convert a skeleton mask into a deterministic graph."""
    binary = _normalize_skeleton(skeleton)
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")

    sknw_available = _is_sknw_available() if use_sknw_if_available else False
    backend = "fallback"
    pixels = _sorted_pixels(binary)
    if not pixels:
        return SkeletonGraph(
            nodes=(),
            edges=(),
            shape=binary.shape,
            component_count=0,
            backend=backend,
            sknw_available=sknw_available,
        )

    labels, component_count = _component_labels(binary, connectivity)
    neighbors = {pixel: _neighbors_for(pixel, binary, connectivity) for pixel in pixels}
    nodes: list[SkeletonNode] = []
    edges: list[SkeletonEdge] = []

    for component_id in range(1, component_count + 1):
        component_pixels = [pixel for pixel in pixels if labels[pixel] == component_id]
        node_pixels = [
            pixel
            for pixel in component_pixels
            if _node_type_from_degree(len(neighbors[pixel])) is not None
        ]
        if not node_pixels:
            cycle_nodes, cycle_edges = _build_pure_cycle(component_pixels, neighbors, component_id, len(nodes), len(edges))
            nodes.extend(cycle_nodes)
            edges.extend(cycle_edges)
            continue

        node_id_by_pixel: dict[Pixel, str] = {}
        for pixel in sorted(node_pixels):
            node_type = _node_type_from_degree(len(neighbors[pixel]))
            if node_type is None:
                continue
            node_id = f"n{len(nodes):04d}"
            node_id_by_pixel[pixel] = node_id
            nodes.append(
                SkeletonNode(
                    id=node_id,
                    pixel=pixel,
                    node_type=node_type,
                    component_id=component_id - 1,
                    degree=len(neighbors[pixel]),
                )
            )

        visited_links: set[frozenset[Pixel]] = set()
        node_pixel_set = set(node_id_by_pixel)
        for start_pixel in sorted(node_pixel_set):
            for neighbor in neighbors[start_pixel]:
                link = frozenset((start_pixel, neighbor))
                if link in visited_links:
                    continue
                walked_pixels, end_pixel = _walk_edge(start_pixel, neighbor, node_pixel_set, neighbors, visited_links)
                if not walked_pixels:
                    continue
                if end_pixel is None:
                    continue
                start_id = node_id_by_pixel[start_pixel]
                end_id = node_id_by_pixel[end_pixel]
                edge_id = f"e{len(edges):04d}"
                edges.append(
                    SkeletonEdge(
                        id=edge_id,
                        start_node_id=start_id,
                        end_node_id=end_id,
                        pixels=tuple(walked_pixels),
                        component_id=component_id - 1,
                        is_cycle=start_id == end_id and len(walked_pixels) > 2,
                    )
                )

    nodes = sorted(nodes, key=lambda node: (node.component_id, node.pixel, node.id))
    node_id_map = {node.id: f"n{index:04d}" for index, node in enumerate(nodes)}
    remapped_nodes = tuple(
        SkeletonNode(
            id=node_id_map[node.id],
            pixel=node.pixel,
            node_type=node.node_type,
            component_id=node.component_id,
            degree=node.degree,
        )
        for node in nodes
    )
    remapped_edges = tuple(
        SkeletonEdge(
            id=f"e{index:04d}",
            start_node_id=node_id_map[edge.start_node_id],
            end_node_id=node_id_map[edge.end_node_id],
            pixels=edge.pixels,
            component_id=edge.component_id,
            is_cycle=edge.is_cycle,
            metadata=edge.metadata,
        )
        for index, edge in enumerate(
            sorted(
                edges,
                key=lambda edge: (
                    edge.component_id,
                    min(edge.pixels),
                    max(edge.pixels),
                    edge.start_node_id,
                    edge.end_node_id,
                ),
            )
        )
    )

    warnings = _validate_graph(remapped_nodes, remapped_edges, binary.shape, strict)
    return SkeletonGraph(
        nodes=remapped_nodes,
        edges=remapped_edges,
        shape=binary.shape,
        component_count=component_count,
        backend=backend,
        sknw_available=sknw_available,
        warnings=tuple(warnings),
    )


def _build_pure_cycle(
    component_pixels: list[Pixel],
    neighbors: dict[Pixel, list[Pixel]],
    component_id: int,
    node_offset: int,
    edge_offset: int,
) -> tuple[list[SkeletonNode], list[SkeletonEdge]]:
    if not component_pixels:
        return [], []
    start = min(component_pixels)
    ordered = _walk_cycle(start, neighbors)
    node_id = f"n{node_offset:04d}"
    edge_id = f"e{edge_offset:04d}"
    node = SkeletonNode(
        id=node_id,
        pixel=start,
        node_type=SkeletonNodeType.CYCLE_ANCHOR,
        component_id=component_id - 1,
        degree=len(neighbors[start]),
    )
    edge = SkeletonEdge(
        id=edge_id,
        start_node_id=node_id,
        end_node_id=node_id,
        pixels=tuple(ordered),
        component_id=component_id - 1,
        is_cycle=True,
    )
    return [node], [edge]


def _walk_cycle(start: Pixel, neighbors: dict[Pixel, list[Pixel]]) -> list[Pixel]:
    ordered = [start]
    if not neighbors[start]:
        return ordered
    previous = start
    current = neighbors[start][0]
    visited_links = {frozenset((previous, current))}
    guard = 0
    while current != start and guard <= len(neighbors) + 1:
        ordered.append(current)
        choices = [pixel for pixel in neighbors[current] if pixel != previous]
        if not choices:
            break
        next_pixel = sorted(choices, key=lambda pixel: (frozenset((current, pixel)) in visited_links, pixel))[0]
        visited_links.add(frozenset((current, next_pixel)))
        previous, current = current, next_pixel
        guard += 1
    return ordered


def _walk_edge(
    start_pixel: Pixel,
    first_pixel: Pixel,
    node_pixels: set[Pixel],
    neighbors: dict[Pixel, list[Pixel]],
    visited_links: set[frozenset[Pixel]],
) -> tuple[list[Pixel], Pixel | None]:
    pixels = [start_pixel]
    previous = start_pixel
    current = first_pixel
    visited_links.add(frozenset((start_pixel, first_pixel)))
    guard = 0

    while guard <= len(neighbors) + 1:
        pixels.append(current)
        if current in node_pixels:
            return pixels, current
        candidates = [pixel for pixel in neighbors[current] if pixel != previous]
        if not candidates:
            return pixels, None
        unvisited = [pixel for pixel in candidates if frozenset((current, pixel)) not in visited_links]
        if not unvisited:
            return pixels, None
        next_pixel = sorted(unvisited)[0]
        visited_links.add(frozenset((current, next_pixel)))
        previous, current = current, next_pixel
        guard += 1
    return pixels, None


def _normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    array = np.asarray(skeleton)
    if array.size == 0:
        raise ValueError("skeleton must not be empty.")
    if array.ndim != 2:
        raise ValueError("skeleton must be 2D.")
    if not np.all(np.isfinite(array)):
        raise ValueError("skeleton values must be finite.")
    return array > 0


def _component_labels(binary: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=connectivity)
    return labels, max(0, int(count) - 1)


def _sorted_pixels(binary: np.ndarray) -> list[Pixel]:
    rows, cols = np.nonzero(binary)
    return sorted((int(row), int(col)) for row, col in zip(rows, cols, strict=True))


def _neighbors_for(pixel: Pixel, binary: np.ndarray, connectivity: int) -> list[Pixel]:
    row, col = pixel
    offsets = _neighbor_offsets(connectivity)
    height, width = binary.shape
    neighbors: list[Pixel] = []
    for dy, dx in offsets:
        y = row + dy
        x = col + dx
        if 0 <= y < height and 0 <= x < width and binary[y, x]:
            neighbors.append((int(y), int(x)))
    return sorted(neighbors)


def _neighbor_offsets(connectivity: int) -> tuple[Pixel, ...]:
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    raise ValueError("connectivity must be 4 or 8.")


def _node_type_from_degree(degree: int) -> SkeletonNodeType | None:
    if degree == 0:
        return SkeletonNodeType.ISOLATED
    if degree == 1:
        return SkeletonNodeType.ENDPOINT
    if degree >= 3:
        return SkeletonNodeType.JUNCTION
    return None


def _validate_graph(
    nodes: tuple[SkeletonNode, ...],
    edges: tuple[SkeletonEdge, ...],
    shape: tuple[int, int],
    strict: bool,
) -> list[str]:
    warnings: list[str] = []
    node_by_id = {node.id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        warnings.append("duplicate node id")
    seen_edges: set[tuple[str, str, tuple[Pixel, ...]]] = set()
    for node in nodes:
        if not _pixel_in_shape(node.pixel, shape):
            warnings.append(f"node {node.id} is outside image")
        if node.node_type is SkeletonNodeType.ENDPOINT and node.degree != 1:
            warnings.append(f"endpoint {node.id} has degree {node.degree}")
        if node.node_type is SkeletonNodeType.JUNCTION and node.degree < 3:
            warnings.append(f"junction {node.id} has degree {node.degree}")
    for edge in edges:
        if edge.start_node_id not in node_by_id or edge.end_node_id not in node_by_id:
            warnings.append(f"edge {edge.id} references missing node")
        if not edge.pixels:
            warnings.append(f"edge {edge.id} has no pixels")
        for pixel in edge.pixels:
            if not _pixel_in_shape(pixel, shape):
                warnings.append(f"edge {edge.id} contains pixel outside image")
        key = tuple(sorted((edge.start_node_id, edge.end_node_id))), tuple(edge.pixels)
        flat_key = (key[0][0], key[0][1], key[1])
        if flat_key in seen_edges:
            warnings.append(f"edge {edge.id} duplicates another edge")
        seen_edges.add(flat_key)
        if edge.is_cycle and len(edge.pixels) < 3:
            warnings.append(f"cycle edge {edge.id} has too few pixels")
    if warnings and strict:
        raise ValueError("; ".join(sorted(set(warnings))))
    return sorted(set(warnings))


def _is_sknw_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("sknw") is not None
    except Exception:
        return False


def _validate_pixel(name: str, pixel: Pixel) -> None:
    if len(pixel) != 2:
        raise ValueError(f"{name} must contain row and column.")
    row, col = pixel
    if int(row) < 0 or int(col) < 0:
        raise ValueError(f"{name} coordinates must be non-negative.")


def _pixel_in_shape(pixel: Pixel, shape: tuple[int, int]) -> bool:
    row, col = pixel
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def component_pixel_counts(skeleton: np.ndarray, connectivity: int = 8) -> dict[int, int]:
    """Return foreground pixel counts keyed by zero-based component id."""
    binary = _normalize_skeleton(skeleton)
    labels, component_count = _component_labels(binary, connectivity)
    return {
        component_id - 1: int(np.count_nonzero(labels == component_id))
        for component_id in range(1, component_count + 1)
    }


__all__ = [
    "Pixel",
    "SkeletonEdge",
    "SkeletonGraph",
    "SkeletonNode",
    "SkeletonNodeType",
    "build_skeleton_graph",
    "component_pixel_counts",
]
