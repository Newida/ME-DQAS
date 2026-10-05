from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class CompiledMaxCut:
    num_variables: int
    edge_u: np.ndarray
    edge_v: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.edge_u.size)


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def compile_edges(edges: Sequence[Sequence[int]], num_variables: int) -> CompiledMaxCut:
    n = _validate_positive_int(num_variables, "num_variables")
    edges = list(edges)
    if not edges:
        raise ValueError("edges must contain at least one edge.")

    canonical_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_edge in edges:
        if len(raw_edge) != 2:
            raise ValueError(f"edge {raw_edge!r} does not contain exactly two vertices.")
        u, v = (int(raw_edge[0]), int(raw_edge[1]))
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge {(u, v)!r} contains a vertex outside [0, {n}).")
        if u == v:
            raise ValueError("self-loops are not supported for MaxCut.")
        edge = (u, v) if u < v else (v, u)
        if edge in seen:
            raise ValueError("edges contain a duplicate undirected edge.")
        seen.add(edge)
        canonical_edges.append(edge)

    edge_array = np.asarray(canonical_edges, dtype=np.intp)
    return CompiledMaxCut(
        num_variables=n,
        edge_u=edge_array[:, 0].copy(),
        edge_v=edge_array[:, 1].copy(),
    )


def _validate_compiled(compiled: CompiledMaxCut) -> None:
    if not isinstance(compiled, CompiledMaxCut):
        raise TypeError("compiled must be an instance of CompiledMaxCut.")


def _as_2d_bool_samples(samples, num_variables: int) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(samples, dtype=bool)
    if array.ndim == 1:
        if array.shape[0] != num_variables:
            raise ValueError(
                f"samples have {array.shape[0]} variables, expected {num_variables}."
            )
        return array[None, :], ()
    if array.ndim < 2:
        raise ValueError("samples must have shape (..., num_variables).")
    if array.shape[-1] != num_variables:
        raise ValueError(
            f"samples have {array.shape[-1]} variables, expected {num_variables}."
        )
    leading_shape = array.shape[:-1]
    return array.reshape(-1, num_variables), leading_shape


def count_cut_edges(samples, compiled: CompiledMaxCut) -> np.ndarray:
    _validate_compiled(compiled)
    matrix, leading_shape = _as_2d_bool_samples(samples, compiled.num_variables)
    cut = matrix[:, compiled.edge_u] != matrix[:, compiled.edge_v]
    values = np.sum(cut, axis=1, dtype=np.int32)
    if leading_shape == ():
        return values
    return values.reshape(leading_shape)


def count_uncut_edges(samples, compiled: CompiledMaxCut) -> np.ndarray:
    _validate_compiled(compiled)
    matrix, leading_shape = _as_2d_bool_samples(samples, compiled.num_variables)
    uncut = matrix[:, compiled.edge_u] == matrix[:, compiled.edge_v]
    values = np.sum(uncut, axis=1, dtype=np.int32)
    if leading_shape == ():
        return values
    return values.reshape(leading_shape)


def estimate_mean_uncut_edges(samples, compiled: CompiledMaxCut) -> np.ndarray | float:
    if isinstance(samples, list):
        return np.asarray(
            [float(np.mean(count_uncut_edges(sample, compiled))) for sample in samples],
            dtype=np.float64,
        )
    values = count_uncut_edges(samples, compiled)
    if values.ndim == 1:
        return float(np.mean(values, dtype=np.float64))
    return np.mean(values, axis=1, dtype=np.float64)
