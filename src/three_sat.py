"""Utilities for generating, compiling, and scoring 3-SAT instances."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CompiledThreeSAT:
    num_variables: int
    clause_var_idx: np.ndarray
    clause_is_negated: np.ndarray


def _validate_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def generate_dimacs_3sat(
    num_variables: int,
    num_clauses: int,
    seed: int | None = None,
) -> str:
    n = _validate_positive_int(num_variables, "num_variables")
    m = _validate_positive_int(num_clauses, "num_clauses")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer when provided.")

    rng = np.random.default_rng(seed)
    lines = [f"p cnf {n} {m}"]
    signs = np.array([-1, 1], dtype=np.int8)
    for _ in range(m):
        variables = rng.choice(n, size=3, replace=n < 3) + 1
        literals = variables.astype(np.int64) * rng.choice(signs, size=3)
        lines.append(f"{int(literals[0])} {int(literals[1])} {int(literals[2])} 0")
    return "\n".join(lines)


def canonicalize_clause(clause: Sequence[int]) -> tuple[int, int, int]:
    if len(clause) != 3:
        raise ValueError("clause must contain exactly three literals.")

    normalized: list[int] = []
    for literal in clause:
        if not isinstance(literal, (int, np.integer)) or isinstance(literal, bool):
            raise ValueError("clause literals must be integers.")
        literal = int(literal)
        normalized.append(literal)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (item if item >= 0 else -item - 1, item < 0),
        )
    )


def compile_internal_clauses(
    clauses: Sequence[Sequence[int]],
    num_variables: int,
) -> CompiledThreeSAT:
    num_variables = _validate_positive_int(num_variables, "num_variables")
    clauses = list(clauses)
    if not clauses:
        raise ValueError("clauses must contain at least one clause.")

    clause_var_idx = np.empty((len(clauses), 3), dtype=np.intp)
    clause_is_negated = np.empty((len(clauses), 3), dtype=bool)
    seen_clauses: set[tuple[int, int, int]] = set()

    for row, clause in enumerate(clauses):
        canonical = canonicalize_clause(clause)
        if canonical in seen_clauses:
            raise ValueError("clauses contain a duplicate clause.")
        seen_clauses.add(canonical)

        for column, literal in enumerate(clause):
            literal = int(literal)
            variable = literal if literal >= 0 else -literal - 1
            if variable >= num_variables:
                raise ValueError("clause contains out-of-range variable indices.")
            clause_var_idx[row, column] = variable
            clause_is_negated[row, column] = literal < 0

    clause_var_idx.setflags(write=False)
    clause_is_negated.setflags(write=False)
    compiled = CompiledThreeSAT(
        num_variables=num_variables,
        clause_var_idx=clause_var_idx,
        clause_is_negated=clause_is_negated,
    )
    _validate_compiled(compiled)
    return compiled


def _validate_compiled(compiled: CompiledThreeSAT) -> None:
    if not isinstance(compiled, CompiledThreeSAT):
        raise TypeError("compiled must be an instance of CompiledThreeSAT.")
    if (
        not isinstance(compiled.num_variables, int)
        or isinstance(compiled.num_variables, bool)
        or compiled.num_variables <= 0
    ):
        raise ValueError("compiled.num_variables must be a positive integer.")
    if not isinstance(compiled.clause_var_idx, np.ndarray):
        raise TypeError("compiled.clause_var_idx must be a numpy.ndarray.")
    if not isinstance(compiled.clause_is_negated, np.ndarray):
        raise TypeError("compiled.clause_is_negated must be a numpy.ndarray.")
    if compiled.clause_var_idx.dtype != np.intp:
        raise ValueError("compiled.clause_var_idx must have dtype np.intp.")
    if compiled.clause_is_negated.dtype != np.bool_:
        raise ValueError("compiled.clause_is_negated must have dtype bool.")
    if compiled.clause_var_idx.ndim != 2 or compiled.clause_var_idx.shape[1] != 3:
        raise ValueError("compiled.clause_var_idx must have shape (m, 3).")
    if compiled.clause_is_negated.ndim != 2 or compiled.clause_is_negated.shape[1] != 3:
        raise ValueError("compiled.clause_is_negated must have shape (m, 3).")
    if compiled.clause_var_idx.shape[0] == 0:
        raise ValueError("compiled must contain at least one clause.")
    if compiled.clause_var_idx.shape != compiled.clause_is_negated.shape:
        raise ValueError("compiled clause arrays must have identical shapes.")
    if np.any(compiled.clause_var_idx < 0) or np.any(
        compiled.clause_var_idx >= compiled.num_variables
    ):
        raise ValueError("compiled.clause_var_idx contains out-of-range variable indices.")


def _validate_sample_matrix(
    sample_matrix: np.ndarray,
    num_variables: int,
    name: str,
) -> None:
    if not isinstance(sample_matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if sample_matrix.dtype != np.bool_:
        raise ValueError(f"{name} must have dtype bool.")
    if sample_matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (shots, n).")
    if sample_matrix.shape[-1] != num_variables:
        raise ValueError(
            f"{name} last dimension must equal num_variables={num_variables}, "
            f"got {sample_matrix.shape[-1]}."
        )


def _normalize_samples(
    samples: np.ndarray | Sequence[np.ndarray],
    num_variables: int,
) -> tuple[str, np.ndarray | list[np.ndarray]]:
    if isinstance(samples, np.ndarray):
        if samples.dtype != np.bool_:
            raise ValueError("samples must have dtype bool.")
        if samples.ndim not in (2, 3):
            raise ValueError(
                "samples must have shape (shots, n), (M, shots, n), "
                "or be a sequence of (shots_m, n) arrays."
            )
        if samples.shape[-1] != num_variables:
            raise ValueError(
                f"samples last dimension must equal num_variables={num_variables}, "
                f"got {samples.shape[-1]}."
            )
        return ("2d" if samples.ndim == 2 else "3d"), samples

    if isinstance(samples, (str, bytes)):
        raise TypeError(
            "samples must be a numpy.ndarray or a sequence of numpy.ndarray objects."
        )
    try:
        samples = list(samples)
    except TypeError as exc:
        raise TypeError(
            "samples must be a numpy.ndarray or a sequence of numpy.ndarray objects."
        ) from exc
    if not samples:
        raise ValueError("samples sequence must be non-empty.")
    for index, sample_matrix in enumerate(samples):
        _validate_sample_matrix(sample_matrix, num_variables, f"samples[{index}]")
    return "sequence", samples


def _count_violated_for_2d_samples(
    samples: np.ndarray,
    compiled: CompiledThreeSAT,
) -> np.ndarray:
    literal_values = samples[:, compiled.clause_var_idx]
    literal_truth = np.logical_xor(literal_values, compiled.clause_is_negated)
    clause_violated = ~np.any(literal_truth, axis=2)
    return clause_violated.sum(axis=1, dtype=np.int32).astype(np.int32, copy=False)


def count_violated_clauses(
    samples: np.ndarray | Sequence[np.ndarray],
    compiled: CompiledThreeSAT,
) -> np.ndarray | list[np.ndarray]:
    _validate_compiled(compiled)
    mode, samples = _normalize_samples(samples, compiled.num_variables)
    if mode == "2d":
        return _count_violated_for_2d_samples(samples, compiled)
    if mode == "3d":
        leading_shape = samples.shape[:2]
        flat_samples = samples.reshape(-1, compiled.num_variables)
        return _count_violated_for_2d_samples(flat_samples, compiled).reshape(leading_shape)
    return [
        _count_violated_for_2d_samples(sample_matrix, compiled)
        for sample_matrix in samples
    ]


def estimate_mean_violations(
    samples: np.ndarray | Sequence[np.ndarray],
    compiled: CompiledThreeSAT,
) -> float | np.ndarray:
    violated = count_violated_clauses(samples=samples, compiled=compiled)
    if isinstance(violated, list):
        return np.asarray(
            [np.mean(values, dtype=np.float64) for values in violated],
            dtype=np.float64,
        )
    if violated.ndim == 1:
        return float(np.mean(violated, dtype=np.float64))
    return np.mean(violated, axis=1, dtype=np.float64)


def estimate_global_mean_violations(
    samples: np.ndarray | Sequence[np.ndarray],
    compiled: CompiledThreeSAT,
) -> float:
    per_circuit = estimate_mean_violations(samples=samples, compiled=compiled)
    return float(np.mean(per_circuit, dtype=np.float64))


def count_violated_clauses_device(
    samples: object,
    compiled: CompiledThreeSAT,
) -> object:
    _validate_compiled(compiled)
    xp = np
    try:
        import cupy as cp  # type: ignore

        xp = cp.get_array_module(samples)
    except Exception:
        pass

    if not all(hasattr(samples, attribute) for attribute in ("dtype", "ndim", "shape")):
        raise TypeError("samples must be a NumPy-like or CuPy-like array.")
    if samples.dtype != xp.bool_:
        raise ValueError("samples must have dtype bool.")
    if samples.ndim not in (2, 3):
        raise ValueError("samples must have shape (shots, n) or (M, shots, n).")
    if samples.shape[-1] != compiled.num_variables:
        raise ValueError(
            f"samples last dimension must equal num_variables={compiled.num_variables}, "
            f"got {samples.shape[-1]}."
        )

    clause_var_idx = xp.asarray(compiled.clause_var_idx)
    clause_is_negated = xp.asarray(compiled.clause_is_negated)
    leading_shape = samples.shape[:-1]
    flat_samples = samples.reshape(-1, compiled.num_variables)
    literal_values = flat_samples[:, clause_var_idx]
    literal_truth = xp.logical_xor(literal_values, clause_is_negated)
    result = (~xp.any(literal_truth, axis=2)).sum(axis=1, dtype=xp.int32)
    return result.reshape(leading_shape).astype(xp.int32, copy=False)


def estimate_mean_violations_device(
    samples: object,
    compiled: CompiledThreeSAT,
) -> object:
    violated = count_violated_clauses_device(samples=samples, compiled=compiled)
    xp = np
    try:
        import cupy as cp  # type: ignore

        xp = cp.get_array_module(violated)
    except Exception:
        pass
    if violated.ndim == 1:
        return xp.mean(violated, dtype=xp.float64)
    return xp.mean(violated, axis=1, dtype=xp.float64)
