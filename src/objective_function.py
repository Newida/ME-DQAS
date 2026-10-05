from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from . import simulator
from .maxcut import CompiledMaxCut, count_uncut_edges, estimate_mean_uncut_edges
from .three_sat import CompiledThreeSAT, count_violated_clauses, estimate_mean_violations


_PARAMETRIC_GATES = {"rx", "ry", "rz"}


class ObjectiveFunction(ABC):
    is_parametric: bool = False

    @abstractmethod
    def evaluate(self, samples, S_f) -> np.ndarray:
        pass

    def parameter_shift_gradient(self, samples, S_f, S_theta, **kwargs):
        if self.is_parametric:
            raise NotImplementedError(
                "Parametric objective functions must implement parameter_shift_gradient."
            )
        return None


class ThreeSATObjectiveFunction(ObjectiveFunction):
    def __init__(
        self,
        gate_set: Sequence[Any],
        is_parametric: bool,
        clauses: CompiledThreeSAT,
        *,
        theta: np.ndarray | None = None,
        seed: int = 0,
        precision: str = "single",
        simulator_backend=simulator,
    ):
        gate_set = tuple(gate_set)
        if not gate_set:
            raise ValueError("gate_set must contain at least one gate.")
        self.gate_set = gate_set
        self.is_parametric = bool(is_parametric)
        self.compiled_clauses = clauses
        self.theta = None if theta is None else np.asarray(theta, dtype=np.float64)
        self.seed = int(seed)
        self.precision = precision
        self.simulator = simulator_backend

        self.num_qubits = self.compiled_clauses.num_variables
        self._seed_offset = 0

    @staticmethod
    def _is_parametric_gate(gate: Any) -> bool:
        return isinstance(gate, str) and gate.lower() in _PARAMETRIC_GATES

    @staticmethod
    def _normalize_samples(samples) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.intp)
        if samples.ndim == 1:
            return samples[None, :]
        if samples.ndim != 2:
            raise ValueError("samples must have shape (batch_size, num_positions).")
        return samples

    def _next_seed(self) -> int:
        seed = self.seed + self._seed_offset
        self._seed_offset += 1
        return seed

    def _validate_theta(self, num_positions: int) -> np.ndarray | None:
        if self.theta is None:
            return None
        if self.theta.shape in {
            (num_positions,),
            (num_positions, len(self.gate_set)),
        }:
            return self.theta
        raise ValueError(
            "theta must have shape (num_positions,) or (num_positions, num_gates)."
        )

    @staticmethod
    def _theta_value(
        theta: np.ndarray | None,
        position: int,
        gate_index: int,
    ) -> float:
        if theta is None:
            return 0.0
        if theta.ndim == 1:
            return float(theta[position])
        return float(theta[position, gate_index])

    def build_circuit(self, choices, theta: np.ndarray | None = None):
        circuit = QuantumCircuit(self.num_qubits)
        for position, raw_choice in enumerate(choices):
            gate_index = int(raw_choice)
            gate = self.gate_set[gate_index]
            if not isinstance(gate, str):
                raise TypeError("ThreeSATObjectiveFunction expects gate_set entries to be strings.")

            gate_name = gate.lower()
            qubit = position % self.num_qubits
            if gate_name == "h":
                circuit.h(qubit)
            elif gate_name == "x":
                circuit.x(qubit)
            elif gate_name == "y":
                circuit.y(qubit)
            elif gate_name == "z":
                circuit.z(qubit)
            elif gate_name == "s":
                circuit.s(qubit)
            elif gate_name == "sdg":
                circuit.sdg(qubit)
            elif gate_name == "t":
                circuit.t(qubit)
            elif gate_name == "tdg":
                circuit.tdg(qubit)
            elif gate_name == "rx":
                circuit.rx(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name == "ry":
                circuit.ry(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name == "rz":
                circuit.rz(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name in {"cx", "cnot"}:
                circuit.cx(qubit, (qubit + 1) % self.num_qubits)
            elif gate_name == "cz":
                circuit.cz(qubit, (qubit + 1) % self.num_qubits)
            elif gate_name.startswith("cz_o"):
                offset = int(gate_name.split("o", 1)[1])
                circuit.cz(qubit, (qubit + offset) % self.num_qubits)
            else:
                raise ValueError(f"Unsupported gate in objective circuit builder: {gate!r}.")
        return circuit

    def build_circuits(self, samples, theta: np.ndarray | None = None) -> list[object]:
        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1]) if theta is None else theta
        return [self.build_circuit(sample, theta=theta) for sample in samples]

    def parameter_shift_pair_count(self, samples) -> int:
        if not self.is_parametric:
            return 0
        samples = self._normalize_samples(samples)
        count = 0
        for sample in samples:
            for raw_gate_index in sample:
                gate_index = int(raw_gate_index)
                if self._is_parametric_gate(self.gate_set[gate_index]):
                    count += 1
        return count

    def parameter_shift_shot_cost(self, samples, S_theta: int) -> int:
        if not isinstance(S_theta, int) or isinstance(S_theta, bool) or S_theta <= 0:
            raise ValueError("S_theta must be a positive integer.")
        return 2 * self.parameter_shift_pair_count(samples) * S_theta

    def _proposal_objective(self, measurements: np.ndarray) -> float:
        return float(
            estimate_mean_violations(
                samples=measurements,
                compiled=self.compiled_clauses,
            )
        )

    def sample_costs(self, samples) -> np.ndarray:
        return count_violated_clauses(samples=samples, compiled=self.compiled_clauses)

    def _proposal_values(self, measurements: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        values = estimate_mean_violations(
            samples=measurements,
            compiled=self.compiled_clauses,
        )
        return np.atleast_1d(values)

    def evaluate(self, samples, S_f) -> np.ndarray:
        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1])
        circuits = [self.build_circuit(sample, theta=theta) for sample in samples]
        measurements = self.simulator.simulate_and_sample_batch(
            circuits=circuits,
            shots=S_f,
            seed=self._next_seed(),
            precision=self.precision,
        )
        return self._proposal_values(measurements)

    def parameter_shift_gradient(self, samples, S_f, S_theta, *, max_shifted_pairs: int | None = None):
        if not self.is_parametric:
            return None
        if not isinstance(S_theta, int) or isinstance(S_theta, bool) or S_theta <= 0:
            raise ValueError("S_theta must be a positive integer.")
        if max_shifted_pairs is not None:
            if (
                not isinstance(max_shifted_pairs, int)
                or isinstance(max_shifted_pairs, bool)
                or max_shifted_pairs < 0
            ):
                raise ValueError("max_shifted_pairs must be a non-negative integer or None.")

        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1])
        if theta is None:
            raise ValueError("theta is required for parameter_shift_gradient.")

        plus_circuits: list[object] = []
        minus_circuits: list[object] = []
        coordinates: list[int | tuple[int, int]] = []
        for sample in samples:
            for position, raw_gate_index in enumerate(sample):
                if max_shifted_pairs is not None and len(coordinates) >= max_shifted_pairs:
                    break
                gate_index = int(raw_gate_index)
                if not self._is_parametric_gate(self.gate_set[gate_index]):
                    continue

                theta_plus = theta.copy()
                theta_minus = theta.copy()
                if theta.ndim == 1:
                    coordinate: int | tuple[int, int] = position
                    theta_plus[position] += np.pi / 2
                    theta_minus[position] -= np.pi / 2
                else:
                    coordinate = (position, gate_index)
                    theta_plus[position, gate_index] += np.pi / 2
                    theta_minus[position, gate_index] -= np.pi / 2

                plus_circuits.append(self.build_circuit(sample, theta=theta_plus))
                minus_circuits.append(self.build_circuit(sample, theta=theta_minus))
                coordinates.append(coordinate)
            if max_shifted_pairs is not None and len(coordinates) >= max_shifted_pairs:
                break

        gradient = np.zeros_like(theta, dtype=np.float64)
        if not coordinates:
            return gradient

        num_shifted_circuits = len(plus_circuits)
        measurements = self.simulator.simulate_and_sample_batch(
            circuits=plus_circuits + minus_circuits,
            shots=S_theta,
            seed=self._next_seed(),
            precision=self.precision,
        )
        plus_values = self._proposal_values(measurements[:num_shifted_circuits])
        minus_values = self._proposal_values(measurements[num_shifted_circuits:])

        for coordinate, value in zip(coordinates, 0.5 * (plus_values - minus_values)):
            gradient[coordinate] += value
        return gradient / samples.shape[0]


class MaxCutObjectiveFunction(ObjectiveFunction):
    """MaxCut objective minimized as the number of uncut edges."""

    def __init__(
        self,
        gate_set: Sequence[Any],
        is_parametric: bool,
        graph: CompiledMaxCut,
        *,
        theta: np.ndarray | None = None,
        seed: int = 0,
        precision: str = "single",
        simulator_backend=simulator,
    ):
        gate_set = tuple(gate_set)
        if not gate_set:
            raise ValueError("gate_set must contain at least one gate.")
        self.gate_set = gate_set
        self.is_parametric = bool(is_parametric)
        self.compiled_graph = graph
        self.theta = None if theta is None else np.asarray(theta, dtype=np.float64)
        self.seed = int(seed)
        self.precision = precision
        self.simulator = simulator_backend

        self.num_qubits = self.compiled_graph.num_variables
        self._seed_offset = 0

    @staticmethod
    def _is_parametric_gate(gate: Any) -> bool:
        return isinstance(gate, str) and gate.lower() in _PARAMETRIC_GATES

    @staticmethod
    def _normalize_samples(samples) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.intp)
        if samples.ndim == 1:
            return samples[None, :]
        if samples.ndim != 2:
            raise ValueError("samples must have shape (batch_size, num_positions).")
        return samples

    def _next_seed(self) -> int:
        seed = self.seed + self._seed_offset
        self._seed_offset += 1
        return seed

    def _validate_theta(self, num_positions: int) -> np.ndarray | None:
        if self.theta is None:
            return None
        if self.theta.shape in {
            (num_positions,),
            (num_positions, len(self.gate_set)),
        }:
            return self.theta
        raise ValueError(
            "theta must have shape (num_positions,) or (num_positions, num_gates)."
        )

    @staticmethod
    def _theta_value(
        theta: np.ndarray | None,
        position: int,
        gate_index: int,
    ) -> float:
        if theta is None:
            return 0.0
        if theta.ndim == 1:
            return float(theta[position])
        return float(theta[position, gate_index])

    def build_circuit(self, choices, theta: np.ndarray | None = None):
        circuit = QuantumCircuit(self.num_qubits)
        for position, raw_choice in enumerate(choices):
            gate_index = int(raw_choice)
            gate = self.gate_set[gate_index]
            if not isinstance(gate, str):
                raise TypeError("MaxCutObjectiveFunction expects gate_set entries to be strings.")

            gate_name = gate.lower()
            qubit = position % self.num_qubits
            if gate_name == "h":
                circuit.h(qubit)
            elif gate_name == "x":
                circuit.x(qubit)
            elif gate_name == "y":
                circuit.y(qubit)
            elif gate_name == "z":
                circuit.z(qubit)
            elif gate_name == "s":
                circuit.s(qubit)
            elif gate_name == "sdg":
                circuit.sdg(qubit)
            elif gate_name == "t":
                circuit.t(qubit)
            elif gate_name == "tdg":
                circuit.tdg(qubit)
            elif gate_name == "rx":
                circuit.rx(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name == "ry":
                circuit.ry(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name == "rz":
                circuit.rz(self._theta_value(theta, position, gate_index), qubit)
            elif gate_name in {"cx", "cnot"}:
                circuit.cx(qubit, (qubit + 1) % self.num_qubits)
            elif gate_name == "cz":
                circuit.cz(qubit, (qubit + 1) % self.num_qubits)
            elif gate_name.startswith("cz_o"):
                offset = int(gate_name.split("o", 1)[1])
                circuit.cz(qubit, (qubit + offset) % self.num_qubits)
            else:
                raise ValueError(f"Unsupported gate in objective circuit builder: {gate!r}.")
        return circuit

    def build_circuits(self, samples, theta: np.ndarray | None = None) -> list[object]:
        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1]) if theta is None else theta
        return [self.build_circuit(sample, theta=theta) for sample in samples]

    def parameter_shift_pair_count(self, samples) -> int:
        if not self.is_parametric:
            return 0
        samples = self._normalize_samples(samples)
        count = 0
        for sample in samples:
            for raw_gate_index in sample:
                gate_index = int(raw_gate_index)
                if self._is_parametric_gate(self.gate_set[gate_index]):
                    count += 1
        return count

    def parameter_shift_shot_cost(self, samples, S_theta: int) -> int:
        if not isinstance(S_theta, int) or isinstance(S_theta, bool) or S_theta <= 0:
            raise ValueError("S_theta must be a positive integer.")
        return 2 * self.parameter_shift_pair_count(samples) * S_theta

    def _proposal_objective(self, measurements: np.ndarray) -> float:
        return float(
            estimate_mean_uncut_edges(
                samples=measurements,
                compiled=self.compiled_graph,
            )
        )

    def sample_costs(self, samples) -> np.ndarray:
        return count_uncut_edges(samples=samples, compiled=self.compiled_graph)

    def _proposal_values(self, measurements: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        values = estimate_mean_uncut_edges(
            samples=measurements,
            compiled=self.compiled_graph,
        )
        return np.atleast_1d(values)

    def evaluate(self, samples, S_f) -> np.ndarray:
        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1])
        circuits = [self.build_circuit(sample, theta=theta) for sample in samples]
        measurements = self.simulator.simulate_and_sample_batch(
            circuits=circuits,
            shots=S_f,
            seed=self._next_seed(),
            precision=self.precision,
        )
        return self._proposal_values(measurements)

    def parameter_shift_gradient(self, samples, S_f, S_theta, *, max_shifted_pairs: int | None = None):
        if not self.is_parametric:
            return None
        if not isinstance(S_theta, int) or isinstance(S_theta, bool) or S_theta <= 0:
            raise ValueError("S_theta must be a positive integer.")
        if max_shifted_pairs is not None:
            if (
                not isinstance(max_shifted_pairs, int)
                or isinstance(max_shifted_pairs, bool)
                or max_shifted_pairs < 0
            ):
                raise ValueError("max_shifted_pairs must be a non-negative integer or None.")

        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1])
        if theta is None:
            raise ValueError("theta is required for parameter_shift_gradient.")

        plus_circuits: list[object] = []
        minus_circuits: list[object] = []
        coordinates: list[int | tuple[int, int]] = []
        for sample in samples:
            for position, raw_gate_index in enumerate(sample):
                if max_shifted_pairs is not None and len(coordinates) >= max_shifted_pairs:
                    break
                gate_index = int(raw_gate_index)
                if not self._is_parametric_gate(self.gate_set[gate_index]):
                    continue

                theta_plus = theta.copy()
                theta_minus = theta.copy()
                if theta.ndim == 1:
                    coordinate: int | tuple[int, int] = position
                    theta_plus[position] += np.pi / 2
                    theta_minus[position] -= np.pi / 2
                else:
                    coordinate = (position, gate_index)
                    theta_plus[position, gate_index] += np.pi / 2
                    theta_minus[position, gate_index] -= np.pi / 2

                plus_circuits.append(self.build_circuit(sample, theta=theta_plus))
                minus_circuits.append(self.build_circuit(sample, theta=theta_minus))
                coordinates.append(coordinate)
            if max_shifted_pairs is not None and len(coordinates) >= max_shifted_pairs:
                break

        gradient = np.zeros_like(theta, dtype=np.float64)
        if not coordinates:
            return gradient

        num_shifted_circuits = len(plus_circuits)
        measurements = self.simulator.simulate_and_sample_batch(
            circuits=plus_circuits + minus_circuits,
            shots=S_theta,
            seed=self._next_seed(),
            precision=self.precision,
        )
        plus_values = self._proposal_values(measurements[:num_shifted_circuits])
        minus_values = self._proposal_values(measurements[num_shifted_circuits:])

        for coordinate, value in zip(coordinates, 0.5 * (plus_values - minus_values)):
            gradient[coordinate] += value
        return gradient / samples.shape[0]
