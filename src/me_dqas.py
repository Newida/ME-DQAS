"""Measurement-efficient DQAS extensions for diagonal cost Hamiltonians.

The extension implements the one-shift bit-flip estimator from the local
IEEE manuscript. For an eligible Pauli rotation followed only by diagonal
gates, the theta gradient is estimated from one shifted circuit by measuring

    0.5 * (f(x) - f(x xor s(P)))

instead of using two full-Hamiltonian parameter-shift measurements. Gates that
do not satisfy the diagonal-suffix condition can still use a one-shift Pauli
term estimator when their suffix can be pushed through as a Clifford circuit.
All other gates fall back to ordinary parameter shift, so this module can be
used as a drop-in extension without changing the existing DQAS behavior outside
the measurement-efficient regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from .dqas import DQAS
from .objective_function import MaxCutObjectiveFunction, ThreeSATObjectiveFunction


_ME_GATES = {"rx", "ry"}
_DIAGONAL_GATES = {"z", "s", "sdg", "t", "tdg", "rz", "cz"}
_PARAMETRIC_ROTATION_GATES = {"rx", "ry", "rz"}
_CLIFFORD_ANGLE_ATOL = 1e-9


@dataclass(frozen=True)
class _ZPauliTerm:
    coefficient: float
    z_mask: int


@dataclass(frozen=True)
class _GradientJob:
    kind: Literal["measurement_efficient", "parameter_shift"]
    estimator: Literal["diagonal_bit_flip", "clifford_pauli", "parameter_shift"]
    sample: np.ndarray
    coordinate: int | tuple[int, int]
    position: int
    gate_index: int
    qubit: int
    active_term_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SelectedJob:
    job: _GradientJob
    shots_per_shift_circuit: int


@dataclass(frozen=True)
class MeasurementEfficientGradientResult:
    gradient: np.ndarray | None
    shots_spent: int
    requested_parametric_terms: int
    requested_jobs: int
    requested_measurement_efficient_jobs: int
    requested_parameter_shift_jobs: int
    requested_diagonal_suffix_jobs: int
    requested_clifford_suffix_jobs: int
    zero_gradient_jobs: int
    executed_parametric_terms: int
    executed_jobs: int
    executed_measurement_efficient_jobs: int
    executed_parameter_shift_jobs: int
    executed_diagonal_suffix_jobs: int
    executed_clifford_suffix_jobs: int
    conventional_full_shot_cost: int
    measurement_efficient_full_shot_cost: int
    zero_gradient_shots_saved_for_full_step: int
    one_shift_shots_saved_for_full_step: int
    diagonal_suffix_shots_saved_for_full_step: int
    clifford_suffix_shots_saved_for_full_step: int
    partial_budget_limited: bool


class MeasurementEfficientDiagonalObjectiveMixin:
    """Mixin implementing the measurement-efficient theta-gradient estimator."""

    @staticmethod
    def _is_diagonal_gate_name(gate_name: str) -> bool:
        return gate_name in _DIAGONAL_GATES or gate_name.startswith("cz_o")

    @staticmethod
    def _bit(qubit: int) -> int:
        return 1 << int(qubit)

    @staticmethod
    def _mask_qubits(mask: int) -> tuple[int, ...]:
        qubits: list[int] = []
        bit_index = 0
        while mask:
            if mask & 1:
                qubits.append(bit_index)
            bit_index += 1
            mask >>= 1
        return tuple(qubits)

    @staticmethod
    def _term_parity(measurements: np.ndarray, z_mask: int) -> np.ndarray:
        if z_mask == 0:
            return np.ones(measurements.shape[0], dtype=np.float64)
        selected = measurements[:, MeasurementEfficientDiagonalObjectiveMixin._mask_qubits(z_mask)]
        odd = np.count_nonzero(selected, axis=1) % 2 == 1
        return np.where(odd, -1.0, 1.0)

    @staticmethod
    def _pauli_label(x_mask: int, z_mask: int, qubit: int) -> str:
        bit = 1 << int(qubit)
        has_x = bool(x_mask & bit)
        has_z = bool(z_mask & bit)
        if has_x and has_z:
            return "y"
        if has_x:
            return "x"
        if has_z:
            return "z"
        return "i"

    @staticmethod
    def _pauli_commutes_with_axis(x_mask: int, z_mask: int, axis: str, qubit: int) -> bool:
        label = MeasurementEfficientDiagonalObjectiveMixin._pauli_label(x_mask, z_mask, qubit)
        return label in {"i", axis}

    @staticmethod
    def _anticommutes_with_generator(
        x_mask: int,
        z_mask: int,
        generator: str,
        qubit: int,
    ) -> bool:
        bit = 1 << int(qubit)
        has_x = bool(x_mask & bit)
        has_z = bool(z_mask & bit)
        if generator == "x":
            return has_z
        if generator == "y":
            return has_x ^ has_z
        if generator == "z":
            return has_x
        raise ValueError(f"Unsupported Pauli generator: {generator!r}.")

    @staticmethod
    def _quarter_turn_index(angle: float) -> int | None:
        scaled = float(angle) / (np.pi / 2.0)
        rounded = int(np.rint(scaled))
        if np.isclose(float(angle), rounded * np.pi / 2.0, rtol=0.0, atol=_CLIFFORD_ANGLE_ATOL):
            return rounded
        return None

    @staticmethod
    def _conjugate_single_qubit_quarter_turn(
        x_mask: int,
        z_mask: int,
        *,
        axis: str,
        qubit: int,
        quarter_turns: int,
    ) -> tuple[int, int]:
        if quarter_turns % 2 == 0:
            return x_mask, z_mask

        bit = 1 << int(qubit)
        if axis == "x":
            if z_mask & bit:
                x_mask ^= bit
        elif axis == "y":
            x_bit = x_mask & bit
            z_bit = z_mask & bit
            if x_bit != z_bit:
                x_mask ^= bit
                z_mask ^= bit
        elif axis == "z":
            if x_mask & bit:
                z_mask ^= bit
        else:
            raise ValueError(f"Unsupported rotation axis: {axis!r}.")
        return x_mask, z_mask

    @staticmethod
    def _conjugate_h(x_mask: int, z_mask: int, qubit: int) -> tuple[int, int]:
        bit = 1 << int(qubit)
        x_bit = x_mask & bit
        z_bit = z_mask & bit
        if x_bit != z_bit:
            x_mask ^= bit
            z_mask ^= bit
        return x_mask, z_mask

    @staticmethod
    def _conjugate_cx(
        x_mask: int,
        z_mask: int,
        control: int,
        target: int,
    ) -> tuple[int, int]:
        control_bit = 1 << int(control)
        target_bit = 1 << int(target)
        control_has_x = bool(x_mask & control_bit)
        target_has_z = bool(z_mask & target_bit)
        if control_has_x:
            x_mask ^= target_bit
        if target_has_z:
            z_mask ^= control_bit
        return x_mask, z_mask

    @staticmethod
    def _conjugate_cz(
        x_mask: int,
        z_mask: int,
        first: int,
        second: int,
    ) -> tuple[int, int]:
        first_bit = 1 << int(first)
        second_bit = 1 << int(second)
        first_has_x = bool(x_mask & first_bit)
        second_has_x = bool(x_mask & second_bit)
        if first_has_x:
            z_mask ^= second_bit
        if second_has_x:
            z_mask ^= first_bit
        return x_mask, z_mask

    def _suffix_is_diagonal(self, sample: np.ndarray, position: int) -> bool:
        for raw_gate_index in sample[position + 1 :]:
            gate = self.gate_set[int(raw_gate_index)]
            if not isinstance(gate, str):
                return False
            if not self._is_diagonal_gate_name(gate.lower()):
                return False
        return True

    def _gate_qubits(self, gate_name: str, position: int) -> tuple[int, ...]:
        qubit = position % self.num_qubits
        if gate_name in {"cx", "cnot", "cz"}:
            return (qubit, (qubit + 1) % self.num_qubits)
        if gate_name.startswith("cz_o"):
            offset = int(gate_name.split("o", 1)[1])
            return (qubit, (qubit + offset) % self.num_qubits)
        return (qubit,)

    @staticmethod
    def _rotation_generator(gate_name: str) -> str | None:
        if gate_name == "rx":
            return "x"
        if gate_name == "ry":
            return "y"
        if gate_name == "rz":
            return "z"
        return None

    def _hamiltonian_depends_on_qubit(self, qubit: int) -> bool:
        return True

    def _pauli_generator_commutes_with_hamiltonian(self, pauli: str, qubit: int) -> bool:
        if pauli == "z":
            return True
        return not self._hamiltonian_depends_on_qubit(qubit)

    def _gate_commutes_with_hamiltonian(self, gate_name: str, position: int) -> bool:
        gate_name = gate_name.lower()
        if self._is_diagonal_gate_name(gate_name):
            return True
        return all(
            not self._hamiltonian_depends_on_qubit(qubit)
            for qubit in self._gate_qubits(gate_name, position)
        )

    @staticmethod
    def _pauli_commutes_with_same_qubit_gate(pauli: str, gate_name: str) -> bool:
        if gate_name in {"x", "rx"}:
            return pauli == "x"
        if gate_name in {"y", "ry"}:
            return pauli == "y"
        if gate_name in {"z", "s", "sdg", "t", "tdg", "rz"}:
            return pauli == "z"
        return False

    def _pauli_generator_commutes_with_gate(
        self,
        pauli: str,
        qubit: int,
        gate_name: str,
        position: int,
    ) -> bool:
        gate_name = gate_name.lower()
        gate_qubits = self._gate_qubits(gate_name, position)
        if qubit not in gate_qubits:
            return True
        if gate_name == "cz" or gate_name.startswith("cz_o"):
            return pauli == "z"
        if gate_name in {"cx", "cnot"}:
            control, target = gate_qubits
            return (qubit == control and pauli == "z") or (
                qubit == target and pauli == "x"
            )
        return self._pauli_commutes_with_same_qubit_gate(pauli, gate_name)

    def _suffix_commutes_with_generator(
        self,
        sample: np.ndarray,
        position: int,
        pauli: str,
        qubit: int,
    ) -> bool:
        for later_position, raw_gate_index in enumerate(
            sample[position + 1 :],
            start=position + 1,
        ):
            gate = self.gate_set[int(raw_gate_index)]
            if not isinstance(gate, str):
                return False
            if not self._pauli_generator_commutes_with_gate(
                pauli,
                qubit,
                gate,
                later_position,
            ):
                return False
        return True

    def _suffix_commutes_with_hamiltonian(self, sample: np.ndarray, position: int) -> bool:
        for later_position, raw_gate_index in enumerate(
            sample[position + 1 :],
            start=position + 1,
        ):
            gate = self.gate_set[int(raw_gate_index)]
            if not isinstance(gate, str):
                return False
            if not self._gate_commutes_with_hamiltonian(gate, later_position):
                return False
        return True

    def _has_structural_zero_gradient(
        self,
        sample: np.ndarray,
        position: int,
        gate_name: str,
        qubit: int,
    ) -> bool:
        """Return true when commutation proves the theta derivative is zero."""
        pauli = self._rotation_generator(gate_name)
        if pauli is None:
            return False
        if not self._pauli_generator_commutes_with_hamiltonian(pauli, qubit):
            return False
        return self._suffix_commutes_with_generator(
            sample,
            position,
            pauli,
            qubit,
        ) or self._suffix_commutes_with_hamiltonian(sample, position)

    @staticmethod
    def _coordinate(theta: np.ndarray, position: int, gate_index: int) -> int | tuple[int, int]:
        if theta.ndim == 1:
            return position
        return (position, gate_index)

    @staticmethod
    def _theta_value_for_gate(theta: np.ndarray, position: int, gate_index: int) -> float:
        if theta.ndim == 1:
            return float(theta[position])
        return float(theta[position, gate_index])

    @staticmethod
    def _shifted_theta(
        theta: np.ndarray,
        coordinate: int | tuple[int, int],
        shift: float,
    ) -> np.ndarray:
        shifted = theta.copy()
        shifted[coordinate] += shift
        return shifted

    @cached_property
    def _z_pauli_terms(self) -> tuple[_ZPauliTerm, ...]:
        terms = self._build_z_pauli_terms()
        return tuple(
            term
            for term in terms
            if term.z_mask != 0 and not np.isclose(term.coefficient, 0.0, rtol=0.0, atol=0.0)
        )

    def _build_z_pauli_terms(self) -> tuple[_ZPauliTerm, ...]:
        raise NotImplementedError("Subclasses must expose the Z-only Pauli decomposition.")

    def _push_pauli_through_gate(
        self,
        x_mask: int,
        z_mask: int,
        *,
        gate_name: str,
        position: int,
        gate_index: int,
        theta: np.ndarray,
    ) -> tuple[int, int] | None:
        gate_name = gate_name.lower()
        qubits = self._gate_qubits(gate_name, position)

        if gate_name in _PARAMETRIC_ROTATION_GATES:
            axis = self._rotation_generator(gate_name)
            if axis is None:
                return None
            qubit = qubits[0]
            if self._pauli_commutes_with_axis(x_mask, z_mask, axis, qubit):
                return x_mask, z_mask
            quarter_turns = self._quarter_turn_index(
                self._theta_value_for_gate(theta, position, gate_index)
            )
            if quarter_turns is None:
                return None
            return self._conjugate_single_qubit_quarter_turn(
                x_mask,
                z_mask,
                axis=axis,
                qubit=qubit,
                quarter_turns=quarter_turns,
            )

        if gate_name == "h":
            return self._conjugate_h(x_mask, z_mask, qubits[0])
        if gate_name in {"x", "y", "z"}:
            return x_mask, z_mask
        if gate_name == "s":
            return self._conjugate_single_qubit_quarter_turn(
                x_mask,
                z_mask,
                axis="z",
                qubit=qubits[0],
                quarter_turns=1,
            )
        if gate_name == "sdg":
            return self._conjugate_single_qubit_quarter_turn(
                x_mask,
                z_mask,
                axis="z",
                qubit=qubits[0],
                quarter_turns=-1,
            )
        if gate_name in {"t", "tdg"}:
            if self._pauli_commutes_with_axis(x_mask, z_mask, "z", qubits[0]):
                return x_mask, z_mask
            return None
        if gate_name in {"cx", "cnot"}:
            return self._conjugate_cx(x_mask, z_mask, qubits[0], qubits[1])
        if gate_name == "cz" or gate_name.startswith("cz_o"):
            return self._conjugate_cz(x_mask, z_mask, qubits[0], qubits[1])
        return None

    def _clifford_suffix_active_term_indices(
        self,
        sample: np.ndarray,
        position: int,
        theta: np.ndarray,
        generator: str,
        qubit: int,
    ) -> tuple[int, ...] | None:
        active: list[int] = []
        for term_index, term in enumerate(self._z_pauli_terms):
            x_mask = 0
            z_mask = int(term.z_mask)
            for later_position in range(len(sample) - 1, position, -1):
                gate_index = int(sample[later_position])
                gate = self.gate_set[gate_index]
                if not isinstance(gate, str):
                    return None
                pushed = self._push_pauli_through_gate(
                    x_mask,
                    z_mask,
                    gate_name=gate,
                    position=later_position,
                    gate_index=gate_index,
                    theta=theta,
                )
                if pushed is None:
                    return None
                x_mask, z_mask = pushed
            if self._anticommutes_with_generator(x_mask, z_mask, generator, qubit):
                active.append(term_index)
        return tuple(active)

    def _gradient_jobs(
        self,
        samples: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[list[_GradientJob], int]:
        jobs: list[_GradientJob] = []
        zero_gradient_jobs = 0
        for sample in samples:
            for position, raw_gate_index in enumerate(sample):
                gate_index = int(raw_gate_index)
                gate = self.gate_set[gate_index]
                if not isinstance(gate, str):
                    continue

                gate_name = gate.lower()
                if not self._is_parametric_gate(gate_name):
                    continue

                coordinate = self._coordinate(theta, position, gate_index)
                qubit = position % self.num_qubits
                suffix_is_diagonal = self._suffix_is_diagonal(sample, position)

                generator = self._rotation_generator(gate_name)
                if self._has_structural_zero_gradient(sample, position, gate_name, qubit):
                    zero_gradient_jobs += 1
                    continue

                if suffix_is_diagonal and gate_name in _ME_GATES:
                    kind: Literal["measurement_efficient", "parameter_shift"] = "measurement_efficient"
                    estimator: Literal["diagonal_bit_flip", "clifford_pauli", "parameter_shift"] = "diagonal_bit_flip"
                    active_term_indices: tuple[int, ...] = ()
                elif generator is not None:
                    active_term_indices = self._clifford_suffix_active_term_indices(
                        sample,
                        position,
                        theta,
                        generator,
                        qubit,
                    )
                    if active_term_indices is None:
                        kind = "parameter_shift"
                        estimator = "parameter_shift"
                        active_term_indices = ()
                    elif not active_term_indices:
                        zero_gradient_jobs += 1
                        continue
                    else:
                        kind = "measurement_efficient"
                        estimator = "clifford_pauli"
                else:
                    kind = "parameter_shift"
                    estimator = "parameter_shift"
                    active_term_indices = ()

                jobs.append(
                    _GradientJob(
                        kind=kind,
                        estimator=estimator,
                        sample=np.array(sample, dtype=np.intp, copy=True),
                        coordinate=coordinate,
                        position=position,
                        gate_index=gate_index,
                        qubit=qubit,
                        active_term_indices=active_term_indices,
                    )
                )
        return jobs, zero_gradient_jobs

    @staticmethod
    def _select_jobs(
        jobs: list[_GradientJob],
        S_theta: int,
        shot_budget_remaining: int | None,
    ) -> tuple[list[_SelectedJob], int, bool]:
        if shot_budget_remaining is None:
            shot_budget_remaining = sum(
                S_theta if job.kind == "measurement_efficient" else 2 * S_theta
                for job in jobs
            )
        if shot_budget_remaining < 0:
            raise ValueError("shot_budget_remaining must be non-negative or None.")

        selected: list[_SelectedJob] = []
        shots_spent = 0
        partial_budget_limited = False
        remaining = int(shot_budget_remaining)

        for job in jobs:
            if job.kind == "measurement_efficient":
                if remaining >= S_theta:
                    shots = S_theta
                elif remaining >= 1:
                    shots = remaining
                    partial_budget_limited = True
                else:
                    partial_budget_limited = True
                    break
                cost = shots
            else:
                if remaining >= 2 * S_theta:
                    shots = S_theta
                elif remaining >= 2:
                    shots = remaining // 2
                    partial_budget_limited = True
                else:
                    partial_budget_limited = True
                    break
                cost = 2 * shots

            selected.append(_SelectedJob(job=job, shots_per_shift_circuit=int(shots)))
            shots_spent += int(cost)
            remaining -= int(cost)
            if partial_budget_limited:
                break

        if len(selected) < len(jobs):
            partial_budget_limited = True
        return selected, shots_spent, partial_budget_limited

    def _bit_flip_gradient_estimate(self, measurements: np.ndarray, qubit: int) -> float:
        flipped = np.array(measurements, dtype=bool, copy=True)
        flipped[:, qubit] = ~flipped[:, qubit]
        original_cost = self.sample_costs(measurements).astype(np.float64, copy=False)
        flipped_cost = self.sample_costs(flipped).astype(np.float64, copy=False)
        return float(np.mean(0.5 * (original_cost - flipped_cost), dtype=np.float64))

    def _pauli_subset_gradient_estimate(
        self,
        measurements: np.ndarray,
        active_term_indices: tuple[int, ...],
    ) -> float:
        if not active_term_indices:
            return 0.0
        values = np.zeros(measurements.shape[0], dtype=np.float64)
        terms = self._z_pauli_terms
        for term_index in active_term_indices:
            term = terms[term_index]
            values += term.coefficient * self._term_parity(measurements, term.z_mask)
        return float(np.mean(values, dtype=np.float64))

    def measurement_efficient_parameter_gradient(
        self,
        samples,
        S_theta: int,
        *,
        shot_budget_remaining: int | None = None,
    ) -> MeasurementEfficientGradientResult:
        if not self.is_parametric:
            return MeasurementEfficientGradientResult(
                gradient=None,
                shots_spent=0,
                requested_parametric_terms=0,
                requested_jobs=0,
                requested_measurement_efficient_jobs=0,
                requested_parameter_shift_jobs=0,
                requested_diagonal_suffix_jobs=0,
                requested_clifford_suffix_jobs=0,
                zero_gradient_jobs=0,
                executed_parametric_terms=0,
                executed_jobs=0,
                executed_measurement_efficient_jobs=0,
                executed_parameter_shift_jobs=0,
                executed_diagonal_suffix_jobs=0,
                executed_clifford_suffix_jobs=0,
                conventional_full_shot_cost=0,
                measurement_efficient_full_shot_cost=0,
                zero_gradient_shots_saved_for_full_step=0,
                one_shift_shots_saved_for_full_step=0,
                diagonal_suffix_shots_saved_for_full_step=0,
                clifford_suffix_shots_saved_for_full_step=0,
                partial_budget_limited=False,
            )
        if not isinstance(S_theta, int) or isinstance(S_theta, bool) or S_theta <= 0:
            raise ValueError("S_theta must be a positive integer.")

        samples = self._normalize_samples(samples)
        theta = self._validate_theta(samples.shape[1])
        if theta is None:
            raise ValueError("theta is required for measurement_efficient_parameter_gradient.")

        jobs, zero_gradient_jobs = self._gradient_jobs(samples, theta)
        requested_me_jobs = sum(job.kind == "measurement_efficient" for job in jobs)
        requested_ps_jobs = len(jobs) - requested_me_jobs
        requested_diagonal_suffix_jobs = sum(
            job.estimator == "diagonal_bit_flip" for job in jobs
        )
        requested_clifford_suffix_jobs = sum(
            job.estimator == "clifford_pauli" for job in jobs
        )
        requested_parametric_terms = len(jobs) + zero_gradient_jobs
        conventional_full_shot_cost = 2 * S_theta * requested_parametric_terms
        measurement_efficient_full_shot_cost = (
            S_theta * requested_me_jobs + 2 * S_theta * requested_ps_jobs
        )
        zero_gradient_shots_saved_for_full_step = 2 * S_theta * zero_gradient_jobs
        one_shift_shots_saved_for_full_step = S_theta * requested_me_jobs
        diagonal_suffix_shots_saved_for_full_step = S_theta * requested_diagonal_suffix_jobs
        clifford_suffix_shots_saved_for_full_step = S_theta * requested_clifford_suffix_jobs

        gradient = np.zeros_like(theta, dtype=np.float64)
        selected_jobs, shots_spent, partial_budget_limited = self._select_jobs(
            jobs=jobs,
            S_theta=S_theta,
            shot_budget_remaining=shot_budget_remaining,
        )
        if not selected_jobs:
            return MeasurementEfficientGradientResult(
                gradient=gradient,
                shots_spent=0,
                requested_parametric_terms=requested_parametric_terms,
                requested_jobs=len(jobs),
                requested_measurement_efficient_jobs=requested_me_jobs,
                requested_parameter_shift_jobs=requested_ps_jobs,
                requested_diagonal_suffix_jobs=requested_diagonal_suffix_jobs,
                requested_clifford_suffix_jobs=requested_clifford_suffix_jobs,
                zero_gradient_jobs=zero_gradient_jobs,
                executed_parametric_terms=zero_gradient_jobs,
                executed_jobs=0,
                executed_measurement_efficient_jobs=0,
                executed_parameter_shift_jobs=0,
                executed_diagonal_suffix_jobs=0,
                executed_clifford_suffix_jobs=0,
                conventional_full_shot_cost=conventional_full_shot_cost,
                measurement_efficient_full_shot_cost=measurement_efficient_full_shot_cost,
                zero_gradient_shots_saved_for_full_step=zero_gradient_shots_saved_for_full_step,
                one_shift_shots_saved_for_full_step=one_shift_shots_saved_for_full_step,
                diagonal_suffix_shots_saved_for_full_step=diagonal_suffix_shots_saved_for_full_step,
                clifford_suffix_shots_saved_for_full_step=clifford_suffix_shots_saved_for_full_step,
                partial_budget_limited=partial_budget_limited,
            )

        circuits: list[object] = []
        shot_plan: list[int] = []
        measurement_specs: list[tuple[str, _GradientJob]] = []
        for selected in selected_jobs:
            job = selected.job
            shots = selected.shots_per_shift_circuit
            theta_plus = self._shifted_theta(theta, job.coordinate, np.pi / 2)
            circuits.append(self.build_circuit(job.sample, theta=theta_plus))
            shot_plan.append(shots)
            measurement_specs.append(
                (
                    "measurement_efficient"
                    if job.kind == "measurement_efficient"
                    else "plus",
                    job,
                )
            )

            if job.kind == "parameter_shift":
                theta_minus = self._shifted_theta(theta, job.coordinate, -np.pi / 2)
                circuits.append(self.build_circuit(job.sample, theta=theta_minus))
                shot_plan.append(shots)
                measurement_specs.append(("minus", job))

        measurements = self.simulator.simulate_and_sample_batch(
            circuits=circuits,
            shots=shot_plan,
            seed=self._next_seed(),
            precision=self.precision,
        )
        if not isinstance(measurements, list):
            measurements = [measurements[index] for index in range(measurements.shape[0])]

        pending_parameter_shift: dict[int | tuple[int, int], float] = {}
        for measurement, (kind, job) in zip(measurements, measurement_specs):
            if kind == "measurement_efficient":
                if job.estimator == "diagonal_bit_flip":
                    value = self._bit_flip_gradient_estimate(measurement, job.qubit)
                else:
                    value = self._pauli_subset_gradient_estimate(
                        measurement,
                        job.active_term_indices,
                    )
                gradient[job.coordinate] += value
            elif kind == "plus":
                pending_parameter_shift[job.coordinate] = self._proposal_objective(measurement)
            else:
                plus_value = pending_parameter_shift.pop(job.coordinate)
                minus_value = self._proposal_objective(measurement)
                gradient[job.coordinate] += 0.5 * (plus_value - minus_value)

        executed_me_jobs = sum(
            selected.job.kind == "measurement_efficient" for selected in selected_jobs
        )
        executed_ps_jobs = len(selected_jobs) - executed_me_jobs
        executed_diagonal_suffix_jobs = sum(
            selected.job.estimator == "diagonal_bit_flip" for selected in selected_jobs
        )
        executed_clifford_suffix_jobs = sum(
            selected.job.estimator == "clifford_pauli" for selected in selected_jobs
        )
        return MeasurementEfficientGradientResult(
            gradient=gradient / samples.shape[0],
            shots_spent=shots_spent,
            requested_parametric_terms=requested_parametric_terms,
            requested_jobs=len(jobs),
            requested_measurement_efficient_jobs=requested_me_jobs,
            requested_parameter_shift_jobs=requested_ps_jobs,
            requested_diagonal_suffix_jobs=requested_diagonal_suffix_jobs,
            requested_clifford_suffix_jobs=requested_clifford_suffix_jobs,
            zero_gradient_jobs=zero_gradient_jobs,
            executed_parametric_terms=len(selected_jobs) + zero_gradient_jobs,
            executed_jobs=len(selected_jobs),
            executed_measurement_efficient_jobs=executed_me_jobs,
            executed_parameter_shift_jobs=executed_ps_jobs,
            executed_diagonal_suffix_jobs=executed_diagonal_suffix_jobs,
            executed_clifford_suffix_jobs=executed_clifford_suffix_jobs,
            conventional_full_shot_cost=conventional_full_shot_cost,
            measurement_efficient_full_shot_cost=measurement_efficient_full_shot_cost,
            zero_gradient_shots_saved_for_full_step=zero_gradient_shots_saved_for_full_step,
            one_shift_shots_saved_for_full_step=one_shift_shots_saved_for_full_step,
            diagonal_suffix_shots_saved_for_full_step=diagonal_suffix_shots_saved_for_full_step,
            clifford_suffix_shots_saved_for_full_step=clifford_suffix_shots_saved_for_full_step,
            partial_budget_limited=partial_budget_limited,
        )


class MeasurementEfficientThreeSATObjectiveFunction(
    MeasurementEfficientDiagonalObjectiveMixin,
    ThreeSATObjectiveFunction,
):
    """3-SAT objective with the measurement-efficient theta-gradient estimator."""

    def _hamiltonian_depends_on_qubit(self, qubit: int) -> bool:
        return bool(np.any(self.compiled_clauses.clause_var_idx == int(qubit)))

    def _build_z_pauli_terms(self) -> tuple[_ZPauliTerm, ...]:
        coefficients: dict[int, float] = {}
        for variables, negated in zip(
            self.compiled_clauses.clause_var_idx,
            self.compiled_clauses.clause_is_negated,
        ):
            clause_terms: dict[int, float] = {0: 1.0}
            for variable, is_negated in zip(variables, negated):
                sign = -1.0 if bool(is_negated) else 1.0
                bit = self._bit(int(variable))
                next_terms: dict[int, float] = {}
                for mask, coefficient in clause_terms.items():
                    next_terms[mask] = next_terms.get(mask, 0.0) + 0.5 * coefficient
                    toggled = mask ^ bit
                    next_terms[toggled] = (
                        next_terms.get(toggled, 0.0)
                        + 0.5 * sign * coefficient
                    )
                clause_terms = next_terms
            for mask, coefficient in clause_terms.items():
                coefficients[mask] = coefficients.get(mask, 0.0) + coefficient
        return tuple(
            _ZPauliTerm(coefficient=float(coefficient), z_mask=int(mask))
            for mask, coefficient in sorted(coefficients.items())
        )


class MeasurementEfficientMaxCutObjectiveFunction(
    MeasurementEfficientDiagonalObjectiveMixin,
    MaxCutObjectiveFunction,
):
    """MaxCut objective with the measurement-efficient theta-gradient estimator."""

    def _hamiltonian_depends_on_qubit(self, qubit: int) -> bool:
        qubit = int(qubit)
        return bool(
            np.any(
                (self.compiled_graph.edge_u == qubit)
                | (self.compiled_graph.edge_v == qubit)
            )
        )

    def _build_z_pauli_terms(self) -> tuple[_ZPauliTerm, ...]:
        coefficients: dict[int, float] = {}
        for u, v in zip(self.compiled_graph.edge_u, self.compiled_graph.edge_v):
            mask = self._bit(int(u)) ^ self._bit(int(v))
            coefficients[0] = coefficients.get(0, 0.0) + 0.5
            coefficients[mask] = coefficients.get(mask, 0.0) + 0.5
        return tuple(
            _ZPauliTerm(coefficient=float(coefficient), z_mask=int(mask))
            for mask, coefficient in sorted(coefficients.items())
        )


class MeasurementEfficientDQAS(DQAS):
    """DQAS loop that spends shots according to the measurement-efficient estimator."""

    def _run_parameter_shift_step(self, sample_batch: np.ndarray, step_index: int) -> dict[str, object]:
        estimator = getattr(
            self.objective_function,
            "measurement_efficient_parameter_gradient",
            None,
        )
        if not callable(estimator):
            return super()._run_parameter_shift_step(sample_batch, step_index)

        event: dict[str, object] = {
            "kind": "measurement_efficient_theta_update",
            "step_index": step_index,
            "budget_before": self.shot_budget.used,
            "budget_after": self.shot_budget.used,
            "shots_spent": 0,
            "reason": None,
            "requested_shifted_pairs": 0,
            "executed_shifted_pairs": 0,
            "shots_per_shift_circuit": self.S_theta,
            "theta_gradient_norm": None,
            "requested_parametric_terms": 0,
            "executed_parametric_terms": 0,
            "requested_measured_gradient_jobs": 0,
            "executed_gradient_jobs": 0,
            "requested_measurement_efficient_jobs": 0,
            "requested_parameter_shift_jobs": 0,
            "requested_diagonal_suffix_jobs": 0,
            "requested_clifford_suffix_jobs": 0,
            "zero_gradient_jobs": 0,
            "executed_measurement_efficient_jobs": 0,
            "executed_parameter_shift_jobs": 0,
            "executed_diagonal_suffix_jobs": 0,
            "executed_clifford_suffix_jobs": 0,
            "conventional_full_shot_cost": 0,
            "measurement_efficient_full_shot_cost": 0,
            "zero_gradient_shots_saved_for_full_step": 0,
            "one_shift_shots_saved_for_full_step": 0,
            "diagonal_suffix_shots_saved_for_full_step": 0,
            "clifford_suffix_shots_saved_for_full_step": 0,
            "estimated_shots_saved_for_full_step": 0,
        }

        if not self.objective_function.is_parametric or self.shot_budget.depleted:
            event["reason"] = (
                "objective_not_parametric"
                if not self.objective_function.is_parametric
                else "budget_already_depleted"
            )
            return event

        result = estimator(
            sample_batch,
            self.S_theta,
            shot_budget_remaining=self.shot_budget.remaining,
        )
        self._spend_after(result.shots_spent)
        if result.gradient is not None and getattr(self.objective_function, "theta", None) is not None:
            self.objective_function.theta -= self.theta_lr * result.gradient

        if result.requested_jobs == 0 and result.zero_gradient_jobs == 0:
            reason = "no_parametric_gates_sampled"
        elif result.partial_budget_limited:
            reason = "partial_measurement_efficient_theta_step_limited_by_budget"
        else:
            reason = "full_measurement_efficient_theta_step"

        event.update(
            {
                "budget_after": self.shot_budget.used,
                "shots_spent": result.shots_spent,
                "reason": reason,
                "requested_shifted_pairs": result.requested_parametric_terms,
                "executed_shifted_pairs": result.executed_parametric_terms,
                "theta_gradient_norm": self._gradient_norm(result.gradient),
                "requested_parametric_terms": result.requested_parametric_terms,
                "executed_parametric_terms": result.executed_parametric_terms,
                "requested_measured_gradient_jobs": result.requested_jobs,
                "executed_gradient_jobs": result.executed_jobs,
                "requested_measurement_efficient_jobs": result.requested_measurement_efficient_jobs,
                "requested_parameter_shift_jobs": result.requested_parameter_shift_jobs,
                "requested_diagonal_suffix_jobs": result.requested_diagonal_suffix_jobs,
                "requested_clifford_suffix_jobs": result.requested_clifford_suffix_jobs,
                "zero_gradient_jobs": result.zero_gradient_jobs,
                "executed_measurement_efficient_jobs": result.executed_measurement_efficient_jobs,
                "executed_parameter_shift_jobs": result.executed_parameter_shift_jobs,
                "executed_diagonal_suffix_jobs": result.executed_diagonal_suffix_jobs,
                "executed_clifford_suffix_jobs": result.executed_clifford_suffix_jobs,
                "conventional_full_shot_cost": result.conventional_full_shot_cost,
                "measurement_efficient_full_shot_cost": result.measurement_efficient_full_shot_cost,
                "zero_gradient_shots_saved_for_full_step": (
                    result.zero_gradient_shots_saved_for_full_step
                ),
                "one_shift_shots_saved_for_full_step": (
                    result.one_shift_shots_saved_for_full_step
                ),
                "diagonal_suffix_shots_saved_for_full_step": (
                    result.diagonal_suffix_shots_saved_for_full_step
                ),
                "clifford_suffix_shots_saved_for_full_step": (
                    result.clifford_suffix_shots_saved_for_full_step
                ),
                "estimated_shots_saved_for_full_step": (
                    result.conventional_full_shot_cost - result.measurement_efficient_full_shot_cost
                ),
            }
        )
        return event


MEDQAS = MeasurementEfficientDQAS
