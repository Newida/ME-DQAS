"""GPU simulator API contract (canonical for agents).

Public entrypoints:
- `simulate_and_sample_batch(circuits, shots, seed, precision="single")`
- `simulate_and_sample(circuit, shots, seed, precision="single")`

Input requirements:
- `circuits` must be a non-empty sequence of unmeasured Qiskit
  `QuantumCircuit` objects.
- Do not pass a single `QuantumCircuit` object to the batch API.
- All circuits in one batch must have the same qubit count and at least one
  qubit.
- `shots` accepts either:
  - a single positive integer, or
  - a sequence of positive integers with one entry per circuit.
- `seed` is required and must be an integer.
- `precision` must be `"single"` or `"double"`.

Return contract:
- Scalar `shots` input returns `np.ndarray[bool]` with shape `(M, shots, n)`.
- Sequence `shots` input returns `list[np.ndarray[bool]]` where entry `m` has
  shape `(shots[m], n)`.
- Single-circuit wrapper returns `np.ndarray[bool]` with shape `(shots, n)`.

Execution notes:
- Variable-shot batches are grouped by unique shot counts to reduce backend
  launches.
- Output qubit axis maps directly to logical qubit indices.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import platform

_VALID_PRECISIONS = {"single", "double"}


def _import_quantum_circuit() -> type:
    """
    Import Qiskit's circuit class without requiring Aer. The statevector
    fallback uses this path when qiskit-aer is not installed.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:  # pragma: no cover - qiskit missing
        raise ImportError("qiskit is required. Install with: pip install qiskit qiskit-aer") from exc
    return QuantumCircuit


def _import_qiskit_aer() -> Tuple[type, type]:
    """
    Import qiskit and AerSimulator in a way that supports either the CPU Aer
    package (qiskit-aer) or the GPU-enabled aer package (qiskit-aer-gpu).
    """
    QuantumCircuit = _import_quantum_circuit()
    # Prefer qiskit_aer module (some installs expose this); otherwise use qiskit.providers.aer
    try:
        # qiskit-aer (or qiskit-aer-gpu) may expose qiskit_aer
        from qiskit_aer import AerSimulator  # type: ignore
    except Exception:
        try:
            from qiskit.providers.aer import AerSimulator
        except Exception as exc:  # pragma: no cover - Aer missing
            raise ImportError(
                "qiskit-aer (CPU) or qiskit-aer-gpu (GPU) is required. Install with: pip install qiskit-aer"
            ) from exc

    return QuantumCircuit, AerSimulator


def _import_statevector() -> type:
    try:
        from qiskit.quantum_info import Statevector
    except Exception as exc:  # pragma: no cover - qiskit missing/misconfigured
        raise ImportError("qiskit.quantum_info.Statevector is required for the local simulator fallback.") from exc
    return Statevector


def _create_gpu_backend(precision: str, preferred_device: str = "GPU"):
    """
    Create an Aer statevector backend.

    On macOS prefer CPU (no CUDA). On other platforms attempt GPU first then fall back to CPU/default.
    """
    _, AerSimulator = _import_qiskit_aer()

    is_macos = platform.system() == "Darwin"

    # If running on macOS, skip attempting GPU to avoid "GPU not supported" errors.
    if is_macos:
        try:
            return AerSimulator(method="statevector", device="CPU", precision=precision)
        except TypeError:
            # Some Aer builds don't accept 'device' kwarg
            try:
                return AerSimulator(method="statevector", precision=precision)
            except Exception as exc:  # pragma: no cover - platform/install dependent
                raise RuntimeError(
                    "Failed to create Aer statevector backend on macOS. Ensure qiskit-aer is installed."
                ) from exc

    normalized_device = preferred_device.upper()
    if normalized_device not in {"GPU", "CPU"}:
        raise ValueError("preferred_device must be 'GPU' or 'CPU'.")

    try:
        return AerSimulator(method="statevector", device=normalized_device, precision=precision)
    except Exception:
        if normalized_device == "GPU":
            try:
                return AerSimulator(method="statevector", device="CPU", precision=precision)
            except TypeError:
                try:
                    return AerSimulator(method="statevector", precision=precision)
                except Exception as exc:  # pragma: no cover - platform/install dependent
                    raise RuntimeError(
                        "Failed to create an Aer statevector backend. Ensure qiskit-aer (or qiskit-aer-gpu) is installed."
                    ) from exc
        try:
            return AerSimulator(method="statevector", precision=precision)
        except Exception as exc:  # pragma: no cover - platform/install dependent
            raise RuntimeError(
                "Failed to create an Aer statevector backend. Ensure qiskit-aer (or qiskit-aer-gpu) is installed."
            ) from exc


def _normalize_shots(
    shots: int | Sequence[int],
    num_circuits: int,
) -> Tuple[List[int], bool]:
    # Returns a positive sequence of integers, which are the shots per circuit
    # Broadcasting to equal shots across all circuits if a single integer is provided.
    # The boolean flag indicates whether the input was a scalar (True) or a sequence (False)
    if isinstance(shots, int) and not isinstance(shots, bool):
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")
        return [shots] * num_circuits, True

    if isinstance(shots, (str, bytes)):
        raise TypeError("shots must be a positive integer or a sequence of positive integers.")

    try:
        shots_seq = list(shots)
    except TypeError as exc:
        raise TypeError("shots must be a positive integer or a sequence of positive integers.") from exc

    if len(shots_seq) != num_circuits:
        raise ValueError("shots sequence length must match number of circuits.")

    normalized: List[int] = []
    for index, shot_count in enumerate(shots_seq):
        if not isinstance(shot_count, int) or isinstance(shot_count, bool) or shot_count <= 0:
            raise ValueError(f"shots[{index}] must be a positive integer.")
        normalized.append(int(shot_count))

    return normalized, False


def _spawn_child_seeds(seed: int, num_children: int) -> List[int]:
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(seed).spawn(num_children)
    ]


def _validate_inputs(
    circuits: Sequence[object],
    shots: int | Sequence[int],
    seed: int,
    precision: str,
) -> Tuple[List[object], int, List[int], bool]:
    if precision not in _VALID_PRECISIONS:
        raise ValueError("precision must be 'single' or 'double'.")
    if isinstance(shots, bool):
        raise ValueError("shots must be a positive integer.")
    if isinstance(shots, int) and shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    QuantumCircuit = _import_quantum_circuit()

    if isinstance(circuits, QuantumCircuit):
        raise TypeError("circuits must be a sequence of QuantumCircuit objects, not a single circuit.")

    circuits_list = list(circuits)
    if not circuits_list:
        raise ValueError("circuits must be a non-empty sequence of QuantumCircuit objects.")

    first = circuits_list[0]
    if not isinstance(first, QuantumCircuit):
        raise TypeError("All entries in circuits must be QuantumCircuit objects.")
    num_qubits = int(first.num_qubits)
    if num_qubits < 1:
        raise ValueError("Circuits must have at least one qubit.")

    for index, circuit in enumerate(circuits_list):
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                f"All entries in circuits must be QuantumCircuit objects; "
                f"found {type(circuit).__name__} at index {index}."
            )
        if int(circuit.num_qubits) != num_qubits:
            raise ValueError(
                f"All circuits must have the same number of qubits. "
                f"Expected {num_qubits}, got {circuit.num_qubits} at index {index}."
            )
        if circuit.count_ops().get("measure", 0) > 0:
            raise ValueError(
                f"Input circuit at index {index} already contains measurement operations. "
                "Provide unmeasured circuits; the simulator appends measurements internally."
            )

    shots_per_circuit, scalar_shots_input = _normalize_shots(shots, len(circuits_list))
    return circuits_list, num_qubits, shots_per_circuit, scalar_shots_input


def _build_measured_batch(circuits: Sequence[object]) -> List[object]:
    measured_batch: List[object] = []
    for circuit in circuits:
        # TODO: If circuits are guaranteed single-use, consider in-place measurement to avoid per-batch copy overhead.
        measured = circuit.copy()
        measured.measure_all()
        measured_batch.append(measured)
    return measured_batch


def _memory_to_bool_array(memory: Sequence[str], num_qubits: int) -> np.ndarray:
    shots = len(memory)
    out = np.empty((shots, num_qubits), dtype=bool)
    for shot_index, raw_bits in enumerate(memory):
        bits = str(raw_bits).replace(" ", "")
        if len(bits) != num_qubits:
            raise RuntimeError(
                f"Expected bitstrings of length {num_qubits}, got length {len(bits)}: '{raw_bits}'"
            )
        if any(ch not in {"0", "1"} for ch in bits):
            raise RuntimeError(f"Expected binary bitstrings, got '{raw_bits}'")
        # Qiskit memory strings are MSB-left; reverse to make axis q map to logical qubit q.
        out[shot_index] = np.fromiter((ch == "1" for ch in bits[::-1]), dtype=bool, count=num_qubits)
    return out


def _run_and_collect_memory(
    backend: object,
    measured_batch: Sequence[object],
    shots: int,
    seed: int,
    num_qubits: int,
) -> List[np.ndarray]:
    try:
        result = backend.run(
            measured_batch,
            shots=shots,
            memory=True,
            seed_simulator=seed,
        ).result()
    except Exception as exc:
        raise RuntimeError(
            "GPU statevector simulation failed. Ensure a CUDA-capable GPU is available "
            "and qiskit-aer-gpu is installed with GPU support enabled."
        ) from exc

    sampled: List[np.ndarray] = []
    # Extract and convert memory bitstrings to boolean arrays for each circuit in the batch.
    # Since Aer returns strings, this introduces a GPU-to-CPU conversion boundary.
    for circuit_index in range(len(measured_batch)):
        memory = result.get_memory(circuit_index)
        sampled.append(_memory_to_bool_array(memory=memory, num_qubits=num_qubits))
    return sampled


def _sample_statevector_circuit(
    circuit: object,
    shots: int,
    seed: int,
    num_qubits: int,
) -> np.ndarray:
    if num_qubits > 63:
        raise ValueError("Statevector fallback supports at most 63 qubits.")

    Statevector = _import_statevector()
    state = Statevector.from_instruction(circuit)
    probabilities = np.asarray(state.probabilities(), dtype=np.float64)
    total_probability = float(np.sum(probabilities))
    if not np.isclose(total_probability, 1.0):
        probabilities = probabilities / total_probability

    rng = np.random.default_rng(seed)
    basis_states = rng.choice(probabilities.size, size=shots, p=probabilities).astype(np.uint64, copy=False)
    qubit_offsets = np.arange(num_qubits, dtype=np.uint64)
    return ((basis_states[:, None] >> qubit_offsets[None, :]) & 1).astype(bool, copy=False)


def _simulate_and_sample_batch_statevector(
    circuits: Sequence[object],
    shots_per_circuit: Sequence[int],
    scalar_shots_input: bool,
    seed: int,
    num_qubits: int,
) -> np.ndarray | List[np.ndarray]:
    child_seeds = _spawn_child_seeds(seed=seed, num_children=len(circuits))
    sampled = [
        _sample_statevector_circuit(
            circuit=circuit,
            shots=shot_count,
            seed=child_seed,
            num_qubits=num_qubits,
        )
        for circuit, shot_count, child_seed in zip(circuits, shots_per_circuit, child_seeds)
    ]
    if scalar_shots_input:
        return np.stack(sampled, axis=0)
    return sampled


def _is_cuda_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "no cuda device available",
            "cuda backend not available",
            "cuda-capable gpu",
            "not supported on this system",
        )
    )


def simulate_and_sample_batch(
    circuits: Sequence[object],
    shots: int | Sequence[int],
    seed: int,
    precision: str = "single",
) -> np.ndarray | List[np.ndarray]:
    """
    Simulate a batch of circuits with GPU statevector backend and draw measurement samples.

    Args:
        circuits: Non-empty sequence of unmeasured `QuantumCircuit` objects
            with equal qubit count. Pass a sequence, not a single circuit.
        shots: Either a single positive integer for all circuits or a sequence of
            per-circuit positive integers with length `M`.
        seed: Required integer seed used by simulator sampling.
        precision: Aer statevector precision, either "single" or "double".

    Returns:
        - If `shots` is an integer: boolean tensor of shape `(M, shots, n)`.
        - If `shots` is a sequence: list of length `M`, each entry a boolean
          matrix with shape `(shots[m], n)`.
        In both modes, qubit axis `[..., q]` corresponds to logical qubit index `q`.
    """
    circuits_list, num_qubits, shots_per_circuit, scalar_shots_input = _validate_inputs(
        circuits,
        shots=shots,
        seed=seed,
        precision=precision,
    )
    measured_batch = _build_measured_batch(circuits_list)
    def _execute_with_backend(backend: object) -> np.ndarray | List[np.ndarray]:
        if scalar_shots_input:
            # If all circuits share the same shot count, we can run them as a single batch for performance benefits.
            sampled = _run_and_collect_memory(
                backend=backend,
                measured_batch=measured_batch,
                shots=shots_per_circuit[0],
                seed=seed,
                num_qubits=num_qubits,
            )
            return np.stack(sampled, axis=0)

        # For per-circuit shots, batch circuits that share shot counts to reduce backend launches.
        grouped_indices_by_shots: dict[int, List[int]] = {}
        for circuit_index, shot_count in enumerate(shots_per_circuit):
            grouped_indices_by_shots.setdefault(shot_count, []).append(circuit_index)

        group_seeds = _spawn_child_seeds(seed=seed, num_children=len(grouped_indices_by_shots))
        sampled_variable: List[np.ndarray | None] = [None] * len(measured_batch)
        for group_seed, (shot_count, grouped_indices) in zip(group_seeds, grouped_indices_by_shots.items()):
            grouped_circuits = [measured_batch[index] for index in grouped_indices]
            grouped_samples = _run_and_collect_memory(
                backend=backend,
                measured_batch=grouped_circuits,
                shots=shot_count,
                seed=group_seed,
                num_qubits=num_qubits,
            )
            for group_offset, original_index in enumerate(grouped_indices):
                sampled_variable[original_index] = grouped_samples[group_offset]

        finalized_samples: List[np.ndarray] = []
        for sample in sampled_variable:
            if sample is None:  # pragma: no cover - defensive guard for internal consistency
                raise RuntimeError("Internal error: incomplete grouped sampling results.")
            finalized_samples.append(sample)
        return finalized_samples

    try:
        backend = _create_gpu_backend(precision=precision, preferred_device="GPU")
    except ImportError:
        return _simulate_and_sample_batch_statevector(
            circuits=circuits_list,
            shots_per_circuit=shots_per_circuit,
            scalar_shots_input=scalar_shots_input,
            seed=seed,
            num_qubits=num_qubits,
        )
    try:
        return _execute_with_backend(backend)
    except RuntimeError as exc:
        if exc.__cause__ is None or not _is_cuda_unavailable_error(exc.__cause__):
            raise
        try:
            cpu_backend = _create_gpu_backend(precision=precision, preferred_device="CPU")
        except ImportError:
            return _simulate_and_sample_batch_statevector(
                circuits=circuits_list,
                shots_per_circuit=shots_per_circuit,
                scalar_shots_input=scalar_shots_input,
                seed=seed,
                num_qubits=num_qubits,
            )
        return _execute_with_backend(cpu_backend)


def simulate_and_sample(
    circuit: object,
    shots: int,
    seed: int,
    precision: str = "single",
) -> np.ndarray:
    """
    Convenience wrapper for sampling a single circuit.

    This wraps the circuit into a singleton batch and returns a boolean matrix
    of shape `(shots, n)`.
    """
    batch_samples = simulate_and_sample_batch(
        circuits=[circuit],
        shots=shots,
        seed=seed,
        precision=precision,
    )
    return batch_samples[0] if isinstance(batch_samples, list) else batch_samples[0]
