from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .objective_function import ObjectiveFunction
from .probability_distribution import ProbabilityDistribution


DEFAULT_SHOT_BUDGET = 10_000_000


@dataclass
class ShotBudget:
    total: int = DEFAULT_SHOT_BUDGET
    used: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.total, int) or isinstance(self.total, bool) or self.total <= 0:
            raise ValueError("total shot budget must be a positive integer.")
        if not isinstance(self.used, int) or isinstance(self.used, bool) or self.used < 0:
            raise ValueError("used shots must be a non-negative integer.")
        if self.used > self.total:
            raise ValueError("used shots cannot exceed the total shot budget.")

    @property
    def remaining(self) -> int:
        return self.total - self.used

    @property
    def depleted(self) -> bool:
        return self.remaining <= 0

    def spend(self, shots: int) -> None:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 0:
            raise ValueError("shots must be a non-negative integer.")
        if shots > self.remaining:
            raise RuntimeError("shot budget would be exceeded.")
        self.used += shots


class MovingAverageBaseline:
    def __init__(self, momentum: float = 0.9, initial_value: float | None = None):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the interval [0, 1).")
        self.momentum = float(momentum)
        self.value = None if initial_value is None else float(initial_value)

    def update(self, values) -> None:
        batch_mean = float(np.mean(np.asarray(values, dtype=np.float64)))
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * batch_mean

    def __float__(self) -> float:
        if self.value is None:
            raise TypeError("baseline has not been initialized yet.")
        return self.value


Baseline = MovingAverageBaseline


class DQAS:
    def __init__(
        self,
        f: ObjectiveFunction,
        C_QPU: Callable[[], int] | None = None,
        B_QPU: int = DEFAULT_SHOT_BUDGET,
        prob_dis: ProbabilityDistribution | None = None,
        baseline: MovingAverageBaseline | None = None,
        p_lr: float = 0.1,
        theta_lr: float = 0.1,
        *,
        shot_budget: int | ShotBudget | None = None,
        batch_size: int = 500,
        parameter_steps: int = 2,
        S_f: int = 1_000,
        S_theta: int = 1_000,
    ):
        if prob_dis is None:
            raise ValueError("prob_dis must be provided.")
        self.objective_function = f
        self.probability_distribution = prob_dis
        self.baseline = baseline if baseline is not None else MovingAverageBaseline()

        self.p_lr = float(p_lr)
        self.theta_lr = float(theta_lr)
        self.batch_size = self._positive_int(batch_size, "batch_size")
        self.parameter_steps = self._non_negative_int(parameter_steps, "parameter_steps")
        self.S_f = self._positive_int(S_f, "S_f")
        self.S_theta = self._positive_int(S_theta, "S_theta")

        initial_used = 0
        if C_QPU is not None:
            initial_used = self._non_negative_int(int(C_QPU()), "C_QPU()")
        budget = B_QPU if shot_budget is None else shot_budget
        if isinstance(budget, ShotBudget):
            self.shot_budget = budget
        else:
            self.shot_budget = ShotBudget(total=self._positive_int(int(budget), "shot_budget"), used=initial_used)

        self.C_QPU = lambda: self.shot_budget.used
        self.B_QPU = self.shot_budget.total
        self.best_circuit = None
        self.best_value = float("inf")
        self.selected_circuit = None
        self.iterations_completed = 0
        self.run_logs: list[dict[str, object]] = []
        self.iteration_logs: list[dict[str, object]] = []
        self.loss_history: list[float] = []
        self.batch_min_history: list[float] = []
        self.batch_max_history: list[float] = []
        self.theta_gradient_norm_history: list[float] = []
        self.phi_gradient_norm_history: list[float] = []

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _non_negative_int(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    def _baseline_value(self):
        return None if getattr(self.baseline, "value", None) is None else float(self.baseline.value)

    def _spend_after(self, shots: int) -> None:
        self.shot_budget.spend(int(shots))

    @staticmethod
    def _gradient_norm(gradient) -> float | None:
        if gradient is None:
            return None
        return float(np.linalg.norm(np.asarray(gradient, dtype=np.float64)))

    def _parameter_shift_pair_count(self, samples) -> int:
        counter = getattr(self.objective_function, "parameter_shift_pair_count", None)
        if callable(counter):
            return int(counter(samples))
        return 0

    def _run_parameter_shift_step(self, sample_batch: np.ndarray, step_index: int) -> dict[str, object]:
        event: dict[str, object] = {
            "kind": "theta_parameter_shift",
            "step_index": step_index,
            "budget_before": self.shot_budget.used,
            "budget_after": self.shot_budget.used,
            "shots_spent": 0,
            "reason": None,
            "requested_shifted_pairs": 0,
            "executed_shifted_pairs": 0,
            "shots_per_shift_circuit": 0,
            "theta_gradient_norm": None,
        }

        if not self.objective_function.is_parametric or self.shot_budget.depleted:
            event["reason"] = (
                "objective_not_parametric"
                if not self.objective_function.is_parametric
                else "budget_already_depleted"
            )
            return event

        pair_count = self._parameter_shift_pair_count(sample_batch)
        event["requested_shifted_pairs"] = pair_count
        if pair_count == 0:
            g_theta = self.objective_function.parameter_shift_gradient(
                sample_batch,
                self.S_f,
                self.S_theta,
            )
            event["reason"] = "no_parametric_gates_sampled"
            event["theta_gradient_norm"] = self._gradient_norm(g_theta)
            if g_theta is not None and getattr(self.objective_function, "theta", None) is not None:
                self.objective_function.theta -= self.theta_lr * g_theta
            return event

        remaining = self.shot_budget.remaining
        if remaining < 2:
            event["reason"] = "insufficient_budget_for_parameter_shift_pair"
            return event

        full_pair_cost = 2 * self.S_theta
        if remaining >= pair_count * full_pair_cost:
            shots_per_shift = self.S_theta
            max_shifted_pairs = pair_count
            reason = "full_theta_parameter_shift_step"
        else:
            max_full_pairs = remaining // full_pair_cost
            if max_full_pairs > 0:
                shots_per_shift = self.S_theta
                max_shifted_pairs = int(max_full_pairs)
                reason = "partial_theta_parameter_shift_step_limited_by_budget"
            else:
                shots_per_shift = remaining // 2
                if shots_per_shift <= 0:
                    event["reason"] = "insufficient_budget_for_parameter_shift_pair"
                    return event
                max_shifted_pairs = 1
                reason = "partial_theta_parameter_shift_step_limited_by_budget"

        g_theta = self.objective_function.parameter_shift_gradient(
            sample_batch,
            self.S_f,
            shots_per_shift,
            max_shifted_pairs=max_shifted_pairs,
        )
        shots_spent = 2 * shots_per_shift * max_shifted_pairs
        self._spend_after(shots_spent)
        event.update(
            {
                "budget_after": self.shot_budget.used,
                "shots_spent": shots_spent,
                "reason": reason,
                "executed_shifted_pairs": max_shifted_pairs,
                "shots_per_shift_circuit": shots_per_shift,
                "theta_gradient_norm": self._gradient_norm(g_theta),
            }
        )
        if g_theta is not None and getattr(self.objective_function, "theta", None) is not None:
            self.objective_function.theta -= self.theta_lr * g_theta
        return event

    def _evaluation_plan(self, num_samples: int) -> tuple[int | list[int], int]:
        remaining = self.shot_budget.remaining
        if remaining <= 0:
            return [], 0

        full_cost = num_samples * self.S_f
        if remaining >= full_cost:
            return self.S_f, num_samples

        full_samples = remaining // self.S_f
        leftover = remaining % self.S_f
        shot_plan = [self.S_f] * int(full_samples)
        if leftover > 0:
            shot_plan.append(int(leftover))
        return shot_plan, len(shot_plan)

    def _evaluate_and_update_architecture(self, sample_batch: np.ndarray) -> dict[str, object]:
        event: dict[str, object] = {
            "kind": "objective_and_phi_update",
            "budget_before": self.shot_budget.used,
            "budget_after": self.shot_budget.used,
            "shots_spent": 0,
            "reason": None,
            "requested_samples": len(sample_batch),
            "evaluated_samples": 0,
            "shot_plan": [],
            "loss": None,
            "value_min": None,
            "value_max": None,
            "values": [],
            "phi_gradient_norm": None,
            "updated": False,
        }
        shot_plan, evaluated_count = self._evaluation_plan(len(sample_batch))
        if evaluated_count == 0:
            event["reason"] = "no_budget_for_objective_evaluation"
            return event

        evaluated_samples = sample_batch[:evaluated_count]
        values = self.objective_function.evaluate(evaluated_samples, shot_plan)
        values = np.asarray(values, dtype=np.float64)
        shots_spent = evaluated_count * shot_plan if isinstance(shot_plan, int) else sum(shot_plan)
        self._spend_after(int(shots_spent))

        g_phi = self.probability_distribution.reinforce_gradient(
            evaluated_samples,
            values,
            self._baseline_value(),
        )
        phi_gradient_norm = self._gradient_norm(g_phi)
        self.probability_distribution.update(g_phi, self.p_lr)
        self.baseline.update(values)

        best_index = int(np.argmin(values))
        if values[best_index] < self.best_value:
            self.best_value = float(values[best_index])
            self.best_circuit = np.array(evaluated_samples[best_index], dtype=np.intp, copy=True)
        event.update(
            {
                "budget_after": self.shot_budget.used,
                "shots_spent": int(shots_spent),
                "reason": (
                    "full_objective_batch"
                    if evaluated_count == len(sample_batch)
                    else "partial_objective_batch_limited_by_budget"
                ),
                "evaluated_samples": evaluated_count,
                "shot_plan": shot_plan if isinstance(shot_plan, int) else list(shot_plan),
                "loss": float(np.mean(values, dtype=np.float64)),
                "value_min": float(np.min(values)),
                "value_max": float(np.max(values)),
                "values": values.tolist(),
                "phi_gradient_norm": phi_gradient_norm,
                "updated": True,
            }
        )
        return event

    def _select_final_architecture(self):
        selector = getattr(self.probability_distribution, "most_likely", None)
        if callable(selector):
            return selector()
        return self.best_circuit

    def run(
        self,
        num_iterations: int = 10_000,
        *,
        M_t: int | None = None,
        step_k: int | None = None,
        S_f: int | None = None,
        S_theta: int | None = None,
    ):
        num_iterations = self._positive_int(num_iterations, "num_iterations")
        iteration_logs: list[dict[str, object]] = []
        loss_history: list[float] = []
        batch_min_history: list[float] = []
        batch_max_history: list[float] = []
        theta_gradient_norms: list[float] = []
        phi_gradient_norms: list[float] = []
        run_log: dict[str, object] = {
            "run_index": len(self.run_logs),
            "budget_total": self.shot_budget.total,
            "budget_start": self.shot_budget.used,
            "budget_end": self.shot_budget.used,
            "budget_spent": 0,
            "iterations_requested": num_iterations,
            "iterations_completed": 0,
            "stop_reason": None,
            "iterations": iteration_logs,
            "loss_history": loss_history,
            "batch_min_history": batch_min_history,
            "batch_max_history": batch_max_history,
            "theta_gradient_norms": theta_gradient_norms,
            "phi_gradient_norms": phi_gradient_norms,
            "best_value": self.best_value,
            "best_circuit": None,
            "selected_circuit": None,
        }
        self.run_logs.append(run_log)
        self.iteration_logs = iteration_logs
        self.loss_history = loss_history
        self.batch_min_history = batch_min_history
        self.batch_max_history = batch_max_history
        self.theta_gradient_norm_history = theta_gradient_norms
        self.phi_gradient_norm_history = phi_gradient_norms

        original = (self.batch_size, self.parameter_steps, self.S_f, self.S_theta)
        if M_t is not None:
            self.batch_size = self._positive_int(M_t, "M_t")
        if step_k is not None:
            self.parameter_steps = self._non_negative_int(step_k, "step_k")
        if S_f is not None:
            self.S_f = self._positive_int(S_f, "S_f")
        if S_theta is not None:
            self.S_theta = self._positive_int(S_theta, "S_theta")
        run_log.update(
            {
                "batch_size": self.batch_size,
                "parameter_steps": self.parameter_steps,
                "S_f": self.S_f,
                "S_theta": self.S_theta,
            }
        )

        stop_reason = "max_iterations_reached"
        try:
            for run_iteration in range(num_iterations):
                if self.shot_budget.depleted:
                    stop_reason = "shot_budget_depleted_before_iteration"
                    break

                iteration_log: dict[str, object] = {
                    "run_iteration": run_iteration,
                    "global_iteration": self.iterations_completed,
                    "budget_before": self.shot_budget.used,
                    "budget_after": self.shot_budget.used,
                    "budget_spent": 0,
                    "remaining_budget": self.shot_budget.remaining,
                    "sampled_circuits": self.batch_size,
                    "evaluated_circuits": 0,
                    "spend_events": [],
                    "loss": None,
                    "value_min": None,
                    "value_max": None,
                    "batch_values": [],
                    "theta_gradient_norms": [],
                    "phi_gradient_norm": None,
                    "stop_reason": None,
                }
                sample_batch = np.asarray(
                    self.probability_distribution.sample(self.batch_size),
                    dtype=np.intp,
                )

                for step_index in range(self.parameter_steps):
                    if self.shot_budget.depleted:
                        break
                    theta_event = self._run_parameter_shift_step(sample_batch, step_index)
                    iteration_log["spend_events"].append(theta_event)
                    theta_norm = theta_event["theta_gradient_norm"]
                    if theta_norm is not None:
                        iteration_log["theta_gradient_norms"].append(theta_norm)
                        theta_gradient_norms.append(float(theta_norm))

                if self.shot_budget.depleted:
                    iteration_log["stop_reason"] = "shot_budget_depleted_after_theta_updates"
                    iteration_log["budget_after"] = self.shot_budget.used
                    iteration_log["budget_spent"] = (
                        int(iteration_log["budget_after"]) - int(iteration_log["budget_before"])
                    )
                    iteration_log["remaining_budget"] = self.shot_budget.remaining
                    iteration_logs.append(iteration_log)
                    self.iterations_completed += 1
                    stop_reason = "shot_budget_depleted_after_theta_updates"
                    break

                phi_event = self._evaluate_and_update_architecture(sample_batch)
                iteration_log["spend_events"].append(phi_event)
                iteration_log["evaluated_circuits"] = phi_event["evaluated_samples"]
                iteration_log["loss"] = phi_event["loss"]
                iteration_log["value_min"] = phi_event["value_min"]
                iteration_log["value_max"] = phi_event["value_max"]
                iteration_log["batch_values"] = phi_event["values"]
                iteration_log["phi_gradient_norm"] = phi_event["phi_gradient_norm"]
                if phi_event["loss"] is not None:
                    loss_history.append(float(phi_event["loss"]))
                if phi_event["value_min"] is not None:
                    batch_min_history.append(float(phi_event["value_min"]))
                if phi_event["value_max"] is not None:
                    batch_max_history.append(float(phi_event["value_max"]))
                if phi_event["phi_gradient_norm"] is not None:
                    phi_gradient_norms.append(float(phi_event["phi_gradient_norm"]))

                updated = bool(phi_event["updated"])
                if not updated:
                    iteration_log["stop_reason"] = phi_event["reason"]
                    stop_reason = str(phi_event["reason"])
                elif self.shot_budget.depleted:
                    iteration_log["stop_reason"] = "shot_budget_depleted_after_objective_evaluation"
                    stop_reason = "shot_budget_depleted_after_objective_evaluation"
                else:
                    iteration_log["stop_reason"] = "completed"

                iteration_log["budget_after"] = self.shot_budget.used
                iteration_log["budget_spent"] = (
                    int(iteration_log["budget_after"]) - int(iteration_log["budget_before"])
                )
                iteration_log["remaining_budget"] = self.shot_budget.remaining
                iteration_logs.append(iteration_log)
                self.iterations_completed += 1
                if not updated or self.shot_budget.depleted:
                    break
        finally:
            self.batch_size, self.parameter_steps, self.S_f, self.S_theta = original

        self.selected_circuit = self._select_final_architecture()
        run_log["budget_end"] = self.shot_budget.used
        run_log["budget_spent"] = int(run_log["budget_end"]) - int(run_log["budget_start"])
        run_log["iterations_completed"] = len(iteration_logs)
        run_log["stop_reason"] = stop_reason
        run_log["best_value"] = self.best_value
        if self.best_circuit is not None:
            run_log["best_circuit"] = np.asarray(self.best_circuit, dtype=np.intp).tolist()
        if self.selected_circuit is not None:
            run_log["selected_circuit"] = np.asarray(self.selected_circuit, dtype=np.intp).tolist()
        return self.probability_distribution, self.best_circuit, self.best_value
