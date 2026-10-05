#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dqas import ShotBudget
from src.me_dqas import MeasurementEfficientDQAS, MeasurementEfficientThreeSATObjectiveFunction
from src.probability_distribution import FactorizedCategoricalProbabilityDistribution
from src.three_sat import compile_internal_clauses, count_violated_clauses


def clause_count(n: int, alpha: float) -> int:
    if n < 3:
        raise ValueError("--n must be at least 3 for 3-SAT.")
    if alpha <= 0.0:
        raise ValueError("--alpha must be positive.")
    return int(round(alpha * n))


def planted_easy_clauses(n: int, alpha: float, seed: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    m = clause_count(n, alpha)
    max_clauses = math.comb(n, 3)
    if m > max_clauses:
        raise ValueError(
            f"planted-easy mode can generate at most C(n, 3)={max_clauses} unique clauses; "
            f"requested {m}. Lower --alpha or increase --n."
        )
    clauses: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    while len(clauses) < m:
        variables = tuple(sorted(rng.choice(n, size=3, replace=False).tolist()))
        clause = tuple(-(v + 1) for v in variables)
        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)
    return clauses


def random_3sat_clauses(n: int, alpha: float, seed: int) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    m = clause_count(n, alpha)
    max_clauses = math.comb(n, 3) * 8
    if m > max_clauses:
        raise ValueError(
            f"random mode can generate at most C(n, 3)*8={max_clauses} unique clauses; "
            f"requested {m}. Lower --alpha or increase --n."
        )
    clauses: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    while len(clauses) < m:
        variables = tuple(sorted(rng.choice(n, size=3, replace=False).tolist()))
        signs = rng.choice([-1, 1], size=3)
        clause = tuple(
            int(variable if sign > 0 else -(variable + 1))
            for variable, sign in zip(variables, signs)
        )
        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)
    return clauses


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def exact_3sat_stats(compiled, max_n: int = 20) -> dict[str, object]:
    n = compiled.num_variables
    if n > max_n:
        return {
            "exact_enumerated": False,
            "exact_min_violations": None,
            "exact_max_violations": None,
            "num_exact_min_assignments": None,
        }
    assignments = np.asarray(list(itertools.product([False, True], repeat=n)), dtype=bool)
    values = np.asarray(count_violated_clauses(assignments, compiled), dtype=np.int32)
    minimum = int(np.min(values))
    return {
        "exact_enumerated": True,
        "exact_min_violations": minimum,
        "exact_max_violations": int(np.max(values)),
        "num_exact_min_assignments": int(np.count_nonzero(values == minimum)),
    }


def plot_loss(loss: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(loss):
        ax.plot(np.arange(1, len(loss) + 1), loss, linewidth=1.8)
        ax.scatter([len(loss)], [loss[-1]], color="#d62728", s=24, zorder=3)
    else:
        ax.text(
            0.5,
            0.55,
            "No objective loss was logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.text(
            0.5,
            0.43,
            "The shot budget was depleted before an objective update.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("ME-DQAS iteration")
    ax.set_ylabel("Mean violated clauses")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_min_mean_max(
    batch_min: np.ndarray,
    batch_mean: np.ndarray,
    batch_max: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(batch_mean):
        x = np.arange(1, len(batch_mean) + 1)
        ax.plot(x, batch_mean, linewidth=1.8, label="mean")
        ax.plot(x, batch_min, linewidth=1.2, label="batch min")
        ax.plot(x, batch_max, linewidth=1.2, label="batch max")
        ax.fill_between(x, batch_min, batch_max, alpha=0.15)
        ax.legend()
    else:
        ax.text(
            0.5,
            0.55,
            "No objective values were logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.text(
            0.5,
            0.43,
            "The shot budget was depleted before an objective update.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("ME-DQAS iteration")
    ax.set_ylabel("Mean violated clauses")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def measurement_efficiency_summary(run_log: dict[str, object]) -> dict[str, int | float | None]:
    events = [
        event
        for iteration in run_log.get("iterations", [])
        for event in iteration.get("spend_events", [])
        if event.get("kind") == "measurement_efficient_theta_update"
    ]
    conventional = int(sum(event.get("conventional_full_shot_cost", 0) for event in events))
    efficient = int(sum(event.get("measurement_efficient_full_shot_cost", 0) for event in events))
    actual = int(sum(event.get("shots_spent", 0) for event in events))
    saved = int(sum(event.get("estimated_shots_saved_for_full_step", 0) for event in events))
    s_theta = int(run_log["S_theta"])
    zero_gradient_savings = int(
        sum(
            event.get(
                "zero_gradient_shots_saved_for_full_step",
                int(event.get("zero_gradient_jobs", 0)) * 2 * s_theta,
            )
            for event in events
        )
    )
    one_shift_savings = int(
        sum(
            event.get(
                "one_shift_shots_saved_for_full_step",
                int(event.get("requested_measurement_efficient_jobs", 0)) * s_theta,
            )
            for event in events
        )
    )
    diagonal_suffix_savings = int(
        sum(
            event.get(
                "diagonal_suffix_shots_saved_for_full_step",
                int(event.get("requested_diagonal_suffix_jobs", 0)) * s_theta,
            )
            for event in events
        )
    )
    clifford_suffix_savings = int(
        sum(
            event.get(
                "clifford_suffix_shots_saved_for_full_step",
                int(event.get("requested_clifford_suffix_jobs", 0)) * s_theta,
            )
            for event in events
        )
    )
    if diagonal_suffix_savings == 0 and clifford_suffix_savings == 0:
        diagonal_suffix_savings = one_shift_savings
    split_savings = zero_gradient_savings + one_shift_savings
    requested_me = int(sum(event.get("requested_measurement_efficient_jobs", 0) for event in events))
    requested_ps = int(sum(event.get("requested_parameter_shift_jobs", 0) for event in events))
    requested_diagonal = int(sum(event.get("requested_diagonal_suffix_jobs", 0) for event in events))
    requested_clifford = int(sum(event.get("requested_clifford_suffix_jobs", 0) for event in events))
    zero_jobs = int(sum(event.get("zero_gradient_jobs", 0) for event in events))
    requested_parametric_terms = int(sum(event.get("requested_parametric_terms", 0) for event in events))
    requested_measured_jobs = int(sum(event.get("requested_measured_gradient_jobs", 0) for event in events))
    executed_parametric_terms = int(sum(event.get("executed_parametric_terms", 0) for event in events))
    executed_jobs = int(sum(event.get("executed_gradient_jobs", 0) for event in events))
    executed_me = int(sum(event.get("executed_measurement_efficient_jobs", 0) for event in events))
    executed_ps = int(sum(event.get("executed_parameter_shift_jobs", 0) for event in events))
    executed_diagonal = int(sum(event.get("executed_diagonal_suffix_jobs", 0) for event in events))
    executed_clifford = int(sum(event.get("executed_clifford_suffix_jobs", 0) for event in events))
    ratio = None if conventional == 0 else efficient / conventional
    return {
        "me_theta_events": len(events),
        "me_requested_parametric_terms": requested_parametric_terms,
        "me_requested_measured_gradient_jobs": requested_measured_jobs,
        "me_requested_measurement_efficient_jobs": requested_me,
        "me_requested_parameter_shift_jobs": requested_ps,
        "me_requested_diagonal_suffix_jobs": requested_diagonal,
        "me_requested_clifford_suffix_jobs": requested_clifford,
        "me_zero_gradient_jobs": zero_jobs,
        "me_executed_parametric_terms": executed_parametric_terms,
        "me_executed_gradient_jobs": executed_jobs,
        "me_executed_measurement_efficient_jobs": executed_me,
        "me_executed_parameter_shift_jobs": executed_ps,
        "me_executed_diagonal_suffix_jobs": executed_diagonal,
        "me_executed_clifford_suffix_jobs": executed_clifford,
        "me_conventional_full_theta_shot_cost": conventional,
        "me_measurement_efficient_full_theta_shot_cost": efficient,
        "me_actual_theta_shots_spent": actual,
        "me_estimated_theta_shots_saved_for_full_steps": saved,
        "me_zero_gradient_shots_saved_for_full_steps": zero_gradient_savings,
        "me_one_shift_shots_saved_for_full_steps": one_shift_savings,
        "me_diagonal_suffix_shots_saved_for_full_steps": diagonal_suffix_savings,
        "me_clifford_suffix_shots_saved_for_full_steps": clifford_suffix_savings,
        "me_zero_gradient_savings_fraction": (
            None if split_savings == 0 else zero_gradient_savings / split_savings
        ),
        "me_one_shift_savings_fraction": (
            None if split_savings == 0 else one_shift_savings / split_savings
        ),
        "me_full_theta_cost_ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measurement-efficient DQAS on generated 3-SAT.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("-L", "--length", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument(
        "--problem-mode",
        choices=("planted-easy", "random"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--s-f", type=int, default=50)
    parser.add_argument("--s-theta", type=int, default=50)
    parser.add_argument("--parameter-steps", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--p-lr", type=float, default=0.3)
    parser.add_argument("--theta-lr", type=float, default=0.15)
    parser.add_argument("--theta-init-scale", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.length is not None and args.length <= 0:
        raise ValueError("--length/-L must be a positive integer.")

    if args.problem_mode == "planted-easy":
        clauses = planted_easy_clauses(args.n, args.alpha, args.seed)
    else:
        clauses = random_3sat_clauses(args.n, args.alpha, args.seed)
    compiled = compile_internal_clauses(clauses, args.n)
    exact_stats = exact_3sat_stats(compiled)
    zero_assignment = np.zeros((1, args.n), dtype=bool)
    zero_violations = int(count_violated_clauses(zero_assignment, compiled)[0])

    gate_set = ["rx", "ry", "rz", "cz"]
    num_positions = args.n if args.length is None else args.length
    if args.theta_init_scale < 0.0:
        raise ValueError("--theta-init-scale must be non-negative.")
    theta_rng = np.random.default_rng(args.seed + 3)
    theta = theta_rng.normal(
        loc=0.0,
        scale=args.theta_init_scale,
        size=(num_positions, len(gate_set)),
    )
    objective = MeasurementEfficientThreeSATObjectiveFunction(
        gate_set=gate_set,
        is_parametric=True,
        clauses=compiled,
        theta=theta,
        seed=args.seed + 1,
    )
    prob = FactorizedCategoricalProbabilityDistribution(
        num_positions=num_positions,
        num_choices=len(gate_set),
        seed=args.seed + 2,
    )

    trainer = MeasurementEfficientDQAS(
        objective,
        prob_dis=prob,
        shot_budget=ShotBudget(total=args.budget),
        p_lr=args.p_lr,
        theta_lr=args.theta_lr,
        batch_size=args.batch_size,
        S_f=args.s_f,
        S_theta=args.s_theta,
        parameter_steps=args.parameter_steps,
    )
    trainer.run(num_iterations=args.num_iterations)

    run_log = trainer.run_logs[-1]
    loss = np.asarray(trainer.loss_history, dtype=float)
    batch_min = np.asarray(trainer.batch_min_history, dtype=float)
    batch_max = np.asarray(trainer.batch_max_history, dtype=float)
    tag = args.tag or (
        f"{args.problem_mode}_n{args.n}_L{num_positions}_alpha{args.alpha:g}_batch{run_log['batch_size']}_"
        f"Sf{run_log['S_f']}_Stheta{run_log['S_theta']}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"me_dqas_3sat_{tag}_loss.png"
    min_mean_max_plot_path = args.output_dir / f"me_dqas_3sat_{tag}_min_mean_max.png"
    log_path = args.output_dir / f"me_dqas_3sat_{tag}_run_log.json"
    summary_path = args.output_dir / f"me_dqas_3sat_{tag}_summary.json"

    plot_loss(
        loss,
        plot_path,
        f"ME-DQAS loss on {args.problem_mode} 3-SAT (n={args.n}, L={num_positions}, alpha={args.alpha:g})",
    )
    plot_min_mean_max(
        batch_min,
        loss,
        batch_max,
        min_mean_max_plot_path,
        f"ME-DQAS batch objective range on {args.problem_mode} 3-SAT",
    )

    summary = {
        "algorithm": "ME-DQAS",
        "n": args.n,
        "alpha": args.alpha,
        "problem_mode": args.problem_mode,
        "seed": args.seed,
        "problem_seed": args.seed,
        "objective_seed": args.seed + 1,
        "probability_seed": args.seed + 2,
        "theta_seed": args.seed + 3,
        "num_clauses": len(clauses),
        "clauses": clauses,
        **exact_stats,
        "zero_assignment_violations": zero_violations,
        "planted_solution": [0] * args.n if args.problem_mode == "planted-easy" else None,
        "planted_solution_violations": zero_violations if args.problem_mode == "planted-easy" else None,
        "gate_set": gate_set,
        "num_positions": num_positions,
        "circuit_length": num_positions,
        "shot_budget_total": trainer.shot_budget.total,
        "shot_budget_used": trainer.shot_budget.used,
        "shot_budget_remaining": trainer.shot_budget.remaining,
        "iterations_completed": run_log["iterations_completed"],
        "stop_reason": run_log["stop_reason"],
        "batch_size": run_log["batch_size"],
        "parameter_steps": run_log["parameter_steps"],
        "S_f": run_log["S_f"],
        "S_theta": run_log["S_theta"],
        "theta_init_scale": args.theta_init_scale,
        "loss_points": len(loss),
        "initial_loss": float(loss[0]) if len(loss) else None,
        "final_loss": float(loss[-1]) if len(loss) else None,
        "best_logged_batch_loss": float(np.min(loss)) if len(loss) else None,
        "initial_batch_min": float(batch_min[0]) if len(batch_min) else None,
        "initial_batch_max": float(batch_max[0]) if len(batch_max) else None,
        "final_batch_min": float(batch_min[-1]) if len(batch_min) else None,
        "final_batch_max": float(batch_max[-1]) if len(batch_max) else None,
        "best_batch_min": float(np.min(batch_min)) if len(batch_min) else None,
        "worst_batch_max": float(np.max(batch_max)) if len(batch_max) else None,
        "best_value": trainer.best_value if np.isfinite(trainer.best_value) else None,
        "best_circuit": trainer.best_circuit.tolist() if trainer.best_circuit is not None else None,
        "selected_circuit": trainer.selected_circuit.tolist()
        if trainer.selected_circuit is not None
        else None,
        "theta_gradient_norm_count": len(trainer.theta_gradient_norm_history),
        "phi_gradient_norm_count": len(trainer.phi_gradient_norm_history),
        **measurement_efficiency_summary(run_log),
        "loss_plot": str(plot_path),
        "min_mean_max_plot": str(min_mean_max_plot_path),
        "run_log": str(log_path),
    }

    log_path.write_text(json.dumps(run_log, indent=2, default=json_default) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
