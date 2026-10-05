#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_dqas_3sat_experiment import (  # noqa: E402
    exact_3sat_stats,
    json_default,
    planted_easy_clauses,
    random_3sat_clauses,
)
from src.dqas import DQAS, ShotBudget  # noqa: E402
from src.me_dqas import MeasurementEfficientDQAS, MeasurementEfficientThreeSATObjectiveFunction  # noqa: E402
from src.objective_function import ThreeSATObjectiveFunction  # noqa: E402
from src.probability_distribution import FactorizedCategoricalProbabilityDistribution  # noqa: E402
from src.three_sat import compile_internal_clauses, count_violated_clauses  # noqa: E402


GATE_SET = ["rx", "ry", "rz", "cz"]


def parse_seeds(seed: int, num_seeds: int, seeds: str | None) -> list[int]:
    if seeds is not None:
        parsed = [int(item.strip()) for item in seeds.split(",") if item.strip()]
        if not parsed:
            raise ValueError("--seeds must contain at least one integer seed.")
        return parsed
    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    return [seed + offset for offset in range(num_seeds)]


def finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean_or_none(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.mean(finite, dtype=np.float64))


def generate_clauses(problem_mode: str, n: int, alpha: float, seed: int) -> list[tuple[int, int, int]]:
    if problem_mode == "planted-easy":
        return planted_easy_clauses(n, alpha, seed)
    if problem_mode == "random":
        return random_3sat_clauses(n, alpha, seed)
    raise ValueError(f"Unsupported problem mode: {problem_mode!r}.")


def make_theta(seed: int, num_positions: int, theta_init_scale: float) -> np.ndarray:
    if theta_init_scale < 0.0:
        raise ValueError("--theta-init-scale must be non-negative.")
    rng = np.random.default_rng(seed + 3)
    return rng.normal(
        loc=0.0,
        scale=theta_init_scale,
        size=(num_positions, len(GATE_SET)),
    )


def run_algorithm(
    *,
    algorithm: str,
    compiled,
    theta0: np.ndarray,
    seed: int,
    budget: int,
    batch_size: int,
    s_f: int,
    s_theta: int,
    parameter_steps: int,
    num_iterations: int,
    p_lr: float,
    theta_lr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if algorithm == "DQAS":
        objective_cls = ThreeSATObjectiveFunction
        trainer_cls = DQAS
    elif algorithm == "ME-DQAS":
        objective_cls = MeasurementEfficientThreeSATObjectiveFunction
        trainer_cls = MeasurementEfficientDQAS
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}.")

    objective = objective_cls(
        gate_set=GATE_SET,
        is_parametric=True,
        clauses=compiled,
        theta=theta0.copy(),
        seed=seed + 1,
    )
    probability_distribution = FactorizedCategoricalProbabilityDistribution(
        num_positions=theta0.shape[0],
        num_choices=len(GATE_SET),
        seed=seed + 2,
    )
    trainer = trainer_cls(
        f=objective,
        prob_dis=probability_distribution,
        shot_budget=ShotBudget(total=budget),
        p_lr=p_lr,
        theta_lr=theta_lr,
        batch_size=batch_size,
        S_f=s_f,
        S_theta=s_theta,
        parameter_steps=parameter_steps,
    )
    trainer.run(num_iterations=num_iterations)
    return trainer.run_logs[-1], summarize_run(algorithm, seed, trainer.run_logs[-1])


def theta_events(run_log: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for iteration in run_log.get("iterations", [])
        for event in iteration.get("spend_events", [])
        if event.get("kind") in {"theta_parameter_shift", "measurement_efficient_theta_update"}
    ]


def objective_events(run_log: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for iteration in run_log.get("iterations", [])
        for event in iteration.get("spend_events", [])
        if event.get("kind") == "objective_and_phi_update"
    ]


def summarize_run(algorithm: str, seed: int, run_log: dict[str, Any]) -> dict[str, Any]:
    s_theta = int(run_log["S_theta"])
    theta = theta_events(run_log)
    objective = objective_events(run_log)
    losses = [float(value) for value in run_log.get("loss_history", [])]
    batch_mins = [float(value) for value in run_log.get("batch_min_history", [])]
    batch_maxes = [float(value) for value in run_log.get("batch_max_history", [])]

    actual_theta_shots = int(sum(event.get("shots_spent", 0) for event in theta))
    objective_shots = int(sum(event.get("shots_spent", 0) for event in objective))
    conventional_theta_cost = int(
        sum(
            event.get(
                "conventional_full_shot_cost",
                int(event.get("requested_shifted_pairs", 0)) * 2 * s_theta,
            )
            for event in theta
        )
    )
    efficient_full_theta_cost = int(
        sum(
            event.get(
                "measurement_efficient_full_shot_cost",
                int(event.get("requested_shifted_pairs", 0)) * 2 * s_theta,
            )
            for event in theta
        )
    )
    zero_gradient_savings = int(
        sum(
            event.get(
                "zero_gradient_shots_saved_for_full_step",
                int(event.get("zero_gradient_jobs", 0)) * 2 * s_theta,
            )
            for event in theta
        )
    )
    one_shift_savings = int(
        sum(
            event.get(
                "one_shift_shots_saved_for_full_step",
                int(event.get("requested_measurement_efficient_jobs", 0)) * s_theta,
            )
            for event in theta
        )
    )
    diagonal_suffix_savings = int(
        sum(
            event.get(
                "diagonal_suffix_shots_saved_for_full_step",
                int(event.get("requested_diagonal_suffix_jobs", 0)) * s_theta,
            )
            for event in theta
        )
    )
    clifford_suffix_savings = int(
        sum(
            event.get(
                "clifford_suffix_shots_saved_for_full_step",
                int(event.get("requested_clifford_suffix_jobs", 0)) * s_theta,
            )
            for event in theta
        )
    )
    if diagonal_suffix_savings == 0 and clifford_suffix_savings == 0:
        diagonal_suffix_savings = one_shift_savings
    split_savings = zero_gradient_savings + one_shift_savings
    theta_cost_ratio = (
        None if conventional_theta_cost == 0 else efficient_full_theta_cost / conventional_theta_cost
    )

    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "seed": seed,
        "iterations_completed": int(run_log["iterations_completed"]),
        "objective_updates": len(losses),
        "stop_reason": run_log["stop_reason"],
        "budget_total": int(run_log["budget_total"]),
        "budget_used": int(run_log["budget_spent"]),
        "budget_remaining": int(run_log["budget_total"]) - int(run_log["budget_end"]),
        "theta_shots_spent": actual_theta_shots,
        "objective_shots_spent": objective_shots,
        "conventional_theta_shot_cost": conventional_theta_cost,
        "efficient_full_theta_shot_cost": efficient_full_theta_cost,
        "theta_full_cost_ratio": theta_cost_ratio,
        "theta_shots_saved_vs_conventional_full": conventional_theta_cost - efficient_full_theta_cost,
        "final_loss": None if not losses else losses[-1],
        "best_logged_batch_loss": None if not losses else float(np.min(losses)),
        "best_batch_min": None if not batch_mins else float(np.min(batch_mins)),
        "worst_batch_max": None if not batch_maxes else float(np.max(batch_maxes)),
        "best_value": finite_or_none(run_log.get("best_value")),
        "theta_gradient_norm_count": len(run_log.get("theta_gradient_norms", [])),
        "phi_gradient_norm_count": len(run_log.get("phi_gradient_norms", [])),
    }
    if algorithm == "ME-DQAS":
        summary.update(
            {
                "me_requested_parametric_terms": int(
                    sum(event.get("requested_parametric_terms", 0) for event in theta)
                ),
                "me_executed_parametric_terms": int(
                    sum(event.get("executed_parametric_terms", 0) for event in theta)
                ),
                "me_requested_measured_gradient_jobs": int(
                    sum(event.get("requested_measured_gradient_jobs", 0) for event in theta)
                ),
                "me_executed_gradient_jobs": int(
                    sum(event.get("executed_gradient_jobs", 0) for event in theta)
                ),
                "me_requested_measurement_efficient_jobs": int(
                    sum(event.get("requested_measurement_efficient_jobs", 0) for event in theta)
                ),
                "me_requested_diagonal_suffix_jobs": int(
                    sum(event.get("requested_diagonal_suffix_jobs", 0) for event in theta)
                ),
                "me_requested_clifford_suffix_jobs": int(
                    sum(event.get("requested_clifford_suffix_jobs", 0) for event in theta)
                ),
                "me_executed_measurement_efficient_jobs": int(
                    sum(event.get("executed_measurement_efficient_jobs", 0) for event in theta)
                ),
                "me_executed_diagonal_suffix_jobs": int(
                    sum(event.get("executed_diagonal_suffix_jobs", 0) for event in theta)
                ),
                "me_executed_clifford_suffix_jobs": int(
                    sum(event.get("executed_clifford_suffix_jobs", 0) for event in theta)
                ),
                "me_zero_gradient_jobs": int(sum(event.get("zero_gradient_jobs", 0) for event in theta)),
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
            }
        )
    return summary


def loss_curve(run_log: dict[str, Any]) -> dict[str, list[float]]:
    iteration_axis: list[float] = []
    shot_axis: list[float] = []
    losses: list[float] = []
    batch_min: list[float] = []
    batch_max: list[float] = []
    for iteration in run_log.get("iterations", []):
        if iteration.get("loss") is None:
            continue
        iteration_axis.append(float(iteration["run_iteration"] + 1))
        shot_axis.append(float(iteration["budget_after"]))
        losses.append(float(iteration["loss"]))
        batch_min.append(float(iteration["value_min"]))
        batch_max.append(float(iteration["value_max"]))
    return {
        "iteration": iteration_axis,
        "shots": shot_axis,
        "loss": losses,
        "batch_min": batch_min,
        "batch_max": batch_max,
    }


def plot_loss_curves(
    curves: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"DQAS": "#1f77b4", "ME-DQAS": "#d95f02"}
    line_styles = {"DQAS": "-", "ME-DQAS": "--"}
    for curve_record in curves:
        curve = curve_record["curve"]
        if not curve["loss"]:
            continue
        label = f"{curve_record['algorithm']} seed {curve_record['seed']}"
        ax.plot(
            curve[x_key],
            curve["loss"],
            color=colors[curve_record["algorithm"]],
            linestyle=line_styles[curve_record["algorithm"]],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean violated clauses")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_metrics(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    algorithms = ["DQAS", "ME-DQAS"]
    metrics = [
        ("final_loss", "Final loss"),
        ("best_value", "Best observed value"),
        ("iterations_completed", "Iterations completed"),
        ("theta_shots_spent", "Theta shots spent"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        means = [
            mean_or_none([row.get(metric) for row in rows if row["algorithm"] == algorithm])
            for algorithm in algorithms
        ]
        heights = [0.0 if value is None else value for value in means]
        ax.bar(algorithms, heights, color=["#1f77b4", "#d95f02"], alpha=0.85)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.25)
        for index, value in enumerate(means):
            text = "n/a" if value is None else f"{value:.3g}"
            ax.text(index, heights[index], text, ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_algorithm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)

    aggregate: dict[str, Any] = {}
    for algorithm, algorithm_rows in by_algorithm.items():
        prefix = algorithm.lower().replace("-", "_")
        for metric in (
            "final_loss",
            "best_logged_batch_loss",
            "best_batch_min",
            "best_value",
            "iterations_completed",
            "objective_updates",
            "theta_shots_spent",
            "objective_shots_spent",
            "theta_full_cost_ratio",
        ):
            aggregate[f"{prefix}_{metric}_mean"] = mean_or_none(
                [row.get(metric) for row in algorithm_rows]
            )

    dqas_final = aggregate.get("dqas_final_loss_mean")
    me_final = aggregate.get("me_dqas_final_loss_mean")
    if dqas_final is not None and me_final is not None:
        aggregate["final_loss_delta_dqas_minus_me_dqas"] = dqas_final - me_final

    dqas_best = aggregate.get("dqas_best_value_mean")
    me_best = aggregate.get("me_dqas_best_value_mean")
    if dqas_best is not None and me_best is not None:
        aggregate["best_value_delta_dqas_minus_me_dqas"] = dqas_best - me_best

    return aggregate


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQAS and ME-DQAS on matched 3-SAT runs.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("-L", "--length", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument(
        "--problem-mode",
        choices=("planted-easy", "random"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--seeds", default=None, help="Comma-separated problem seeds. Overrides --seed/--num-seeds.")
    parser.add_argument("--num-seeds", type=int, default=1)
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
    num_positions = args.n if args.length is None else args.length
    seeds = parse_seeds(args.seed, args.num_seeds, args.seeds)
    tag = args.tag or (
        f"compare_{args.problem_mode}_n{args.n}_L{num_positions}_alpha{args.alpha:g}_"
        f"seeds{len(seeds)}_budget{args.budget}_batch{args.batch_size}_Sf{args.s_f}_Stheta{args.s_theta}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    full_runs: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    for seed in seeds:
        clauses = generate_clauses(args.problem_mode, args.n, args.alpha, seed)
        compiled = compile_internal_clauses(clauses, args.n)
        zero_assignment = np.zeros((1, args.n), dtype=bool)
        zero_violations = int(count_violated_clauses(zero_assignment, compiled)[0])
        problem = {
            "seed": seed,
            "problem_seed": seed,
            "objective_seed": seed + 1,
            "probability_seed": seed + 2,
            "theta_seed": seed + 3,
            "clauses": clauses,
            "num_clauses": len(clauses),
            "zero_assignment_violations": zero_violations,
            **exact_3sat_stats(compiled),
        }
        problems.append(problem)

        theta0 = make_theta(seed, num_positions, args.theta_init_scale)
        for algorithm in ("DQAS", "ME-DQAS"):
            run_log, summary = run_algorithm(
                algorithm=algorithm,
                compiled=compiled,
                theta0=theta0,
                seed=seed,
                budget=args.budget,
                batch_size=args.batch_size,
                s_f=args.s_f,
                s_theta=args.s_theta,
                parameter_steps=args.parameter_steps,
                num_iterations=args.num_iterations,
                p_lr=args.p_lr,
                theta_lr=args.theta_lr,
            )
            summary.update(
                {
                    "n": args.n,
                    "alpha": args.alpha,
                    "problem_mode": args.problem_mode,
                    "num_clauses": len(clauses),
                    "gate_set": " ".join(GATE_SET),
                    "num_positions": num_positions,
                    "S_f": args.s_f,
                    "S_theta": args.s_theta,
                    "batch_size": args.batch_size,
                    "parameter_steps": args.parameter_steps,
                    "theta_init_scale": args.theta_init_scale,
                }
            )
            rows.append(summary)
            curves.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "curve": loss_curve(run_log),
                }
            )
            full_runs.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "summary": summary,
                    "run_log": run_log,
                }
            )

    aggregate = aggregate_rows(rows)
    comparison_path = args.output_dir / f"dqas_vs_me_dqas_3sat_{tag}_comparison.json"
    summary_csv_path = args.output_dir / f"dqas_vs_me_dqas_3sat_{tag}_summary.csv"
    loss_updates_plot_path = args.output_dir / f"dqas_vs_me_dqas_3sat_{tag}_loss_vs_updates.png"
    loss_shots_plot_path = args.output_dir / f"dqas_vs_me_dqas_3sat_{tag}_loss_vs_shots.png"
    final_metrics_plot_path = args.output_dir / f"dqas_vs_me_dqas_3sat_{tag}_final_metrics.png"

    bundle = {
        "config": {
            "n": args.n,
            "alpha": args.alpha,
            "problem_mode": args.problem_mode,
            "seeds": seeds,
            "budget": args.budget,
            "batch_size": args.batch_size,
            "S_f": args.s_f,
            "S_theta": args.s_theta,
            "parameter_steps": args.parameter_steps,
            "num_iterations": args.num_iterations,
            "p_lr": args.p_lr,
            "theta_lr": args.theta_lr,
            "theta_init_scale": args.theta_init_scale,
            "gate_set": GATE_SET,
            "num_positions": num_positions,
        },
        "problems": problems,
        "runs": full_runs,
        "aggregate": aggregate,
        "outputs": {
            "summary_csv": str(summary_csv_path),
            "loss_vs_updates_plot": str(loss_updates_plot_path),
            "loss_vs_shots_plot": str(loss_shots_plot_path),
            "final_metrics_plot": str(final_metrics_plot_path),
        },
    }

    comparison_path.write_text(json.dumps(bundle, indent=2, default=json_default) + "\n")
    write_summary_csv(rows, summary_csv_path)
    plot_loss_curves(
        curves,
        x_key="iteration",
        x_label="Objective update",
        path=loss_updates_plot_path,
        title="DQAS vs ME-DQAS loss by objective update",
    )
    plot_loss_curves(
        curves,
        x_key="shots",
        x_label="Cumulative shots spent",
        path=loss_shots_plot_path,
        title="DQAS vs ME-DQAS loss by shot budget",
    )
    plot_final_metrics(
        rows,
        path=final_metrics_plot_path,
        title="DQAS vs ME-DQAS aggregate metrics",
    )

    report = {
        "comparison": str(comparison_path),
        "summary_csv": str(summary_csv_path),
        "loss_vs_updates_plot": str(loss_updates_plot_path),
        "loss_vs_shots_plot": str(loss_shots_plot_path),
        "final_metrics_plot": str(final_metrics_plot_path),
        "aggregate": aggregate,
        "rows": rows,
    }
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    main()
