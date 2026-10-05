#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxcut_common import (  # noqa: E402
    GATE_SET,
    add_maxcut_args,
    aggregate_rows,
    json_default,
    loss_curve,
    make_theta,
    parse_seeds,
    plot_final_metrics,
    plot_loss_curves,
    problem_metadata,
    run_algorithm,
    write_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQAS and ME-DQAS on matched MaxCut runs.")
    add_maxcut_args(parser, compare=True)
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
    rows: list[dict] = []
    curves: list[dict] = []
    full_runs: list[dict] = []
    problems: list[dict] = []

    for seed in seeds:
        _, compiled, problem = problem_metadata(args.problem_mode, args.n, args.alpha, seed)
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
                    "num_edges": problem["num_edges"],
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
    comparison_path = args.output_dir / f"dqas_vs_me_dqas_maxcut_{tag}_comparison.json"
    summary_csv_path = args.output_dir / f"dqas_vs_me_dqas_maxcut_{tag}_summary.csv"
    loss_updates_plot_path = args.output_dir / f"dqas_vs_me_dqas_maxcut_{tag}_loss_vs_updates.png"
    loss_shots_plot_path = args.output_dir / f"dqas_vs_me_dqas_maxcut_{tag}_loss_vs_shots.png"
    final_metrics_plot_path = args.output_dir / f"dqas_vs_me_dqas_maxcut_{tag}_final_metrics.png"

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
        "summary_rows": rows,
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
        title="DQAS vs ME-DQAS MaxCut loss by objective update",
    )
    plot_loss_curves(
        curves,
        x_key="shots",
        x_label="Cumulative shots spent",
        path=loss_shots_plot_path,
        title="DQAS vs ME-DQAS MaxCut loss by shot budget",
    )
    plot_final_metrics(
        rows,
        path=final_metrics_plot_path,
        title="DQAS vs ME-DQAS MaxCut aggregate metrics",
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
