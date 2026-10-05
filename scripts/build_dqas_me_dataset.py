#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_dqas_me_dqas_3sat import (  # noqa: E402
    GATE_SET,
    aggregate_rows,
    generate_clauses,
    json_default,
    loss_curve,
    make_theta,
    plot_final_metrics,
    plot_loss_curves,
    run_algorithm,
    write_summary_csv,
)
from run_dqas_3sat_experiment import exact_3sat_stats  # noqa: E402
from src.three_sat import compile_internal_clauses, count_violated_clauses  # noqa: E402


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    alpha: float
    seed: int


DEFAULT_CANDIDATES = (
    CandidateSpec("medium_alpha1p5_seed20260627", 1.5, 20260627),
    CandidateSpec("medium_alpha1p5_seed20260628", 1.5, 20260628),
    CandidateSpec("medium_alpha2_seed20260628", 2.0, 20260628),
    CandidateSpec("medium_alpha2_seed20260629", 2.0, 20260629),
)


def parse_alpha_grid(raw: str) -> list[float]:
    alphas = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alpha-grid must contain at least one alpha.")
    if any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("--alpha-grid values must be positive.")
    return alphas


def generated_candidates(
    *,
    seed_start: int,
    candidate_limit: int,
    alpha_grid: list[float],
) -> list[CandidateSpec]:
    if candidate_limit <= 0:
        raise ValueError("--candidate-limit must be positive.")
    candidates: list[CandidateSpec] = []
    seed = seed_start
    while len(candidates) < candidate_limit:
        for alpha in alpha_grid:
            if len(candidates) >= candidate_limit:
                break
            candidates.append(
                CandidateSpec(
                    label=f"search_alpha{alpha:g}_seed{seed}",
                    alpha=alpha,
                    seed=seed,
                )
            )
        seed += 1
    return candidates


def parse_candidates(raw: str | None) -> list[CandidateSpec] | None:
    if raw is None:
        return None
    candidates: list[CandidateSpec] = []
    for index, item in enumerate(raw.split(","), start=1):
        item = item.strip()
        if not item:
            continue
        try:
            seed_text, alpha_text = item.split(":", 1)
            seed = int(seed_text)
            alpha = float(alpha_text)
        except ValueError as exc:
            raise ValueError("--candidates entries must look like seed:alpha") from exc
        label = f"candidate_{index:02d}_alpha{alpha:g}_seed{seed}"
        candidates.append(CandidateSpec(label, alpha, seed))
    if not candidates:
        raise ValueError("--candidates did not contain any entries.")
    return candidates


def candidate_list(args: argparse.Namespace) -> list[CandidateSpec]:
    explicit = parse_candidates(args.candidates)
    if explicit is not None:
        return explicit
    return generated_candidates(
        seed_start=args.seed_start,
        candidate_limit=args.candidate_limit,
        alpha_grid=parse_alpha_grid(args.alpha_grid),
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def algorithm_curve_metrics(curve: dict[str, list[float]]) -> dict[str, float | int | None]:
    losses = np.asarray(curve["loss"], dtype=np.float64)
    batch_min = np.asarray(curve["batch_min"], dtype=np.float64)
    batch_max = np.asarray(curve["batch_max"], dtype=np.float64)
    if losses.size == 0:
        return {
            "objective_updates": 0,
            "initial_loss": None,
            "final_loss": None,
            "best_loss": None,
            "loss_span": 0.0,
            "loss_drop_to_best": 0.0,
            "batch_range_span": 0.0,
            "mean_batch_width": 0.0,
        }
    return {
        "objective_updates": int(losses.size),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "best_loss": float(np.min(losses)),
        "loss_span": float(np.max(losses) - np.min(losses)),
        "loss_drop_to_best": float(losses[0] - np.min(losses)),
        "batch_range_span": float(np.max(batch_max) - np.min(batch_min)),
        "mean_batch_width": float(np.mean(batch_max - batch_min, dtype=np.float64)),
    }


def candidate_rejection_reasons(
    rows: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    exact_stats: dict[str, Any],
    *,
    min_objective_updates: int,
    min_loss_span: float,
    min_batch_range_span: float,
    min_gap_from_optimum: float,
    min_gap_from_worst: float,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    reasons: list[str] = []
    by_algorithm = {row["algorithm"]: row for row in rows}
    metrics = {
        curve["algorithm"]: algorithm_curve_metrics(curve["curve"])
        for curve in curves
    }
    exact_min = exact_stats.get("exact_min_violations")
    exact_max = exact_stats.get("exact_max_violations")

    for algorithm in ("DQAS", "ME-DQAS"):
        row = by_algorithm[algorithm]
        curve_metric = metrics[algorithm]
        if curve_metric["objective_updates"] < min_objective_updates:
            reasons.append(f"{algorithm}: too few objective updates")
        if curve_metric["loss_span"] < min_loss_span:
            reasons.append(f"{algorithm}: loss curve too flat")
        if curve_metric["batch_range_span"] < min_batch_range_span:
            reasons.append(f"{algorithm}: batch min/max range too flat")
        if exact_min is not None and row["best_value"] <= exact_min + min_gap_from_optimum:
            reasons.append(f"{algorithm}: solved too easily")
        if exact_max is not None and row["best_value"] >= exact_max - min_gap_from_worst:
            reasons.append(f"{algorithm}: best circuit too close to worst-case region")
    return reasons, metrics


def plot_batch_ranges(curves: list[dict[str, Any]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"DQAS": "#1f77b4", "ME-DQAS": "#d95f02"}
    line_styles = {"DQAS": "-", "ME-DQAS": "--"}
    for curve_record in curves:
        algorithm = curve_record["algorithm"]
        curve = curve_record["curve"]
        if not curve["loss"]:
            continue
        x = np.asarray(curve["iteration"], dtype=np.float64)
        loss = np.asarray(curve["loss"], dtype=np.float64)
        batch_min = np.asarray(curve["batch_min"], dtype=np.float64)
        batch_max = np.asarray(curve["batch_max"], dtype=np.float64)
        color = colors[algorithm]
        ax.plot(
            x,
            loss,
            color=color,
            linestyle=line_styles[algorithm],
            linewidth=1.8,
            label=f"{algorithm} mean",
        )
        ax.fill_between(x, batch_min, batch_max, color=color, alpha=0.13)
    ax.set_xlabel("Objective update")
    ax.set_ylabel("Mean violated clauses")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_problem_json(
    path: Path,
    *,
    config: dict[str, Any],
    candidate: CandidateSpec,
    clauses: list[tuple[int, int, int]],
    exact_stats: dict[str, Any],
    zero_assignment_violations: int,
    metrics: dict[str, dict[str, Any]],
) -> None:
    problem = {
        "label": candidate.label,
        "seed": candidate.seed,
        "problem_seed": candidate.seed,
        "objective_seed": candidate.seed + 1,
        "probability_seed": candidate.seed + 2,
        "theta_seed": candidate.seed + 3,
        "n": config["n"],
        "alpha": candidate.alpha,
        "problem_mode": config["problem_mode"],
        "num_clauses": len(clauses),
        "clauses": clauses,
        "gate_set": GATE_SET,
        "circuit_length": config["length"],
        "batch_size": config["batch_size"],
        "S_f": config["s_f"],
        "S_theta": config["s_theta"],
        "budget": config["budget"],
        "parameter_steps": config["parameter_steps"],
        "num_iterations": config["num_iterations"],
        "theta_init_scale": config["theta_init_scale"],
        "zero_assignment_violations": zero_assignment_violations,
        **exact_stats,
        "selection_metrics": metrics,
    }
    path.write_text(json.dumps(problem, indent=2, default=json_default) + "\n")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# DQAS vs ME-DQAS 3-SAT Dataset",
        "",
        "These instances were selected to be neither flat nor immediately solved under the saved hyperparameters.",
        "Each problem uses a high architecture batch size so the ME-DQAS shot-budget advantage is visible.",
        "",
        "## Default Run Settings",
        "",
        f"- n: {manifest['config']['n']}",
        f"- circuit length L: {manifest['config']['length']}",
        f"- gate set: {' '.join(manifest['config']['gate_set'])}",
        f"- batch size: {manifest['config']['batch_size']}",
        f"- S_f: {manifest['config']['S_f']}",
        f"- S_theta: {manifest['config']['S_theta']}",
        f"- budget: {manifest['config']['budget']}",
        "",
        "## Problems",
        "",
        "| problem | alpha | seed | DQAS updates | ME updates | DQAS final | ME final | ME theta ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for problem in manifest["accepted_problems"]:
        rows = {row["algorithm"]: row for row in problem["summary_rows"]}
        lines.append(
            "| {name} | {alpha:g} | {seed} | {d_updates} | {me_updates} | "
            "{d_final:.4g} | {me_final:.4g} | {ratio:.4g} |".format(
                name=problem["name"],
                alpha=problem["alpha"],
                seed=problem["seed"],
                d_updates=rows["DQAS"]["objective_updates"],
                me_updates=rows["ME-DQAS"]["objective_updates"],
                d_final=rows["DQAS"]["final_loss"],
                me_final=rows["ME-DQAS"]["final_loss"],
                ratio=rows["ME-DQAS"]["theta_full_cost_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "Each problem directory contains:",
            "",
            "- `problem.json`: clauses, seeds, exact brute-force stats, and selection metrics",
            "- `comparison.json`: full paired DQAS and ME-DQAS run logs",
            "- `summary.csv`: one row per algorithm",
            "- `loss_vs_updates.png`, `loss_vs_shots.png`, `batch_min_mean_max.png`, `final_metrics.png`",
            "",
            "The comparison is stochastic; use the saved seeds for reproducibility.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_candidate(candidate: CandidateSpec, config: dict[str, Any]) -> dict[str, Any]:
    clauses = generate_clauses(config["problem_mode"], config["n"], candidate.alpha, candidate.seed)
    compiled = compile_internal_clauses(clauses, config["n"])
    stats = exact_3sat_stats(compiled)
    zero_assignment = np.zeros((1, config["n"]), dtype=bool)
    zero_assignment_violations = int(count_violated_clauses(zero_assignment, compiled)[0])
    theta0 = make_theta(candidate.seed, config["length"], config["theta_init_scale"])

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for algorithm in ("DQAS", "ME-DQAS"):
        run_log, summary = run_algorithm(
            algorithm=algorithm,
            compiled=compiled,
            theta0=theta0,
            seed=candidate.seed,
            budget=config["budget"],
            batch_size=config["batch_size"],
            s_f=config["s_f"],
            s_theta=config["s_theta"],
            parameter_steps=config["parameter_steps"],
            num_iterations=config["num_iterations"],
            p_lr=config["p_lr"],
            theta_lr=config["theta_lr"],
        )
        summary.update(
            {
                "n": config["n"],
                "alpha": candidate.alpha,
                "problem_mode": config["problem_mode"],
                "num_clauses": len(clauses),
                "gate_set": " ".join(GATE_SET),
                "num_positions": config["length"],
                "S_f": config["s_f"],
                "S_theta": config["s_theta"],
                "batch_size": config["batch_size"],
                "parameter_steps": config["parameter_steps"],
                "theta_init_scale": config["theta_init_scale"],
            }
        )
        curve = loss_curve(run_log)
        rows.append(summary)
        curves.append({"algorithm": algorithm, "seed": candidate.seed, "curve": curve})
        runs.append({"algorithm": algorithm, "seed": candidate.seed, "summary": summary, "run_log": run_log})

    reasons, metrics = candidate_rejection_reasons(
        rows,
        curves,
        stats,
        min_objective_updates=config["min_objective_updates"],
        min_loss_span=config["min_loss_span"],
        min_batch_range_span=config["min_batch_range_span"],
        min_gap_from_optimum=config["min_gap_from_optimum"],
        min_gap_from_worst=config["min_gap_from_worst"],
    )
    return {
        "candidate": candidate,
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "clauses": clauses,
        "exact_stats": stats,
        "zero_assignment_violations": zero_assignment_violations,
        "rows": rows,
        "curves": curves,
        "runs": runs,
        "metrics": metrics,
        "aggregate": aggregate_rows(rows),
    }


def run_existing_problem(problem: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidate = CandidateSpec(
        label=str(problem["label"]),
        alpha=float(problem["alpha"]),
        seed=int(problem["seed"]),
    )
    clauses = [tuple(int(literal) for literal in clause) for clause in problem["clauses"]]
    compiled = compile_internal_clauses(clauses, int(problem["n"]))
    stats = exact_3sat_stats(compiled)
    zero_assignment = np.zeros((1, int(problem["n"])), dtype=bool)
    zero_assignment_violations = int(count_violated_clauses(zero_assignment, compiled)[0])
    theta0 = make_theta(candidate.seed, int(config["length"]), float(config["theta_init_scale"]))

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for algorithm in ("DQAS", "ME-DQAS"):
        run_log, summary = run_algorithm(
            algorithm=algorithm,
            compiled=compiled,
            theta0=theta0,
            seed=candidate.seed,
            budget=int(config["budget"]),
            batch_size=int(config["batch_size"]),
            s_f=int(config["s_f"]),
            s_theta=int(config["s_theta"]),
            parameter_steps=int(config["parameter_steps"]),
            num_iterations=int(config["num_iterations"]),
            p_lr=float(config["p_lr"]),
            theta_lr=float(config["theta_lr"]),
        )
        summary.update(
            {
                "n": int(config["n"]),
                "alpha": candidate.alpha,
                "problem_mode": config["problem_mode"],
                "num_clauses": len(clauses),
                "gate_set": " ".join(GATE_SET),
                "num_positions": int(config["length"]),
                "S_f": int(config["s_f"]),
                "S_theta": int(config["s_theta"]),
                "batch_size": int(config["batch_size"]),
                "parameter_steps": int(config["parameter_steps"]),
                "theta_init_scale": float(config["theta_init_scale"]),
            }
        )
        curve = loss_curve(run_log)
        rows.append(summary)
        curves.append({"algorithm": algorithm, "seed": candidate.seed, "curve": curve})
        runs.append({"algorithm": algorithm, "seed": candidate.seed, "summary": summary, "run_log": run_log})

    reasons, metrics = candidate_rejection_reasons(
        rows,
        curves,
        stats,
        min_objective_updates=int(config["min_objective_updates"]),
        min_loss_span=float(config["min_loss_span"]),
        min_batch_range_span=float(config["min_batch_range_span"]),
        min_gap_from_optimum=float(config["min_gap_from_optimum"]),
        min_gap_from_worst=float(config["min_gap_from_worst"]),
    )
    return {
        "candidate": candidate,
        "accepted": True,
        "rejection_reasons": reasons,
        "clauses": clauses,
        "exact_stats": stats,
        "zero_assignment_violations": zero_assignment_violations,
        "rows": rows,
        "curves": curves,
        "runs": runs,
        "metrics": metrics,
        "aggregate": aggregate_rows(rows),
    }


def write_dataset_problem(index: int, result: dict[str, Any], config: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    candidate: CandidateSpec = result["candidate"]
    width = max(2, len(str(config.get("max_problems", index))))
    name = f"problem_{index:0{width}d}_{safe_name(candidate.label)}"
    problem_dir = dataset_dir / name
    problem_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = problem_dir / "comparison.json"
    problem_path = problem_dir / "problem.json"
    summary_path = problem_dir / "summary.csv"
    loss_updates_path = problem_dir / "loss_vs_updates.png"
    loss_shots_path = problem_dir / "loss_vs_shots.png"
    batch_range_path = problem_dir / "batch_min_mean_max.png"
    final_metrics_path = problem_dir / "final_metrics.png"

    write_problem_json(
        problem_path,
        config=config,
        candidate=candidate,
        clauses=result["clauses"],
        exact_stats=result["exact_stats"],
        zero_assignment_violations=result["zero_assignment_violations"],
        metrics=result["metrics"],
    )
    comparison = {
        "problem": str(problem_path),
        "config": config,
        "aggregate": result["aggregate"],
        "summary_rows": result["rows"],
        "runs": result["runs"],
        "outputs": {
            "summary_csv": str(summary_path),
            "loss_vs_updates_plot": str(loss_updates_path),
            "loss_vs_shots_plot": str(loss_shots_path),
            "batch_min_mean_max_plot": str(batch_range_path),
            "final_metrics_plot": str(final_metrics_path),
        },
    }
    comparison_path.write_text(json.dumps(comparison, indent=2, default=json_default) + "\n")
    write_summary_csv(result["rows"], summary_path)
    plot_loss_curves(
        result["curves"],
        x_key="iteration",
        x_label="Objective update",
        path=loss_updates_path,
        title=f"{name}: DQAS vs ME-DQAS loss by update",
    )
    plot_loss_curves(
        result["curves"],
        x_key="shots",
        x_label="Cumulative shots spent",
        path=loss_shots_path,
        title=f"{name}: DQAS vs ME-DQAS loss by shots",
    )
    plot_batch_ranges(
        result["curves"],
        path=batch_range_path,
        title=f"{name}: batch min / mean / max",
    )
    plot_final_metrics(
        result["rows"],
        path=final_metrics_path,
        title=f"{name}: final metrics",
    )
    return {
        "name": name,
        "path": str(problem_dir),
        "alpha": candidate.alpha,
        "seed": candidate.seed,
        "problem_json": str(problem_path),
        "comparison_json": str(comparison_path),
        "summary_csv": str(summary_path),
        "plots": {
            "loss_vs_updates": str(loss_updates_path),
            "loss_vs_shots": str(loss_shots_path),
            "batch_min_mean_max": str(batch_range_path),
            "final_metrics": str(final_metrics_path),
        },
        "selection_metrics": result["metrics"],
        "summary_rows": result["rows"],
        "aggregate": result["aggregate"],
    }


def resolve_manifest_path(path_text: str, dataset_dir: Path) -> Path:
    path = Path(path_text)
    candidates = [
        path,
        ROOT / path,
        dataset_dir / path,
        dataset_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def rerun_existing_dataset(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Existing dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest["config"])
    accepted_manifest = list(manifest.get("accepted_problems", []))
    if not accepted_manifest:
        raise RuntimeError(f"No accepted problems found in {manifest_path}")

    accepted: list[dict[str, Any]] = []
    for index, existing in enumerate(accepted_manifest, start=1):
        problem_path = resolve_manifest_path(existing["problem_json"], dataset_dir)
        problem = json.loads(problem_path.read_text())
        expected_width = max(2, len(str(config.get("max_problems", index))))
        expected_name = f"problem_{index:0{expected_width}d}_{safe_name(problem['label'])}"
        if expected_name != existing["name"]:
            raise RuntimeError(
                f"Refusing to create a new problem directory for {problem_path}: "
                f"expected {existing['name']!r}, reconstructed {expected_name!r}."
            )
        print(
            f"[{index}/{len(accepted_manifest)}] rerunning {existing['name']}",
            flush=True,
        )
        result = run_existing_problem(problem, config)
        accepted.append(write_dataset_problem(index, result, config, dataset_dir))

    manifest["accepted_count"] = len(accepted)
    manifest["accepted_problems"] = accepted
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default) + "\n")
    write_readme(dataset_dir / "README.md", manifest)
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "manifest": str(manifest_path),
                "rerun_count": len(accepted),
            },
            indent=2,
            default=json_default,
        )
    )


def write_candidate_audit(path: Path, audit_rows: list[dict[str, Any]]) -> None:
    if not audit_rows:
        return
    fieldnames = sorted({key for row in audit_rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a curated DQAS/ME-DQAS 3-SAT dataset.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun the accepted problems in an existing dataset directory in place.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated seed:alpha entries. If omitted, a deterministic seed/alpha search is used.",
    )
    parser.add_argument("--seed-start", type=int, default=20260627)
    parser.add_argument("--candidate-limit", type=int, default=250)
    parser.add_argument("--alpha-grid", default="1.5,2.0")
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("-L", "--length", type=int, default=8)
    parser.add_argument("--problem-mode", choices=("random", "planted-easy"), default="random")
    parser.add_argument("--budget", type=int, default=750_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--s-f", type=int, default=50)
    parser.add_argument("--s-theta", type=int, default=50)
    parser.add_argument("--parameter-steps", type=int, default=1)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--p-lr", type=float, default=0.3)
    parser.add_argument("--theta-lr", type=float, default=0.15)
    parser.add_argument("--theta-init-scale", type=float, default=0.4)
    parser.add_argument("--min-objective-updates", type=int, default=6)
    parser.add_argument("--min-loss-span", type=float, default=0.03)
    parser.add_argument("--min-batch-range-span", type=float, default=0.4)
    parser.add_argument("--min-gap-from-optimum", type=float, default=0.25)
    parser.add_argument("--min-gap-from-worst", type=float, default=0.05)
    args = parser.parse_args()

    if args.rerun_existing:
        rerun_existing_dataset(args.output_dir)
        return

    if args.max_problems <= 0:
        raise ValueError("--max-problems must be positive.")
    config = {
        "n": args.n,
        "length": args.length,
        "problem_mode": args.problem_mode,
        "budget": args.budget,
        "batch_size": args.batch_size,
        "S_f": args.s_f,
        "S_theta": args.s_theta,
        "s_f": args.s_f,
        "s_theta": args.s_theta,
        "parameter_steps": args.parameter_steps,
        "num_iterations": args.num_iterations,
        "p_lr": args.p_lr,
        "theta_lr": args.theta_lr,
        "theta_init_scale": args.theta_init_scale,
        "gate_set": GATE_SET,
        "min_objective_updates": args.min_objective_updates,
        "min_loss_span": args.min_loss_span,
        "min_batch_range_span": args.min_batch_range_span,
        "min_gap_from_optimum": args.min_gap_from_optimum,
        "min_gap_from_worst": args.min_gap_from_worst,
        "max_problems": args.max_problems,
        "seed_start": args.seed_start,
        "candidate_limit": args.candidate_limit,
        "alpha_grid": parse_alpha_grid(args.alpha_grid),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidates = candidate_list(args)
    for candidate_index, candidate in enumerate(candidates, start=1):
        if len(accepted) >= args.max_problems:
            break
        print(
            f"[{candidate_index}/{len(candidates)}] testing {candidate.label} "
            f"(accepted {len(accepted)}/{args.max_problems})",
            flush=True,
        )
        result = run_candidate(candidate, config)
        for row in result["rows"]:
            metric = result["metrics"][row["algorithm"]]
            audit_rows.append(
                {
                    "label": candidate.label,
                    "accepted": result["accepted"],
                    "rejection_reasons": "; ".join(result["rejection_reasons"]),
                    "algorithm": row["algorithm"],
                    "seed": candidate.seed,
                    "alpha": candidate.alpha,
                    "objective_updates": row["objective_updates"],
                    "final_loss": row["final_loss"],
                    "best_value": row["best_value"],
                    "theta_full_cost_ratio": row["theta_full_cost_ratio"],
                    "loss_span": metric["loss_span"],
                    "batch_range_span": metric["batch_range_span"],
                }
            )
        if result["accepted"] and len(accepted) < args.max_problems:
            accepted.append(write_dataset_problem(len(accepted) + 1, result, config, args.output_dir))
            print(
                f"  accepted -> {accepted[-1]['name']} "
                f"({len(accepted)}/{args.max_problems})",
                flush=True,
            )
        elif not result["accepted"]:
            rejected.append(
                {
                    "label": candidate.label,
                    "alpha": candidate.alpha,
                    "seed": candidate.seed,
                    "rejection_reasons": result["rejection_reasons"],
                    "metrics": result["metrics"],
                }
            )
            print(
                "  rejected: " + "; ".join(result["rejection_reasons"]),
                flush=True,
            )

    manifest = {
        "config": config,
        "accepted_count": len(accepted),
        "accepted_problems": accepted,
        "rejected_candidates": rejected,
        "candidate_count_tested": len(audit_rows) // 2,
    }
    manifest_path = args.output_dir / "manifest.json"
    readme_path = args.output_dir / "README.md"
    audit_path = args.output_dir / "candidate_audit.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default) + "\n")
    write_readme(readme_path, manifest)
    write_candidate_audit(audit_path, audit_rows)

    print(
        json.dumps(
            {
                "dataset_dir": str(args.output_dir),
                "manifest": str(manifest_path),
                "readme": str(readme_path),
                "candidate_audit": str(audit_path),
                "accepted_count": len(accepted),
                "accepted": [
                    {
                        "name": item["name"],
                        "alpha": item["alpha"],
                        "seed": item["seed"],
                        "path": item["path"],
                    }
                    for item in accepted
                ],
                "rejected_count": len(rejected),
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
