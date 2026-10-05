#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ALGORITHMS = ("DQAS", "ME-DQAS")
COLORS = {"DQAS": "#2C5AA0", "ME-DQAS": "#D65F00"}
LINESTYLES = {"DQAS": "-", "ME-DQAS": "-"}


def resolve_path(path_text: str, dataset_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        path,
        dataset_dir / path,
        ROOT / path,
        dataset_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def exact_range(problem: dict[str, Any]) -> tuple[float | None, float | None]:
    for min_key, max_key in (
        ("exact_min_violations", "exact_max_violations"),
        ("exact_min_uncut_edges", "exact_max_uncut_edges"),
        ("exact_min_loss", "exact_max_loss"),
    ):
        exact_min = problem.get(min_key)
        exact_max = problem.get(max_key)
        if exact_min is not None and exact_max is not None:
            return float(exact_min), float(exact_max)
    return None, None


def dataset_objective_label(dataset_dir: Path) -> str:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        label = manifest.get("config", {}).get("objective_label")
        if isinstance(label, str) and label:
            return label
    for comparison_path in discover_comparisons(dataset_dir):
        try:
            comparison = load_json(comparison_path)
            problem_path = resolve_path(comparison["problem"], dataset_dir)
            problem = load_json(problem_path)
        except Exception:
            continue
        label = problem.get("objective_label")
        if isinstance(label, str) and label:
            return label
        if "exact_min_uncut_edges" in problem:
            return "mean uncut edges"
    return "mean violated clauses"


def discover_comparisons(dataset_dir: Path) -> list[Path]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        paths = [
            resolve_path(problem["comparison_json"], dataset_dir)
            for problem in manifest.get("accepted_problems", [])
            if "comparison_json" in problem
        ]
        paths = [path for path in paths if path.exists()]
        if paths:
            return paths
    return sorted(dataset_dir.glob("problem_*/comparison.json"))


def run_loss_points(
    run: dict[str, Any],
    problem: dict[str, Any],
    *,
    x_axis: str,
    metric: str,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for update_index, iteration in enumerate(run["run_log"].get("iterations", []), start=1):
        loss = iteration.get("loss")
        if loss is None:
            continue
        if x_axis == "shots":
            x_values.append(float(iteration["budget_after"]))
        else:
            x_values.append(float(update_index))
        y_values.append(float(loss))

    if not y_values:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    y = np.asarray(y_values, dtype=np.float64)
    if metric == "best-so-far":
        y = np.minimum.accumulate(y)

    if normalize == "exact-range":
        exact_min, exact_max = exact_range(problem)
        if exact_min is not None and exact_max is not None and exact_max > exact_min:
            y = (y - float(exact_min)) / (float(exact_max) - float(exact_min))

    return np.asarray(x_values, dtype=np.float64), y


def load_dataset_records(
    dataset_dir: Path,
    *,
    x_axis: str,
    metric: str,
    normalize: str,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    records: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {algorithm: [] for algorithm in ALGORITHMS}
    skipped: list[str] = []
    for comparison_path in discover_comparisons(dataset_dir):
        try:
            comparison = load_json(comparison_path)
            problem_path = resolve_path(comparison["problem"], dataset_dir)
            problem = load_json(problem_path)
            runs = comparison["runs"]
        except Exception as exc:
            skipped.append(f"{comparison_path}: {exc}")
            continue

        for run in runs:
            algorithm = run.get("algorithm")
            if algorithm not in records:
                continue
            x, y = run_loss_points(
                run,
                problem,
                x_axis=x_axis,
                metric=metric,
                normalize=normalize,
            )
            if x.size:
                records[algorithm].append((x, y))

    if skipped:
        print("Skipped unreadable comparison files:", file=sys.stderr)
        for item in skipped[:10]:
            print(f"  {item}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more", file=sys.stderr)
    return records


def common_grid(
    records: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    x_axis: str,
    num_points: int,
    support: str,
) -> np.ndarray:
    starts = [float(x[0]) for curves in records.values() for x, _ in curves if x.size]
    stops = [float(x[-1]) for curves in records.values() for x, _ in curves if x.size]
    if not starts or not stops:
        raise RuntimeError("No usable loss curves found.")
    if support == "intersection":
        start_value = max(starts)
        stop_value = min(stops)
    else:
        start_value = min(starts)
        stop_value = max(stops)
    if stop_value <= start_value:
        raise RuntimeError("No non-empty shared x-axis support found.")
    if x_axis == "updates":
        start = int(np.ceil(start_value))
        stop = int(np.floor(stop_value))
        return np.arange(start, stop + 1, dtype=np.float64)
    return np.linspace(start_value, stop_value, num_points, dtype=np.float64)


def interpolate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> np.ndarray:
    matrix = np.full((len(curves), len(grid)), np.nan, dtype=np.float64)
    for row, (x, y) in enumerate(curves):
        matrix[row] = np.interp(grid, x, y, left=np.nan, right=float(y[-1]))
    return matrix


def mean_and_ci(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix)
    n = valid.sum(axis=0)
    sums = np.nansum(matrix, axis=0)
    mean = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, n, out=mean, where=n > 0)

    centered = np.where(valid, matrix - mean[None, :], 0.0)
    variance = np.zeros_like(mean)
    np.divide(
        np.sum(centered * centered, axis=0),
        n - 1,
        out=variance,
        where=n > 1,
    )
    std = np.sqrt(variance)
    ci = np.zeros_like(mean)
    np.divide(1.96 * std, np.sqrt(n), out=ci, where=n > 1)
    return mean, ci, n


def set_neurips_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.25, 3.65),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def y_axis_label(metric: str, normalize: str, objective_label: str) -> str:
    prefix = "Best-so-far " if metric == "best-so-far" else ""
    if normalize == "exact-range":
        return f"{prefix}normalized {objective_label}"
    return f"{prefix}{objective_label}"


def plot_mean_ci(
    records: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    output_base: Path,
    x_axis: str,
    metric: str,
    normalize: str,
    num_points: int,
    formats: list[str],
    support: str,
    objective_label: str,
) -> dict[str, Any]:
    set_neurips_style()
    grid = common_grid(records, x_axis, num_points, support)
    x_plot = grid / 1_000_000.0 if x_axis == "shots" else grid

    fig, ax = plt.subplots()
    summary: dict[str, Any] = {
        "x_axis": x_axis,
        "metric": metric,
        "normalize": normalize,
        "support": support,
        "num_curves": {},
        "final_mean": {},
        "final_ci95": {},
    }
    for algorithm in ALGORITHMS:
        curves = records[algorithm]
        if not curves:
            continue
        matrix = interpolate_curves(curves, grid)
        mean, ci, n = mean_and_ci(matrix)
        ax.plot(x_plot, mean, color=COLORS[algorithm], linestyle=LINESTYLES[algorithm], label=algorithm)
        ax.fill_between(x_plot, mean - ci, mean + ci, color=COLORS[algorithm], alpha=0.16, linewidth=0)
        last = np.where(n > 0)[0][-1]
        summary["num_curves"][algorithm] = len(curves)
        summary["final_mean"][algorithm] = float(mean[last])
        summary["final_ci95"][algorithm] = float(ci[last])

    ax.set_xlabel("Cumulative shots (millions)" if x_axis == "shots" else "Objective update")
    ax.set_ylabel(y_axis_label(metric, normalize, objective_label))
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7)
    ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    curve_counts = [len(curves) for curves in records.values() if curves]
    if curve_counts:
        ax.text(
            0.02,
            0.04,
            f"{min(curve_counts)} problems, mean +/- 95% CI",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#555555",
            fontsize=8,
        )

    fig.tight_layout(pad=0.35)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for file_format in formats:
        path = output_base.with_suffix(f".{file_format}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    summary["outputs"] = paths
    return summary


def parse_formats(raw: str) -> list[str]:
    formats = [item.strip().lstrip(".") for item in raw.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must contain at least one format.")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean +/- confidence interval for a DQAS/ME-DQAS dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "100_problem_search")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--x-axis", choices=("shots", "updates"), default="shots")
    parser.add_argument("--metric", choices=("loss", "best-so-far"), default="loss")
    parser.add_argument("--normalize", choices=("none", "exact-range"), default="exact-range")
    parser.add_argument(
        "--support",
        choices=("intersection", "union"),
        default="intersection",
        help="Use shared x-support by default to keep confidence intervals comparable.",
    )
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument("--formats", default="pdf,png")
    args = parser.parse_args()

    if args.num_points <= 1:
        raise ValueError("--num-points must be greater than 1.")
    output_base = args.output
    if output_base is None:
        output_base = args.dataset_dir / f"mean_ci_{args.metric}_vs_{args.x_axis}"
    else:
        output_base = output_base.with_suffix("")

    records = load_dataset_records(
        args.dataset_dir,
        x_axis=args.x_axis,
        metric=args.metric,
        normalize=args.normalize,
    )
    objective_label = dataset_objective_label(args.dataset_dir)
    summary = plot_mean_ci(
        records,
        output_base=output_base,
        x_axis=args.x_axis,
        metric=args.metric,
        normalize=args.normalize,
        num_points=args.num_points,
        formats=parse_formats(args.formats),
        support=args.support,
        objective_label=objective_label,
    )
    summary["objective_label"] = objective_label
    summary_path = output_base.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
