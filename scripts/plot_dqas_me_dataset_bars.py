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
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_dqas_me_dataset_ci import (  # noqa: E402
    ALGORITHMS,
    COLORS,
    common_grid,
    discover_comparisons,
    interpolate_curves,
    load_dataset_records,
    load_json,
    mean_and_ci,
    resolve_path,
    set_neurips_style,
)


def parse_formats(raw: str) -> list[str]:
    formats = [item.strip().lstrip(".") for item in raw.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must contain at least one format.")
    return formats


def mean_ci(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise RuntimeError("Cannot summarize an empty metric.")
    if array.size == 1:
        ci95 = 0.0
    else:
        ci95 = float(1.96 * np.std(array, ddof=1) / np.sqrt(array.size))
    return {
        "mean": float(np.mean(array, dtype=np.float64)),
        "ci95": ci95,
        "n": int(array.size),
    }


def paired_delta(dqas_values: list[float], me_values: list[float]) -> dict[str, float | int]:
    count = min(len(dqas_values), len(me_values))
    if count == 0:
        raise RuntimeError("Cannot compute paired delta for empty metrics.")
    return mean_ci((np.asarray(dqas_values[:count]) - np.asarray(me_values[:count])).tolist())


def final_best_so_far_values(dataset_dir: Path) -> dict[str, list[float]]:
    records = load_dataset_records(
        dataset_dir,
        x_axis="shots",
        metric="best-so-far",
        normalize="exact-range",
    )
    grid = common_grid(records, "shots", num_points=300, support="intersection")
    values: dict[str, list[float]] = {}
    for algorithm in ALGORITHMS:
        matrix = interpolate_curves(records[algorithm], grid)
        values[algorithm] = matrix[:, -1].astype(float).tolist()
    return values


def summary_row_metrics(dataset_dir: Path) -> dict[str, dict[str, list[float]]]:
    metrics: dict[str, dict[str, list[float]]] = {
        "objective_updates": {algorithm: [] for algorithm in ALGORITHMS},
        "theta_full_cost_ratio": {algorithm: [] for algorithm in ALGORITHMS},
    }
    for comparison_path in discover_comparisons(dataset_dir):
        comparison = load_json(comparison_path)
        problem_path = resolve_path(comparison["problem"], dataset_dir)
        if not problem_path.exists():
            raise RuntimeError(f"Missing problem file referenced by {comparison_path}: {problem_path}")
        rows = {row["algorithm"]: row for row in comparison["summary_rows"]}
        for algorithm in ALGORITHMS:
            metrics["objective_updates"][algorithm].append(float(rows[algorithm]["objective_updates"]))
            metrics["theta_full_cost_ratio"][algorithm].append(float(rows[algorithm]["theta_full_cost_ratio"]))
    return metrics


def collect_metrics(dataset_dir: Path) -> dict[str, dict[str, list[float]]]:
    metrics = summary_row_metrics(dataset_dir)
    metrics["final_best_so_far_normalized_loss"] = final_best_so_far_values(dataset_dir)
    return metrics


def metric_summary(metrics: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric, by_algorithm in metrics.items():
        metric_entry: dict[str, Any] = {
            algorithm: mean_ci(by_algorithm[algorithm])
            for algorithm in ALGORITHMS
        }
        metric_entry["paired_delta_dqas_minus_me_dqas"] = paired_delta(
            by_algorithm["DQAS"],
            by_algorithm["ME-DQAS"],
        )
        summary[metric] = metric_entry
    return summary


BAR_HEIGHT = 0.46


PANEL_SPECS = [
    {
        "metric": "final_best_so_far_normalized_loss",
        "title": "Final quality",
        "subtitle": "lower is better",
        "xlabel": "Normalized loss",
    },
    {
        "metric": "objective_updates",
        "title": "Budget updates",
        "subtitle": "higher is better",
        "xlabel": "Objective updates",
    },
    {
        "metric": "theta_full_cost_ratio",
        "title": "Theta-shot cost",
        "subtitle": "lower is better",
        "xlabel": "Full-step cost ratio",
    },
]


def format_bar_value(metric: str, value: float) -> str:
    if metric == "objective_updates":
        return f"{value:.1f}"
    return f"{value:.3f}"


def x_axis_limit(values: list[float], errors: list[float]) -> tuple[float, float]:
    upper = max(value + error for value, error in zip(values, errors))
    if upper <= 0.0:
        return 0.0, 1.0
    return 0.0, upper * 1.24


def set_axis_style(ax) -> None:
    ax.set_facecolor("#FBFBFC")
    ax.grid(True, axis="x", color="#E1E5EB", linewidth=0.72)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B8BEC8")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, colors="#3A3D42", pad=2)
    ax.tick_params(axis="y", length=0, colors="#17191C", pad=5)


def annotate_bars(ax, bars, values: list[float], errors: list[float], metric: str) -> None:
    x_min, x_max = ax.get_xlim()
    offset = 0.018 * (x_max - x_min)
    for bar, value, error in zip(bars, values, errors):
        text = format_bar_value(metric, value)
        ax.text(
            value + error + offset,
            bar.get_y() + bar.get_height() / 2.0,
            text,
            ha="left",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#181A1F",
            clip_on=False,
        )


def plot_bars(
    summary: dict[str, Any],
    *,
    output_base: Path,
    formats: list[str],
) -> list[str]:
    set_neurips_style()
    plt.rcParams.update(
        {
            "figure.figsize": (6.85, 1.72),
            "figure.facecolor": "white",
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.3,
        }
    )

    fig, axes = plt.subplots(1, len(PANEL_SPECS))
    fig.subplots_adjust(left=0.112, right=0.992, top=0.72, bottom=0.27, wspace=0.34)

    y = np.arange(len(ALGORITHMS), dtype=np.float64)
    bar_colors = [COLORS[algorithm] for algorithm in ALGORITHMS]

    for panel_index, (ax, panel) in enumerate(zip(axes, PANEL_SPECS)):
        metric = panel["metric"]
        values = [summary[metric][algorithm]["mean"] for algorithm in ALGORITHMS]
        errors = [summary[metric][algorithm]["ci95"] for algorithm in ALGORITHMS]
        ax.set_xlim(*x_axis_limit(values, errors))
        bars = ax.barh(
            y,
            values,
            height=BAR_HEIGHT,
            xerr=errors,
            capsize=2.4,
            error_kw={"elinewidth": 0.95, "capthick": 0.95, "ecolor": "#20232A"},
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
        )
        set_axis_style(ax)
        ax.set_title(
            f"{panel['title']}\n{panel['subtitle']}",
            loc="left",
            pad=4,
            color="#111317",
            fontweight="bold",
        )
        ax.set_xlabel(panel["xlabel"], color="#343841", labelpad=4)
        ax.set_yticks(y, ALGORITHMS if panel_index == 0 else [])
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        annotate_bars(ax, bars, values, errors, metric)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for file_format in formats:
        path = output_base.with_suffix(f".{file_format}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a NeurIPS-style bar summary for DQAS vs ME-DQAS.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "100_problem_search")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--formats", default="pdf,png")
    args = parser.parse_args()

    output_base = args.output
    if output_base is None:
        output_base = args.dataset_dir / "bar_summary"
    else:
        output_base = output_base.with_suffix("")

    metrics = collect_metrics(args.dataset_dir)
    summary = metric_summary(metrics)
    outputs = plot_bars(summary, output_base=output_base, formats=parse_formats(args.formats))
    summary["outputs"] = outputs
    summary["dataset_dir"] = str(args.dataset_dir)
    summary["num_problems"] = len(metrics["final_best_so_far_normalized_loss"]["DQAS"])

    summary_path = output_base.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
