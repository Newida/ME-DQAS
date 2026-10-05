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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_dqas_me_dataset_bars import (  # noqa: E402
    ALGORITHMS,
    COLORS,
    PANEL_SPECS,
    collect_metrics,
    format_bar_value,
    metric_summary,
    parse_formats,
    set_axis_style,
    set_neurips_style,
)


def parse_dataset(raw: str) -> tuple[Path, str]:
    if ":" not in raw:
        path = Path(raw)
        return path, path.name
    path_text, label = raw.split(":", 1)
    return Path(path_text), label.strip() or Path(path_text).name


def collect_summary(dataset_dir: Path) -> dict[str, Any]:
    return metric_summary(collect_metrics(dataset_dir))


def metric_xlim(summaries: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    upper = max(
        summaries[row][metric][algorithm]["mean"] + summaries[row][metric][algorithm]["ci95"]
        for row in range(len(summaries))
        for algorithm in ALGORITHMS
    )
    if upper <= 0.0:
        return 0.0, 1.0
    return 0.0, upper * 1.22


def annotate_bars(ax, bars, values: list[float], errors: list[float], metric: str) -> None:
    x_min, x_max = ax.get_xlim()
    offset = 0.018 * (x_max - x_min)
    for bar, value, error in zip(bars, values, errors):
        ax.text(
            value + error + offset,
            bar.get_y() + bar.get_height() / 2.0,
            format_bar_value(metric, value),
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#181A1F",
            clip_on=False,
        )


def plot_paper_summary(
    records: list[tuple[str, dict[str, Any]]],
    *,
    output_base: Path,
    formats: list[str],
) -> list[str]:
    if not records:
        raise ValueError("At least one dataset is required.")

    set_neurips_style()
    plt.rcParams.update(
        {
            "figure.figsize": (6.85, 2.58),
            "figure.facecolor": "white",
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.2,
        }
    )

    summaries = [summary for _, summary in records]
    fig, axes = plt.subplots(len(records), len(PANEL_SPECS), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.992, top=0.82, bottom=0.19, wspace=0.34, hspace=0.44)

    y = np.arange(len(ALGORITHMS), dtype=np.float64)
    colors = [COLORS[algorithm] for algorithm in ALGORITHMS]
    xlims = {
        panel["metric"]: metric_xlim(summaries, panel["metric"])
        for panel in PANEL_SPECS
    }

    for row_index, (dataset_label, summary) in enumerate(records):
        for col_index, panel in enumerate(PANEL_SPECS):
            ax = axes[row_index][col_index]
            metric = panel["metric"]
            values = [summary[metric][algorithm]["mean"] for algorithm in ALGORITHMS]
            errors = [summary[metric][algorithm]["ci95"] for algorithm in ALGORITHMS]
            ax.set_xlim(*xlims[metric])
            bars = ax.barh(
                y,
                values,
                height=0.44,
                xerr=errors,
                capsize=2.3,
                error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#20232A"},
                color=colors,
                edgecolor="white",
                linewidth=0.75,
            )
            set_axis_style(ax)
            if row_index == 0:
                ax.set_title(
                    f"{panel['title']}\n{panel['subtitle']}",
                    loc="left",
                    pad=4,
                    color="#111317",
                    fontweight="bold",
                )
            if row_index == len(records) - 1:
                ax.set_xlabel(panel["xlabel"], color="#343841", labelpad=4)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.set_yticks(y, ALGORITHMS if col_index == 0 else [])
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            annotate_bars(ax, bars, values, errors, metric)

        axes[row_index][0].text(
            -0.36,
            0.5,
            dataset_label,
            transform=axes[row_index][0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.3,
            fontweight="bold",
            color="#111317",
        )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for file_format in formats:
        path = output_base.with_suffix(f".{file_format}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the paper-ready combined DQAS/ME-DQAS bar summary.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[
            str(ROOT / "dataset" / "100_problem_search") + ":3-SAT",
            str(ROOT / "dataset" / "100_maxcut_search") + ":MaxCut",
        ],
        help="Dataset as path:label. May be repeated.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "bar_summary_paper")
    parser.add_argument("--formats", default="pdf,png,svg")
    args = parser.parse_args()

    records = [
        (label, collect_summary(dataset_dir))
        for dataset_dir, label in (parse_dataset(raw) for raw in args.dataset)
    ]
    outputs = plot_paper_summary(records, output_base=args.output.with_suffix(""), formats=parse_formats(args.formats))
    summary = {
        "datasets": [label for label, _ in records],
        "outputs": outputs,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
