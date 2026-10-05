#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def comparison_paths(dataset_dir: Path) -> list[Path]:
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


def ci95(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(values.size))


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values, dtype=np.float64))


def collect_dataset(dataset_dir: Path, label: str) -> dict[str, Any]:
    zero_reductions: list[float] = []
    diagonal_suffix_reductions: list[float] = []
    clifford_suffix_reductions: list[float] = []
    one_shift_reductions: list[float] = []
    total_reductions: list[float] = []
    conventional_costs: list[int] = []
    zero_savings: list[int] = []
    diagonal_suffix_savings: list[int] = []
    clifford_suffix_savings: list[int] = []
    one_shift_savings: list[int] = []

    for comparison_path in comparison_paths(dataset_dir):
        comparison = load_json(comparison_path)
        rows = {row["algorithm"]: row for row in comparison.get("summary_rows", [])}
        row = rows.get("ME-DQAS")
        if row is None:
            continue

        conventional = int(row.get("conventional_theta_shot_cost", 0))
        if conventional <= 0:
            continue
        zero = int(row.get("me_zero_gradient_shots_saved_for_full_steps", 0))
        one_shift = int(row.get("me_one_shift_shots_saved_for_full_steps", 0))
        diagonal_suffix = int(
            row.get("me_diagonal_suffix_shots_saved_for_full_steps", one_shift)
        )
        clifford_suffix = int(
            row.get(
                "me_clifford_suffix_shots_saved_for_full_steps",
                max(0, one_shift - diagonal_suffix),
            )
        )
        total = int(row.get("theta_shots_saved_vs_conventional_full", zero + one_shift))

        conventional_costs.append(conventional)
        zero_savings.append(zero)
        diagonal_suffix_savings.append(diagonal_suffix)
        clifford_suffix_savings.append(clifford_suffix)
        one_shift_savings.append(one_shift)
        zero_reductions.append(zero / conventional)
        diagonal_suffix_reductions.append(diagonal_suffix / conventional)
        clifford_suffix_reductions.append(clifford_suffix / conventional)
        one_shift_reductions.append(one_shift / conventional)
        total_reductions.append(total / conventional)

    zero_array = np.asarray(zero_reductions, dtype=np.float64)
    diagonal_suffix_array = np.asarray(diagonal_suffix_reductions, dtype=np.float64)
    clifford_suffix_array = np.asarray(clifford_suffix_reductions, dtype=np.float64)
    one_shift_array = np.asarray(one_shift_reductions, dtype=np.float64)
    total_array = np.asarray(total_reductions, dtype=np.float64)

    return {
        "label": label,
        "dataset_dir": str(dataset_dir),
        "problem_count": int(total_array.size),
        "mean_zero_gradient_reduction": finite_mean(zero_array),
        "mean_diagonal_suffix_reduction": finite_mean(diagonal_suffix_array),
        "mean_clifford_suffix_reduction": finite_mean(clifford_suffix_array),
        "mean_one_shift_reduction": finite_mean(one_shift_array),
        "mean_total_reduction": finite_mean(total_array),
        "ci95_total_reduction": ci95(total_array),
        "total_conventional_theta_shot_cost": int(sum(conventional_costs)),
        "total_zero_gradient_shots_saved": int(sum(zero_savings)),
        "total_diagonal_suffix_shots_saved": int(sum(diagonal_suffix_savings)),
        "total_clifford_suffix_shots_saved": int(sum(clifford_suffix_savings)),
        "total_one_shift_shots_saved": int(sum(one_shift_savings)),
        "total_shots_saved": int(sum(zero_savings) + sum(one_shift_savings)),
    }


def parse_dataset_specs(raw_specs: list[str]) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for raw in raw_specs:
        if ":" in raw:
            path_text, label = raw.split(":", 1)
            label = label.strip()
        else:
            path_text = raw
            label = Path(raw).name
        path = Path(path_text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        specs.append((path, label or path.name))
    if not specs:
        raise ValueError("At least one --dataset entry is required.")
    return specs


def parse_formats(raw: str) -> list[str]:
    formats = [item.strip().lstrip(".") for item in raw.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must contain at least one format.")
    return formats


def plot_breakdown(
    records: list[dict[str, Any]],
    output_base: Path,
    formats: list[str] | tuple[str, ...] = ("png", "pdf"),
) -> dict[str, str]:
    labels = [record["label"] for record in records]
    diagonal_suffix = np.asarray(
        [
            record.get("mean_diagonal_suffix_reduction", record["mean_one_shift_reduction"])
            for record in records
        ],
        dtype=np.float64,
    )
    clifford_suffix = np.asarray(
        [record.get("mean_clifford_suffix_reduction", 0.0) for record in records],
        dtype=np.float64,
    )
    one_shift = np.asarray([record["mean_one_shift_reduction"] for record in records], dtype=np.float64)
    zero = np.asarray([record["mean_zero_gradient_reduction"] for record in records], dtype=np.float64)
    totals = np.asarray([record["mean_total_reduction"] for record in records], dtype=np.float64)
    total_ci = np.asarray([record["ci95_total_reduction"] for record in records], dtype=np.float64)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 1.92))
    y = np.arange(len(records), dtype=np.float64)
    height = 0.46
    diagonal_color = "#2C7FB8"
    clifford_color = "#7A5195"
    zero_color = "#F28E2B"

    ax.barh(
        y,
        diagonal_suffix,
        height=height,
        color=diagonal_color,
        edgecolor="white",
        linewidth=0.55,
    )
    ax.barh(
        y,
        clifford_suffix,
        height=height,
        left=diagonal_suffix,
        color=clifford_color,
        edgecolor="white",
        linewidth=0.55,
    )
    ax.barh(
        y,
        zero,
        height=height,
        left=one_shift,
        color=zero_color,
        edgecolor="white",
        linewidth=0.55,
    )
    ax.errorbar(
        totals,
        y,
        xerr=total_ci,
        fmt="none",
        ecolor="#20232A",
        elinewidth=0.9,
        capsize=2.5,
        capthick=0.9,
    )

    for index, total in enumerate(totals):
        ax.text(
            total + total_ci[index] + 0.006,
            y[index],
            f"{100.0 * total:.1f}%",
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="#181A1F",
            clip_on=False,
        )
        if diagonal_suffix[index] > 0.025:
            ax.text(
                diagonal_suffix[index] / 2.0,
                y[index],
                f"{100.0 * diagonal_suffix[index]:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=6.4,
                fontweight="bold",
            )
        if zero[index] > 0.025:
            ax.text(
                one_shift[index] + zero[index] / 2.0,
                y[index],
                f"{100.0 * zero[index]:.1f}%",
                ha="center",
                va="center",
                color="#111317",
                fontsize=6.4,
                fontweight="bold",
            )

    upper = max(0.05, float(np.max(totals + total_ci)) + 0.055)
    ax.set_xlim(0.0, upper)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean reduction in requested theta-gradient shots")
    ax.xaxis.set_major_formatter(lambda value, _: f"{100.0 * value:.0f}%")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_facecolor("#FBFBFC")
    ax.grid(True, axis="x", color="#E1E5EB", linewidth=0.72)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B8BEC8")
    ax.tick_params(axis="x", length=0, colors="#3A3D42", pad=2)
    ax.tick_params(axis="y", length=0, colors="#17191C", pad=5)

    legend_handles = [
        Patch(facecolor=diagonal_color, label="Diagonal one-shift"),
        Patch(facecolor=zero_color, label="Zero-gradient skip"),
        Patch(facecolor=clifford_color, label="Clifford one-shift"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.05,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.20, right=0.94, top=0.76, bottom=0.30)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for file_format in formats:
        path = output_base.with_suffix(f".{file_format}")
        if file_format == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        else:
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        outputs[file_format] = str(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ME-DQAS theta-shot reduction by source.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset directory, optionally followed by ':Label'. May be repeated.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("dataset/me_dqas_reduction_breakdown"),
    )
    parser.add_argument("--formats", default="pdf,png")
    args = parser.parse_args()

    records = [
        collect_dataset(dataset_dir, label)
        for dataset_dir, label in parse_dataset_specs(args.dataset)
    ]
    outputs = plot_breakdown(records, args.output_base, formats=parse_formats(args.formats))
    summary = {"records": records, "outputs": outputs}
    summary_path = args.output_base.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
