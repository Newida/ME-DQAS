#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from plot_dqas_me_dataset_bars import (  # noqa: E402
    collect_metrics as collect_bar_metrics,
    metric_summary as bar_metric_summary,
    plot_bars,
)
from plot_dqas_me_dataset_ci import (  # noqa: E402
    dataset_objective_label,
    discover_comparisons,
    load_dataset_records,
    parse_formats,
    plot_mean_ci,
)
from plot_me_dqas_reduction_breakdown import (  # noqa: E402
    collect_dataset as collect_reduction_dataset,
    plot_breakdown,
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    path: Path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def safe_key(label: str) -> str:
    key = label.lower().replace("-", "")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if not key:
        raise ValueError(f"Cannot derive an output key from label {label!r}.")
    return key


def parse_extra_dataset(raw: str) -> DatasetSpec:
    parts = raw.split(":", 2)
    if len(parts) == 1:
        path = Path(parts[0]).expanduser()
        label = path.name
        key = safe_key(label)
    elif len(parts) == 2:
        path = Path(parts[0]).expanduser()
        label = parts[1].strip() or path.name
        key = safe_key(label)
    else:
        path = Path(parts[0]).expanduser()
        label = parts[1].strip() or path.name
        key = parts[2].strip() or safe_key(label)
    return DatasetSpec(key=key, label=label, path=path)


def default_dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = [
        DatasetSpec("3sat", "3-SAT", args.three_sat_dir),
        DatasetSpec("maxcut", "MaxCut", args.maxcut_dir),
    ]
    specs.extend(parse_extra_dataset(raw) for raw in args.dataset)
    return specs


def validate_dataset(spec: DatasetSpec) -> int:
    if not spec.path.exists():
        raise FileNotFoundError(f"{spec.label} dataset directory does not exist: {spec.path}")
    if not spec.path.is_dir():
        raise NotADirectoryError(f"{spec.label} dataset path is not a directory: {spec.path}")
    comparisons = discover_comparisons(spec.path)
    if not comparisons:
        raise RuntimeError(f"No comparison files found for {spec.label} in {spec.path}")
    return len(comparisons)


def output_dir_for_dataset(spec: DatasetSpec, args: argparse.Namespace) -> Path:
    if args.in_place:
        return spec.path
    return args.output_dir / spec.key


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default) + "\n")


def make_bar_summary(spec: DatasetSpec, output_base: Path, formats: list[str]) -> dict[str, Any]:
    metrics = collect_bar_metrics(spec.path)
    summary = bar_metric_summary(metrics)
    outputs = plot_bars(summary, output_base=output_base, formats=formats)
    summary["outputs"] = outputs
    summary["dataset_dir"] = str(spec.path)
    summary["num_problems"] = len(metrics["final_best_so_far_normalized_loss"]["DQAS"])
    summary_path = output_base.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return {
        "summary": str(summary_path),
        "outputs": outputs,
        "num_problems": summary["num_problems"],
    }


def make_mean_ci(
    spec: DatasetSpec,
    output_base: Path,
    *,
    x_axis: str,
    metric: str,
    normalize: str,
    support: str,
    num_points: int,
    formats: list[str],
) -> dict[str, Any]:
    records = load_dataset_records(
        spec.path,
        x_axis=x_axis,
        metric=metric,
        normalize=normalize,
    )
    objective_label = dataset_objective_label(spec.path)
    summary = plot_mean_ci(
        records,
        output_base=output_base,
        x_axis=x_axis,
        metric=metric,
        normalize=normalize,
        num_points=num_points,
        formats=formats,
        support=support,
        objective_label=objective_label,
    )
    summary["objective_label"] = objective_label
    summary["dataset_dir"] = str(spec.path)
    summary_path = output_base.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return {"summary": str(summary_path), **summary}


def make_dataset_results(
    spec: DatasetSpec,
    args: argparse.Namespace,
    formats: list[str],
) -> dict[str, Any]:
    comparison_count = validate_dataset(spec)
    dataset_output_dir = output_dir_for_dataset(spec, args)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {
        "label": spec.label,
        "dataset_dir": str(spec.path),
        "comparison_count": comparison_count,
    }
    if not args.skip_bars:
        artifacts["bar_summary"] = make_bar_summary(
            spec,
            dataset_output_dir / "bar_summary",
            formats,
        )
    if not args.skip_ci:
        ci_artifacts: dict[str, Any] = {}
        for x_axis in args.x_axis:
            for metric in args.metric:
                metric_name = metric.replace("-", "_")
                key = f"{metric_name}_vs_{x_axis}"
                ci_artifacts[key] = make_mean_ci(
                    spec,
                    dataset_output_dir / f"mean_ci_{metric_name}_vs_{x_axis}",
                    x_axis=x_axis,
                    metric=metric,
                    normalize=args.normalize,
                    support=args.support,
                    num_points=args.num_points,
                    formats=formats,
                )
        artifacts["mean_ci"] = ci_artifacts
    return artifacts


def make_reduction_breakdown(
    specs: list[DatasetSpec],
    args: argparse.Namespace,
    formats: list[str],
) -> dict[str, Any] | None:
    if args.skip_reduction_breakdown:
        return None
    output_base = (
        ROOT / "dataset" / "me_dqas_reduction_breakdown"
        if args.in_place
        else args.output_dir / "me_dqas_reduction_breakdown"
    )
    records = [
        collect_reduction_dataset(spec.path, spec.label)
        for spec in specs
    ]
    outputs = plot_breakdown(records, output_base, formats=formats)
    summary = {"records": records, "outputs": outputs}
    summary_path = output_base.with_suffix(".summary.json")
    write_json(summary_path, summary)
    return {"summary": str(summary_path), **summary}


def parse_csv_choices(raw: str, allowed: set[str], name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one value.")
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Invalid --{name} value(s): {', '.join(invalid)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate aggregate result artifacts from saved DQAS/ME-DQAS "
            "dataset comparison files. This does not rerun experiments."
        )
    )
    parser.add_argument("--three-sat-dir", type=Path, default=ROOT / "dataset" / "100_problem_search")
    parser.add_argument("--maxcut-dir", type=Path, default=ROOT / "dataset" / "100_maxcut_search")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Extra dataset as path[:Label[:key]]. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset" / "results")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write artifacts next to each dataset, matching the older plotting-script defaults.",
    )
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--x-axis", default="shots", help="Comma-separated: shots,updates.")
    parser.add_argument("--metric", default="loss,best-so-far", help="Comma-separated: loss,best-so-far.")
    parser.add_argument("--normalize", choices=("none", "exact-range"), default="exact-range")
    parser.add_argument("--support", choices=("intersection", "union"), default="intersection")
    parser.add_argument("--num-points", type=int, default=300)
    parser.add_argument("--skip-bars", action="store_true")
    parser.add_argument("--skip-ci", action="store_true")
    parser.add_argument("--skip-reduction-breakdown", action="store_true")
    args = parser.parse_args()

    if args.num_points <= 1:
        raise ValueError("--num-points must be greater than 1.")
    args.x_axis = parse_csv_choices(args.x_axis, {"shots", "updates"}, "x-axis")
    args.metric = parse_csv_choices(args.metric, {"loss", "best-so-far"}, "metric")
    formats = parse_formats(args.formats)

    specs = default_dataset_specs(args)
    keys = [spec.key for spec in specs]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"Dataset output keys must be unique; duplicates: {', '.join(duplicate_keys)}")

    summary: dict[str, Any] = {
        "datasets": {},
        "output_dir": None if args.in_place else str(args.output_dir),
        "in_place": bool(args.in_place),
        "formats": formats,
    }
    for spec in specs:
        summary["datasets"][spec.key] = make_dataset_results(spec, args, formats)

    reduction = make_reduction_breakdown(specs, args, formats)
    if reduction is not None:
        summary["reduction_breakdown"] = reduction

    summary_path = (
        ROOT / "dataset" / "results_from_dataset.summary.json"
        if args.in_place
        else args.output_dir / "results_from_dataset.summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
