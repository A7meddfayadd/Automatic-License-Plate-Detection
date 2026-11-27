"""Evaluate the Week-10 baselines on the development split.

Run with ``python -m src.evaluate``.  The test split is intentionally not
evaluated here: it stays untouched until the final evaluation in Week 13.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

from . import data, metrics, pipeline

BASELINE_RUNNERS = {
    "trivial": pipeline.trivial_threshold,
    "simple": pipeline.simple_otsu_opening,
}
LOG_FIELDS = [
    "run_id",
    "date",
    "author",
    "pipeline",
    "stage",
    "method",
    "parameters",
    "seed",
    "processing_time_s",
    "metric_name",
    "metric_value",
    "notes",
]


def load_config(path: str | Path) -> dict:
    """Read the YAML configuration."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_one(
    baseline_name: str,
    sample: data.Sample,
    settings: dict,
    hit_iou: float,
) -> dict:
    """Apply one baseline to one image and score it."""
    gray = data.load_image(sample.image_path, color=False)
    runner = BASELINE_RUNNERS[baseline_name]
    params = dict(settings)
    params.pop("opening_kernel", None)
    if baseline_name == "simple":
        params["kernel_size"] = settings.get("opening_kernel", 3)

    begin = time.perf_counter()
    result = runner(gray, **params)
    elapsed_ms = (time.perf_counter() - begin) * 1000.0

    best_iou, best_index = metrics.best_box_iou(result.bbox, sample.boxes)
    predicted_count = 0 if result.bbox is None else 1
    record = {
        "image": sample.name,
        "pipeline": baseline_name,
        "n_gt": int(len(sample.boxes)),
        "iou": best_iou,
        "hit": int(best_iou >= hit_iou),
        "count_error": metrics.count_error(predicted_count, len(sample.boxes)),
        "time_ms": elapsed_ms,
        "pred_bbox": "" if result.bbox is None else " ".join(str(v) for v in result.bbox),
        "best_gt_bbox": (
            ""
            if best_index < 0
            else " ".join(str(int(round(v))) for v in sample.boxes[best_index])
        ),
    }
    return record


def summarize(records: list[dict], hit_iou: float) -> dict:
    """Aggregate per-image records into reportable statistics."""
    ious = np.array([row["iou"] for row in records], dtype=np.float64)
    best_index = int(np.argmax(ious))
    worst_index = int(np.argmin(ious))
    return {
        "n": len(records),
        "mean_iou": float(ious.mean()),
        "std_iou": float(ious.std()),
        "median_iou": float(np.median(ious)),
        "hit_rate": float(np.mean(ious >= hit_iou)),
        "mean_count_error": float(np.mean([row["count_error"] for row in records])),
        "mean_ms": float(np.mean([row["time_ms"] for row in records])),
        "total_s": float(np.sum([row["time_ms"] for row in records]) / 1000.0),
        "best_image": records[best_index]["image"],
        "best_iou": float(ious[best_index]),
        "worst_image": records[worst_index]["image"],
        "worst_iou": float(ious[worst_index]),
    }


def write_per_image_csv(rows: list[dict], path: Path) -> None:
    """Write the combined per-image results table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    baselines: dict[str, dict],
    path: Path,
    split: dict,
    config: dict,
) -> None:
    """Write a small, report-ready summary table."""
    lines = [
        "# Week 10 - baseline results (development split)",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"Split seed: {split['seed']}  ",
        f"Development scenes: {len(split['dev'])}; test scenes: {len(split['test'])}; "
        f"derived from {split['n_unique_scenes']} unique scenes among "
        f"{split['n_files']} files (duplicate groups kept together)  ",
        "Metric: bounding-box IoU of the single predicted plate candidate "
        "against the best reference box; hit rate at IoU >= 0.5.",
        "",
        "| Pipeline | Mean IoU | Std IoU | Median IoU | Hit@0.5 | "
        "Mean count error | ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in baselines.items():
        lines.append(
            f"| {name} | {summary['mean_iou']:.4f} | {summary['std_iou']:.4f} | "
            f"{summary['median_iou']:.4f} | {summary['hit_rate']:.3f} | "
            f"{summary['mean_count_error']:.3f} | {summary['mean_ms']:.2f} |"
        )
    lines += [
        "",
        "Best and worst cases:",
        "",
    ]
    for name, summary in baselines.items():
        lines.append(
            f"- {name}: best {summary['best_image']} (IoU {summary['best_iou']:.3f}), "
            f"worst {summary['worst_image']} (IoU {summary['worst_iou']:.3f})"
        )
    lines += [
        "",
        "Both baselines threshold raw grayscale intensity, as specified for "
        "segmentation tasks, and then keep the largest connected component. "
        "This is deliberately naive for plate localization: bright road, sky or "
        "car-body regions often dominate the mask. The result motivates the "
        "gradient-based Pipeline A.",
        "",
        f"Configuration: `{config['_path']}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_log(
    baselines: dict[str, dict],
    config: dict,
    path: Path,
    split: dict,
) -> None:
    """Upsert this run's two rows into the shared experiment log."""
    date = datetime.now().astimezone().isoformat(timespec="seconds")
    run_rows = [
        {
            "run_id": "W10-BASE-TRIVIAL-DEV",
            "date": date,
            "author": config["author"],
            "pipeline": "baselines",
            "stage": "baseline",
            "method": "fixed_threshold",
            "parameters": "threshold=127; min_area=100; selection=largest_component",
            "seed": config["seed"],
            "processing_time_s": f"{baselines['trivial']['total_s']:.3f}",
            "metric_name": "mean_bbox_iou_dev",
            "metric_value": f"{baselines['trivial']['mean_iou']:.4f}",
            "notes": (
                f"dev split; {baselines['trivial']['n']} unique scenes; "
                f"hit@0.5={baselines['trivial']['hit_rate']:.3f}; "
                f"duplicate groups of {split['n_files']} files kept together"
            ),
        },
        {
            "run_id": "W10-BASE-SIMPLE-DEV",
            "date": date,
            "author": config["author"],
            "pipeline": "baselines",
            "stage": "baseline",
            "method": "otsu+opening",
            "parameters": "otsu; opening=3x3; min_area=100; selection=largest_component",
            "seed": config["seed"],
            "processing_time_s": f"{baselines['simple']['total_s']:.3f}",
            "metric_name": "mean_bbox_iou_dev",
            "metric_value": f"{baselines['simple']['mean_iou']:.4f}",
            "notes": (
                f"dev split; {baselines['simple']['n']} unique scenes; "
                f"hit@0.5={baselines['simple']['hit_rate']:.3f}; "
                f"duplicate groups of {split['n_files']} files kept together"
            ),
        },
    ]
    existing: list[dict] = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    replaced = {row["run_id"] for row in run_rows}
    kept = [row for row in existing if row.get("run_id") not in replaced]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(kept + run_rows)


def save_example_figure(
    records: dict[str, list[dict]],
    summaries: dict[str, dict],
    samples_by_name: dict[str, data.Sample],
    path: Path,
) -> None:
    """Show the best and worst image of each baseline with boxes drawn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for name, rows in records.items():
        best = next(row for row in rows if row["image"] == summaries[name]["best_image"])
        worst = next(row for row in rows if row["image"] == summaries[name]["worst_image"])
        panels.append((name, "best", best))
        panels.append((name, "worst", worst))

    figure, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for axis, (name, label, row) in zip(axes.ravel(), panels):
        sample = samples_by_name[row["image"]]
        image = data.load_image(sample.image_path, color=True)
        axis.imshow(image)
        for box in sample.boxes:
            axis.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.8,
                )
            )
        if row["pred_bbox"]:
            box = [float(value) for value in row["pred_bbox"].split()]
            axis.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor="red",
                    linewidth=1.8,
                    linestyle="--",
                )
            )
        axis.set_title(f"{name} - {label} ({row['image']}, IoU {row['iou']:.3f})", fontsize=9)
        axis.axis("off")
    figure.suptitle("Baselines: green = reference, red dashed = predicted", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(data.PROJECT_ROOT / "configs" / "baselines.yaml"),
        help="path to the baseline YAML configuration",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    config["_path"] = str(config_path.relative_to(data.PROJECT_ROOT))

    samples = data.discover_samples()
    split = data.ensure_split(samples, seed=config["seed"])
    dev_samples = data.get_split_samples("dev", samples=samples, split=split)
    samples_by_name = {sample.name: sample for sample in samples}
    hit_iou = float(config["evaluation"]["hit_iou"])
    print(
        f"Loaded {len(samples)} files -> {split['n_unique_scenes']} unique scenes; "
        f"dev={len(dev_samples)} test={len(split['test'])}"
    )

    all_rows: list[dict] = []
    by_baseline: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for name, settings in config["baselines"].items():
        rows = [run_one(name, sample, settings, hit_iou) for sample in dev_samples]
        by_baseline[name] = rows
        summaries[name] = summarize(rows, hit_iou)
        all_rows.extend(rows)
        summary = summaries[name]
        print(
            f"{name:7s} mean IoU={summary['mean_iou']:.4f} "
            f"std={summary['std_iou']:.4f} hit@0.5={summary['hit_rate']:.3f} "
            f"{summary['mean_ms']:.2f} ms/image"
        )

    output = config["output"]
    write_per_image_csv(all_rows, data.PROJECT_ROOT / output["results_csv"])
    write_summary_markdown(
        summaries,
        data.PROJECT_ROOT / output["summary_md"],
        split,
        config,
    )
    write_experiment_log(
        summaries,
        config,
        data.PROJECT_ROOT / output["log_csv"],
        split,
    )
    save_example_figure(
        by_baseline,
        summaries,
        samples_by_name,
        data.PROJECT_ROOT / output["figure"],
    )
    print("Wrote results CSV, summary, experiment log and example figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
