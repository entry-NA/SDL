"""Create and summarize the fixed-format AEEM multi-seed result sheet."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


GROUPS = ("baseline", "naive_sam2", "full_m4", "reliable_replacement")
SEEDS = (42, 3407, 2025)
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
METRICS = ("E_MEAN", "F_MAX", "SMeasure", "MAE", "WFM")
INPUT_COLUMNS = ("group", "seed", "dataset", *METRICS)


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS)
        writer.writeheader()
        for group in GROUPS:
            for seed in SEEDS:
                for dataset in DATASETS:
                    writer.writerow({"group": group, "seed": seed, "dataset": dataset})


def read_complete_results(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INPUT_COLUMNS:
            raise ValueError(f"Expected columns: {', '.join(INPUT_COLUMNS)}")
        rows = list(reader)

    expected = {(group, seed, dataset) for group in GROUPS for seed in SEEDS for dataset in DATASETS}
    parsed = []
    seen = set()
    for line_number, row in enumerate(rows, start=2):
        key = (row["group"], int(row["seed"]), row["dataset"])
        if key not in expected:
            raise ValueError(f"Unexpected group/seed/dataset at CSV line {line_number}: {key}")
        if key in seen:
            raise ValueError(f"Duplicate row at CSV line {line_number}: {key}")
        seen.add(key)
        values = {metric: float(row[metric]) for metric in METRICS}
        parsed.append({"group": key[0], "seed": key[1], "dataset": key[2], **values})

    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"CSV is incomplete; missing {len(missing)} rows, first missing row: {missing[0]}")
    return parsed


def summarize(rows: list[dict[str, object]], output: Path) -> None:
    values = defaultdict(lambda: defaultdict(list))
    per_seed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row["group"])
        dataset = str(row["dataset"])
        seed = int(row["seed"])
        for metric in METRICS:
            value = float(row[metric])
            values[(group, dataset)][metric].append(value)
            per_seed[(group, seed)][metric].append(value)

    for (group, _seed), metric_values in per_seed.items():
        for metric in METRICS:
            values[(group, "MACRO")][metric].append(statistics.fmean(metric_values[metric]))

    columns = ["group", "dataset", "n_seeds"]
    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group in GROUPS:
            for dataset in (*DATASETS, "MACRO"):
                metric_values = values[(group, dataset)]
                row = {"group": group, "dataset": dataset, "n_seeds": len(metric_values[METRICS[0]])}
                for metric in METRICS:
                    samples = metric_values[metric]
                    row[f"{metric}_mean"] = f"{statistics.fmean(samples):.6f}"
                    row[f"{metric}_std"] = f"{statistics.stdev(samples):.6f}"
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.write_template is None and (args.input is None or args.output is None):
        parser.error("use --write-template, or provide both --input and --output")
    return args


def main() -> None:
    args = parse_args()
    if args.write_template is not None:
        write_template(args.write_template)
        print(f"Template written: {args.write_template.resolve()}")
    if args.input is not None:
        summarize(read_complete_results(args.input), args.output)
        print(f"Summary written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
