"""Build a fixed 12-image AEEM v2 cohort without reading GT."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeem_v2.dataset import build_sample_records, load_coarse_mask, load_feature_map
from aeem_v2.semantic import compute_semantic_localization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a no-GT 12-image AEEM cohort.")
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/aeem_v2_m1_cohort12.json")
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/RefCOD"))
    parser.add_argument("--dataset", default="TR-CAMO+TR-COD10K")
    parser.add_argument(
        "--coarse-dir",
        type=Path,
        default=Path("datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K"),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("datasets/cache/features_cache/dinov2/train/TR-CAMO+TR-COD10K"),
    )
    parser.add_argument("--balanced-per-dataset-route", type=int, default=0)
    parser.add_argument("--balanced-per-dataset", type=int, default=0)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--exclude-cohort", action="append", type=Path, default=[])
    return parser.parse_args()


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(count - 1, 0)


def _quantile_choices(records: List[Dict], count: int) -> List[Dict]:
    if len(records) <= count:
        return list(records)
    positions = np.linspace(0, len(records) - 1, count)
    return [records[int(round(position))] for position in positions]


def _replace_for_coverage(selected: List[Dict], candidate: Dict, reason: str) -> None:
    if candidate in selected:
        candidate["selection_reasons"].append(reason)
        return
    same_route = [item for item in selected if item["route"] == candidate["route"]]
    replaceable = [
        item for item in same_route
        if not any(key in item["selection_reasons"] for key in ("empty", "small", "multi"))
    ]
    if not replaceable:
        return
    replaced = replaceable[-1]
    selected[selected.index(replaced)] = candidate
    candidate["selection_reasons"].append(reason)


def select_cohort(records: List[Dict]) -> List[Dict]:
    selected: List[Dict] = []
    for route in ("low", "medium", "high"):
        route_records = sorted(
            (record for record in records if record["route"] == route),
            key=lambda record: (record["localization_reliability"], record["index"]),
        )
        choices = _quantile_choices(route_records, 4)
        for choice in choices:
            choice["selection_reasons"].append(f"{route}_reliability_coverage")
        selected.extend(choices)

    if len(selected) != 12:
        raise ValueError(
            f"Expected four samples per route, got {len(selected)}; route distribution is insufficient"
        )

    empty_candidates = [record for record in records if record["is_empty"]]
    if empty_candidates:
        _replace_for_coverage(selected, empty_candidates[0], "empty")
    nonempty = [record for record in records if not record["is_empty"]]
    if nonempty:
        _replace_for_coverage(
            selected,
            min(nonempty, key=lambda record: (record["area_ratio"], record["index"])),
            "small",
        )
    multi = [record for record in nonempty if record["component_count"] > 1]
    if multi:
        _replace_for_coverage(
            selected,
            max(multi, key=lambda record: (record["component_count"], -record["index"])),
            "multi",
        )
    return sorted(selected, key=lambda record: record["index"])


def select_balanced_cohort(
    records: List[Dict],
    dataset_names,
    per_dataset_route: int,
) -> List[Dict]:
    if per_dataset_route <= 0:
        raise ValueError("per_dataset_route must be positive")
    selected: List[Dict] = []
    for dataset_name in dataset_names:
        for route in ("low", "medium", "high"):
            group = sorted(
                (
                    record for record in records
                    if record["dataset"] == dataset_name and record["route"] == route
                ),
                key=lambda record: (
                    record["localization_reliability"], record["index"]
                ),
            )
            if len(group) < per_dataset_route:
                raise ValueError(
                    f"Insufficient samples for {dataset_name}/{route}: "
                    f"{len(group)} < {per_dataset_route}"
                )
            choices = _quantile_choices(group, per_dataset_route)
            for choice in choices:
                choice["selection_reasons"].append(
                    f"balanced_{dataset_name}_{route}_coverage"
                )
            selected.extend(choices)
    return sorted(selected, key=lambda record: record["index"])


def select_dataset_balanced_cohort(
    records: List[Dict],
    dataset_names,
    per_dataset: int,
) -> List[Dict]:
    if per_dataset <= 0:
        raise ValueError("per_dataset must be positive")
    routes = ("low", "medium", "high")
    selected: List[Dict] = []
    for dataset_name in dataset_names:
        groups = {
            route: sorted(
                (
                    record for record in records
                    if record["dataset"] == dataset_name and record["route"] == route
                ),
                key=lambda record: (
                    record["localization_reliability"], record["index"]
                ),
            )
            for route in routes
        }
        available_count = sum(len(group) for group in groups.values())
        if available_count < per_dataset:
            raise ValueError(
                f"Insufficient samples for {dataset_name}: "
                f"{available_count} < {per_dataset}"
            )
        base_quota = per_dataset // len(routes)
        quotas = {
            route: min(len(groups[route]), base_quota)
            for route in routes
        }
        remaining = per_dataset - sum(quotas.values())
        while remaining:
            eligible_routes = [
                route for route in routes if quotas[route] < len(groups[route])
            ]
            if not eligible_routes:
                raise ValueError(f"Unable to allocate route quotas for {dataset_name}")
            route = min(
                eligible_routes,
                key=lambda item: (quotas[item], routes.index(item)),
            )
            quotas[route] += 1
            remaining -= 1

        for route in routes:
            choices = _quantile_choices(groups[route], quotas[route])
            for choice in choices:
                choice["selection_reasons"].append(
                    f"dataset_balanced_{dataset_name}_{route}_coverage"
                )
            selected.extend(choices)
    return sorted(selected, key=lambda record: record["index"])


def select_all_cohort(records: List[Dict]) -> List[Dict]:
    selected = sorted(records, key=lambda record: record["index"])
    for record in selected:
        record["selection_reasons"].append("all_eligible_samples")
    return selected


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.balanced_per_dataset_route < 0:
        raise ValueError("balanced-per-dataset-route must be non-negative")
    if args.balanced_per_dataset < 0:
        raise ValueError("balanced-per-dataset must be non-negative")
    selection_mode_count = sum((
        bool(args.all_samples),
        bool(args.balanced_per_dataset),
        bool(args.balanced_per_dataset_route),
    ))
    if selection_mode_count > 1:
        raise ValueError(
            "all-samples, balanced-per-dataset, and balanced-per-dataset-route "
            "are mutually exclusive"
        )
    excluded_stems = set()
    for cohort_path in args.exclude_cohort:
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        if cohort.get("generated_without_gt") is not True:
            raise ValueError(f"Excluded cohort is not no-GT: {cohort_path}")
        excluded_stems.update(
            sample["image_name"] for sample in cohort.get("samples", [])
        )
    records = build_sample_records(
        dataset_dir=args.dataset_dir,
        dataset_names=args.dataset.split("+"),
        coarse_dir=args.coarse_dir,
        feature_dir=args.feature_dir,
    )
    diagnostics: List[Dict] = []
    for position, record in enumerate(records):
        coarse = load_coarse_mask(record.coarse_path)
        feature = load_feature_map(record.feature_path)
        localization = compute_semantic_localization(coarse, feature)
        coarse_binary = coarse > 0.5
        diagnostics.append({
            "area_ratio": float(coarse_binary.mean()),
            "component_count": _component_count(coarse_binary),
            "dataset": record.dataset,
            "image_name": record.image_name,
            "index": record.index,
            "is_empty": bool(not coarse_binary.any()),
            "localization_components": localization.components,
            "localization_reliability": localization.reliability,
            "route": localization.route,
            "selection_reasons": [],
        })
        if (position + 1) % 500 == 0:
            print(f"Scanned {position + 1}/{len(records)}")

    eligible = [
        record for record in diagnostics if record["image_name"] not in excluded_stems
    ]
    dataset_names = args.dataset.split("+")
    if args.all_samples:
        cohort = select_all_cohort(eligible)
        selection = "all eligible samples in deterministic dataset index order"
    elif args.balanced_per_dataset:
        cohort = select_dataset_balanced_cohort(
            eligible,
            dataset_names=dataset_names,
            per_dataset=args.balanced_per_dataset,
        )
        selection = (
            f"{args.balanced_per_dataset} reliability-stratified samples per dataset; "
            "scarce route quotas redistributed without GT"
        )
    elif args.balanced_per_dataset_route:
        cohort = select_balanced_cohort(
            eligible,
            dataset_names=dataset_names,
            per_dataset_route=args.balanced_per_dataset_route,
        )
        selection = (
            f"{args.balanced_per_dataset_route} reliability-stratified samples "
            "per dataset and route"
        )
    else:
        cohort = select_cohort(eligible)
        selection = (
            "four reliability-stratified samples per route plus "
            "empty/small/multi coverage"
        )
    payload = {
        "cohort_size": len(cohort),
        "dataset": args.dataset,
        "dataset_counts": {
            dataset_name: sum(item["dataset"] == dataset_name for item in cohort)
            for dataset_name in dataset_names
        },
        "excluded_cohorts": [str(path.resolve()) for path in args.exclude_cohort],
        "excluded_sample_count": len(excluded_stems),
        "generated_without_gt": True,
        "route_counts": {
            route: sum(item["route"] == route for item in cohort)
            for route in ("low", "medium", "high")
        },
        "samples": cohort,
        "selection": selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["route_counts"], indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
