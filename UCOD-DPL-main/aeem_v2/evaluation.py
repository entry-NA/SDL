"""Parameterized, GT-only diagnostic evaluation for AEEM pseudo-labels."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from engine.utils.metrics.metric import Smeasure
from experiments.utils_metrics import bootstrap_ci, compute_bfscore, compute_localized_metrics

from .artifacts import capture_git_state, file_record, sha256_file, utc_timestamp, write_json


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(directory: Path) -> Dict[str, Path]:
    image_paths = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise ValueError(f"No mask images found in {directory}")
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate mask stems found in {directory}")
    return {path.stem: path for path in image_paths}


def _load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _resize_soft(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.float32) / 255.0
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(mask, mode="L").resize(
        (target_shape[1], target_shape[0]),
        resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _binary_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _boundary_region(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    eroded = cv2.erode(
        padded,
        np.ones((3, 3), dtype=np.uint8),
        iterations=width,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    boundary = np.logical_and(padded, np.logical_not(eroded.astype(bool)))
    return boundary[1:-1, 1:-1]


def _boundary_iou(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    width: int,
) -> float:
    prediction_boundary = _boundary_region(prediction, width)
    ground_truth_boundary = _boundary_region(ground_truth, width)
    return _binary_iou(prediction_boundary, ground_truth_boundary)


def _largest_component_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return None
    largest_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    centroid = centroids[largest_index]
    return float(centroid[0]), float(centroid[1])


def _centroid_shift(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_centroid = _largest_component_centroid(prediction)
    gt_centroid = _largest_component_centroid(ground_truth)
    if pred_centroid is None and gt_centroid is None:
        return 0.0
    if pred_centroid is None or gt_centroid is None:
        return 1.0
    diagonal = math.hypot(*ground_truth.shape)
    return float(math.dist(pred_centroid, gt_centroid) / diagonal)


def _component_count(mask: np.ndarray) -> int:
    component_count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(max(component_count - 1, 0))


def _s_measure(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    measure = Smeasure()
    measure.step(
        pred=prediction.astype(np.uint8) * 255,
        gt=ground_truth.astype(np.uint8) * 255,
    )
    return float(measure.get_results()["sm"])


def _mean_ci(values: Sequence[float], n_bootstrap: int) -> Dict[str, float]:
    mean, lower, upper = bootstrap_ci(values, n_bootstrap=n_bootstrap)
    return {"mean": float(mean), "ci_lower": float(lower), "ci_upper": float(upper)}


def _summarize_rows(rows: Sequence[Dict[str, Any]], n_bootstrap: int) -> Dict[str, Any]:
    metrics = (
        "iou", "boundary_iou", "bf", "boundary_precision", "boundary_recall",
        "s_measure", "mae", "centroid_shift", "area_ratio",
        "absolute_area_error", "component_count", "component_count_delta",
        "component_count_error",
    )
    summary: Dict[str, Any] = {"count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[metric] = _mean_ci(values, n_bootstrap)
    band_metrics = sorted(
        key for key in rows[0]
        if key.startswith("band_") and key.endswith(("_iou", "_bf"))
    )
    for metric in band_metrics:
        values = [float(row[metric]) for row in rows]
        summary[metric] = _mean_ci(values, n_bootstrap)
    return summary


def _comparison_summary(
    rows_by_key: Mapping[Tuple[str, str, str], Dict[str, Any]],
    prediction_name: str,
    baseline_name: str,
    dataset_name: Optional[str],
    n_bootstrap: int,
) -> Dict[str, Any]:
    pairs = []
    for (dataset, image_name, name), row in rows_by_key.items():
        if name != prediction_name or (dataset_name is not None and dataset != dataset_name):
            continue
        baseline = rows_by_key[(dataset, image_name, baseline_name)]
        pairs.append((row, baseline))
    iou_deltas = [pair[0]["iou"] - pair[1]["iou"] for pair in pairs]
    bf_deltas = [pair[0]["bf"] - pair[1]["bf"] for pair in pairs]
    boundary_iou_deltas = [
        pair[0]["boundary_iou"] - pair[1]["boundary_iou"] for pair in pairs
    ]
    mae_deltas = [pair[0]["mae"] - pair[1]["mae"] for pair in pairs]
    centroid_deltas = [
        pair[0]["centroid_shift"] - pair[1]["centroid_shift"] for pair in pairs
    ]
    area_error_deltas = [
        pair[0]["absolute_area_error"] - pair[1]["absolute_area_error"]
        for pair in pairs
    ]
    component_error_deltas = [
        pair[0]["component_count_error"] - pair[1]["component_count_error"]
        for pair in pairs
    ]
    return {
        "absolute_area_error_delta": _mean_ci(area_error_deltas, n_bootstrap),
        "bf_delta": _mean_ci(bf_deltas, n_bootstrap),
        "boundary_iou_delta": _mean_ci(boundary_iou_deltas, n_bootstrap),
        "catastrophic_failure_rate": float(
            np.mean(np.asarray(iou_deltas) < -0.2)
        ),
        "centroid_shift_delta": _mean_ci(centroid_deltas, n_bootstrap),
        "component_count_error_delta": _mean_ci(
            component_error_deltas, n_bootstrap
        ),
        "count": len(pairs),
        "iou_delta": _mean_ci(iou_deltas, n_bootstrap),
        "mae_delta": _mean_ci(mae_deltas, n_bootstrap),
    }


def _write_report(path: Path, summary: Mapping[str, Any], baseline_name: Optional[str]) -> None:
    lines = ["# AEEM v2 Label Quality Evaluation", ""]
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Prediction Summary")
    lines.append("")
    lines.append(
        "| Prediction | Dataset | Count | IoU | Boundary IoU | BF-score | "
        "S-measure | MAE | Abs. Area Error | CC Error |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for prediction_name, datasets in summary["predictions"].items():
        for dataset_name, metrics in datasets.items():
            lines.append(
                f"| {prediction_name} | {dataset_name} | {metrics['count']} | "
                f"{metrics['iou']['mean']:.6f} | "
                f"{metrics['boundary_iou']['mean']:.6f} | "
                f"{metrics['bf']['mean']:.6f} | "
                f"{metrics['s_measure']['mean']:.6f} | "
                f"{metrics['mae']['mean']:.6f} | "
                f"{metrics['absolute_area_error']['mean']:.6f} | "
                f"{metrics['component_count_error']['mean']:.6f} |"
            )
    if baseline_name is not None:
        lines.extend(["", f"## Comparisons Against `{baseline_name}`", ""])
        lines.append(
            "| Prediction | Dataset | ΔIoU | ΔBoundary IoU | ΔBF | ΔMAE | "
            "ΔAbs. Area Error | ΔCC Error | Catastrophic Rate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for prediction_name, datasets in summary["comparisons"].items():
            for dataset_name, metrics in datasets.items():
                lines.append(
                    f"| {prediction_name} | {dataset_name} | "
                    f"{metrics['iou_delta']['mean']:.6f} | "
                    f"{metrics['boundary_iou_delta']['mean']:.6f} | "
                    f"{metrics['bf_delta']['mean']:.6f} | "
                    f"{metrics['mae_delta']['mean']:.6f} | "
                    f"{metrics['absolute_area_error_delta']['mean']:.6f} | "
                    f"{metrics['component_count_error_delta']['mean']:.6f} | "
                    f"{metrics['catastrophic_failure_rate']:.6f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_predictions(
    gt_sets: Mapping[str, Path],
    predictions: Mapping[str, Path],
    output_dir: Path,
    repo_root: Path,
    baseline_name: Optional[str] = None,
    threshold: float = 0.5,
    band_widths: Sequence[int] = (5, 10, 20),
    n_bootstrap: int = 1000,
    include_stems: Optional[Sequence[str]] = None,
    source_files: Sequence[Path] = (),
    prediction_fallbacks: Optional[Mapping[str, Path]] = None,
    boundary_width_ratio: float = 0.02,
) -> Path:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if baseline_name is not None and baseline_name not in predictions:
        raise ValueError(f"Unknown baseline prediction: {baseline_name}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not band_widths or any(width <= 0 for width in band_widths):
        raise ValueError("band_widths must contain positive integers")
    if boundary_width_ratio <= 0.0:
        raise ValueError("boundary_width_ratio must be positive")

    prediction_fallbacks = dict(prediction_fallbacks or {})
    unknown_fallbacks = set(prediction_fallbacks) - set(predictions)
    if unknown_fallbacks:
        raise ValueError(
            f"Fallbacks reference unknown predictions: {sorted(unknown_fallbacks)}"
        )

    gt_maps = {name: _collect_images(path.resolve()) for name, path in gt_sets.items()}
    primary_prediction_maps = {
        name: _collect_images(path.resolve()) for name, path in predictions.items()
    }
    fallback_maps = {
        name: _collect_images(path.resolve())
        for name, path in prediction_fallbacks.items()
    }
    if include_stems is not None:
        requested_stems = set(include_stems)
        if not requested_stems:
            raise ValueError("include_stems must not be empty")
        available_gt_stems = {stem for paths in gt_maps.values() for stem in paths}
        missing_gt_stems = requested_stems - available_gt_stems
        if missing_gt_stems:
            raise ValueError(
                f"Requested stems missing from GT: {sorted(missing_gt_stems)}"
            )
        gt_maps = {
            name: {stem: path for stem, path in paths.items() if stem in requested_stems}
            for name, paths in gt_maps.items()
        }
        primary_prediction_maps = {
            name: {stem: path for stem, path in paths.items() if stem in requested_stems}
            for name, paths in primary_prediction_maps.items()
        }
        fallback_maps = {
            name: {stem: path for stem, path in paths.items() if stem in requested_stems}
            for name, paths in fallback_maps.items()
        }
    gt_stems = [stem for paths in gt_maps.values() for stem in paths]
    if len(gt_stems) != len(set(gt_stems)):
        raise ValueError("GT image stems must be unique across datasets")
    gt_stem_set = set(gt_stems)
    prediction_maps: Dict[str, Dict[str, Path]] = {}
    prediction_fallback_counts: Dict[str, int] = {}
    mismatches = {}
    for prediction_name, primary_paths in primary_prediction_maps.items():
        fallback_paths = fallback_maps.get(prediction_name, {})
        available_stems = set(primary_paths) | set(fallback_paths)
        missing = sorted(gt_stem_set - available_stems)
        extra = sorted(available_stems - gt_stem_set)
        if missing or extra:
            mismatches[prediction_name] = {"extra": extra, "missing": missing}
            continue
        resolved_paths = dict(fallback_paths)
        resolved_paths.update(primary_paths)
        prediction_maps[prediction_name] = {
            stem: resolved_paths[stem] for stem in gt_stems
        }
        prediction_fallback_counts[prediction_name] = len(
            gt_stem_set - set(primary_paths)
        )
    if mismatches:
        details = "; ".join(
            f"{name}: {len(values['missing'])} missing, {len(values['extra'])} extra"
            for name, values in mismatches.items()
        )
        raise ValueError(f"Prediction directories do not match GT: {details}")

    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_timestamp()
    config = {
        "band_widths": list(band_widths),
        "baseline_name": baseline_name,
        "boundary_width_ratio": boundary_width_ratio,
        "gt_sets": {name: str(path.resolve()) for name, path in gt_sets.items()},
        "include_stems": sorted(include_stems) if include_stems is not None else None,
        "n_bootstrap": n_bootstrap,
        "predictions": {name: str(path.resolve()) for name, path in predictions.items()},
        "prediction_fallbacks": {
            name: str(path.resolve()) for name, path in prediction_fallbacks.items()
        },
        "started_at": started_at,
        "threshold": threshold,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "manifest.json", {"status": "running", "started_at": started_at})
    git_state = capture_git_state(repo_root.resolve(), output_dir)

    input_hashes: Dict[str, Any] = {
        "ground_truth": {
            dataset: [file_record(path) for path in paths.values()]
            for dataset, paths in gt_maps.items()
        },
        "predictions": {
            name: [file_record(path) for path in paths.values()]
            for name, paths in prediction_maps.items()
        },
        "prediction_primaries": {
            name: [file_record(path) for path in paths.values()]
            for name, paths in primary_prediction_maps.items()
        },
        "prediction_fallbacks": {
            name: [file_record(path) for path in paths.values()]
            for name, paths in fallback_maps.items()
        },
        "sources": [file_record(path.resolve()) for path in source_files],
    }
    write_json(output_dir / "input_hashes.json", input_hashes)

    rows: List[Dict[str, Any]] = []
    with tqdm(
        total=len(gt_stems),
        desc="Evaluating labels",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for dataset_name, gt_map in gt_maps.items():
            for image_name, gt_path in sorted(gt_map.items()):
                gt_u8 = _load_grayscale(gt_path)
                gt_binary = gt_u8 > 127
                gt_area = int(gt_binary.sum())
                gt_component_count = _component_count(gt_binary)
                boundary_width = max(
                    1,
                    int(round(boundary_width_ratio * math.hypot(*gt_binary.shape))),
                )
                for prediction_name, prediction_map in prediction_maps.items():
                    pred_u8 = _load_grayscale(prediction_map[image_name])
                    pred_soft = _resize_soft(pred_u8, gt_binary.shape)
                    pred_binary = pred_soft > threshold
                    bf = compute_bfscore(pred_binary, gt_binary)
                    pred_area = int(pred_binary.sum())
                    area_error = float((pred_area - gt_area) / max(gt_area, 1))
                    pred_component_count = _component_count(pred_binary)
                    component_count_delta = pred_component_count - gt_component_count
                    row: Dict[str, Any] = {
                        "absolute_area_error": abs(area_error),
                        "area_ratio": float(pred_area / max(gt_area, 1)),
                        "bf": float(bf["bf"]),
                        "boundary_iou": _boundary_iou(
                            pred_binary, gt_binary, boundary_width
                        ),
                        "boundary_precision": float(bf["p_b"]),
                        "boundary_recall": float(bf["r_b"]),
                        "centroid_shift": _centroid_shift(pred_binary, gt_binary),
                        "component_count": pred_component_count,
                        "component_count_delta": component_count_delta,
                        "component_count_error": abs(component_count_delta),
                        "dataset": dataset_name,
                        "image_name": image_name,
                        "iou": _binary_iou(pred_binary, gt_binary),
                        "mae": float(np.mean(np.abs(
                            pred_soft - gt_binary.astype(np.float32)
                        ))),
                        "prediction": prediction_name,
                        "s_measure": _s_measure(pred_binary, gt_binary),
                    }
                    for width in band_widths:
                        localized = compute_localized_metrics(
                            pred_binary, gt_binary, band_width=width
                        )
                        row[f"band_{width}_iou"] = float(
                            localized["local_iou"]
                            if localized is not None else 0.0
                        )
                        row[f"band_{width}_bf"] = float(
                            localized["local_bf"]
                            if localized is not None else 0.0
                        )
                    rows.append(row)
                progress.set_postfix_str(
                    f"{dataset_name}/{image_name}",
                    refresh=False,
                )
                progress.update(1)

    fieldnames = list(rows[0].keys())
    with (output_dir / "metrics_per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prediction"], row["dataset"])].append(row)
        grouped[(row["prediction"], "ALL")].append(row)
    prediction_summary: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (prediction_name, dataset_name), group_rows in grouped.items():
        prediction_summary[prediction_name][dataset_name] = _summarize_rows(
            group_rows, n_bootstrap
        )

    rows_by_key = {
        (row["dataset"], row["image_name"], row["prediction"]): row for row in rows
    }
    comparisons: Dict[str, Dict[str, Any]] = defaultdict(dict)
    if baseline_name is not None:
        for prediction_name in predictions:
            if prediction_name == baseline_name:
                continue
            for dataset_name in [*gt_sets.keys(), "ALL"]:
                comparisons[prediction_name][dataset_name] = _comparison_summary(
                    rows_by_key,
                    prediction_name,
                    baseline_name,
                    None if dataset_name == "ALL" else dataset_name,
                    n_bootstrap,
                )

    summary = {
        "baseline_name": baseline_name,
        "comparisons": comparisons,
        "generated_at": utc_timestamp(),
        "predictions": prediction_summary,
    }
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary, baseline_name)
    manifest = {
        "completed_at": utc_timestamp(),
        "git": git_state,
        "gt_count": len(gt_stems),
        "input_hashes_sha256": sha256_file(output_dir / "input_hashes.json"),
        "prediction_count": len(predictions),
        "prediction_fallback_counts": {
            name: count
            for name, count in prediction_fallback_counts.items()
            if count > 0
        },
        "row_count": len(rows),
        "started_at": started_at,
        "status": "complete",
        "summary_sha256": sha256_file(output_dir / "summary.json"),
    }
    write_json(output_dir / "manifest.json", manifest)
    return output_dir
