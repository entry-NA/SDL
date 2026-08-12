"""Audit current pseudo-label assets and confidence-gating behavior."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "datasets" / "cache"
OUTPUT = ROOT / "experiments" / "output"

LABEL_DIRS = (
    "naive_sam2_labels",
    "refined_pseudo_labels",
    "refined_pseudo_labels_broken",
    "refined_pseudo_labels_vloose",
)

EXPECTED_DIRS = (
    "raw_sam2_outputs",
    "refined_pseudo_labels_backup",
    "refined_pseudo_labels_backup_old",
)


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) >= 128


def summarize_directory(path):
    files = sorted(path.glob("*.png"))
    area_ratios = []
    empty = 0
    full = 0
    for file_path in files:
        mask = load_mask(file_path)
        ratio = float(mask.mean())
        area_ratios.append(ratio)
        empty += ratio == 0.0
        full += ratio == 1.0

    if not area_ratios:
        return {"count": 0, "empty": 0, "full": 0}

    values = np.asarray(area_ratios)
    return {
        "count": len(files),
        "empty": empty,
        "full": full,
        "area_mean": float(values.mean()),
        "area_quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
    }


def compare_directories(reference, candidate):
    reference_files = {path.name: path for path in reference.glob("*.png")}
    candidate_files = {path.name: path for path in candidate.glob("*.png")}
    common = sorted(reference_files.keys() & candidate_files.keys())
    differences = []
    ious = []
    equal = 0

    for name in common:
        left = load_mask(reference_files[name])
        right = load_mask(candidate_files[name])
        if left.shape != right.shape:
            resized = Image.fromarray(right.astype(np.uint8) * 255).resize(
                (left.shape[1], left.shape[0]), Image.Resampling.NEAREST
            )
            right = np.asarray(resized) >= 128

        equal += bool(np.array_equal(left, right))
        differences.append(float(np.mean(left != right)))
        intersection = np.logical_and(left, right).sum()
        union = np.logical_or(left, right).sum()
        ious.append(float(intersection / union) if union else 1.0)

    if not common:
        return {"common": 0}

    difference_values = np.asarray(differences)
    iou_values = np.asarray(ious)
    return {
        "common": len(common),
        "equal": equal,
        "different": len(common) - equal,
        "pixel_difference_mean": float(difference_values.mean()),
        "pixel_difference_median": float(np.median(difference_values)),
        "iou_mean": float(iou_values.mean()),
        "iou_median": float(np.median(iou_values)),
    }


def summarize_stats(path):
    if not path.exists():
        return {"exists": False}

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_image", payload)
    values = list(rows.values())
    result = {
        "exists": True,
        "count": len(values),
        "gates": dict(Counter(row.get("gate_decision") for row in values)),
        "selected_mask_idx_logged": dict(
            Counter(str(row.get("selected_mask_idx")) for row in values)
        ),
    }
    for key in ("S_score", "IoU_ori", "EdgeAlign", "IoU_pred", "coarse_area_ratio"):
        series = np.asarray(
            [float(row[key]) for row in values if row.get(key) is not None]
        )
        if series.size:
            result[key] = {
                "min": float(series.min()),
                "p05": float(np.quantile(series, 0.05)),
                "median": float(np.median(series)),
                "p95": float(np.quantile(series, 0.95)),
                "max": float(series.max()),
                "mean": float(series.mean()),
            }
    scores = [float(row["S_score"]) for row in values if row.get("S_score") is not None]
    result["score_over_1"] = sum(score > 1.0 for score in scores)
    result["score_over_999"] = sum(score > 999.0 for score in scores)
    return result


def gating_truth_table():
    table = []
    for score in (0.1, 0.3, 0.6, 1.0, 1.2, 2.0, 5.0):
        outputs = {}
        for sam_value, coarse_value in ((0, 0), (0, 1), (1, 0), (1, 1)):
            fused = score * sam_value + (1.0 - score) * coarse_value
            outputs[f"sam={sam_value},coarse={coarse_value}"] = {
                "value": fused,
                "binary": int(fused > 0.5),
            }
        table.append({"score": score, "outputs": outputs})
    return table


def render_markdown(report):
    lines = [
        "# Current Pseudo-Label State Audit",
        "",
        "Generated from the current filesystem and code-adjacent artifacts.",
        "",
        "## Label Directories",
        "",
        "| Directory | PNG count | Empty | Full | Mean foreground ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report["label_directories"].items():
        lines.append(
            f"| `{name}` | {summary.get('count', 0)} | {summary.get('empty', 0)} | "
            f"{summary.get('full', 0)} | {summary.get('area_mean', 0.0):.6f} |"
        )

    lines.extend(["", "## Compared With `naive_sam2_labels`", ""])
    for name, comparison in report["comparisons"].items():
        lines.append(
            f"- `{name}`: common={comparison.get('common', 0)}, "
            f"equal={comparison.get('equal', 0)}, "
            f"mean pixel difference={comparison.get('pixel_difference_mean', 0.0):.6f}, "
            f"mean IoU={comparison.get('iou_mean', 0.0):.6f}."
        )

    lines.extend(["", "## Missing Referenced Assets", ""])
    for name, exists in report["expected_directories"].items():
        lines.append(f"- `{name}`: {'present' if exists else 'MISSING'}")

    stats = report["per_image_stats"]
    lines.extend(["", "## Recorded Gate Statistics", ""])
    if stats.get("exists"):
        lines.append(f"- Entries: {stats['count']}")
        lines.append(f"- Gate counts: `{stats['gates']}`")
        lines.append(f"- Scores greater than 1: {stats['score_over_1']}")
        lines.append(f"- Scores greater than 999: {stats['score_over_999']}")
        lines.append(f"- S-score summary: `{stats.get('S_score', {})}`")
    else:
        lines.append("- `per_image_stats.json` is missing.")

    lines.extend(
        [
            "",
            "## Gating Algebra Finding",
            "",
            "For `S > 1`, the current formula `S*SAM + (1-S)*coarse` gives the coarse "
            "mask a negative coefficient. After thresholding, pixels present only in the coarse "
            "mask are always removed, so the operation is not a convex fusion.",
            "",
            "## Interpretation Rule",
            "",
            "This report describes the current disk state only. It does not prove which label "
            "directory produced a historical checkpoint unless that run recorded an immutable "
            "configuration and input manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    directories = {name: CACHE / name for name in LABEL_DIRS}
    report = {
        "label_directories": {
            name: summarize_directory(path) for name, path in directories.items()
        },
        "comparisons": {
            name: compare_directories(directories["naive_sam2_labels"], path)
            for name, path in directories.items()
            if name != "naive_sam2_labels"
        },
        "expected_directories": {
            name: (CACHE / name).exists() for name in EXPECTED_DIRS
        },
        "per_image_stats": summarize_stats(
            CACHE / "refined_pseudo_labels" / "per_image_stats.json"
        ),
        "gating_truth_table": gating_truth_table(),
    }

    json_path = OUTPUT / "current_state_audit.json"
    markdown_path = OUTPUT / "current_state_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(markdown_path)


if __name__ == "__main__":
    main()
