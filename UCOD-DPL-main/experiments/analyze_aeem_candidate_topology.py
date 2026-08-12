"""Replay cached AEEM candidates for no-GT topology diagnosis."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeem_v2.artifacts import (
    capture_git_state,
    create_experiment_directory,
    file_record,
    sha256_file,
    utc_timestamp,
    write_json,
)
from aeem_v2.dataset import build_sample_records, load_coarse_mask
from aeem_v2.refinement import (
    CandidateAssessment,
    MaskCandidate,
    build_boundary_constraint,
    fuse_boundary_residual,
    load_candidate_cache,
)
from aeem_v2.topology import component_profile, quantized_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose cached AEEM candidates without reading GT."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        default=Path("artifacts/aeem_v2/m1_cohort12_20260724_v5_final"),
    )
    parser.add_argument(
        "--cohort", type=Path, default=Path("experiments/aeem_v2_m1_cohort12.json")
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/aeem_v2/diagnostics"),
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
    parser.add_argument("--quality-window", type=float, default=0.05)
    return parser.parse_args()


def _resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.clip(cv2.resize(
        probability.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ), 0.0, 1.0)


def _load_probability(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _assessment(mask: np.ndarray, metadata: Dict) -> CandidateAssessment:
    candidate = MaskCandidate(
        mask=mask,
        sam_score=float(metadata["sam_score"]),
        prompt_name=metadata["prompt_name"],
        mask_index=int(metadata["mask_index"]),
    )
    return CandidateAssessment(
        candidate=candidate,
        quality=float(metadata["quality"]),
        q_semantic=float(metadata["q_semantic"]),
        q_stability=float(metadata["q_stability"]),
        q_edge=float(metadata["q_edge"]),
        q_safety=float(metadata["q_safety"]),
        valid=bool(metadata["valid"]),
    )


def _profile(mask: np.ndarray, expected_components: int) -> Dict:
    return component_profile(mask, expected_components=expected_components).as_dict()


def _mean(rows: Sequence[Dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.quality_window <= 1.0:
        raise ValueError("quality-window must be in [0,1]")

    cohort_path = args.cohort.resolve()
    candidate_artifact = args.candidate_artifact.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("generated_without_gt") is not True:
        raise ValueError("Cohort must declare generated_without_gt=true")
    audits = {
        row["image_name"]: row
        for row in (
            json.loads(line)
            for line in (candidate_artifact / "audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    }
    records = build_sample_records(
        dataset_dir=args.dataset_dir,
        dataset_names=args.dataset.split("+"),
        coarse_dir=args.coarse_dir,
        feature_dir=args.feature_dir,
    )
    experiment_dir = create_experiment_directory(
        args.artifact_root.resolve(), args.experiment_id
    )
    started_at = utc_timestamp()
    write_json(experiment_dir / "config.json", {
        "candidate_artifact": str(candidate_artifact),
        "cohort": str(cohort_path),
        "experiment_id": args.experiment_id,
        "generated_without_gt": True,
        "quality_window": args.quality_window,
        "started_at": started_at,
    })
    write_json(
        experiment_dir / "manifest.json",
        {"started_at": started_at, "status": "running"},
    )
    git_state = capture_git_state(PROJECT_ROOT, experiment_dir)

    candidate_rows: List[Dict] = []
    image_rows: List[Dict] = []
    input_records: List[Dict] = []
    for sample in cohort["samples"]:
        record = records[sample["index"]]
        audit = audits[record.image_name]
        if record.image_name != sample["image_name"]:
            raise ValueError(f"Cohort drift at index {record.index}")
        image = np.array(Image.open(record.image_path).convert("RGB"), copy=True)
        height, width = image.shape[:2]
        coarse = _resize_probability(
            load_coarse_mask(record.coarse_path), width, height
        )
        semantic_path = candidate_artifact / "semantic" / f"{record.image_name}.png"
        semantic = _load_probability(semantic_path)
        cache_path = candidate_artifact / "candidates" / f"{record.image_name}.npz"
        masks, cache = load_candidate_cache(cache_path)
        assessments = [
            _assessment(mask, metadata)
            for mask, metadata in zip(masks, cache["assessments"])
        ]
        coarse_initial = component_profile(
            quantized_binary(coarse), expected_components=0
        )
        expected_components = coarse_initial.effective_component_count
        coarse_profile = _profile(quantized_binary(coarse), expected_components)
        constraint = build_boundary_constraint(coarse, semantic, audit["route"])

        selected_metadata = audit.get("selected")
        selected_key = None
        if selected_metadata is not None:
            selected_key = (
                selected_metadata["prompt_name"],
                int(selected_metadata["mask_index"]),
            )
        per_image_rows: List[Dict] = []
        for assessment in assessments:
            result = fuse_boundary_residual(
                assessment,
                assessments,
                coarse,
                semantic,
                image,
                constraint,
            )
            candidate_profile = _profile(
                assessment.candidate.mask, expected_components
            )
            fused_profile = _profile(
                quantized_binary(result.refined), expected_components
            )
            threshold_growth = {}
            for threshold in (0.45, 0.50, 0.55):
                if threshold == 0.50:
                    coarse_binary = quantized_binary(coarse)
                    refined_binary = quantized_binary(result.refined)
                else:
                    coarse_binary = coarse > threshold
                    refined_binary = result.refined > threshold
                coarse_at_threshold = component_profile(
                    coarse_binary, expected_components=0
                )
                refined_at_threshold = component_profile(
                    refined_binary,
                    expected_components=coarse_at_threshold.effective_component_count,
                )
                threshold_growth[f"growth_{threshold:.2f}"] = (
                    refined_at_threshold.effective_component_count
                    - coarse_at_threshold.effective_component_count
                )
            key = (
                assessment.candidate.prompt_name,
                assessment.candidate.mask_index,
            )
            row = {
                "candidate_effective_components": candidate_profile[
                    "effective_component_count"
                ],
                "candidate_extra_mass_ratio": candidate_profile[
                    "extra_component_mass_ratio"
                ],
                "candidate_raw_components": candidate_profile["component_count"],
                "coarse_effective_components": coarse_profile[
                    "effective_component_count"
                ],
                "dataset": record.dataset,
                "fused_effective_components": fused_profile[
                    "effective_component_count"
                ],
                "fused_extra_mass_ratio": fused_profile[
                    "extra_component_mass_ratio"
                ],
                "fused_growth": (
                    fused_profile["effective_component_count"]
                    - coarse_profile["effective_component_count"]
                ),
                "image_name": record.image_name,
                "index": record.index,
                "is_selected": key == selected_key,
                "mask_index": assessment.candidate.mask_index,
                "prompt_name": assessment.candidate.prompt_name,
                "quality": assessment.quality,
                "route": audit["route"],
                "valid": assessment.valid,
                **threshold_growth,
            }
            candidate_rows.append(row)
            per_image_rows.append(row)

        selected_row = next(
            (row for row in per_image_rows if row["is_selected"]), None
        )
        alternative = None
        if selected_row is not None:
            eligible = [
                row for row in per_image_rows
                if row["valid"]
                and row["quality"] >= selected_row["quality"] - args.quality_window
            ]
            alternative = min(
                eligible,
                key=lambda row: (
                    row["fused_growth"],
                    row["fused_extra_mass_ratio"],
                    -row["quality"],
                ),
            )
        image_rows.append({
            "alternative": alternative,
            "candidate_count": len(per_image_rows),
            "coarse_profile": coarse_profile,
            "dataset": record.dataset,
            "image_name": record.image_name,
            "route": audit["route"],
            "selected": selected_row,
        })
        input_records.append({
            "candidate_cache": file_record(cache_path),
            "coarse": file_record(record.coarse_path),
            "image": file_record(record.image_path),
            "semantic": file_record(semantic_path),
        })

    fieldnames = list(candidate_rows[0].keys())
    with (experiment_dir / "candidate_topology.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    write_json(experiment_dir / "image_summary.json", image_rows)

    selected_rows = [row for row in candidate_rows if row["is_selected"]]
    summaries = {}
    for dataset_name in ["ALL", *args.dataset.split("+")]:
        rows = (
            selected_rows if dataset_name == "ALL"
            else [row for row in selected_rows if row["dataset"] == dataset_name]
        )
        summaries[dataset_name] = {
            "candidate_extra_mass_ratio_mean": _mean(
                rows, "candidate_extra_mass_ratio"
            ),
            "fused_extra_mass_ratio_mean": _mean(
                rows, "fused_extra_mass_ratio"
            ),
            "fused_growth_mean": _mean(rows, "fused_growth"),
            "growth_0.45_mean": _mean(rows, "growth_0.45"),
            "growth_0.50_mean": _mean(rows, "growth_0.50"),
            "growth_0.55_mean": _mean(rows, "growth_0.55"),
            "sample_count": len(rows),
        }
    alternative_count = sum(
        row["selected"] is not None
        and row["alternative"] is not None
        and row["alternative"]["fused_growth"] < row["selected"]["fused_growth"]
        for row in image_rows
    )
    summary = {
        "alternative_with_lower_growth_count": alternative_count,
        "candidate_count": len(candidate_rows),
        "generated_without_gt": True,
        "hypothesis_indicators": summaries,
        "image_count": len(image_rows),
        "selected_candidate_count": len(selected_rows),
    }
    write_json(experiment_dir / "summary.json", summary)
    source_paths = [
        PROJECT_ROOT / "aeem_v2/topology.py",
        PROJECT_ROOT / "experiments/analyze_aeem_candidate_topology.py",
    ]
    write_json(experiment_dir / "input_hashes.json", {
        "candidate_artifact_manifest": file_record(
            candidate_artifact / "manifest.json"
        ),
        "cohort": file_record(cohort_path),
        "records": input_records,
        "sources": [file_record(path) for path in source_paths],
    })
    manifest = {
        "candidate_count": len(candidate_rows),
        "completed_at": utc_timestamp(),
        "experiment_id": args.experiment_id,
        "git": git_state,
        "image_count": len(image_rows),
        "input_hashes_sha256": sha256_file(experiment_dir / "input_hashes.json"),
        "started_at": started_at,
        "status": "complete",
        "summary_sha256": sha256_file(experiment_dir / "summary.json"),
    }
    write_json(experiment_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(experiment_dir.resolve())


if __name__ == "__main__":
    main()
