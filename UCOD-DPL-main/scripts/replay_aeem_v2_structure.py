"""Replay cached AEEM v2 candidates with no-GT topology safety."""

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
from aeem_v2.dataset import (
    build_sample_records,
    load_coarse_mask,
    load_feature_map,
)
from aeem_v2.refinement import (
    CandidateAssessment,
    MaskCandidate,
    build_boundary_constraint,
    load_candidate_cache,
)
from aeem_v2.structure import apply_structure_calibration
from aeem_v2.topology import component_profile, quantized_binary
from aeem_v2.semantic import compute_semantic_localization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay cached AEEM candidates without SAM2 or GT access."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/aeem_v2/structure_replays"),
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
    parser.add_argument("--maximum-effective-component-growth", type=int, default=1)
    parser.add_argument("--maximum-extra-mass-ratio", type=float, default=0.05)
    return parser.parse_args()


def _resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.clip(cv2.resize(
        probability.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ), 0.0, 1.0)


def _save_probability(path: Path, probability: np.ndarray) -> None:
    pixels = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _assessment(mask: np.ndarray, metadata: Dict) -> CandidateAssessment:
    return CandidateAssessment(
        candidate=MaskCandidate(
            mask=mask,
            sam_score=float(metadata["sam_score"]),
            prompt_name=metadata["prompt_name"],
            mask_index=int(metadata["mask_index"]),
        ),
        quality=float(metadata["quality"]),
        q_semantic=float(metadata["q_semantic"]),
        q_stability=float(metadata["q_stability"]),
        q_edge=float(metadata["q_edge"]),
        q_safety=float(metadata["q_safety"]),
        valid=bool(metadata["valid"]),
    )


def _selected_assessment(
    assessments: Sequence[CandidateAssessment],
    selected_metadata: Optional[Dict],
) -> Optional[CandidateAssessment]:
    if selected_metadata is None:
        return None
    selected_key = (
        selected_metadata["prompt_name"],
        int(selected_metadata["mask_index"]),
    )
    matches = [
        assessment
        for assessment in assessments
        if (
            assessment.candidate.prompt_name,
            assessment.candidate.mask_index,
        ) == selected_key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected candidate for key {selected_key}")
    return matches[0]


def _mean(rows: Sequence[Dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    if args.maximum_effective_component_growth < 0:
        raise ValueError("maximum-effective-component-growth must be non-negative")
    if not 0.0 <= args.maximum_extra_mass_ratio <= 1.0:
        raise ValueError("maximum-extra-mass-ratio must be in [0,1]")

    cohort_path = args.cohort.resolve()
    candidate_artifact = args.candidate_artifact.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("generated_without_gt") is not True:
        raise ValueError("Cohort must declare generated_without_gt=true")
    if cohort.get("cohort_size") != len(cohort.get("samples", [])):
        raise ValueError("Cohort size does not match its sample list")
    if cohort.get("dataset") != args.dataset:
        raise ValueError(f"Cohort dataset does not match {args.dataset}")

    source_manifest = json.loads(
        (candidate_artifact / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "complete":
        raise ValueError("Candidate artifact is not complete")
    source_audits = {
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
    output_dirs = {
        "backbone": experiment_dir / "connectivity_backbone",
        "cleaned": experiment_dir / "cleaned_selected_candidates",
        "confidence": experiment_dir / "confidence",
        "refined": experiment_dir / "refined_pseudo_labels",
    }
    for directory in output_dirs.values():
        directory.mkdir()

    started_at = utc_timestamp()
    write_json(experiment_dir / "config.json", {
        "artifact_type": "aeem_v2_cached_structure_replay",
        "candidate_artifact": str(candidate_artifact),
        "cohort": str(cohort_path),
        "dataset": args.dataset,
        "encoded_image_count": 0,
        "experiment_id": args.experiment_id,
        "generated_without_gt": True,
        "maximum_effective_component_growth": (
            args.maximum_effective_component_growth
        ),
        "maximum_extra_mass_ratio": args.maximum_extra_mass_ratio,
        "platform": platform.platform(),
        "python": sys.version,
        "started_at": started_at,
    })
    write_json(
        experiment_dir / "manifest.json",
        {"started_at": started_at, "status": "running"},
    )
    git_state = capture_git_state(PROJECT_ROOT, experiment_dir)

    audit_rows: List[Dict] = []
    input_records: List[Dict] = []
    output_records: List[Dict] = []
    try:
        for position, sample in enumerate(cohort["samples"]):
            record = records[sample["index"]]
            if record.image_name != sample["image_name"] or record.dataset != sample["dataset"]:
                raise ValueError(f"Frozen cohort drift at index {record.index}")
            source_audit = source_audits[record.image_name]
            if source_audit["route"] != sample["route"]:
                raise ValueError(f"Route drift for {record.image_name}")

            image = np.asarray(Image.open(record.image_path).convert("RGB"))
            height, width = image.shape[:2]
            coarse_low = load_coarse_mask(record.coarse_path)
            coarse = _resize_probability(coarse_low, width, height)
            semantic_path = (
                candidate_artifact / "semantic" / f"{record.image_name}.png"
            )
            semantic_low_path = (
                candidate_artifact
                / "semantic_probability_low"
                / f"{record.image_name}.npy"
            )
            if semantic_low_path.is_file():
                semantic_low = np.load(semantic_low_path, allow_pickle=False)
                semantic_source_path = semantic_low_path
            else:
                localization = compute_semantic_localization(
                    coarse_low, load_feature_map(record.feature_path)
                )
                if localization.route != source_audit["route"]:
                    raise ValueError(f"Route drift for {record.image_name}")
                semantic_low = localization.probability
                semantic_source_path = record.feature_path
            semantic = _resize_probability(semantic_low, width, height)
            cache_path = (
                candidate_artifact / "candidates" / f"{record.image_name}.npz"
            )
            masks, cache = load_candidate_cache(cache_path)
            assessments = [
                _assessment(mask, metadata)
                for mask, metadata in zip(masks, cache["assessments"])
            ]
            selected = _selected_assessment(
                assessments, source_audit.get("selected")
            )
            constraint = build_boundary_constraint(
                coarse, semantic, source_audit["route"]
            )
            calibrated = apply_structure_calibration(
                selected,
                assessments,
                coarse,
                semantic,
                image,
                constraint,
                maximum_effective_component_growth=(
                    args.maximum_effective_component_growth
                ),
                maximum_extra_mass_ratio=args.maximum_extra_mass_ratio,
            )
            result = calibrated.fusion

            refined_path = output_dirs["refined"] / f"{record.image_name}.png"
            confidence_path = output_dirs["confidence"] / f"{record.image_name}.png"
            backbone_path = output_dirs["backbone"] / f"{record.image_name}.png"
            cleaned_path = output_dirs["cleaned"] / f"{record.image_name}.png"
            _save_probability(refined_path, result.refined)
            _save_probability(confidence_path, result.confidence)
            _save_probability(
                backbone_path, calibrated.connectivity_backbone.astype(np.float32)
            )
            cleaned_mask = (
                calibrated.cleaned_selected.candidate.mask
                if calibrated.cleaned_selected is not None
                else np.zeros_like(coarse, dtype=bool)
            )
            _save_probability(cleaned_path, cleaned_mask.astype(np.float32))

            coarse_profile = component_profile(
                quantized_binary(coarse), expected_components=0
            )
            expected_components = coarse_profile.effective_component_count
            refined_profile = component_profile(
                quantized_binary(result.refined),
                expected_components=expected_components,
            )
            final_growth = (
                refined_profile.effective_component_count
                - coarse_profile.effective_component_count
            )
            outside_band_exact = bool(np.array_equal(
                result.refined[~constraint.uncertainty_band],
                coarse[~constraint.uncertainty_band],
            ))
            foreground_core_exact = bool(np.array_equal(
                result.refined[constraint.foreground_core],
                coarse[constraint.foreground_core],
            ))
            background_core_exact = bool(np.array_equal(
                result.refined[constraint.background_core],
                coarse[constraint.background_core],
            ))
            low_route_exact = bool(
                source_audit["route"] != "low"
                or np.array_equal(result.refined, coarse)
            )
            cleanup = calibrated.cleanup
            audit = {
                "background_core_exact": background_core_exact,
                "backbone_ratio": float(calibrated.connectivity_backbone.mean()),
                "candidate_count": len(assessments),
                "cleanup": None if cleanup is None else {
                    "component_count_after": cleanup.component_count_after,
                    "component_count_before": cleanup.component_count_before,
                    "removed_area_ratio": cleanup.removed_area_ratio,
                    "removed_component_count": cleanup.removed_component_count,
                    "risk_detected": cleanup.risk_detected,
                },
                "dataset": record.dataset,
                "final_effective_component_growth": final_growth,
                "final_extra_component_mass_ratio": (
                    refined_profile.extra_component_mass_ratio
                ),
                "foreground_core_exact": foreground_core_exact,
                "image_name": record.image_name,
                "index": record.index,
                "low_route_exact": low_route_exact,
                "outside_band_exact": outside_band_exact,
                "pre_fallback_effective_component_growth": (
                    calibrated.pre_fallback_effective_component_growth
                ),
                "route": source_audit["route"],
                "selected": (
                    calibrated.cleaned_selected.as_dict()
                    if calibrated.cleaned_selected is not None
                    else None
                ),
                "source_fallback_reason": source_audit.get("fallback_reason"),
                "structure_fallback_reason": calibrated.fallback_reason,
            }
            audit_rows.append(audit)
            print(
                f"[{position + 1:02d}/{len(cohort['samples']):02d}] "
                f"{record.image_name} route={source_audit['route']} "
                f"growth={calibrated.pre_fallback_effective_component_growth}"
                f"->{final_growth} fallback={calibrated.fallback_reason}"
            )
            input_records.append({
                "candidate_cache": file_record(cache_path),
                "coarse": file_record(record.coarse_path),
                "image": file_record(record.image_path),
                "semantic": file_record(semantic_path),
                "semantic_float_source": file_record(semantic_source_path),
            })
            output_records.append({
                "backbone": file_record(backbone_path),
                "cleaned_candidate": file_record(cleaned_path),
                "confidence": file_record(confidence_path),
                "image_name": record.image_name,
                "refined": file_record(refined_path),
            })

        with (experiment_dir / "audit.jsonl").open("w", encoding="utf-8") as file_handle:
            for row in audit_rows:
                file_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

        selected_rows = [row for row in audit_rows if row["selected"] is not None]
        summaries = {}
        for dataset_name in ["ALL", *args.dataset.split("+")]:
            rows = (
                selected_rows
                if dataset_name == "ALL"
                else [row for row in selected_rows if row["dataset"] == dataset_name]
            )
            summaries[dataset_name] = {
                "final_effective_component_growth_mean": _mean(
                    rows, "final_effective_component_growth"
                ),
                "final_extra_component_mass_ratio_mean": _mean(
                    rows, "final_extra_component_mass_ratio"
                ),
                "pre_fallback_effective_component_growth_mean": _mean(
                    rows, "pre_fallback_effective_component_growth"
                ),
                "sample_count": len(rows),
                "structure_fallback_count": sum(
                    row["structure_fallback_reason"]
                    == "excess_effective_component_growth"
                    for row in rows
                ),
            }
        summary = {
            "cleanup_removed_component_count": sum(
                row["cleanup"]["removed_component_count"]
                for row in audit_rows
                if row["cleanup"] is not None
            ),
            "encoded_image_count": 0,
            "generated_without_gt": True,
            "image_count": len(audit_rows),
            "invariant_failure_count": sum(
                not (
                    row["outside_band_exact"]
                    and row["foreground_core_exact"]
                    and row["background_core_exact"]
                    and row["low_route_exact"]
                )
                for row in audit_rows
            ),
            "selected_candidate_count": len(selected_rows),
            "topology": summaries,
        }
        write_json(experiment_dir / "summary.json", summary)
        write_json(experiment_dir / "input_hashes.json", {
            "candidate_artifact_manifest": file_record(
                candidate_artifact / "manifest.json"
            ),
            "cohort": file_record(cohort_path),
            "records": input_records,
            "sources": [
                file_record(PROJECT_ROOT / path)
                for path in (
                    "aeem_v2/refinement.py",
                    "aeem_v2/structure.py",
                    "aeem_v2/topology.py",
                    "scripts/replay_aeem_v2_structure.py",
                )
            ],
        })
        write_json(
            experiment_dir / "output_hashes.json", {"records": output_records}
        )
        write_json(experiment_dir / "manifest.json", {
            "completed_at": utc_timestamp(),
            "encoded_image_count": 0,
            "experiment_id": args.experiment_id,
            "git": git_state,
            "input_count": len(audit_rows),
            "input_hashes_sha256": sha256_file(
                experiment_dir / "input_hashes.json"
            ),
            "output_count": len(list(output_dirs["refined"].glob("*.png"))),
            "output_hashes_sha256": sha256_file(
                experiment_dir / "output_hashes.json"
            ),
            "started_at": started_at,
            "status": "complete",
            "summary_sha256": sha256_file(experiment_dir / "summary.json"),
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(experiment_dir.resolve())
    except Exception as error:
        write_json(experiment_dir / "manifest.json", {
            "error": repr(error),
            "experiment_id": args.experiment_id,
            "failed_at": utc_timestamp(),
            "started_at": started_at,
            "status": "failed",
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
