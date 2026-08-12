"""Run a frozen AEEM v2 boundary-safe cohort."""

import argparse
import json
import platform
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


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
    SampleRecord,
    build_sample_records,
    load_coarse_mask,
    load_feature_map,
)
from aeem_v2.pipeline import ordered_staged_map
from aeem_v2.refinement import (
    FusionResult,
    MaskCandidate,
    PromptVariant,
    assess_candidates,
    build_boundary_constraint,
    build_prompt_variants,
    fuse_boundary_residual,
    save_candidate_cache,
    select_candidate,
)
from aeem_v2.sam2_adapter import SAM2Adapter
from aeem_v2.semantic import SemanticLocalization, compute_semantic_localization
from aeem_v2.structure import apply_structure_calibration


@dataclass(frozen=True)
class PreparedSample:
    position: int
    record: SampleRecord
    image: np.ndarray
    coarse: np.ndarray
    semantic: np.ndarray
    localization: SemanticLocalization
    prompts: Sequence[PromptVariant]


@dataclass(frozen=True)
class CompletedSample:
    position: int
    audit: Dict
    input_record: Dict
    output_record: Dict
    route: str
    fallback_reason: Optional[str]
    progress_detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen no-GT AEEM v2 cohort.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--cohort", type=Path, default=Path("experiments/aeem_v2_m1_cohort12.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/aeem_v2")
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-quality", type=float, default=0.35)
    parser.add_argument("--structure-calibration", action="store_true")
    parser.add_argument("--maximum-effective-component-growth", type=int, default=1)
    parser.add_argument("--maximum-extra-mass-ratio", type=float, default=0.05)
    parser.add_argument("--postprocess-workers", type=int, default=2)
    parser.add_argument("--pipeline-buffer", type=int, default=4)
    return parser.parse_args()


def _load_cohort(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("generated_without_gt") is not True:
        raise ValueError("Cohort must declare generated_without_gt=true")
    if payload.get("cohort_size") != len(payload.get("samples", [])):
        raise ValueError("Cohort size does not match its sample list")
    return payload


def _resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.clip(cv2.resize(
        probability.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ), 0.0, 1.0)


def _save_probability(path: Path, probability: np.ndarray) -> None:
    image = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _fallback(coarse: np.ndarray) -> FusionResult:
    coarse_binary = np.rint(np.clip(coarse, 0.0, 1.0) * 255.0).astype(np.uint8) > 127
    component_count, _ = cv2.connectedComponents(
        coarse_binary.astype(np.uint8), connectivity=8
    )
    component_count = max(int(component_count) - 1, 0)
    return FusionResult(
        refined=coarse.copy(),
        confidence=np.zeros_like(coarse, dtype=np.float32),
        changed_ratio=0.0,
        component_count_before=component_count,
        component_count_after=component_count,
    )


def _prepare_sample(
    item: Tuple[int, Dict, SampleRecord],
) -> PreparedSample:
    position, sample, record = item
    with Image.open(record.image_path) as source_image:
        image = np.array(source_image.convert("RGB"), dtype=np.uint8, copy=True)
    height, width = image.shape[:2]
    coarse_low = load_coarse_mask(record.coarse_path)
    feature = load_feature_map(record.feature_path)
    localization = compute_semantic_localization(coarse_low, feature)
    if localization.route != sample["route"]:
        raise ValueError(
            f"Route drift for {record.image_name}: "
            f"{sample['route']} -> {localization.route}"
        )
    coarse = _resize_probability(coarse_low, width, height)
    semantic = _resize_probability(localization.probability, width, height)
    prompts = build_prompt_variants(coarse, semantic, localization.route)
    return PreparedSample(
        position=position,
        record=record,
        image=image,
        coarse=coarse,
        semantic=semantic,
        localization=localization,
        prompts=prompts,
    )


def _finish_sample(
    prepared: PreparedSample,
    candidates: Sequence[MaskCandidate],
    output_dirs: Dict[str, Path],
    minimum_quality: float,
    structure_calibration: bool,
    maximum_effective_component_growth: int,
    maximum_extra_mass_ratio: float,
) -> CompletedSample:
    assessments = assess_candidates(
        candidates,
        prepared.coarse,
        prepared.semantic,
        prepared.image,
    )
    selected = select_candidate(assessments, minimum_quality=minimum_quality)
    constraint = build_boundary_constraint(
        prepared.coarse,
        prepared.semantic,
        prepared.localization.route,
    )
    if selected is None:
        result = _fallback(prepared.coarse)
        fallback_reason = (
            "no_prompt" if not prepared.prompts else "no_valid_candidate"
        )
        structure = None
    elif structure_calibration:
        structure = apply_structure_calibration(
            selected,
            assessments,
            prepared.coarse,
            prepared.semantic,
            prepared.image,
            constraint,
            maximum_effective_component_growth=maximum_effective_component_growth,
            maximum_extra_mass_ratio=maximum_extra_mass_ratio,
        )
        result = structure.fusion
        fallback_reason = structure.fallback_reason
    else:
        result = fuse_boundary_residual(
            selected,
            assessments,
            prepared.coarse,
            prepared.semantic,
            prepared.image,
            constraint,
        )
        fallback_reason = None
        structure = None

    record = prepared.record
    refined_path = output_dirs["refined"] / f"{record.image_name}.png"
    confidence_path = output_dirs["confidence"] / f"{record.image_name}.png"
    semantic_path = output_dirs["semantic"] / f"{record.image_name}.png"
    semantic_low_path = output_dirs["semantic_low"] / f"{record.image_name}.npy"
    candidate_path = output_dirs["candidates"] / f"{record.image_name}.npz"
    _save_probability(refined_path, result.refined)
    _save_probability(confidence_path, result.confidence)
    _save_probability(semantic_path, prepared.semantic)
    np.save(
        semantic_low_path,
        prepared.localization.probability.astype(np.float32),
        allow_pickle=False,
    )
    save_candidate_cache(candidate_path, prepared.prompts, assessments)

    absolute_change = np.abs(result.refined - prepared.coarse)
    audit = {
        "band_ratio": float(constraint.uncertainty_band.mean()),
        "band_radius": constraint.radius,
        "candidate_count": len(candidates),
        "changed_ratio": result.changed_ratio,
        "changed_ratio_0_01": float((absolute_change > 0.01).mean()),
        "changed_ratio_0_05": float((absolute_change > 0.05).mean()),
        "changed_ratio_0_10": float((absolute_change > 0.10).mean()),
        "coarse_mean": float(prepared.coarse.mean()),
        "confidence_max": float(result.confidence.max()),
        "confidence_mean": float(result.confidence.mean()),
        "component_count_after": result.component_count_after,
        "component_count_before": result.component_count_before,
        "component_count_growth": (
            result.component_count_after - result.component_count_before
        ),
        "dataset": record.dataset,
        "fallback_reason": fallback_reason,
        "far_background_ratio": float(constraint.background_core.mean()),
        "foreground_core_ratio": float(constraint.foreground_core.mean()),
        "image_name": record.image_name,
        "image_size": [prepared.image.shape[1], prepared.image.shape[0]],
        "index": record.index,
        "localization_components": prepared.localization.components,
        "localization_reliability": prepared.localization.reliability,
        "mean_absolute_change": float(absolute_change.mean()),
        "prompt_count": len(prepared.prompts),
        "prompts": [prompt.as_dict() for prompt in prepared.prompts],
        "refined_mean": float(result.refined.mean()),
        "route": prepared.localization.route,
        "selected": selected.as_dict() if selected is not None else None,
        "structure": None if structure is None else {
            "backbone_ratio": float(structure.connectivity_backbone.mean()),
            "cleanup": {
                "component_count_after": structure.cleanup.component_count_after,
                "component_count_before": structure.cleanup.component_count_before,
                "removed_area_ratio": structure.cleanup.removed_area_ratio,
                "removed_component_count": structure.cleanup.removed_component_count,
                "risk_detected": structure.cleanup.risk_detected,
            },
            "pre_fallback_effective_component_growth": (
                structure.pre_fallback_effective_component_growth
            ),
        },
        "valid_candidate_count": sum(item.valid for item in assessments),
    }
    input_record = {
        "coarse": file_record(record.coarse_path),
        "dataset": record.dataset,
        "feature": file_record(record.feature_path),
        "image": file_record(record.image_path),
        "image_name": record.image_name,
        "index": record.index,
    }
    output_record = {
        "candidates": file_record(candidate_path),
        "confidence": file_record(confidence_path),
        "image_name": record.image_name,
        "refined": file_record(refined_path),
        "semantic": file_record(semantic_path),
        "semantic_low": file_record(semantic_low_path),
    }
    progress_detail = (
        f"{record.image_name} route={prepared.localization.route} "
        f"prompts={len(prepared.prompts)} candidates={len(candidates)} "
        f"changed={result.changed_ratio:.4f} "
        f"components={result.component_count_before}->{result.component_count_after}"
    )
    return CompletedSample(
        position=prepared.position,
        audit=audit,
        input_record=input_record,
        output_record=output_record,
        route=prepared.localization.route,
        fallback_reason=fallback_reason,
        progress_detail=progress_detail,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_quality <= 1.0:
        raise ValueError("minimum-quality must be in [0,1]")
    if args.maximum_effective_component_growth < 0:
        raise ValueError("maximum-effective-component-growth must be non-negative")
    if not 0.0 <= args.maximum_extra_mass_ratio <= 1.0:
        raise ValueError("maximum-extra-mass-ratio must be in [0,1]")
    if args.postprocess_workers <= 0:
        raise ValueError("postprocess-workers must be positive")
    if args.pipeline_buffer < args.postprocess_workers:
        raise ValueError("pipeline-buffer must be at least postprocess-workers")

    cohort_path = args.cohort.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cohort = _load_cohort(cohort_path)
    if cohort.get("dataset") != args.dataset:
        raise ValueError(
            f"Cohort dataset {cohort.get('dataset')} does not match {args.dataset}"
        )

    records = build_sample_records(
        dataset_dir=args.dataset_dir,
        dataset_names=args.dataset.split("+"),
        coarse_dir=args.coarse_dir,
        feature_dir=args.feature_dir,
    )
    samples = cohort["samples"]
    for sample in samples:
        record = records[sample["index"]]
        if record.image_name != sample["image_name"] or record.dataset != sample["dataset"]:
            raise ValueError(f"Frozen cohort no longer matches index {record.index}")

    experiment_dir = create_experiment_directory(
        args.artifact_root.resolve(), args.experiment_id
    )
    output_dirs = {
        "candidates": experiment_dir / "candidates",
        "confidence": experiment_dir / "confidence",
        "refined": experiment_dir / "refined_pseudo_labels",
        "semantic": experiment_dir / "semantic",
        "semantic_low": experiment_dir / "semantic_probability_low",
    }
    for directory in output_dirs.values():
        directory.mkdir()

    started_at = utc_timestamp()
    config = {
        "artifact_type": "aeem_v2_boundary_safe_mvp",
        "checkpoint": str(checkpoint_path),
        "coarse_dir": str(args.coarse_dir.resolve()),
        "cohort": str(cohort_path),
        "dataset": args.dataset,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "device": args.device,
        "experiment_id": args.experiment_id,
        "feature_dir": str(args.feature_dir.resolve()),
        "generated_without_gt": True,
        "maximum_effective_component_growth": (
            args.maximum_effective_component_growth
        ),
        "maximum_extra_mass_ratio": args.maximum_extra_mass_ratio,
        "minimum_quality": args.minimum_quality,
        "pipeline": {
            "max_pending": args.pipeline_buffer,
            "postprocess_workers": args.postprocess_workers,
            "prefetch_workers": 1,
        },
        "platform": platform.platform(),
        "python": sys.version,
        "sam2_config": args.sam2_config,
        "started_at": started_at,
        "structure_calibration": args.structure_calibration,
    }
    write_json(experiment_dir / "config.json", config)
    write_json(
        experiment_dir / "manifest.json",
        {"experiment_id": args.experiment_id, "started_at": started_at, "status": "running"},
    )
    git_state = capture_git_state(PROJECT_ROOT, experiment_dir)

    input_records: List[Dict] = []
    output_records: List[Dict] = []
    route_counts = {route: 0 for route in ("low", "medium", "high")}
    fallback_counts: Dict[str, int] = {}
    encoded_image_count = 0
    audit_path = experiment_dir / "audit.jsonl"

    try:
        adapter = SAM2Adapter(
            checkpoint_path=checkpoint_path,
            config_file=args.sam2_config,
            device=args.device,
        )
        pipeline_items = (
            (position, sample, records[sample["index"]])
            for position, sample in enumerate(samples)
        )

        def infer(prepared: PreparedSample) -> Sequence[MaskCandidate]:
            nonlocal encoded_image_count
            candidates = adapter.predict_candidates(
                prepared.image,
                prepared.prompts,
            )
            if prepared.prompts:
                encoded_image_count += 1
            return candidates

        def finish(
            prepared: PreparedSample,
            candidates: Sequence[MaskCandidate],
        ) -> CompletedSample:
            return _finish_sample(
                prepared,
                candidates,
                output_dirs=output_dirs,
                minimum_quality=args.minimum_quality,
                structure_calibration=args.structure_calibration,
                maximum_effective_component_growth=(
                    args.maximum_effective_component_growth
                ),
                maximum_extra_mass_ratio=args.maximum_extra_mass_ratio,
            )

        with audit_path.open("w", encoding="utf-8") as audit_file, tqdm(
            total=len(samples),
            desc="Refining",
            dynamic_ncols=True,
        ) as progress:
            for completed in ordered_staged_map(
                pipeline_items,
                prepare=_prepare_sample,
                infer=infer,
                finish=finish,
                finish_workers=args.postprocess_workers,
                max_pending=args.pipeline_buffer,
            ):
                route_counts[completed.route] += 1
                if completed.fallback_reason is not None:
                    fallback_counts[completed.fallback_reason] = (
                        fallback_counts.get(completed.fallback_reason, 0) + 1
                    )
                audit_file.write(
                    json.dumps(
                        completed.audit,
                        ensure_ascii=False,
                        sort_keys=True,
                    ) + "\n"
                )
                input_records.append(completed.input_record)
                output_records.append(completed.output_record)
                progress.set_postfix_str(completed.progress_detail, refresh=False)
                progress.update(1)

        source_paths = [
            PROJECT_ROOT / "aeem_v2/dataset.py",
            PROJECT_ROOT / "aeem_v2/pipeline.py",
            PROJECT_ROOT / "aeem_v2/refinement.py",
            PROJECT_ROOT / "aeem_v2/sam2_adapter.py",
            PROJECT_ROOT / "aeem_v2/semantic.py",
            PROJECT_ROOT / "aeem_v2/structure.py",
            PROJECT_ROOT / "aeem_v2/topology.py",
            PROJECT_ROOT / "scripts/run_aeem_v2_mvp.py",
        ]
        write_json(experiment_dir / "input_hashes.json", {
            "checkpoint": file_record(checkpoint_path),
            "cohort": file_record(cohort_path),
            "records": input_records,
            "sources": [file_record(path) for path in source_paths],
        })
        write_json(
            experiment_dir / "output_hashes.json", {"records": output_records}
        )
        manifest = {
            "candidate_cache_count": len(list(output_dirs["candidates"].glob("*.npz"))),
            "completed_at": utc_timestamp(),
            "encoded_image_count": encoded_image_count,
            "experiment_id": args.experiment_id,
            "fallback_counts": fallback_counts,
            "git": git_state,
            "input_count": len(samples),
            "output_count": len(list(output_dirs["refined"].glob("*.png"))),
            "output_hashes_sha256": sha256_file(experiment_dir / "output_hashes.json"),
            "route_counts": route_counts,
            "started_at": started_at,
            "status": "complete",
        }
        write_json(experiment_dir / "manifest.json", manifest)
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
