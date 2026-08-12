"""Compose dataset-isolated AEEM training-label artifacts."""

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from .artifacts import (
    capture_git_state,
    create_experiment_directory,
    file_record,
    sha256_file,
    utc_timestamp,
    write_json,
)


def _load_samples(cohort_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    samples = cohort.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Cohort has no samples: {cohort_path}")
    if cohort.get("cohort_size") != len(samples):
        raise ValueError(
            f"Cohort size mismatch: {cohort.get('cohort_size')} != {len(samples)}"
        )

    image_names = [str(sample.get("image_name", "")) for sample in samples]
    if any(
        not image_name
        or Path(image_name).name != image_name
        or image_name in {".", ".."}
        for image_name in image_names
    ):
        raise ValueError("Cohort contains an invalid image name")
    if len(image_names) != len(set(image_names)):
        raise ValueError("Cohort image names must be unique")
    if any(not sample.get("dataset") for sample in samples):
        raise ValueError("Every cohort sample must include its dataset")
    return cohort, samples


def select_top_fraction_from_audit(
    audit_path: Path,
    dataset: str,
    score_field: str,
    fraction: float,
) -> Tuple[Set[str], Dict[str, Any]]:
    """Select a deterministic top fraction using only frozen audit signals."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    if not dataset_rows:
        raise ValueError(f"No audit rows found for dataset: {dataset}")

    scored: List[Tuple[float, str]] = []
    for row in dataset_rows:
        image_name = str(row.get("image_name", ""))
        value: Any = row
        for part in score_field.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Audit score is not numeric for {image_name}: {score_field}"
            ) from exc
        if not image_name:
            raise ValueError("Audit row has an empty image_name")
        scored.append((score, image_name))

    if len(scored) != len({image_name for _, image_name in scored}):
        raise ValueError(f"Duplicate scored image names in audit: {audit_path}")
    target_count = round(len(dataset_rows) * fraction)
    target_count = max(1, min(target_count, len(scored)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:target_count]
    metadata = {
        "audit_path": str(audit_path.resolve()),
        "dataset": dataset,
        "fraction": fraction,
        "record_count": len(dataset_rows),
        "scored_count": len(scored),
        "score_field": score_field,
        "selected_count": len(selected),
        "selected_score_max": selected[0][0],
        "selected_score_min": selected[-1][0],
    }
    return {image_name for _, image_name in selected}, metadata


def _validate_sources(
    samples: Sequence[Dict[str, Any]],
    aeem_dir: Path,
    soft_dir: Path,
    aeem_datasets: set[str],
    aeem_image_names: Set[str],
) -> List[Dict[str, Any]]:
    cohort_datasets = {str(sample["dataset"]) for sample in samples}
    unknown_datasets = aeem_datasets.difference(cohort_datasets)
    if unknown_datasets:
        raise ValueError(
            "AEEM datasets are not present in the cohort: "
            + ", ".join(sorted(unknown_datasets))
        )
    cohort_image_names = {str(sample["image_name"]) for sample in samples}
    unknown_images = aeem_image_names.difference(cohort_image_names)
    if unknown_images:
        raise ValueError(
            "AEEM image names are not present in the cohort: "
            + ", ".join(sorted(unknown_images)[:5])
        )

    selections: List[Dict[str, Any]] = []
    missing_paths: List[Path] = []
    for sample in samples:
        dataset = str(sample["dataset"])
        image_name = str(sample["image_name"])
        source_type = (
            "aeem"
            if dataset in aeem_datasets or image_name in aeem_image_names
            else "soft"
        )
        source_dir = aeem_dir if source_type == "aeem" else soft_dir
        source_path = source_dir / f"{image_name}.png"
        if not source_path.is_file():
            missing_paths.append(source_path)
        selections.append({
            "dataset": dataset,
            "image_name": image_name,
            "source_path": source_path,
            "source_type": source_type,
        })

    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} selected label files; first paths: {preview}"
        )
    return selections


def compose_label_artifact(
    artifact_root: Path,
    experiment_id: str,
    cohort_path: Path,
    aeem_dir: Path,
    soft_dir: Path,
    aeem_datasets: Iterable[str],
    repo_root: Path,
    source_files: Sequence[Path] = (),
    aeem_image_names: Iterable[str] = (),
    selection_metadata: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
) -> Path:
    """Build one immutable label directory using AEEM only for selected datasets."""
    cohort_path = cohort_path.resolve()
    aeem_dir = aeem_dir.resolve()
    soft_dir = soft_dir.resolve()
    repo_root = repo_root.resolve()
    selected_datasets = {str(dataset) for dataset in aeem_datasets}
    selected_images = {str(image_name) for image_name in aeem_image_names}
    cohort, samples = _load_samples(cohort_path)
    selections = _validate_sources(
        samples=samples,
        aeem_dir=aeem_dir,
        soft_dir=soft_dir,
        aeem_datasets=selected_datasets,
        aeem_image_names=selected_images,
    )

    experiment_dir = create_experiment_directory(artifact_root, experiment_id)
    output_dir = experiment_dir / "refined_pseudo_labels"
    output_dir.mkdir()
    started_at = utc_timestamp()

    config = {
        "aeem_datasets": sorted(selected_datasets),
        "aeem_image_selection": selection_metadata,
        "aeem_dir": str(aeem_dir),
        "artifact_type": "aeem_v2_dataset_source_isolation",
        "cohort": str(cohort_path),
        "experiment_id": experiment_id,
        "platform": platform.platform(),
        "python": sys.version,
        "soft_dir": str(soft_dir),
        "started_at": started_at,
        "training_contract": {
            "cache_dir": "./datasets/cache",
            "refined_pseudo_label_dir": str(output_dir.resolve()),
        },
    }
    write_json(experiment_dir / "config.json", config)
    write_json(
        experiment_dir / "manifest.json",
        {"experiment_id": experiment_id, "started_at": started_at, "status": "running"},
    )
    git_state = capture_git_state(repo_root, experiment_dir)

    input_records: List[Dict[str, Any]] = []
    output_records: List[Dict[str, Any]] = []
    source_counts = {"aeem": 0, "soft": 0}
    audit_path = experiment_dir / "audit.jsonl"
    iterator = tqdm(
        selections,
        desc="Composing labels",
        unit="label",
        disable=not show_progress,
    )
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for selection in iterator:
            source_path = selection["source_path"]
            output_path = output_dir / f"{selection['image_name']}.png"
            source_record = file_record(source_path)
            shutil.copyfile(source_path, output_path)
            output_record = file_record(output_path)
            if source_record["sha256"] != output_record["sha256"]:
                raise OSError(f"Hash mismatch after copying {source_path}")

            source_counts[selection["source_type"]] += 1
            input_records.append({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "source": source_record,
                "source_type": selection["source_type"],
            })
            output_records.append({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "output": output_record,
                "source_type": selection["source_type"],
            })
            audit_file.write(json.dumps({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "output_sha256": output_record["sha256"],
                "source_sha256": source_record["sha256"],
                "source_type": selection["source_type"],
            }, ensure_ascii=False, sort_keys=True) + "\n")

    source_records = [file_record(path.resolve()) for path in source_files]
    write_json(
        experiment_dir / "input_hashes.json",
        {
            "cohort": file_record(cohort_path),
            "records": input_records,
            "sources": source_records,
        },
    )
    write_json(experiment_dir / "output_hashes.json", {"records": output_records})

    dataset_counts = {
        dataset: sum(selection["dataset"] == dataset for selection in selections)
        for dataset in sorted({selection["dataset"] for selection in selections})
    }
    completed_at = utc_timestamp()
    manifest = {
        "aeem_datasets": sorted(selected_datasets),
        "completed_at": completed_at,
        "dataset_counts": dataset_counts,
        "experiment_id": experiment_id,
        "git": git_state,
        "input_count": len(selections),
        "output_count": len(list(output_dir.glob("*.png"))),
        "output_hashes_sha256": sha256_file(experiment_dir / "output_hashes.json"),
        "source_counts": source_counts,
        "explicit_aeem_image_count": len(selected_images),
        "started_at": started_at,
        "status": "complete",
    }
    if manifest["output_count"] != len(samples):
        raise OSError(
            f"Expected {len(samples)} output labels, found {manifest['output_count']}"
        )
    write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir
