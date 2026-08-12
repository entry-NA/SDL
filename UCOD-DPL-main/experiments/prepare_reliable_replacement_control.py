"""Build the frozen reliable-replacement control label artifact."""

import argparse
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from aeem_v2.composition import _load_samples


def _load_selection_rows(selection_audit_path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in selection_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Selection audit has no rows: {selection_audit_path}")

    keys = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        image_name = str(row.get("image_name", ""))
        source_type = row.get("source_type")
        if not dataset or not image_name:
            raise ValueError("Every selection row must include dataset and image_name")
        if source_type not in {"aeem", "soft"}:
            raise ValueError(
                f"Unsupported frozen source_type for {dataset}/{image_name}: "
                f"{source_type!r}"
            )
        keys.append((dataset, image_name))
    if len(keys) != len(set(keys)):
        raise ValueError("Selection audit contains duplicate dataset/image_name rows")
    return rows


def _validate_selections(
    samples: Sequence[Dict[str, Any]],
    selection_rows: Sequence[Dict[str, Any]],
    aeem_dir: Path,
    naive_dir: Path,
    soft_dir: Path,
) -> List[Dict[str, Any]]:
    selection_by_key = {
        (str(row["dataset"]), str(row["image_name"])): row
        for row in selection_rows
    }
    cohort_keys = {
        (str(sample["dataset"]), str(sample["image_name"])) for sample in samples
    }
    selection_keys = set(selection_by_key)
    missing_rows = cohort_keys.difference(selection_keys)
    extra_rows = selection_keys.difference(cohort_keys)
    if missing_rows or extra_rows:
        raise ValueError(
            "Selection audit does not exactly match the cohort: "
            f"missing={len(missing_rows)}, extra={len(extra_rows)}"
        )

    selections: List[Dict[str, Any]] = []
    missing_paths: List[Path] = []
    for sample in samples:
        dataset = str(sample["dataset"])
        image_name = str(sample["image_name"])
        frozen_source = selection_by_key[(dataset, image_name)]["source_type"]
        if frozen_source == "aeem":
            source_type = "aeem"
            source_path = aeem_dir / f"{image_name}.png"
        else:
            naive_path = naive_dir / f"{image_name}.png"
            if naive_path.is_file():
                source_type = "naive_sam2"
                source_path = naive_path
            else:
                source_type = "soft_fallback"
                source_path = soft_dir / f"{image_name}.png"
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


def compose_reliable_replacement_artifact(
    artifact_root: Path,
    experiment_id: str,
    cohort_path: Path,
    selection_audit_path: Path,
    aeem_dir: Path,
    naive_dir: Path,
    soft_dir: Path,
    repo_root: Path,
    show_progress: bool = False,
) -> Path:
    """Compose AEEM/naive/Soft labels using an existing frozen selection audit."""
    cohort_path = cohort_path.resolve()
    selection_audit_path = selection_audit_path.resolve()
    aeem_dir = aeem_dir.resolve()
    naive_dir = naive_dir.resolve()
    soft_dir = soft_dir.resolve()
    repo_root = repo_root.resolve()

    cohort, samples = _load_samples(cohort_path)
    selection_rows = _load_selection_rows(selection_audit_path)
    selections = _validate_selections(
        samples=samples,
        selection_rows=selection_rows,
        aeem_dir=aeem_dir,
        naive_dir=naive_dir,
        soft_dir=soft_dir,
    )

    experiment_dir = create_experiment_directory(artifact_root, experiment_id)
    output_dir = experiment_dir / "refined_pseudo_labels"
    output_dir.mkdir()
    started_at = utc_timestamp()

    write_json(experiment_dir / "config.json", {
        "aeem_dir": str(aeem_dir),
        "artifact_type": "aeem_v2_reliable_replacement_control",
        "cohort": str(cohort_path),
        "experiment_id": experiment_id,
        "naive_dir": str(naive_dir),
        "platform": platform.platform(),
        "python": sys.version,
        "selection_audit": str(selection_audit_path),
        "soft_fallback_dir": str(soft_dir),
        "started_at": started_at,
        "training_contract": {
            "cache_dir": "./datasets/cache",
            "refined_pseudo_label_dir": str(output_dir.resolve()),
        },
    })
    write_json(experiment_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "started_at": started_at,
        "status": "running",
    })
    git_state = capture_git_state(repo_root, experiment_dir)

    input_records: List[Dict[str, Any]] = []
    output_records: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    audit_path = experiment_dir / "audit.jsonl"
    iterator = tqdm(
        selections,
        desc="Composing reliable replacement labels",
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

            source_type = selection["source_type"]
            source_counts[source_type] += 1
            input_records.append({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "source": source_record,
                "source_type": source_type,
            })
            output_records.append({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "output": output_record,
                "source_type": source_type,
            })
            audit_file.write(json.dumps({
                "dataset": selection["dataset"],
                "image_name": selection["image_name"],
                "output_sha256": output_record["sha256"],
                "source_sha256": source_record["sha256"],
                "source_type": source_type,
            }, ensure_ascii=False, sort_keys=True) + "\n")

    write_json(experiment_dir / "input_hashes.json", {
        "cohort": file_record(cohort_path),
        "records": input_records,
        "selection_audit": file_record(selection_audit_path),
        "sources": [
            file_record(Path(__file__).resolve()),
            file_record(repo_root / "aeem_v2" / "artifacts.py"),
        ],
    })
    write_json(experiment_dir / "output_hashes.json", {"records": output_records})

    output_count = len(list(output_dir.glob("*.png")))
    if output_count != len(samples):
        raise OSError(f"Expected {len(samples)} output labels, found {output_count}")
    dataset_counts = Counter(str(sample["dataset"]) for sample in samples)
    manifest = {
        "completed_at": utc_timestamp(),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "experiment_id": experiment_id,
        "git": git_state,
        "input_count": len(selections),
        "output_count": output_count,
        "output_hashes_sha256": sha256_file(experiment_dir / "output_hashes.json"),
        "source_counts": {
            name: source_counts.get(name, 0)
            for name in ("aeem", "naive_sam2", "soft_fallback")
        },
        "started_at": started_at,
        "status": "complete",
    }
    write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose the immutable AEEM-on-naive reliable replacement control."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("experiments/aeem_v2_m2_full4040.json"),
    )
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--aeem-dir", type=Path, required=True)
    parser.add_argument("--naive-dir", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/aeem_v2")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = compose_reliable_replacement_artifact(
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
        cohort_path=args.cohort,
        selection_audit_path=args.selection_audit,
        aeem_dir=args.aeem_dir,
        naive_dir=args.naive_dir,
        soft_dir=args.soft_dir,
        repo_root=PROJECT_ROOT,
        show_progress=True,
    )
    print(f"Reliable replacement artifact completed: {experiment_dir}")


if __name__ == "__main__":
    main()
