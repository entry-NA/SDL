"""Hard- and soft-coarse representation controls for AEEM v2."""

import json
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .artifacts import (
    capture_git_state,
    create_experiment_directory,
    file_record,
    sha256_file,
    utc_timestamp,
    write_json,
)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_image_directory(dataset_root: Path) -> Path:
    for name in ("im", "Image", "images", "JPEGImages"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No image directory found under {dataset_root}")


def discover_images(dataset_dir: Path, dataset_names: Sequence[str]) -> List[Path]:
    image_paths: List[Path] = []
    for dataset_name in dataset_names:
        image_dir = _find_image_directory(dataset_dir / dataset_name)
        image_paths.extend(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    image_paths = sorted(image_paths)
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Image stems must be unique across the selected datasets")
    return image_paths


def load_coarse_mask(path: Path) -> np.ndarray:
    with path.open("rb") as file_handle:
        coarse = pickle.load(file_handle)
    if hasattr(coarse, "detach"):
        coarse = coarse.detach()
    if hasattr(coarse, "cpu"):
        coarse = coarse.cpu()
    if hasattr(coarse, "numpy"):
        coarse = coarse.numpy()
    mask = np.asarray(coarse, dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D coarse mask, got shape {mask.shape}: {path}")
    return np.clip(mask, 0.0, 1.0)


def _load_index(coarse_dir: Path, image_count: int) -> Tuple[Path, List[Path]]:
    index_path = coarse_dir / "index.json"
    index_map = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index_map) != image_count:
        raise ValueError(
            f"Count mismatch: {image_count} images vs {len(index_map)} coarse labels"
        )
    coarse_paths = []
    for index in range(image_count):
        key = str(index)
        if key not in index_map:
            raise ValueError(f"Missing index key {key} in {index_path}")
        coarse_path = coarse_dir / index_map[key]
        if not coarse_path.is_file():
            raise FileNotFoundError(coarse_path)
        coarse_paths.append(coarse_path)
    return index_path, coarse_paths


def _save_png(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask, mode="L").save(path)


def _control_masks(
    coarse: np.ndarray,
    image_size: Tuple[int, int],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    soft = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    soft = np.clip(soft, 0.0, 1.0)
    soft_u8 = np.rint(soft * 255.0).astype(np.uint8)
    hard_u8 = (soft > threshold).astype(np.uint8) * 255
    return hard_u8, soft_u8


def generate_control_artifact(
    artifact_root: Path,
    experiment_id: str,
    dataset_dir: Path,
    dataset_names: Sequence[str],
    coarse_dir: Path,
    repo_root: Path,
    threshold: float = 0.5,
    source_files: Sequence[Path] = (),
) -> Path:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    dataset_dir = dataset_dir.resolve()
    coarse_dir = coarse_dir.resolve()
    repo_root = repo_root.resolve()
    image_paths = discover_images(dataset_dir, dataset_names)
    index_path, coarse_paths = _load_index(coarse_dir, len(image_paths))
    experiment_dir = create_experiment_directory(artifact_root, experiment_id)

    hard_dir = experiment_dir / "controls" / "hard_coarse" / "refined_pseudo_labels"
    soft_dir = experiment_dir / "controls" / "soft_coarse" / "refined_pseudo_labels"
    hard_dir.mkdir(parents=True)
    soft_dir.mkdir(parents=True)

    started_at = utc_timestamp()
    config = {
        "artifact_type": "aeem_v2_representation_controls",
        "coarse_dir": str(coarse_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_names": list(dataset_names),
        "experiment_id": experiment_id,
        "interpolation": "cv2.INTER_LINEAR",
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": started_at,
        "threshold": threshold,
        "training_contract": {
            "cache_dir": "./datasets/cache",
            "hard_refined_pseudo_label_dir": str(hard_dir.resolve()),
            "soft_refined_pseudo_label_dir": str(soft_dir.resolve()),
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
    audit_path = experiment_dir / "audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for index, (image_path, coarse_path) in enumerate(zip(image_paths, coarse_paths)):
            with Image.open(image_path) as image:
                image_size = image.size
            coarse = load_coarse_mask(coarse_path)
            hard_mask, soft_mask = _control_masks(coarse, image_size, threshold)

            hard_path = hard_dir / f"{image_path.stem}.png"
            soft_path = soft_dir / f"{image_path.stem}.png"
            _save_png(hard_mask, hard_path)
            _save_png(soft_mask, soft_path)

            dataset_name = image_path.parent.parent.name
            input_records.append({
                "coarse": file_record(coarse_path),
                "dataset": dataset_name,
                "image": file_record(image_path),
                "image_name": image_path.stem,
                "index": index,
            })
            output_records.append({
                "hard": file_record(hard_path),
                "image_name": image_path.stem,
                "soft": file_record(soft_path),
            })
            audit_file.write(json.dumps({
                "coarse_max": float(coarse.max()),
                "coarse_min": float(coarse.min()),
                "dataset": dataset_name,
                "hard_foreground_ratio": float((hard_mask > 127).mean()),
                "image_name": image_path.stem,
                "image_size": list(image_size),
                "soft_foreground_mean": float(soft_mask.mean() / 255.0),
                "soft_transition_ratio": float(
                    np.logical_and(soft_mask > 25, soft_mask < 230).mean()
                ),
            }, ensure_ascii=False, sort_keys=True) + "\n")

    source_records = [file_record(path.resolve()) for path in source_files]
    input_payload = {
        "index": file_record(index_path),
        "records": input_records,
        "sources": source_records,
    }
    write_json(experiment_dir / "input_hashes.json", input_payload)
    write_json(experiment_dir / "output_hashes.json", {"records": output_records})

    completed_at = utc_timestamp()
    manifest = {
        "completed_at": completed_at,
        "dataset_counts": {
            dataset_name: sum(
                path.parent.parent.name == dataset_name for path in image_paths
            )
            for dataset_name in dataset_names
        },
        "experiment_id": experiment_id,
        "git": git_state,
        "hard_output_count": len(list(hard_dir.glob("*.png"))),
        "input_count": len(image_paths),
        "output_hashes_sha256": sha256_file(experiment_dir / "output_hashes.json"),
        "soft_output_count": len(list(soft_dir.glob("*.png"))),
        "started_at": started_at,
        "status": "complete",
    }
    write_json(experiment_dir / "manifest.json", manifest)
    return experiment_dir
