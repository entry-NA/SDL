"""Dataset index helpers shared by AEEM v2 offline stages."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .controls import discover_images, load_coarse_mask


@dataclass(frozen=True)
class SampleRecord:
    index: int
    dataset: str
    image_path: Path
    coarse_path: Path
    feature_path: Path

    @property
    def image_name(self) -> str:
        return self.image_path.stem


def _indexed_paths(directory: Path, expected_count: int) -> List[Path]:
    index_path = directory / "index.json"
    index_map = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index_map) != expected_count:
        raise ValueError(
            f"Count mismatch: expected {expected_count}, found {len(index_map)} in {index_path}"
        )
    paths = []
    for index in range(expected_count):
        key = str(index)
        if key not in index_map:
            raise ValueError(f"Missing index key {key} in {index_path}")
        path = directory / index_map[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_sample_records(
    dataset_dir: Path,
    dataset_names: Sequence[str],
    coarse_dir: Path,
    feature_dir: Path,
) -> List[SampleRecord]:
    image_paths = discover_images(dataset_dir.resolve(), dataset_names)
    coarse_paths = _indexed_paths(coarse_dir.resolve(), len(image_paths))
    feature_paths = _indexed_paths(feature_dir.resolve(), len(image_paths))
    return [
        SampleRecord(
            index=index,
            dataset=image_path.parent.parent.name,
            image_path=image_path,
            coarse_path=coarse_path,
            feature_path=feature_path,
        )
        for index, (image_path, coarse_path, feature_path) in enumerate(
            zip(image_paths, coarse_paths, feature_paths)
        )
    ]


def load_feature_map(path: Path) -> np.ndarray:
    with path.open("rb") as file_handle:
        feature = pickle.load(file_handle)
    if hasattr(feature, "detach"):
        feature = feature.detach()
    if hasattr(feature, "cpu"):
        feature = feature.cpu()
    if hasattr(feature, "numpy"):
        feature = feature.numpy()
    feature_array = np.asarray(feature, dtype=np.float32).squeeze()
    if feature_array.ndim != 3:
        raise ValueError(f"Expected a 3D feature map, got {feature_array.shape}: {path}")
    return feature_array


__all__ = [
    "SampleRecord",
    "build_sample_records",
    "load_coarse_mask",
    "load_feature_map",
]
