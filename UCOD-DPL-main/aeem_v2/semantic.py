"""Semantic Localization Calibration for AEEM v2."""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticLocalization:
    probability: np.ndarray
    semantic_mask: np.ndarray
    foreground_core: np.ndarray
    background_core: np.ndarray
    reliability: float
    route: str
    components: Dict[str, float]


def _as_chw(feature_map: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature_map, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError(f"Expected 3D feature map, got {feature.shape}")
    if feature.shape[0] >= feature.shape[1] and feature.shape[0] >= feature.shape[2]:
        return feature
    return np.moveaxis(feature, -1, 0)


def _kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return cv2.erode(mask.astype(np.uint8), _kernel(radius)) > 0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius)) > 0


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return float("nan"), float("nan")
    y, x = coordinates.mean(axis=0)
    return float(x), float(y)


def _centroid_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first_centroid = _centroid(first)
    second_centroid = _centroid(second)
    if not np.isfinite(first_centroid).all() or not np.isfinite(second_centroid).all():
        return 0.0
    diagonal = float(np.hypot(*first.shape))
    distance = float(np.hypot(
        first_centroid[0] - second_centroid[0],
        first_centroid[1] - second_centroid[1],
    ))
    return float(np.clip(1.0 - distance / diagonal, 0.0, 1.0))


def _area_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first_area = int(first.sum())
    second_area = int(second.sum())
    if first_area == 0 or second_area == 0:
        return 0.0
    return float(min(first_area, second_area) / max(first_area, second_area))


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(count - 1, 0)


def _component_agreement(first: np.ndarray, second: np.ndarray) -> float:
    difference = abs(_component_count(first) - _component_count(second))
    return float(1.0 / (1.0 + difference))


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-6)


def _route(reliability: float, low_threshold: float, high_threshold: float) -> str:
    if reliability >= high_threshold:
        return "high"
    if reliability >= low_threshold:
        return "medium"
    return "low"


def compute_semantic_localization(
    coarse_soft: np.ndarray,
    feature_map: np.ndarray,
    foreground_threshold: float = 0.8,
    temperature: float = 0.1,
    low_route_threshold: float = 0.33,
    high_route_threshold: float = 0.67,
) -> SemanticLocalization:
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive")
    if not 0.0 <= low_route_threshold < high_route_threshold <= 1.0:
        raise ValueError("route thresholds must satisfy 0 <= low < high <= 1")

    feature = _as_chw(feature_map)
    _, height, width = feature.shape
    coarse = cv2.resize(
        np.asarray(coarse_soft, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    coarse = np.clip(coarse, 0.0, 1.0)
    coarse_binary = coarse > 0.5

    foreground_seed = coarse >= foreground_threshold
    foreground_core = _erode(foreground_seed, 1)
    if not foreground_core.any():
        foreground_core = foreground_seed
    background_core = np.logical_not(_dilate(coarse_binary, 2))
    if not background_core.any():
        background_core = coarse <= 0.1

    if not foreground_core.any() or not background_core.any():
        probability = coarse.copy()
        semantic_mask = probability > 0.5
        components = {
            "area_agreement": 0.0,
            "centroid_agreement": 0.0,
            "component_agreement": 0.0,
            "region_iou": 0.0,
            "semantic_margin": 0.0,
        }
        return SemanticLocalization(
            probability=probability,
            semantic_mask=semantic_mask,
            foreground_core=foreground_core,
            background_core=background_core,
            reliability=0.0,
            route="low",
            components=components,
        )

    vectors = np.moveaxis(feature, 0, -1).reshape(-1, feature.shape[0])
    vectors = _normalize_vectors(vectors)
    foreground_flat = foreground_core.reshape(-1)
    background_flat = background_core.reshape(-1)
    foreground_prototype = _normalize_vectors(
        vectors[foreground_flat].mean(axis=0, keepdims=True)
    )[0]
    background_prototype = _normalize_vectors(
        vectors[background_flat].mean(axis=0, keepdims=True)
    )[0]
    margin = vectors @ foreground_prototype - vectors @ background_prototype
    logits = np.clip(margin / temperature, -20.0, 20.0)
    probability = (1.0 / (1.0 + np.exp(-logits))).reshape(height, width)
    semantic_mask = probability > 0.5

    semantic_margin = float(np.clip(
        probability[foreground_core].mean() - probability[background_core].mean(),
        0.0,
        1.0,
    ))
    components = {
        "area_agreement": _area_agreement(coarse_binary, semantic_mask),
        "centroid_agreement": _centroid_agreement(coarse_binary, semantic_mask),
        "component_agreement": _component_agreement(coarse_binary, semantic_mask),
        "region_iou": _iou(coarse_binary, semantic_mask),
        "semantic_margin": semantic_margin,
    }
    reliability = float(np.mean(list(components.values())))
    return SemanticLocalization(
        probability=probability.astype(np.float32),
        semantic_mask=semantic_mask,
        foreground_core=foreground_core,
        background_core=background_core,
        reliability=reliability,
        route=_route(reliability, low_route_threshold, high_route_threshold),
        components=components,
    )
