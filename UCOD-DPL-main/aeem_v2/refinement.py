"""Prompt routing, candidate assessment, and boundary-safe fusion for AEEM v2."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
BoxXYXY = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PromptVariant:
    name: str
    positive_points: Tuple[Point, ...]
    negative_points: Tuple[Point, ...]
    box_xyxy: Optional[BoxXYXY]

    def as_dict(self) -> Dict:
        return {
            "box_xyxy": list(self.box_xyxy) if self.box_xyxy is not None else None,
            "name": self.name,
            "negative_points": [list(point) for point in self.negative_points],
            "positive_points": [list(point) for point in self.positive_points],
        }


@dataclass
class MaskCandidate:
    mask: np.ndarray
    sam_score: float
    prompt_name: str
    mask_index: int


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: MaskCandidate
    quality: float
    q_semantic: float
    q_stability: float
    q_edge: float
    q_safety: float
    valid: bool

    def as_dict(self) -> Dict:
        mask = np.asarray(self.candidate.mask, dtype=bool)
        coordinates = np.argwhere(mask)
        centroid_xy = None
        if len(coordinates):
            centroid_yx = coordinates.mean(axis=0)
            centroid_xy = [float(centroid_yx[1]), float(centroid_yx[0])]
        component_count, _ = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8
        )
        return {
            "area": int(mask.sum()),
            "area_ratio": float(mask.mean()),
            "centroid_xy": centroid_xy,
            "component_count": max(int(component_count) - 1, 0),
            "mask_index": self.candidate.mask_index,
            "prompt_name": self.candidate.prompt_name,
            "q_edge": self.q_edge,
            "q_safety": self.q_safety,
            "q_semantic": self.q_semantic,
            "q_stability": self.q_stability,
            "quality": self.quality,
            "sam_score": self.candidate.sam_score,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class BoundaryConstraint:
    uncertainty_band: np.ndarray
    foreground_core: np.ndarray
    background_core: np.ndarray
    radius: int


@dataclass(frozen=True)
class FusionResult:
    refined: np.ndarray
    confidence: np.ndarray
    changed_ratio: float
    component_count_before: int = 0
    component_count_after: int = 0


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


def _resize_probability(probability: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if probability.shape == shape:
        return np.clip(probability.astype(np.float32), 0.0, 1.0)
    return np.clip(cv2.resize(
        probability.astype(np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ), 0.0, 1.0)


def _distance_peaks(mask: np.ndarray, count: int) -> Tuple[Point, ...]:
    if not mask.any() or count <= 0:
        return ()
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    suppression_radius = max(3, int(round(math.sqrt(float(mask.sum())) / 4.0)))
    points: List[Point] = []
    working = distance.copy()
    for _ in range(count):
        _, maximum, _, location = cv2.minMaxLoc(working)
        if maximum <= 0:
            break
        points.append((int(location[0]), int(location[1])))
        cv2.circle(working, location, suppression_radius, 0.0, thickness=-1)
    return tuple(points)


def _weak_box(mask: np.ndarray, expansion_ratio: float = 0.1) -> Optional[BoxXYXY]:
    y, x = np.where(mask)
    if len(y) == 0:
        return None
    height, width = mask.shape
    x0, x1 = int(x.min()), int(x.max())
    y0, y1 = int(y.min()), int(y.max())
    padding_x = max(2, int(round((x1 - x0 + 1) * expansion_ratio)))
    padding_y = max(2, int(round((y1 - y0 + 1) * expansion_ratio)))
    return (
        max(0, x0 - padding_x),
        max(0, y0 - padding_y),
        min(width - 1, x1 + padding_x),
        min(height - 1, y1 + padding_y),
    )


def _safe_negative(
    coarse_binary: np.ndarray,
    semantic_probability: np.ndarray,
) -> Tuple[Point, ...]:
    safe_background = np.logical_and(
        np.logical_not(_dilate(coarse_binary, 3)),
        semantic_probability < 0.3,
    )
    if not safe_background.any():
        return ()
    distance = cv2.distanceTransform(
        np.logical_not(coarse_binary).astype(np.uint8), cv2.DIST_L2, 5
    )
    distance[~safe_background] = 0.0
    _, maximum, _, location = cv2.minMaxLoc(distance)
    if maximum <= 0:
        return ()
    return ((int(location[0]), int(location[1])),)


def build_prompt_variants(
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    route: str,
) -> List[PromptVariant]:
    if route not in {"low", "medium", "high"}:
        raise ValueError(f"Unknown route: {route}")
    if route == "low":
        return []

    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    semantic = _resize_probability(semantic_probability, coarse.shape)
    coarse_binary = coarse > 0.5
    foreground_core = np.logical_and(coarse >= 0.8, semantic >= 0.5)
    foreground_core = _erode(foreground_core, 1)
    if not foreground_core.any():
        foreground_core = coarse >= 0.8
    if not foreground_core.any():
        return []

    point_count = 1 if route == "high" else 3
    positive_points = _distance_peaks(foreground_core, point_count)
    if not positive_points:
        return []
    search_radius = max(3, int(round(math.sqrt(float(coarse_binary.sum())) * 0.2)))
    prompt_region = np.logical_or(
        coarse_binary,
        np.logical_and(semantic >= 0.65, _dilate(coarse_binary, search_radius)),
    )
    box = _weak_box(prompt_region)

    prompts = [PromptVariant(
        name="point_only",
        positive_points=(positive_points[0],),
        negative_points=(),
        box_xyxy=None,
    )]
    if box is not None:
        prompts.append(PromptVariant(
            name="weak_box",
            positive_points=(positive_points[0],),
            negative_points=(),
            box_xyxy=box,
        ))
    if route == "medium":
        prompts.append(PromptVariant(
            name="consensus_points",
            positive_points=positive_points,
            negative_points=_safe_negative(coarse_binary, semantic),
            box_xyxy=None,
        ))
    return prompts


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_u8 = mask.astype(np.uint8)
    return cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(int(count) - 1, 0)


def _png_binary(probability: np.ndarray) -> np.ndarray:
    quantized = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return quantized > 127


def _edge_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    scale = float(np.percentile(magnitude, 95))
    if scale <= 1e-6:
        return np.zeros_like(magnitude)
    return np.clip(magnitude / scale, 0.0, 1.0)


def _centroid_score(first: np.ndarray, second: np.ndarray) -> float:
    first_points = np.argwhere(first)
    second_points = np.argwhere(second)
    if len(first_points) == 0 or len(second_points) == 0:
        return 0.0
    first_center = first_points.mean(axis=0)
    second_center = second_points.mean(axis=0)
    diagonal = float(np.hypot(*first.shape))
    distance = float(np.linalg.norm(first_center - second_center))
    return float(np.clip(1.0 - distance / diagonal, 0.0, 1.0))


def _candidate_stability(
    candidate: MaskCandidate,
    candidates: Sequence[MaskCandidate],
) -> float:
    other_prompt_names = {
        item.prompt_name for item in candidates if item.prompt_name != candidate.prompt_name
    }
    if not other_prompt_names:
        return 0.5
    agreements = []
    for prompt_name in other_prompt_names:
        agreements.append(max(
            _iou(candidate.mask, item.mask)
            for item in candidates if item.prompt_name == prompt_name
        ))
    return float(np.mean(agreements))


def assess_candidates(
    candidates: Sequence[MaskCandidate],
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    image_rgb: np.ndarray,
) -> List[CandidateAssessment]:
    if not candidates:
        return []
    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    semantic = _resize_probability(semantic_probability, coarse.shape)
    coarse_binary = coarse > 0.5
    foreground_core = _erode(
        np.logical_and(coarse >= 0.8, semantic >= 0.5), 1
    )
    if not foreground_core.any():
        foreground_core = coarse >= 0.8
    background_core = np.logical_and(
        np.logical_not(_dilate(coarse_binary, 5)), semantic < 0.3
    )
    edges = _edge_map(image_rgb)
    coarse_area = max(int(coarse_binary.sum()), 1)
    image_area = coarse.size

    assessments = []
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != coarse.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (coarse.shape[1], coarse.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            candidate.mask = mask
        area = int(mask.sum())
        inside_semantic = float(semantic[mask].mean()) if area else 0.0
        outside_semantic = float(semantic[~mask].mean()) if area < image_area else 1.0
        q_semantic = float(np.clip(
            0.5 * (inside_semantic + 1.0 - outside_semantic), 0.0, 1.0
        ))
        q_stability = _candidate_stability(candidate, candidates)
        candidate_boundary = _boundary(mask)
        q_edge = float(edges[candidate_boundary].mean()) if candidate_boundary.any() else 0.0

        area_score = float(math.exp(-abs(math.log((area + 1.0) / (coarse_area + 1.0)))))
        centroid_score = _centroid_score(mask, coarse_binary)
        core_coverage = (
            float(mask[foreground_core].mean()) if foreground_core.any() else 0.5
        )
        background_exclusion = (
            float((~mask)[background_core].mean()) if background_core.any() else 0.5
        )
        q_safety = float(np.mean([
            area_score, centroid_score, core_coverage, background_exclusion
        ]))
        quality = float(np.mean([q_semantic, q_stability, q_edge, q_safety]))
        valid = bool(0 < area < image_area and q_safety >= 0.25)
        assessments.append(CandidateAssessment(
            candidate=candidate,
            quality=quality,
            q_semantic=q_semantic,
            q_stability=q_stability,
            q_edge=q_edge,
            q_safety=q_safety,
            valid=valid,
        ))
    return assessments


def select_candidate(
    assessments: Sequence[CandidateAssessment],
    minimum_quality: float = 0.35,
) -> Optional[CandidateAssessment]:
    valid = [item for item in assessments if item.valid]
    if not valid:
        return None
    selected = max(valid, key=lambda item: item.quality)
    return selected if selected.quality >= minimum_quality else None


def build_boundary_constraint(
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    route: str,
) -> BoundaryConstraint:
    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    semantic = _resize_probability(semantic_probability, coarse.shape)
    coarse_binary = coarse > 0.5
    area = max(float(coarse_binary.sum()), 1.0)
    equivalent_radius = math.sqrt(area / math.pi)
    if route == "high":
        radius = int(np.clip(round(0.05 * equivalent_radius), 2, 12))
    elif route == "medium":
        radius = int(np.clip(round(0.10 * equivalent_radius), 4, 20))
    else:
        radius = 0

    foreground_core = _erode(coarse >= 0.8, max(1, radius // 2))
    if not foreground_core.any():
        foreground_core = coarse >= 0.8
    background_core = np.logical_and(
        np.logical_not(_dilate(coarse_binary, max(2, radius * 2))),
        semantic < 0.3,
    )
    if radius == 0:
        uncertainty_band = np.zeros_like(coarse_binary)
    else:
        coarse_band = np.logical_and(
            _dilate(coarse_binary, radius),
            np.logical_not(_erode(coarse_binary, radius)),
        )
        disagreement = np.logical_xor(coarse_binary, semantic >= 0.5)
        semantic_band = _dilate(disagreement, max(1, radius // 2))
        uncertainty_band = np.logical_or(coarse_band, semantic_band)
        uncertainty_band[foreground_core] = False
        uncertainty_band[background_core] = False
    return BoundaryConstraint(
        uncertainty_band=uncertainty_band,
        foreground_core=foreground_core,
        background_core=background_core,
        radius=radius,
    )


def fuse_boundary_residual(
    selected: CandidateAssessment,
    assessments: Sequence[CandidateAssessment],
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    image_rgb: np.ndarray,
    constraint: BoundaryConstraint,
    protected_foreground: Optional[np.ndarray] = None,
) -> FusionResult:
    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    semantic = _resize_probability(semantic_probability, coarse.shape)
    selected_mask = selected.candidate.mask.astype(bool)
    candidate_masks = np.stack([
        assessment.candidate.mask.astype(bool) for assessment in assessments
    ])
    prompt_consensus = np.mean(candidate_masks == selected_mask[None, ...], axis=0)
    semantic_consistency = np.where(selected_mask, semantic, 1.0 - semantic)
    edge_support = cv2.GaussianBlur(_edge_map(image_rgb), (5, 5), 0)
    edge_support = 0.5 + 0.5 * np.clip(edge_support, 0.0, 1.0)
    confidence = (
        selected.quality
        * prompt_consensus
        * semantic_consistency
        * edge_support
    ).astype(np.float32)
    confidence[~constraint.uncertainty_band] = 0.0
    confidence[constraint.foreground_core] = 0.0
    confidence[constraint.background_core] = 0.0
    if protected_foreground is not None:
        protected = np.asarray(protected_foreground, dtype=bool)
        if protected.shape != coarse.shape:
            raise ValueError("protected foreground must match the coarse label shape")
        negative_residual = np.logical_and(protected, ~selected_mask)
        confidence[negative_residual] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0)
    refined = (
        (1.0 - confidence) * coarse
        + confidence * selected_mask.astype(np.float32)
    )
    component_count_before = _component_count(_png_binary(coarse))
    component_count_after = _component_count(_png_binary(refined))
    refined = np.clip(refined, 0.0, 1.0)
    changed_ratio = float((np.abs(refined - coarse) > (1.0 / 255.0)).mean())
    return FusionResult(
        refined=refined,
        confidence=confidence,
        changed_ratio=changed_ratio,
        component_count_before=component_count_before,
        component_count_after=component_count_after,
    )


def save_candidate_cache(
    path: Path,
    prompts: Sequence[PromptVariant],
    assessments: Sequence[CandidateAssessment],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if assessments:
        shape = assessments[0].candidate.mask.shape
        packed_masks = np.stack([
            np.packbits(item.candidate.mask.astype(np.uint8).reshape(-1))
            for item in assessments
        ])
    else:
        shape = (0, 0)
        packed_masks = np.empty((0, 0), dtype=np.uint8)
    metadata = {
        "assessments": [item.as_dict() for item in assessments],
        "prompts": [prompt.as_dict() for prompt in prompts],
    }
    np.savez_compressed(
        path,
        mask_shape=np.asarray(shape, dtype=np.int32),
        packed_masks=packed_masks,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_candidate_cache(path: Path) -> Tuple[List[np.ndarray], Dict]:
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(value) for value in data["mask_shape"])
        packed_masks = data["packed_masks"]
        metadata = json.loads(str(data["metadata_json"]))
    masks = []
    pixel_count = int(np.prod(shape))
    for packed in packed_masks:
        unpacked = np.unpackbits(packed)[:pixel_count]
        masks.append(unpacked.reshape(shape).astype(bool))
    return masks, metadata
