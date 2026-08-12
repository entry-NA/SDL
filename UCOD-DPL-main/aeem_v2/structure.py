"""Topology-safe candidate cleanup and residual fusion for AEEM v2."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import cv2
import numpy as np

from aeem_v2.refinement import (
    BoundaryConstraint,
    CandidateAssessment,
    FusionResult,
    fuse_boundary_residual,
)
from aeem_v2.topology import component_profile, quantized_binary


@dataclass(frozen=True)
class CandidateCleanupResult:
    mask: np.ndarray
    risk_detected: bool
    component_count_before: int
    component_count_after: int
    removed_component_count: int
    removed_area_ratio: float


@dataclass(frozen=True)
class StructureCalibrationResult:
    fusion: FusionResult
    cleanup: Optional[CandidateCleanupResult]
    cleaned_selected: Optional[CandidateAssessment]
    connectivity_backbone: np.ndarray
    pre_fallback_effective_component_growth: int
    fallback_reason: Optional[str]


def _coarse_fallback(coarse_soft: np.ndarray) -> FusionResult:
    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    binary = quantized_binary(coarse)
    count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    component_count = max(int(count) - 1, 0)
    return FusionResult(
        refined=coarse.copy(),
        confidence=np.zeros_like(coarse),
        changed_ratio=0.0,
        component_count_before=component_count,
        component_count_after=component_count,
    )


def apply_structure_calibration(
    selected: Optional[CandidateAssessment],
    assessments: Sequence[CandidateAssessment],
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    image_rgb: np.ndarray,
    constraint: BoundaryConstraint,
    maximum_effective_component_growth: int = 1,
    maximum_extra_mass_ratio: float = 0.05,
) -> StructureCalibrationResult:
    """Clean risky islands, protect coarse connectivity, then apply sample fallback."""
    if maximum_effective_component_growth < 0:
        raise ValueError("maximum_effective_component_growth must be non-negative")
    coarse = np.clip(np.asarray(coarse_soft, dtype=np.float32), 0.0, 1.0)
    if selected is None:
        return StructureCalibrationResult(
            fusion=_coarse_fallback(coarse),
            cleanup=None,
            cleaned_selected=None,
            connectivity_backbone=np.zeros_like(coarse, dtype=bool),
            pre_fallback_effective_component_growth=0,
            fallback_reason="no_selected_candidate",
        )

    cleanup = clean_candidate_islands(
        selected.candidate.mask,
        coarse,
        semantic_probability,
        maximum_extra_mass_ratio=maximum_extra_mass_ratio,
    )
    cleaned_candidate = replace(selected.candidate, mask=cleanup.mask)
    cleaned_selected = replace(selected, candidate=cleaned_candidate)
    selected_key = (
        selected.candidate.prompt_name,
        selected.candidate.mask_index,
    )
    calibrated_assessments = [
        cleaned_selected
        if (
            assessment.candidate.prompt_name,
            assessment.candidate.mask_index,
        ) == selected_key
        else assessment
        for assessment in assessments
    ]
    backbone = build_connectivity_backbone(
        quantized_binary(coarse), constraint.foreground_core
    )
    fusion = fuse_boundary_residual(
        cleaned_selected,
        calibrated_assessments,
        coarse,
        semantic_probability,
        image_rgb,
        constraint,
        protected_foreground=backbone,
    )
    coarse_profile = component_profile(
        quantized_binary(coarse), expected_components=0
    )
    expected_components = coarse_profile.effective_component_count
    refined_profile = component_profile(
        quantized_binary(fusion.refined),
        expected_components=expected_components,
    )
    growth = (
        refined_profile.effective_component_count
        - coarse_profile.effective_component_count
    )
    fallback_reason = None
    if growth > maximum_effective_component_growth:
        fusion = _coarse_fallback(coarse)
        fallback_reason = "excess_effective_component_growth"
    return StructureCalibrationResult(
        fusion=fusion,
        cleanup=cleanup,
        cleaned_selected=cleaned_selected,
        connectivity_backbone=backbone,
        pre_fallback_effective_component_growth=growth,
        fallback_reason=fallback_reason,
    )


def build_connectivity_backbone(
    coarse_mask: np.ndarray,
    foreground_core: np.ndarray,
) -> np.ndarray:
    """Connect reliable core islands within each coarse component in one wavefront."""
    coarse = np.asarray(coarse_mask, dtype=bool)
    core = np.logical_and(np.asarray(foreground_core, dtype=bool), coarse)
    if coarse.shape != core.shape:
        raise ValueError("coarse mask and foreground core must share a shape")
    if not core.any():
        return np.zeros_like(coarse)

    coarse_count, coarse_labels = cv2.connectedComponents(
        coarse.astype(np.uint8), connectivity=8
    )
    core_count, core_labels = cv2.connectedComponents(
        core.astype(np.uint8), connectivity=8
    )
    if core_count <= 2:
        return core.copy()

    coarse_flat = coarse.ravel()
    coarse_label_flat = coarse_labels.ravel()
    owner = core_labels.ravel().astype(np.int32, copy=True)
    core_positions = np.flatnonzero(owner)
    core_to_coarse = np.zeros(core_count, dtype=np.int32)
    core_to_coarse[owner[core_positions]] = coarse_label_flat[core_positions]
    cores_per_coarse = np.bincount(
        core_to_coarse[1:], minlength=coarse_count
    )
    remaining_connections = int(
        np.maximum(cores_per_coarse[1:] - 1, 0).sum()
    )
    if remaining_connections == 0:
        return core.copy()

    parent_pixel = np.full(coarse.size, -1, dtype=np.int64)
    union_parent = np.arange(core_count, dtype=np.int32)
    rank = np.zeros(core_count, dtype=np.uint8)
    backbone = core.ravel().copy()
    height, width = coarse.shape
    queue = deque(int(position) for position in core_positions)

    def find(label: int) -> int:
        root = label
        while union_parent[root] != root:
            root = int(union_parent[root])
        while union_parent[label] != label:
            next_label = int(union_parent[label])
            union_parent[label] = root
            label = next_label
        return root

    def union(first: int, second: int) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        union_parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1
        return True

    def mark_parent_path(position: int) -> None:
        while position >= 0:
            backbone[position] = True
            position = int(parent_pixel[position])

    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    while queue and remaining_connections > 0:
        position = queue.popleft()
        y, x = divmod(position, width)
        position_owner = int(owner[position])
        for delta_y, delta_x in neighbor_offsets:
            neighbor_y = y + delta_y
            neighbor_x = x + delta_x
            if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                continue
            neighbor = neighbor_y * width + neighbor_x
            if not coarse_flat[neighbor]:
                continue
            neighbor_owner = int(owner[neighbor])
            if neighbor_owner == 0:
                owner[neighbor] = position_owner
                parent_pixel[neighbor] = position
                queue.append(neighbor)
                continue
            if neighbor_owner == position_owner:
                continue
            if core_to_coarse[position_owner] != core_to_coarse[neighbor_owner]:
                continue
            if union(position_owner, neighbor_owner):
                mark_parent_path(position)
                mark_parent_path(neighbor)
                remaining_connections -= 1
                if remaining_connections == 0:
                    break

    return backbone.reshape(coarse.shape)


def clean_candidate_islands(
    candidate_mask: np.ndarray,
    coarse_soft: np.ndarray,
    semantic_probability: np.ndarray,
    maximum_extra_mass_ratio: float = 0.05,
) -> CandidateCleanupResult:
    """Remove unsupported islands only when the candidate has topology risk."""
    if not 0.0 <= maximum_extra_mass_ratio <= 1.0:
        raise ValueError("maximum_extra_mass_ratio must be in [0,1]")

    candidate = np.asarray(candidate_mask, dtype=bool)
    coarse = np.asarray(coarse_soft, dtype=np.float32)
    semantic = np.asarray(semantic_probability, dtype=np.float32)
    if candidate.shape != coarse.shape or candidate.shape != semantic.shape:
        raise ValueError("candidate, coarse, and semantic maps must share a shape")

    coarse_profile = component_profile(
        quantized_binary(coarse), expected_components=0
    )
    expected_components = coarse_profile.effective_component_count
    candidate_profile = component_profile(
        candidate, expected_components=expected_components
    )
    risk_detected = bool(
        candidate_profile.effective_component_count > expected_components
        or candidate_profile.extra_component_mass_ratio > maximum_extra_mass_ratio
    )
    if not risk_detected or not candidate.any():
        return CandidateCleanupResult(
            mask=candidate.copy(),
            risk_detected=risk_detected,
            component_count_before=candidate_profile.component_count,
            component_count_after=candidate_profile.component_count,
            removed_component_count=0,
            removed_area_ratio=0.0,
        )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    support = np.logical_or(coarse >= 0.8, semantic >= 0.65)
    keep_labels = {
        label
        for label in range(1, component_count)
        if np.logical_and(labels == label, support).any()
    }
    if not keep_labels and component_count > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep_labels.add(largest_label)

    cleaned = np.isin(labels, tuple(keep_labels))
    removed_area = int(np.logical_and(candidate, ~cleaned).sum())
    cleaned_count = max(len(keep_labels), 0)
    return CandidateCleanupResult(
        mask=cleaned,
        risk_detected=True,
        component_count_before=max(component_count - 1, 0),
        component_count_after=cleaned_count,
        removed_component_count=max(component_count - 1 - cleaned_count, 0),
        removed_area_ratio=float(removed_area / max(int(candidate.sum()), 1)),
    )
