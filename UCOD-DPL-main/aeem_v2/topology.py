"""No-GT topology diagnostics for AEEM v2 masks."""

import math
from dataclasses import asdict, dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentProfile:
    component_count: int
    effective_component_count: int
    expected_components: int
    extra_component_mass_ratio: float
    foreground_area: int
    top_k_mass_ratio: float

    def as_dict(self) -> Dict:
        return asdict(self)


def quantized_binary(probability: np.ndarray) -> np.ndarray:
    quantized = np.rint(
        np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return quantized > 127


def component_profile(
    mask: np.ndarray,
    expected_components: int,
    minimum_area_pixels: int = 16,
    minimum_area_ratio: float = 0.001,
) -> ComponentProfile:
    if expected_components < 0:
        raise ValueError("expected_components must be non-negative")
    if minimum_area_pixels < 1:
        raise ValueError("minimum_area_pixels must be positive")
    if not 0.0 <= minimum_area_ratio <= 1.0:
        raise ValueError("minimum_area_ratio must be in [0,1]")

    binary = np.asarray(mask, dtype=bool)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    areas = np.sort(areas)[::-1]
    foreground_area = int(areas.sum())
    minimum_area = max(
        minimum_area_pixels,
        int(math.ceil(binary.size * minimum_area_ratio)),
    )
    effective_component_count = int((areas >= minimum_area).sum())
    if foreground_area == 0:
        top_k_mass_ratio = 1.0
        extra_component_mass_ratio = 0.0
    else:
        top_k_area = int(areas[:expected_components].sum())
        top_k_mass_ratio = float(top_k_area / foreground_area)
        extra_component_mass_ratio = float(1.0 - top_k_mass_ratio)
    return ComponentProfile(
        component_count=max(int(count) - 1, 0),
        effective_component_count=effective_component_count,
        expected_components=expected_components,
        extra_component_mass_ratio=extra_component_mass_ratio,
        foreground_area=foreground_area,
        top_k_mass_ratio=top_k_mass_ratio,
    )
