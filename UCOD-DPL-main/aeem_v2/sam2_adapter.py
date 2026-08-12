"""Lazy SAM2 adapter for AEEM v2 multi-prompt inference."""

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .refinement import MaskCandidate, PromptVariant


class SAM2Adapter:
    def __init__(
        self,
        checkpoint_path: Path,
        config_file: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_file = config_file
        self.device = device
        self.predictor = self._build_predictor()

    @classmethod
    def from_predictor(cls, predictor) -> "SAM2Adapter":
        adapter = cls.__new__(cls)
        adapter.checkpoint_path = None
        adapter.config_file = None
        adapter.device = None
        adapter.predictor = predictor
        return adapter

    def _build_predictor(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError(
                "SAM2 is not installed in this Python environment. "
                "Run the MVP with the test01 environment."
            ) from error

        model = build_sam2(
            config_file=self.config_file,
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
            mode="eval",
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return SAM2ImagePredictor(model)

    @staticmethod
    def _box_array(prompt: PromptVariant, image_shape) -> np.ndarray:
        box = np.asarray(prompt.box_xyxy, dtype=np.float32)
        if box.shape != (4,):
            raise ValueError(f"Expected one XYXY box, got {box.shape}")
        x0, y0, x1, y1 = box
        height, width = image_shape[:2]
        if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
            raise ValueError(
                f"Invalid XYXY box {tuple(box.tolist())} for image {width}x{height}"
            )
        return box

    def predict_candidates(
        self,
        image_rgb: np.ndarray,
        prompts: Sequence[PromptVariant],
    ) -> List[MaskCandidate]:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
        if not prompts:
            return []

        image = np.array(image, dtype=np.uint8, order="C", copy=True)
        self.predictor.set_image(image)
        candidates: List[MaskCandidate] = []
        for prompt in prompts:
            points = prompt.positive_points + prompt.negative_points
            point_coords = (
                np.asarray(points, dtype=np.float32) if points else None
            )
            point_labels = None
            if points:
                point_labels = np.asarray(
                    [1] * len(prompt.positive_points)
                    + [0] * len(prompt.negative_points),
                    dtype=np.int32,
                )
            box = (
                self._box_array(prompt, image.shape)
                if prompt.box_xyxy is not None else None
            )
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
                return_logits=False,
            )
            mask_array = np.asarray(masks)
            score_array = np.asarray(scores).reshape(-1)
            if mask_array.ndim == 2:
                mask_array = mask_array[None, ...]
            if len(mask_array) != len(score_array):
                raise ValueError(
                    f"SAM2 returned {len(mask_array)} masks and {len(score_array)} scores"
                )
            for mask_index, (mask, score) in enumerate(zip(mask_array, score_array)):
                candidates.append(MaskCandidate(
                    mask=np.asarray(mask, dtype=bool),
                    sam_score=float(score),
                    prompt_name=prompt.name,
                    mask_index=mask_index,
                ))
        return candidates
