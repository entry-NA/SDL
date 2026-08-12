"""Milestone 1 regression tests for AEEM v2."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aeem_v2.sam2_adapter import SAM2Adapter
from aeem_v2.refinement import (
    CandidateAssessment,
    MaskCandidate,
    PromptVariant,
    build_boundary_constraint,
    build_prompt_variants,
    fuse_boundary_residual,
    load_candidate_cache,
    save_candidate_cache,
)
from aeem_v2.semantic import compute_semantic_localization


class SemanticLocalizationTests(unittest.TestCase):
    def test_empty_coarse_label_uses_low_route(self):
        coarse = np.zeros((24, 32), dtype=np.float32)
        feature = np.ones((8, 6, 8), dtype=np.float32)

        localization = compute_semantic_localization(coarse, feature)

        self.assertEqual(localization.route, "low")
        self.assertEqual(localization.reliability, 0.0)

    def test_probability_and_reliability_are_bounded(self):
        coarse = np.zeros((24, 32), dtype=np.float32)
        coarse[6:18, 8:24] = 1.0
        feature = np.zeros((8, 6, 8), dtype=np.float32)
        feature[0, 1:5, 2:6] = 1.0
        feature[1, :, :] = 1.0 - feature[0]

        localization = compute_semantic_localization(coarse, feature)

        self.assertGreaterEqual(float(localization.probability.min()), 0.0)
        self.assertLessEqual(float(localization.probability.max()), 1.0)
        self.assertGreaterEqual(localization.reliability, 0.0)
        self.assertLessEqual(localization.reliability, 1.0)
        for component in localization.components.values():
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)


class PromptRoutingTests(unittest.TestCase):
    def test_low_route_generates_no_prompt(self):
        coarse = np.zeros((32, 40), dtype=np.float32)
        coarse[10:20, 12:24] = 1.0

        prompts = build_prompt_variants(coarse, coarse, route="low")

        self.assertEqual(prompts, [])

    def test_weak_box_is_xyxy(self):
        coarse = np.zeros((30, 40), dtype=np.float32)
        coarse[5:10, 10:20] = 1.0

        prompts = build_prompt_variants(coarse, coarse, route="high")
        weak_box = next(prompt for prompt in prompts if prompt.name == "weak_box")

        self.assertEqual(weak_box.box_xyxy, (8, 3, 21, 11))
        x0, y0, x1, y1 = weak_box.box_xyxy
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)


class BoundaryFusionTests(unittest.TestCase):
    @staticmethod
    def _assessment(mask: np.ndarray, quality: float) -> CandidateAssessment:
        candidate = MaskCandidate(
            mask=mask,
            sam_score=0.8,
            prompt_name="point_only",
            mask_index=0,
        )
        return CandidateAssessment(
            candidate=candidate,
            quality=quality,
            q_semantic=quality,
            q_stability=quality,
            q_edge=quality,
            q_safety=quality,
            valid=True,
        )

    def test_fusion_only_changes_uncertainty_band(self):
        coarse = np.zeros((64, 64), dtype=np.float32)
        coarse[16:48, 16:48] = 1.0
        semantic = coarse.copy()
        candidate_mask = np.zeros_like(coarse, dtype=bool)
        candidate_mask[14:50, 14:50] = True
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        constraint = build_boundary_constraint(coarse, semantic, route="medium")
        selected = self._assessment(candidate_mask, quality=1.0)

        result = fuse_boundary_residual(
            selected,
            [selected],
            coarse,
            semantic,
            image,
            constraint,
        )

        np.testing.assert_array_equal(
            result.refined[~constraint.uncertainty_band],
            coarse[~constraint.uncertainty_band],
        )
        np.testing.assert_array_equal(
            result.refined[constraint.foreground_core],
            coarse[constraint.foreground_core],
        )
        np.testing.assert_array_equal(
            result.refined[constraint.background_core],
            coarse[constraint.background_core],
        )

    def test_zero_candidate_quality_is_exact_coarse_fallback(self):
        coarse = np.zeros((32, 32), dtype=np.float32)
        coarse[8:24, 8:24] = 0.75
        semantic = coarse.copy()
        candidate_mask = np.ones_like(coarse, dtype=bool)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        constraint = build_boundary_constraint(coarse, semantic, route="medium")
        selected = self._assessment(candidate_mask, quality=0.0)

        result = fuse_boundary_residual(
            selected,
            [selected],
            coarse,
            semantic,
            image,
            constraint,
        )

        np.testing.assert_array_equal(result.refined, coarse)
        np.testing.assert_array_equal(result.confidence, np.zeros_like(coarse))

class CandidateCacheTests(unittest.TestCase):
    def test_packbits_round_trip_preserves_masks_and_metadata(self):
        first_mask = np.zeros((9, 11), dtype=bool)
        first_mask[2:7, 3:8] = True
        second_mask = np.logical_not(first_mask)
        prompt = PromptVariant(
            name="point_only",
            positive_points=((5, 4),),
            negative_points=(),
            box_xyxy=None,
        )
        assessments = [
            BoundaryFusionTests._assessment(first_mask, quality=0.8),
            BoundaryFusionTests._assessment(second_mask, quality=0.6),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "candidate.npz"
            save_candidate_cache(cache_path, [prompt], assessments)
            masks, metadata = load_candidate_cache(cache_path)

        np.testing.assert_array_equal(masks[0], first_mask)
        np.testing.assert_array_equal(masks[1], second_mask)
        self.assertEqual(metadata["prompts"][0]["name"], "point_only")
        self.assertEqual(len(metadata["assessments"]), 2)
        self.assertEqual(metadata["assessments"][0]["area"], int(first_mask.sum()))
        self.assertEqual(metadata["assessments"][0]["component_count"], 1)
        self.assertEqual(metadata["assessments"][0]["centroid_xy"], [5.0, 4.0])


class FakePredictor:
    def __init__(self):
        self.set_image_calls = 0
        self.predict_calls = []

    def set_image(self, image):
        self.set_image_calls += 1
        self.image_shape = image.shape
        self.image_writeable = image.flags.writeable

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        height, width = self.image_shape[:2]
        masks = np.zeros((3, height, width), dtype=bool)
        masks[0, 2:5, 3:7] = True
        masks[1, 1:6, 2:8] = True
        masks[2, 3:4, 4:6] = True
        return masks, np.asarray([0.8, 0.7, 0.6]), np.zeros((3, 4, 4))


class SAM2AdapterTests(unittest.TestCase):
    def test_one_image_encoding_serves_all_prompt_variants(self):
        predictor = FakePredictor()
        adapter = SAM2Adapter.from_predictor(predictor)
        prompts = [
            PromptVariant("point_only", ((4, 3),), (), None),
            PromptVariant("weak_box", ((4, 3),), (), (1, 2, 8, 7)),
        ]

        image = np.zeros((10, 12, 3), dtype=np.uint8)
        image.setflags(write=False)
        candidates = adapter.predict_candidates(image, prompts)

        self.assertEqual(predictor.set_image_calls, 1)
        self.assertTrue(predictor.image_writeable)
        self.assertEqual(len(predictor.predict_calls), 2)
        self.assertEqual(len(candidates), 6)
        np.testing.assert_array_equal(
            predictor.predict_calls[1]["box"],
            np.asarray([1, 2, 8, 7], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            predictor.predict_calls[0]["point_coords"],
            np.asarray([[4, 3]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
