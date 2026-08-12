"""Milestone 2 preflight tests for AEEM v2."""

import unittest

import cv2
import numpy as np

from aeem_v2.refinement import (
    BoundaryConstraint,
    CandidateAssessment,
    MaskCandidate,
)
from aeem_v2.topology import component_profile, quantized_binary
from aeem_v2.structure import (
    apply_structure_calibration,
    build_connectivity_backbone,
    clean_candidate_islands,
)
from scripts.build_aeem_v2_cohort import (
    select_all_cohort,
    select_balanced_cohort,
    select_dataset_balanced_cohort,
)


class TopologyProfileTests(unittest.TestCase):
    def test_profile_separates_raw_count_from_effective_components(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        mask[2:4, 2:4] = True
        mask[35:37, 35:37] = True

        profile = component_profile(
            mask,
            expected_components=1,
            minimum_area_pixels=5,
            minimum_area_ratio=0.0,
        )

        self.assertEqual(profile.component_count, 3)
        self.assertEqual(profile.effective_component_count, 1)
        self.assertAlmostEqual(profile.top_k_mass_ratio, 400 / 408)
        self.assertAlmostEqual(profile.extra_component_mass_ratio, 8 / 408)

    def test_quantized_binary_matches_saved_png_threshold(self):
        probability = np.asarray([[0.499, 0.5, 0.501]], dtype=np.float32)

        binary = quantized_binary(probability)

        np.testing.assert_array_equal(
            binary,
            np.asarray([[False, True, True]]),
        )


class StructureCalibrationTests(unittest.TestCase):
    @staticmethod
    def _assessment(mask: np.ndarray) -> CandidateAssessment:
        return CandidateAssessment(
            candidate=MaskCandidate(
                mask=mask,
                sam_score=0.9,
                prompt_name="point_only",
                mask_index=0,
            ),
            quality=1.0,
            q_semantic=1.0,
            q_stability=1.0,
            q_edge=1.0,
            q_safety=1.0,
            valid=True,
        )

    def test_cleanup_removes_only_unsupported_candidate_islands(self):
        coarse = np.zeros((48, 64), dtype=np.float32)
        coarse[14:34, 16:40] = 1.0
        semantic = coarse.copy()
        candidate = coarse.astype(bool)
        candidate[3:9, 52:59] = True
        original = candidate.copy()

        cleanup = clean_candidate_islands(candidate, coarse, semantic)

        np.testing.assert_array_equal(candidate, original)
        np.testing.assert_array_equal(
            cleanup.mask[14:34, 16:40],
            np.ones((20, 24), dtype=bool),
        )
        self.assertFalse(cleanup.mask[3:9, 52:59].any())
        self.assertTrue(cleanup.risk_detected)
        self.assertEqual(cleanup.removed_component_count, 1)

    def test_connectivity_backbone_preserves_narrow_coarse_bridge(self):
        coarse = np.zeros((40, 72), dtype=bool)
        coarse[10:30, 6:26] = True
        coarse[10:30, 46:66] = True
        coarse[19:21, 26:46] = True
        foreground_core = np.zeros_like(coarse)
        foreground_core[14:26, 10:22] = True
        foreground_core[14:26, 50:62] = True

        backbone = build_connectivity_backbone(coarse, foreground_core)

        self.assertTrue(backbone[19:21, 26:46].any(axis=0).all())
        self.assertTrue(np.logical_and(backbone, ~coarse).sum() == 0)
        self.assertLess(int(backbone.sum()), int(coarse.sum()))
        components, _ = cv2.connectedComponents(
            backbone.astype(np.uint8), connectivity=8
        )
        self.assertEqual(components - 1, 1)

    def test_negative_residual_cannot_disconnect_reliable_cores(self):
        coarse = np.zeros((40, 72), dtype=np.float32)
        coarse[10:30, 6:26] = 1.0
        coarse[10:30, 46:66] = 1.0
        coarse[19:21, 26:46] = 1.0
        semantic = coarse.copy()
        semantic[19:21, 26:46] = 0.0
        candidate = coarse.astype(bool)
        candidate[19:21, 26:46] = False
        foreground_core = np.zeros_like(candidate)
        foreground_core[14:26, 10:22] = True
        foreground_core[14:26, 50:62] = True
        constraint = BoundaryConstraint(
            uncertainty_band=np.logical_and(coarse > 0.5, ~foreground_core),
            foreground_core=foreground_core,
            background_core=np.zeros_like(candidate),
            radius=4,
        )
        stripes = ((np.indices(coarse.shape).sum(axis=0) % 2) * 255).astype(np.uint8)
        image = np.repeat(stripes[:, :, None], 3, axis=2)
        selected = self._assessment(candidate)

        calibrated = apply_structure_calibration(
            selected,
            [selected],
            coarse,
            semantic,
            image,
            constraint,
            maximum_effective_component_growth=1,
        )

        refined_binary = quantized_binary(calibrated.fusion.refined)
        components, _ = cv2.connectedComponents(
            refined_binary.astype(np.uint8), connectivity=8
        )
        self.assertEqual(components - 1, 1)
        self.assertTrue(
            np.logical_and(
                calibrated.connectivity_backbone,
                calibrated.fusion.refined == coarse,
            ).all(where=calibrated.connectivity_backbone)
        )

    def test_excess_structure_growth_falls_back_to_soft_coarse(self):
        coarse = np.zeros((64, 64), dtype=np.float32)
        coarse[20:44, 20:44] = 1.0
        semantic = coarse.copy()
        candidate = coarse.astype(bool)
        for y, x in ((4, 4), (4, 52), (52, 4)):
            candidate[y:y + 6, x:x + 6] = True
            semantic[y:y + 6, x:x + 6] = 1.0
        foreground_core = np.zeros_like(candidate)
        foreground_core[24:40, 24:40] = True
        constraint = BoundaryConstraint(
            uncertainty_band=~foreground_core,
            foreground_core=foreground_core,
            background_core=np.zeros_like(candidate),
            radius=8,
        )
        selected = self._assessment(candidate)

        calibrated = apply_structure_calibration(
            selected,
            [selected],
            coarse,
            semantic,
            np.zeros((64, 64, 3), dtype=np.uint8),
            constraint,
            maximum_effective_component_growth=1,
        )

        self.assertGreater(
            calibrated.pre_fallback_effective_component_growth, 1
        )
        self.assertEqual(
            calibrated.fallback_reason,
            "excess_effective_component_growth",
        )
        np.testing.assert_array_equal(calibrated.fusion.refined, coarse)
        np.testing.assert_array_equal(
            calibrated.fusion.confidence,
            np.zeros_like(coarse),
        )

    def test_missing_candidate_keeps_soft_coarse_pixel_exact(self):
        coarse = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
        empty = np.zeros_like(coarse, dtype=bool)
        constraint = BoundaryConstraint(empty, empty, empty, radius=0)

        calibrated = apply_structure_calibration(
            None,
            [],
            coarse,
            coarse,
            np.zeros((5, 7, 3), dtype=np.uint8),
            constraint,
        )

        self.assertEqual(calibrated.fallback_reason, "no_selected_candidate")
        np.testing.assert_array_equal(calibrated.fusion.refined, coarse)
        np.testing.assert_array_equal(
            calibrated.fusion.confidence,
            np.zeros_like(coarse),
        )


class BalancedCohortTests(unittest.TestCase):
    def test_all_cohort_preserves_every_sample_in_index_order(self):
        records = [
            {"index": 9, "selection_reasons": []},
            {"index": 2, "selection_reasons": []},
            {"index": 5, "selection_reasons": []},
        ]

        selected = select_all_cohort(records)

        self.assertEqual([row["index"] for row in selected], [2, 5, 9])
        self.assertTrue(
            all("all_eligible_samples" in row["selection_reasons"] for row in selected)
        )

    def test_selection_is_balanced_by_dataset_and_route(self):
        records = []
        index = 0
        for dataset in ("TR-CAMO", "TR-COD10K"):
            for route in ("low", "medium", "high"):
                for reliability in (0.1, 0.3, 0.6, 0.9):
                    records.append({
                        "dataset": dataset,
                        "index": index,
                        "localization_reliability": reliability,
                        "route": route,
                        "selection_reasons": [],
                    })
                    index += 1

        selected = select_balanced_cohort(
            records,
            dataset_names=("TR-CAMO", "TR-COD10K"),
            per_dataset_route=2,
        )

        self.assertEqual(len(selected), 12)
        for dataset in ("TR-CAMO", "TR-COD10K"):
            for route in ("low", "medium", "high"):
                self.assertEqual(
                    sum(
                        row["dataset"] == dataset and row["route"] == route
                        for row in selected
                    ),
                    2,
                )

    def test_dataset_balance_redistributes_scarce_route_quota(self):
        records = []
        index = 0
        route_counts = {"low": 1, "medium": 40, "high": 40}
        for dataset in ("TR-CAMO", "TR-COD10K"):
            for route, count in route_counts.items():
                for position in range(count):
                    records.append({
                        "dataset": dataset,
                        "index": index,
                        "localization_reliability": position / max(count, 1),
                        "route": route,
                        "selection_reasons": [],
                    })
                    index += 1

        selected = select_dataset_balanced_cohort(
            records,
            dataset_names=("TR-CAMO", "TR-COD10K"),
            per_dataset=60,
        )

        self.assertEqual(len(selected), 120)
        for dataset in ("TR-CAMO", "TR-COD10K"):
            dataset_rows = [row for row in selected if row["dataset"] == dataset]
            self.assertEqual(len(dataset_rows), 60)
            self.assertEqual(sum(row["route"] == "low" for row in dataset_rows), 1)
            self.assertLessEqual(
                abs(
                    sum(row["route"] == "medium" for row in dataset_rows)
                    - sum(row["route"] == "high" for row in dataset_rows)
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
