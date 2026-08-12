"""Milestone 0 regression tests for AEEM v2."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from aeem_v2.controls import generate_control_artifact
from aeem_v2.evaluation import _boundary_iou, evaluate_predictions
from data.datasets.base_dataset import BaseCODDataset, resolve_refined_pseudo_label_dir
from data.datasets import dataloader_utils
from engine.config import CfgNode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}


class RefinedOnlyCache:
    def get_features_cache(self):
        return None

    def get_pseudo_label_cache(self):
        raise AssertionError("PKL fallback must not be used when refined PNG exists")


class PartialRefinedCache:
    def get_features_cache(self):
        return None

    def get_pseudo_label_cache(self):
        return self

    def read_file(self, index):
        return torch.zeros((1, 2, 2), dtype=torch.float32)


class RefinedPseudoLabelPathTests(unittest.TestCase):
    def test_full_training_config_switches_only_refined_label_artifact(self):
        experiment_id = "unit_test_full4040"
        config_path = (
            PROJECT_ROOT
            / "configs"
            / "uscod"
            / "UCOD-DPL_dinov2_aeem_v2_full4040.py"
        )
        with mock.patch.dict(os.environ, {"AEEM_EXPERIMENT_ID": experiment_id}):
            config = CfgNode.load_with_base(str(config_path))

        self.assertEqual(config["dataset_cfg"]["cache_dir"], "./datasets/cache")
        self.assertEqual(
            config["dataset_cfg"]["refined_pseudo_label_dir"],
            f"./artifacts/aeem_v2/{experiment_id}/refined_pseudo_labels",
        )
        self.assertEqual(
            config["exp_name"],
            f"UCOD-DPL_dinov2_aeem_v2_{experiment_id}",
        )

    def test_explicit_path_does_not_change_cache_dir(self):
        self.assertEqual(
            resolve_refined_pseudo_label_dir("cache-root", "artifact/labels"),
            "artifact/labels",
        )
        self.assertEqual(
            resolve_refined_pseudo_label_dir("cache-root", None),
            str(Path("cache-root") / "refined_pseudo_labels"),
        )

    def test_dataloader_passes_optional_refined_path(self):
        config = CfgNode({
            "cache_dir": "cache-root",
            "dataset_dir": "dataset-root",
            "feature_extractor_cfg": {},
            "refined_pseudo_label_dir": "artifact/labels",
            "trainloader_cfg": {"batch_size": 1, "num_workers": 0, "shuffle": False},
            "trainset_cfg": {},
        })
        with mock.patch.object(
            dataloader_utils, "USCODDataset", return_value=DummyDataset()
        ) as dataset_class:
            dataloader_utils.DataLoaderFactory.create_train_loader(config)
        kwargs = dataset_class.call_args.kwargs
        self.assertEqual(kwargs["cache_dir"], "cache-root")
        self.assertEqual(kwargs["refined_pseudo_label_dir"], "artifact/labels")

    def test_dataset_reads_png_from_explicit_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.jpg"
            refined_dir = root / "artifact-labels"
            refined_dir.mkdir()
            Image.new("RGB", (8, 8), color="black").save(image_path)
            refined = np.zeros((8, 8), dtype=np.uint8)
            refined[2:6, 2:6] = 255
            Image.fromarray(refined).save(refined_dir / "sample.png")

            dataset = BaseCODDataset.__new__(BaseCODDataset)
            dataset.image_paths = [image_path]
            dataset.label_paths = []
            dataset.cache_manager = RefinedOnlyCache()
            dataset.config = CfgNode({"feature_size": 4})
            dataset.refined_pseudo_label_dir = str(refined_dir)

            item = dataset[0]
            self.assertEqual(tuple(item["pseudo_label"].shape), (1, 4, 4))
            self.assertGreater(float(item["pseudo_label"].max()), 0.9)

    def test_partial_refined_directory_stacks_png_and_pkl_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refined_dir = root / "partial-labels"
            refined_dir.mkdir()
            image_paths = [root / "refined.jpg", root / "fallback.jpg"]
            for image_path in image_paths:
                Image.new("RGB", (8, 8), color="black").save(image_path)
            Image.new("L", (8, 8), color=255).save(refined_dir / "refined.png")

            dataset = BaseCODDataset.__new__(BaseCODDataset)
            dataset.image_paths = image_paths
            dataset.label_paths = []
            dataset.cache_manager = PartialRefinedCache()
            dataset.config = CfgNode({"feature_size": 4})
            dataset.refined_pseudo_label_dir = str(refined_dir)

            batch = dataloader_utils.collate_fn([dataset[0], dataset[1]])

            self.assertIsInstance(batch["pseudo_label"], torch.Tensor)
            self.assertEqual(tuple(batch["pseudo_label"].shape), (2, 1, 4, 4))


class ControlArtifactTests(unittest.TestCase):
    def _create_control_inputs(self, root: Path):
        dataset_dir = root / "datasets"
        image_dir = dataset_dir / "TR-A" / "im"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (12, 8), color="white").save(image_dir / "a.jpg")
        Image.new("RGB", (10, 6), color="black").save(image_dir / "b.jpg")

        coarse_dir = root / "coarse"
        coarse_dir.mkdir()
        first = np.zeros((1, 4, 4), dtype=np.float32)
        first[:, 1:3, 1:3] = 1.0
        second = np.zeros((1, 4, 4), dtype=np.float32)
        second[:, :, 2:] = 1.0
        for index, mask in enumerate((first, second)):
            with (coarse_dir / f"data_{index}.pkl").open("wb") as file_handle:
                pickle.dump(mask, file_handle)
        (coarse_dir / "index.json").write_text(
            json.dumps({"0": "data_0.pkl", "1": "data_1.pkl"}),
            encoding="utf-8",
        )
        return dataset_dir, coarse_dir

    def test_controls_are_isolated_and_audited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir, coarse_dir = self._create_control_inputs(root)
            artifact_root = root / "artifacts"
            experiment_dir = generate_control_artifact(
                artifact_root=artifact_root,
                experiment_id="control-test",
                dataset_dir=dataset_dir,
                dataset_names=["TR-A"],
                coarse_dir=coarse_dir,
                repo_root=PROJECT_ROOT,
                source_files=[PROJECT_ROOT / "aeem_v2" / "controls.py"],
            )

            hard = np.asarray(Image.open(
                experiment_dir / "controls/hard_coarse/refined_pseudo_labels/a.png"
            ))
            soft = np.asarray(Image.open(
                experiment_dir / "controls/soft_coarse/refined_pseudo_labels/a.png"
            ))
            self.assertEqual(set(np.unique(hard)).difference({0, 255}), set())
            self.assertTrue(np.any((soft > 0) & (soft < 255)))

            manifest = json.loads(
                (experiment_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["hard_output_count"], 2)
            self.assertEqual(manifest["soft_output_count"], 2)
            self.assertTrue((experiment_dir / "input_hashes.json").is_file())
            self.assertTrue((experiment_dir / "output_hashes.json").is_file())

            with self.assertRaises(FileExistsError):
                generate_control_artifact(
                    artifact_root=artifact_root,
                    experiment_id="control-test",
                    dataset_dir=dataset_dir,
                    dataset_names=["TR-A"],
                    coarse_dir=coarse_dir,
                    repo_root=PROJECT_ROOT,
                )


class EvaluationTests(unittest.TestCase):
    def test_boundary_iou_rewards_matching_boundaries(self):
        ground_truth = np.zeros((64, 64), dtype=bool)
        ground_truth[16:48, 16:48] = True
        shifted = np.zeros_like(ground_truth)
        shifted[20:52, 20:52] = True

        self.assertEqual(_boundary_iou(ground_truth, ground_truth, width=2), 1.0)
        self.assertLess(_boundary_iou(shifted, ground_truth, width=2), 1.0)

    def test_parameterized_evaluation_reports_improvement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            baseline_dir = root / "baseline"
            refined_dir = root / "refined"
            for directory in (gt_dir, baseline_dir, refined_dir):
                directory.mkdir()

            gt = np.zeros((32, 32), dtype=np.uint8)
            gt[8:24, 8:24] = 255
            baseline = np.zeros_like(gt)
            baseline[5:21, 5:21] = 255
            Image.fromarray(gt).save(gt_dir / "sample.png")
            Image.fromarray(baseline).save(baseline_dir / "sample.png")
            Image.fromarray(gt).save(refined_dir / "sample.png")
            Image.fromarray(gt).save(gt_dir / "ignored.png")
            Image.fromarray(gt).save(baseline_dir / "ignored.png")
            Image.fromarray(gt).save(refined_dir / "ignored.png")

            output_dir = root / "evaluation"
            evaluate_predictions(
                gt_sets={"TR-A": gt_dir},
                predictions={"baseline": baseline_dir, "refined": refined_dir},
                output_dir=output_dir,
                repo_root=PROJECT_ROOT,
                baseline_name="baseline",
                n_bootstrap=20,
                include_stems={"sample"},
                source_files=[PROJECT_ROOT / "aeem_v2" / "evaluation.py"],
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            comparison = summary["comparisons"]["refined"]["ALL"]
            self.assertGreater(comparison["iou_delta"]["mean"], 0.0)
            self.assertGreater(comparison["bf_delta"]["mean"], 0.0)
            self.assertEqual(comparison["catastrophic_failure_rate"], 0.0)
            self.assertEqual(summary["predictions"]["refined"]["ALL"]["count"], 1)
            self.assertTrue((output_dir / "metrics_per_image.csv").is_file())
            self.assertTrue((output_dir / "report.md").is_file())

    def test_prediction_fallback_completes_partial_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            baseline_dir = root / "baseline"
            refined_dir = root / "refined"
            fallback_dir = root / "fallback"
            for directory in (gt_dir, baseline_dir, refined_dir, fallback_dir):
                directory.mkdir()

            gt = np.zeros((32, 32), dtype=np.uint8)
            gt[8:24, 8:24] = 255
            baseline = np.zeros_like(gt)
            baseline[5:21, 5:21] = 255
            for name in ("primary", "fallback"):
                Image.fromarray(gt).save(gt_dir / f"{name}.png")
                Image.fromarray(baseline).save(baseline_dir / f"{name}.png")
            Image.fromarray(gt).save(refined_dir / "primary.png")
            Image.fromarray(gt).save(fallback_dir / "fallback.png")

            output_dir = root / "evaluation"
            evaluate_predictions(
                gt_sets={"TR-A": gt_dir},
                predictions={"baseline": baseline_dir, "refined": refined_dir},
                prediction_fallbacks={"refined": fallback_dir},
                output_dir=output_dir,
                repo_root=PROJECT_ROOT,
                baseline_name="baseline",
                n_bootstrap=20,
            )

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            comparison = summary["comparisons"]["refined"]["ALL"]
            self.assertEqual(manifest["prediction_fallback_counts"], {"refined": 1})
            self.assertEqual(summary["predictions"]["refined"]["ALL"]["count"], 2)
            self.assertIn("boundary_iou_delta", comparison)
            self.assertIn("absolute_area_error_delta", comparison)
            self.assertIn("component_count_error_delta", comparison)
            self.assertIn("Boundary IoU", (output_dir / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
