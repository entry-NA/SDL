"""Tests for dataset-source isolation label composition."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aeem_v2.composition import (
    compose_label_artifact,
    select_top_fraction_from_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LabelCompositionTests(unittest.TestCase):
    def _create_inputs(self, root: Path):
        aeem_dir = root / "aeem"
        soft_dir = root / "soft"
        aeem_dir.mkdir()
        soft_dir.mkdir()
        for image_name, aeem_value, soft_value in (
            ("sample_a", 25, 125),
            ("sample_b", 50, 200),
        ):
            Image.fromarray(
                np.full((4, 4), aeem_value, dtype=np.uint8)
            ).save(aeem_dir / f"{image_name}.png")
            Image.fromarray(
                np.full((4, 4), soft_value, dtype=np.uint8)
            ).save(soft_dir / f"{image_name}.png")

        cohort_path = root / "cohort.json"
        cohort_path.write_text(json.dumps({
            "cohort_size": 2,
            "dataset_counts": {"TR-A": 1, "TR-B": 1},
            "samples": [
                {"dataset": "TR-A", "image_name": "sample_a"},
                {"dataset": "TR-B", "image_name": "sample_b"},
            ],
        }), encoding="utf-8")
        return cohort_path, aeem_dir, soft_dir

    def test_composes_aeem_for_selected_dataset_and_soft_for_other(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cohort_path, aeem_dir, soft_dir = self._create_inputs(root)
            experiment_dir = compose_label_artifact(
                artifact_root=root / "artifacts",
                experiment_id="isolate-a",
                cohort_path=cohort_path,
                aeem_dir=aeem_dir,
                soft_dir=soft_dir,
                aeem_datasets=["TR-A"],
                repo_root=PROJECT_ROOT,
            )

            output_dir = experiment_dir / "refined_pseudo_labels"
            selected = np.asarray(Image.open(output_dir / "sample_a.png"))
            fallback = np.asarray(Image.open(output_dir / "sample_b.png"))
            self.assertTrue(np.all(selected == 25))
            self.assertTrue(np.all(fallback == 200))

            manifest = json.loads(
                (experiment_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["output_count"], 2)
            self.assertEqual(manifest["source_counts"], {"aeem": 1, "soft": 1})
            self.assertEqual(manifest["aeem_datasets"], ["TR-A"])

            with self.assertRaises(FileExistsError):
                compose_label_artifact(
                    artifact_root=root / "artifacts",
                    experiment_id="isolate-a",
                    cohort_path=cohort_path,
                    aeem_dir=aeem_dir,
                    soft_dir=soft_dir,
                    aeem_datasets=["TR-A"],
                    repo_root=PROJECT_ROOT,
                )

    def test_rejects_unknown_dataset_before_creating_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cohort_path, aeem_dir, soft_dir = self._create_inputs(root)
            artifact_root = root / "artifacts"
            with self.assertRaisesRegex(ValueError, "not present"):
                compose_label_artifact(
                    artifact_root=artifact_root,
                    experiment_id="invalid",
                    cohort_path=cohort_path,
                    aeem_dir=aeem_dir,
                    soft_dir=soft_dir,
                    aeem_datasets=["TR-MISSING"],
                    repo_root=PROJECT_ROOT,
                )
            self.assertFalse((artifact_root / "invalid").exists())

    def test_selects_deterministic_top_fraction_from_nested_audit_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            rows = [
                {"dataset": "TR-B", "image_name": "b", "selected": {"q_semantic": 0.8}},
                {"dataset": "TR-B", "image_name": "a", "selected": {"q_semantic": 0.8}},
                {"dataset": "TR-B", "image_name": "c", "selected": {"q_semantic": 0.9}},
                {"dataset": "TR-B", "image_name": "d", "selected": {"q_semantic": 0.1}},
            ]
            audit_path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            selected, metadata = select_top_fraction_from_audit(
                audit_path=audit_path,
                dataset="TR-B",
                score_field="selected.q_semantic",
                fraction=0.5,
            )
            self.assertEqual(selected, {"a", "c"})
            self.assertEqual(metadata["selected_count"], 2)
            self.assertEqual(metadata["selected_score_min"], 0.8)


if __name__ == "__main__":
    unittest.main()
