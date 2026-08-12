"""Tests for the reliable-replacement verification artifact."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.prepare_reliable_replacement_control import (
    compose_reliable_replacement_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReliableReplacementControlTests(unittest.TestCase):
    def _write_minimal_inputs(self, root):
        aeem_dir = root / "aeem"
        naive_dir = root / "naive"
        soft_dir = root / "soft"
        for directory in (aeem_dir, naive_dir, soft_dir):
            directory.mkdir()
        Image.fromarray(np.full((4, 4), 25, dtype=np.uint8)).save(
            aeem_dir / "selected.png"
        )
        cohort_path = root / "cohort.json"
        cohort_path.write_text(
            json.dumps({
                "cohort_size": 1,
                "samples": [{"dataset": "TR-A", "image_name": "selected"}],
            }),
            encoding="utf-8",
        )
        selection_path = root / "selection.jsonl"
        selection_path.write_text(
            json.dumps({
                "dataset": "TR-A",
                "image_name": "selected",
                "source_type": "aeem",
            }),
            encoding="utf-8",
        )
        return cohort_path, selection_path, aeem_dir, naive_dir, soft_dir

    def test_composes_aeem_naive_and_soft_fallback_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aeem_dir = root / "aeem"
            naive_dir = root / "naive"
            soft_dir = root / "soft"
            for directory in (aeem_dir, naive_dir, soft_dir):
                directory.mkdir()

            for image_name, value in (("selected", 25), ("naive", 50)):
                Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(
                    aeem_dir / f"{image_name}.png"
                )
            Image.fromarray(np.full((4, 4), 125, dtype=np.uint8)).save(
                naive_dir / "naive.png"
            )
            for image_name in ("selected", "naive", "fallback"):
                Image.fromarray(np.full((4, 4), 200, dtype=np.uint8)).save(
                    soft_dir / f"{image_name}.png"
                )

            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps({
                    "cohort_size": 3,
                    "samples": [
                        {"dataset": "TR-A", "image_name": "selected"},
                        {"dataset": "TR-B", "image_name": "naive"},
                        {"dataset": "TR-B", "image_name": "fallback"},
                    ],
                }),
                encoding="utf-8",
            )
            selection_path = root / "selection.jsonl"
            selection_path.write_text(
                "\n".join(json.dumps(row) for row in (
                    {"dataset": "TR-A", "image_name": "selected", "source_type": "aeem"},
                    {"dataset": "TR-B", "image_name": "naive", "source_type": "soft"},
                    {"dataset": "TR-B", "image_name": "fallback", "source_type": "soft"},
                )),
                encoding="utf-8",
            )

            experiment_dir = compose_reliable_replacement_artifact(
                artifact_root=root / "artifacts",
                experiment_id="reliable-control",
                cohort_path=cohort_path,
                selection_audit_path=selection_path,
                aeem_dir=aeem_dir,
                naive_dir=naive_dir,
                soft_dir=soft_dir,
                repo_root=PROJECT_ROOT,
            )

            output_dir = experiment_dir / "refined_pseudo_labels"
            output_values = {
                name: int(np.asarray(Image.open(output_dir / f"{name}.png"))[0, 0])
                for name in ("selected", "naive", "fallback")
            }
            self.assertEqual(
                output_values,
                {"selected": 25, "naive": 125, "fallback": 200},
            )
            manifest = json.loads(
                (experiment_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["output_count"], 3)
            self.assertEqual(
                manifest["source_counts"],
                {"aeem": 1, "naive_sam2": 1, "soft_fallback": 1},
            )

            audit_rows = [
                json.loads(line)
                for line in (experiment_dir / "audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(audit_rows), 3)
            self.assertTrue(all(
                row["source_sha256"] == row["output_sha256"]
                for row in audit_rows
            ))

    def test_refuses_to_overwrite_existing_experiment_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._write_minimal_inputs(root)
            artifact_root = root / "artifacts"
            existing_dir = artifact_root / "reliable-control"
            existing_dir.mkdir(parents=True)
            sentinel = existing_dir / "keep.txt"
            sentinel.write_text("unchanged", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                compose_reliable_replacement_artifact(
                    artifact_root=artifact_root,
                    experiment_id="reliable-control",
                    cohort_path=inputs[0],
                    selection_audit_path=inputs[1],
                    aeem_dir=inputs[2],
                    naive_dir=inputs[3],
                    soft_dir=inputs[4],
                    repo_root=PROJECT_ROOT,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_invalid_selection_fails_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._write_minimal_inputs(root)
            inputs[1].write_text("", encoding="utf-8")
            artifact_root = root / "artifacts"

            with self.assertRaises(ValueError):
                compose_reliable_replacement_artifact(
                    artifact_root=artifact_root,
                    experiment_id="reliable-control",
                    cohort_path=inputs[0],
                    selection_audit_path=inputs[1],
                    aeem_dir=inputs[2],
                    naive_dir=inputs[3],
                    soft_dir=inputs[4],
                    repo_root=PROJECT_ROOT,
                )

            self.assertFalse((artifact_root / "reliable-control").exists())

    def test_missing_selected_source_fails_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = self._write_minimal_inputs(root)
            (inputs[2] / "selected.png").unlink()
            artifact_root = root / "artifacts"

            with self.assertRaises(FileNotFoundError):
                compose_reliable_replacement_artifact(
                    artifact_root=artifact_root,
                    experiment_id="reliable-control",
                    cohort_path=inputs[0],
                    selection_audit_path=inputs[1],
                    aeem_dir=inputs[2],
                    naive_dir=inputs[3],
                    soft_dir=inputs[4],
                    repo_root=PROJECT_ROOT,
                )

            self.assertFalse((artifact_root / "reliable-control").exists())


if __name__ == "__main__":
    unittest.main()
