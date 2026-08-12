import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_multiseed_results import (
    DATASETS,
    GROUPS,
    METRICS,
    SEEDS,
    read_complete_results,
    summarize,
    write_template,
)


class MultiSeedSummaryTests(unittest.TestCase):
    def test_template_and_summary_cover_all_groups_seeds_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.csv"
            output_path = Path(tmpdir) / "summary.csv"
            write_template(input_path)

            with input_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(GROUPS) * len(SEEDS) * len(DATASETS))

            for row in rows:
                seed_offset = SEEDS.index(int(row["seed"])) * 0.01
                for metric in METRICS:
                    row[metric] = str(0.5 + seed_offset)
            with input_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            summarize(read_complete_results(input_path), output_path)
            with output_path.open(newline="", encoding="utf-8-sig") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), len(GROUPS) * (len(DATASETS) + 1))
            macro = next(row for row in summary if row["group"] == "full_m4" and row["dataset"] == "MACRO")
            self.assertEqual(macro["n_seeds"], "3")
            self.assertEqual(macro["E_MEAN_mean"], "0.510000")
            self.assertEqual(macro["E_MEAN_std"], "0.010000")


if __name__ == "__main__":
    unittest.main()
