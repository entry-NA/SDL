"""Compose an AEEM/Soft dataset-source isolation artifact."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeem_v2.composition import (
    compose_label_artifact,
    select_top_fraction_from_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose immutable mixed AEEM and Soft-Coarse training labels."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("experiments/aeem_v2_m2_full4040.json"),
    )
    parser.add_argument("--aeem-dir", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument(
        "--aeem-dataset",
        action="append",
        required=True,
        help="Dataset that uses AEEM labels; repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/aeem_v2")
    )
    parser.add_argument("--ranked-audit", type=Path)
    parser.add_argument("--ranked-dataset")
    parser.add_argument("--ranked-field", default="selected.q_semantic")
    parser.add_argument("--ranked-fraction", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked_images = set()
    selection_metadata = None
    ranked_args = (
        args.ranked_audit,
        args.ranked_dataset,
        args.ranked_fraction,
    )
    if any(value is not None for value in ranked_args):
        if not all(value is not None for value in ranked_args):
            raise ValueError(
                "--ranked-audit, --ranked-dataset and --ranked-fraction "
                "must be provided together"
            )
        ranked_images, selection_metadata = select_top_fraction_from_audit(
            audit_path=args.ranked_audit,
            dataset=args.ranked_dataset,
            score_field=args.ranked_field,
            fraction=args.ranked_fraction,
        )

    experiment_dir = compose_label_artifact(
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
        cohort_path=args.cohort,
        aeem_dir=args.aeem_dir,
        soft_dir=args.soft_dir,
        aeem_datasets=args.aeem_dataset,
        repo_root=PROJECT_ROOT,
        aeem_image_names=ranked_images,
        selection_metadata=selection_metadata,
        source_files=[
            PROJECT_ROOT / "aeem_v2" / "composition.py",
            PROJECT_ROOT / "scripts" / "compose_aeem_v2_labels.py",
            *( [args.ranked_audit] if args.ranked_audit else [] ),
        ],
        show_progress=True,
    )
    print(f"Composition completed: {experiment_dir}")


if __name__ == "__main__":
    main()
