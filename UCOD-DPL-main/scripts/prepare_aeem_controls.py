"""Generate isolated Hard-Coarse and Soft-Coarse AEEM controls."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeem_v2.controls import generate_control_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate immutable Hard-Coarse and Soft-Coarse PNG controls."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/aeem_v2"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/RefCOD"))
    parser.add_argument("--dataset", default="TR-CAMO+TR-COD10K")
    parser.add_argument(
        "--coarse-dir",
        type=Path,
        default=Path("datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = [
        Path(__file__),
        PROJECT_ROOT / "aeem_v2" / "artifacts.py",
        PROJECT_ROOT / "aeem_v2" / "controls.py",
        PROJECT_ROOT / "data" / "datasets" / "base_dataset.py",
    ]
    experiment_dir = generate_control_artifact(
        artifact_root=args.artifact_root,
        experiment_id=args.experiment_id,
        dataset_dir=args.dataset_dir,
        dataset_names=args.dataset.split("+"),
        coarse_dir=args.coarse_dir,
        repo_root=PROJECT_ROOT,
        threshold=args.threshold,
        source_files=source_files,
    )
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "experiment_dir": str(experiment_dir.resolve()),
        "hard_output_count": manifest["hard_output_count"],
        "soft_output_count": manifest["soft_output_count"],
        "status": manifest["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
