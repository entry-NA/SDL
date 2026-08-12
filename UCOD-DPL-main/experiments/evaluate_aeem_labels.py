"""Evaluate AEEM label directories with explicit GT and prediction paths."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeem_v2.evaluation import evaluate_predictions


def parse_named_paths(values: Sequence[str], argument_name: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument_name} must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate name for {argument_name}: {name}")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        parsed[name] = path
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-configuration GT diagnostics for pseudo-label directories."
    )
    parser.add_argument("--gt-set", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--prediction", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument(
        "--prediction-fallback",
        action="append",
        metavar="NAME=PATH",
        help="Fill prediction stems missing from its primary directory.",
    )
    parser.add_argument("--baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--band-width", action="append", type=int, dest="band_widths")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--boundary-width-ratio", type=float, default=0.02)
    parser.add_argument(
        "--cohort",
        type=Path,
        help="Optional no-GT cohort JSON restricting the diagnostic to frozen stems.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_sets = parse_named_paths(args.gt_set, "--gt-set")
    predictions = parse_named_paths(args.prediction, "--prediction")
    prediction_fallbacks = parse_named_paths(
        args.prediction_fallback or [], "--prediction-fallback"
    )
    band_widths = tuple(args.band_widths or (5, 10, 20))
    include_stems = None
    source_files = [Path(__file__), PROJECT_ROOT / "aeem_v2" / "evaluation.py"]
    if args.cohort is not None:
        cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
        if cohort.get("generated_without_gt") is not True:
            raise ValueError("Cohort must declare generated_without_gt=true")
        include_stems = [sample["image_name"] for sample in cohort["samples"]]
        source_files.append(args.cohort)
    output_dir = evaluate_predictions(
        gt_sets=gt_sets,
        predictions=predictions,
        output_dir=args.output_dir,
        repo_root=PROJECT_ROOT,
        baseline_name=args.baseline,
        threshold=args.threshold,
        band_widths=band_widths,
        n_bootstrap=args.bootstrap,
        include_stems=include_stems,
        source_files=source_files,
        prediction_fallbacks=prediction_fallbacks,
        boundary_width_ratio=args.boundary_width_ratio,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "gt_count": manifest["gt_count"],
        "output_dir": str(output_dir.resolve()),
        "row_count": manifest["row_count"],
        "status": manifest["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
