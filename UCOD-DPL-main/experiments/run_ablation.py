"""run_ablation.py — Ablation study for SAM2 refinement mechanisms.

Systematically disables each of the 4 gating mechanisms to measure their
individual contribution. Each variant requires: (1) generate refined labels
with specific mechanisms on/off, (2) train UCOD-DPL first stage,
(3) evaluate on COD10K/CAMO/CHAMELEON/NC4K, (4) record S_m, E_m, F_beta_w, MAE.

Usage:
    python experiments/run_ablation.py  # prints commands for all 6 variants
"""
import os, sys, json

ABLATION_VARIANTS = {
    'baseline': {
        'description': 'UCOD-DPL original (16x16 pkl pseudo-labels)',
        'flags': {'use_original_pkl': True},
    },
    'naive_sam2': {
        'description': 'Naive SAM2 — direct mask prompt, no gating',
        'flags': {'adaptive_prompt': False, 'mask_selection': False,
                  'edge_gating': False, 'local_sam': False},
    },
    'adaptive_prompt': {
        'description': 'Naive SAM2 + Adaptive Prompt + Hierarchical Neg Sampling',
        'flags': {'adaptive_prompt': True, 'mask_selection': False,
                  'edge_gating': False, 'local_sam': False},
    },
    'mask_selection': {
        'description': 'Above + Truncated Multi-Mask Selection',
        'flags': {'adaptive_prompt': True, 'mask_selection': True,
                  'edge_gating': False, 'local_sam': False},
    },
    'edge_gating': {
        'description': 'Above + Edge-Aware Confidence Gating',
        'flags': {'adaptive_prompt': True, 'mask_selection': True,
                  'edge_gating': True, 'local_sam': False},
    },
    'full_model': {
        'description': 'Full model — all mechanisms including Local-SAM',
        'flags': {'adaptive_prompt': True, 'mask_selection': True,
                  'edge_gating': True, 'local_sam': True},
    },
}

RESULTS_CSV_HEADER = "variant,dataset,S_m,E_m,F_beta_w,MAE\n"


def generate_pseudo_labels(variant_name, flags):
    """Generate refined pseudo-labels for a specific ablation variant.

    For 'baseline': use original 16x16 pkl labels directly (skip SAM2 refinement).
    For other variants: run offline_sam2_refine.py with specific mechanisms
    enabled/disabled via a temporary flag file.
    """
    if flags.get('use_original_pkl'):
        print(f"[{variant_name}] Using original 16x16 pkl pseudo-labels — no refinement needed")
        return

    flag_path = f'./experiments/ablation_flags_{variant_name}.json'
    with open(flag_path, 'w') as f:
        json.dump(flags, f)

    output_dir = f'./datasets/cache/refined_pseudo_labels_{variant_name}'
    cmd = (
        f"python scripts/offline_sam2_refine.py "
        f"--flags {flag_path} "
        f"--output_dir {output_dir}"
    )
    print(f"[{variant_name}] Refinement command:\n  {cmd}")


def train_and_eval(variant_name):
    """Print training and evaluation commands for a variant."""
    train_cmd = (
        f"bash ./scripts/launch_train_first_stage.sh "
        f"-c ./configs/uscod/UCOD-DPL_dinov2.py"
    )
    print(f"[{variant_name}] Training command:\n  {train_cmd}")

    for dataset in ['TE-COD10K', 'TE-CAMO', 'CHAMELEON', 'NC4K']:
        eval_cmd = (
            f"bash ./scripts/launch_val_first_stage.sh "
            f"-c ./configs/uscod/UCOD-DPL_dinov2.py "
            f"-m ./work/<exp_name>/ckp/epoch25.pth "
            f"-d {dataset}"
        )
        print(f"[{variant_name}] Eval {dataset}:\n  {eval_cmd}")


def main():
    print("=" * 60)
    print("ABLATION STUDY — SAM2 Refinement Mechanisms")
    print("=" * 60)
    print()

    for variant_name, cfg in ABLATION_VARIANTS.items():
        print(f"\n{'─' * 50}")
        print(f"Variant: {variant_name}")
        print(f"Description: {cfg['description']}")
        print(f"{'─' * 50}")

        # Step 1: Generate refined labels (or use baseline)
        generate_pseudo_labels(variant_name, cfg['flags'])

        # Step 2: Train
        train_and_eval(variant_name)

    print("\n" + "=" * 60)
    print("After running all variants, collect results into CSV:")
    print(f"Header: {RESULTS_CSV_HEADER}")
    print("=" * 60)


if __name__ == '__main__':
    main()
