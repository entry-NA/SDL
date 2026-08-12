"""plot_figure4.py — Generate 5-column visualization for SAM2 refinement.

Columns: (a) Original Image  (b) Coarse 16x16 Pseudo-label
         (c) Raw SAM2 Output (d) Ours (Refined)  (e) Ground Truth

Rows: 4 representative cases (large target, small target, low contrast, multi-object)

Usage:
    python experiments/plot_figure4.py
    # Output: ./experiments/figure4_sam2_refinement.png
"""
import os, sys, pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
REFCOD_BASE = r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD'
PLABLE_DIR = r'C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K'
REFINED_DIR = './datasets/cache/refined_pseudo_labels'
OUTPUT_PATH = './experiments/figure4_sam2_refinement.png'

# ── Representative cases (img_name, dataset, index_in_pkl) ─────────────
# Index mapping: TR-CAMO images come first (indices 0-999),
# TR-COD10K images follow (indices 1000-4039)
CASES = [
    {
        'name': 'Large Target',
        'img': 'camourflage_00001.jpg',
        'dataset': 'TR-CAMO',
        'pkl_idx': 0,
    },
    {
        'name': 'Small Target',
        'img': 'camourflage_00150.jpg',
        'dataset': 'TR-CAMO',
        'pkl_idx': 149,
    },
    {
        'name': 'Low Contrast',
        'img': 'COD10K-CAM-1-Aquatic-3-Bat-242.jpg',
        'dataset': 'TR-COD10K',
        'pkl_idx': 1241,
    },
    {
        'name': 'Multi-Object',
        'img': 'COD10K-CAM-1-Aquatic-1-Crab-88.jpg',
        'dataset': 'TR-COD10K',
        'pkl_idx': 1087,
    },
]


def load_coarse_label(pkl_idx):
    """Load and upsample 16x16 pkl coarse label to visualize."""
    import json
    with open(os.path.join(PLABLE_DIR, 'index.json'), 'r') as f:
        idx_map = json.load(f)
    pkl_path = os.path.join(PLABLE_DIR, idx_map[str(pkl_idx)])
    with open(pkl_path, 'rb') as f:
        coarse = pickle.load(f)
    if hasattr(coarse, 'numpy'):
        coarse = coarse.numpy()
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    return coarse


def load_refined_mask(img_name):
    """Load refined PNG mask if available."""
    png_path = os.path.join(REFINED_DIR, f"{img_name}.png")
    if os.path.exists(png_path):
        from PIL import Image
        img = Image.open(png_path).convert('L')
        return np.array(img)
    return None


def plot_one_case(axs, case):
    """Fill one row of the figure."""
    img_name = os.path.splitext(case['img'])[0]
    img_path = os.path.join(REFCOD_BASE, case['dataset'], 'im', case['img'])
    gt_path = os.path.join(REFCOD_BASE, case['dataset'], 'gt', case['img'].replace('.jpg', '.png'))

    # (a) Original
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    axs[0].imshow(img)
    axs[0].set_title('(a) Original Image', fontsize=9)
    axs[0].axis('off')

    # (b) Coarse 16x16 (upsample to image size for visualization)
    coarse = load_coarse_label(case['pkl_idx'])
    coarse_up = cv2.resize(coarse, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    axs[1].imshow(coarse_up, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('(b) Coarse 16x16', fontsize=9)
    axs[1].axis('off')

    # (c) Raw SAM2 — placeholder (requires running with naive_sam2 flags)
    axs[2].text(0.5, 0.5, 'Raw SAM2\n(run ablation)', ha='center', va='center',
                transform=axs[2].transAxes, fontsize=8, color='gray')
    axs[2].set_title('(c) Raw SAM2 Output', fontsize=9)
    axs[2].axis('off')

    # (d) Ours refined
    refined = load_refined_mask(img_name)
    if refined is not None:
        refined_disp = cv2.resize(refined, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        axs[3].imshow(refined_disp, cmap='gray', vmin=0, vmax=255)
        axs[3].set_title('(d) Ours (Refined)', fontsize=9)
    else:
        axs[3].text(0.5, 0.5, 'Refined masks\nnot yet generated', ha='center',
                    va='center', transform=axs[3].transAxes, fontsize=8, color='gray')
        axs[3].set_title('(d) Ours (Refined)', fontsize=9)
    axs[3].axis('off')

    # (e) Ground Truth
    if os.path.exists(gt_path):
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        axs[4].imshow(gt, cmap='gray', vmin=0, vmax=255)
        axs[4].set_title('(e) Ground Truth', fontsize=9)
    else:
        axs[4].text(0.5, 0.5, 'GT not found', ha='center', va='center',
                    transform=axs[4].transAxes, fontsize=8, color='gray')
        axs[4].set_title('(e) Ground Truth', fontsize=9)
    axs[4].axis('off')

    # Row label
    axs[0].set_ylabel(case['name'], fontsize=11, fontweight='bold', rotation=0,
                      labelpad=40, va='center')


def main():
    print("Generating Figure 4: SAM2 Refinement Visualization...")

    n_rows = len(CASES)
    n_cols = 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))

    # Handle single-row case
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    for row, case in enumerate(CASES):
        print(f"  Row {row+1}/{n_rows}: {case['name']} ({case['img']})")
        plot_one_case(axs[row], case)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
