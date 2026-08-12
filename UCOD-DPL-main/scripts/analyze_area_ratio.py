import os, json, pickle, sys, cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

REFINED_DIR = './datasets/cache/refined_pseudo_labels'
COARSE_DIR = r'C:\Users\23991\Desktop\plable\TR-CAMO+TR-COD10K'
IMAGE_DIRS = [
    r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-CAMO\im',
    r'C:\Users\23991\Desktop\RefCOD (1)\RefCOD\TR-COD10K\im',
]

image_paths = []
for d in IMAGE_DIRS:
    image_paths.extend(sorted(str(p) for p in Path(d).glob('*.jpg')))
    image_paths.extend(sorted(str(p) for p in Path(d).glob('*.png')))

with open(os.path.join(COARSE_DIR, 'index.json'), 'r') as f:
    index_map = json.load(f)

ratios = []
areas_sam_list = []
areas_coarse_list = []
skipped = 0

for idx in tqdm(range(len(image_paths))):
    img_name = os.path.splitext(os.path.basename(image_paths[idx]))[0]
    png_path = os.path.join(REFINED_DIR, f'{img_name}.png')
    if not os.path.exists(png_path):
        skipped += 1
        continue

    sam_mask = np.array(Image.open(png_path).convert('L')) >= 128
    area_sam = sam_mask.sum()
    if area_sam == 0:
        skipped += 1
        continue

    with open(os.path.join(COARSE_DIR, index_map[str(idx)]), 'rb') as f:
        coarse = pickle.load(f)
    if hasattr(coarse, 'numpy'):
        coarse = coarse.numpy()
    coarse = coarse.squeeze()

    # upsample 16x16 -> 原图分辨率 (跟 offline_sam2_refine.py 的 load_coarse_label 一致)
    coarse_up = cv2.resize(
        (coarse * 255).astype(np.uint8),
        (sam_mask.shape[1], sam_mask.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    coarse_bool = coarse_up >= 128
    area_coarse = coarse_bool.sum()
    if area_coarse == 0:
        skipped += 1
        continue

    ratios.append(float(area_sam) / area_coarse)
    areas_sam_list.append(area_sam)
    areas_coarse_list.append(area_coarse)

if len(ratios) == 0:
    print('ERROR: 0 valid samples')
    sys.exit(1)

ratios = np.array(ratios)
areas_sam_arr = np.array(areas_sam_list)
areas_coarse_arr = np.array(areas_coarse_list)

print(f'\nValid: {len(ratios)}  Skipped: {skipped}')
print(f'\n=== 面积统计 (原图像素) ===')
print(f'SAM2 area:   mean={areas_sam_arr.mean():.0f}  median={np.median(areas_sam_arr):.0f}')
print(f'Coarse area: mean={areas_coarse_arr.mean():.0f}  median={np.median(areas_coarse_arr):.0f}')

print(f'\n=== 面积比 r = area_sam / area_coarse ===')
print(f'mean={ratios.mean():.4f}  std={ratios.std():.4f}  median={np.median(ratios):.4f}')
print(f'min={ratios.min():.4f}  max={ratios.max():.4f}')
for t in [5, 10, 25, 50, 75, 90, 95]:
    print(f'P{t} = {np.percentile(ratios, t):.4f}')

bins = [(0,0.5), (0.5,0.7), (0.7,0.75), (0.75,0.8), (0.8,0.85), (0.85,0.9),
        (0.9,0.95), (0.95,1.0), (1.0,1.05), (1.05,1.1), (1.1,1.2), (1.2,1.5),
        (1.5,2.0), (2.0,999)]
print('\nHistogram:')
for lo, hi in bins:
    c = ((ratios >= lo) & (ratios < hi)).sum()
    pct = c / len(ratios) * 100
    bar = '#' * max(1, int(pct))
    print(f'  [{lo:.2f}, {hi:5.1f})  {c:5d} ({pct:5.1f}%) {bar}')

for th in [1.0, 0.9, 0.85, 0.80, 0.75]:
    c = (ratios < th).sum()
    print(f'r < {th:.2f}: {c} ({100*c/len(ratios):.1f}%)')
