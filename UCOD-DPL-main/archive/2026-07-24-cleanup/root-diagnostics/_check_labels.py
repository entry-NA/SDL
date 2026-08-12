import numpy as np
from PIL import Image
import os, glob

label_dir = r'C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main\datasets\cache\refined_pseudo_labels'
files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

# 统计所有标签
zero_count = 0
fg_ratios = []
for f in files:
    img = np.array(Image.open(f))
    ratio = (img > 0).sum() / img.size
    fg_ratios.append(ratio)
    if ratio == 0:
        zero_count += 1

print(f'Total labels: {len(files)}')
print(f'Fully empty (zero foreground): {zero_count}')
print(f'Mean fg ratio: {np.mean(fg_ratios):.6f}')
print(f'Median fg ratio: {np.median(fg_ratios):.6f}')
print(f'Max fg ratio: {np.max(fg_ratios):.6f}')
print()

# 看前10张
for f in files[:10]:
    img = np.array(Image.open(f))
    ratio = (img > 0).sum() / img.size
    print(f'{os.path.basename(f)}: shape={img.shape}, fg_ratio={ratio:.4f}')
