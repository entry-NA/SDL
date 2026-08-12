# === Gamma 扫描 v2: 用 PNG 标签 + coarse 标签计算 S 分布 ===
import sys, os, json
sys.path.insert(0, "scripts")
import offline_sam2_refine as osm
import numpy as np, cv2, pickle
from PIL import Image

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align

# 读 per_image_stats 取 iou_pred 均值
stats_path = "datasets/cache/refined_pseudo_labels/per_image_stats.json"
with open(stats_path) as f:
    per_stats = json.load(f)

# Build name -> (image_path, pkl_path)
coarse_dir = CFG["coarse_dir"]
with open(os.path.join(coarse_dir, "index.json")) as f:
    index_map = json.load(f)

all_images = []
for d in CFG["image_dirs"]:
    for f in sorted(os.listdir(d)):
        if f.endswith((".jpg", ".png")):
            all_images.append((os.path.join(d, f), os.path.splitext(f)[0]))
all_images.sort(key=lambda x: x[1])

name_to_info = {}
for idx, (img_path, name) in enumerate(all_images):
    k = str(idx)
    if k in index_map:
        name_to_info[name] = (img_path, os.path.join(coarse_dir, index_map[k]), idx)

PNG_DIR = "datasets/cache/refined_pseudo_labels"
png_names = set(f.replace(".png", "") for f in os.listdir(PNG_DIR) if f.endswith(".png") and not f.startswith("per_image"))

# Collect: for each image with PNG + coarse, compute iou_ori and edge_align
S_components = []  # (alpha*score, beta*iou_ori, edge_norm)

for name in list(png_names)[:300]:
    if name not in name_to_info:
        continue
    img_path, pkl_path, idx = name_to_info[name]
    
    # Load coarse
    try:
        with open(pkl_path, "rb") as fh:
            coarse_pkl = pickle.load(fh)
    except:
        continue
    if hasattr(coarse_pkl, "numpy"):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    
    # Load image + PNG mask
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    mask = np.array(Image.open(os.path.join(PNG_DIR, name + ".png")))
    mask_bool = mask >= 128
    
    iou_ori = compute_iou(mask_bool, coarse_bool)
    edge_norm = compute_edge_align(mask_bool, image)  # now [0,1]
    
    # Estimate iou_pred from per_image_stats or use mean
    iou_pred = 0.75  # typical SAM2 self-score mean
    if name in per_stats:
        ps = per_stats[name]
        if "sam_score" in ps:
            iou_pred = ps["sam_score"]
    
    S_components.append((0.3 * iou_pred, 0.4 * iou_ori, edge_norm))

alpha_arr = np.array([c[0] for c in S_components])
beta_arr = np.array([c[1] for c in S_components])
edge_arr = np.array([c[2] for c in S_components])

s_low, s_high = 0.20, 0.80

print("=" * 60)
print("Gamma Sweep Calibration v2 (n=%d from PNG labels)" % len(S_components))
print("=" * 60)
print()
print("Component stats:")
print("  alpha*score: mean=%.4f, min=%.4f, max=%.4f" % (alpha_arr.mean(), alpha_arr.min(), alpha_arr.max()))
print("  beta*IoU:    mean=%.4f, min=%.4f, max=%.4f" % (beta_arr.mean(), beta_arr.min(), beta_arr.max()))
print("  edge_norm:   mean=%.4f, min=%.4f, max=%.4f" % (edge_arr.mean(), edge_arr.min(), edge_arr.max()))
print()
print("Thresholds: s_lower=%.2f, s_upper=%.2f" % (s_low, s_high))
print()

fmt = "%-8s %-10s %-10s %-10s %-12s %-12s %-12s"
print(fmt % ("gamma", "S_mean", "S_median", "S_max", "FULL%", "FUSION%", "FALLBACK%"))
print("-" * 72)

for gamma in [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]:
    S_vals = alpha_arr + beta_arr + gamma * edge_arr
    full_pct = 100.0 * (S_vals > s_high).sum() / len(S_vals)
    fusion_pct = 100.0 * ((S_vals >= s_low) & (S_vals <= s_high)).sum() / len(S_vals)
    fallback_pct = 100.0 * (S_vals < s_low).sum() / len(S_vals)
    print("%-8d %-10.4f %-10.4f %-10.4f %-12.1f %-12.1f %-12.1f" % (
        gamma, S_vals.mean(), np.median(S_vals), S_vals.max(),
        full_pct, fusion_pct, fallback_pct))

print()
print("-- 消融实验 v5 参考: FULL=96.4%, FUSION=3.4%, FALLBACK=0.1% --")
print("-- 修前 buggy (gamma=0.3, edge=[0,255]): FULL=97.9%, FUSION=2.0%, FALLBACK=0.1% --")
