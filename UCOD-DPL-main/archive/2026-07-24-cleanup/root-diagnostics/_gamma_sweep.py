# === Gamma 扫描校准脚本 ===
# 在 200 张图上测试 gamma=5~50，输出每个 gamma 下的门控分布
import sys, os
sys.path.insert(0, "scripts")
import offline_sam2_refine as osm
import numpy as np, cv2, json, pickle

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align  # NOW returns [0,1] after fix

RAW_DIR = "datasets/cache/raw_sam2_outputs"
coarse_dir = CFG["coarse_dir"]
with open(os.path.join(coarse_dir, "index.json")) as f:
    index_map = json.load(f)

# Build sorted image list
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

npz_names = set(f.replace(".npz", "") for f in os.listdir(RAW_DIR) if f.endswith(".npz"))

# Collect best mask info for each image (with 3-factor selection)
all_best_infos = []
count = 0
for name in npz_names:
    if name not in name_to_info:
        continue
    img_path, pkl_path, idx = name_to_info[name]
    with open(pkl_path, "rb") as fh:
        coarse_pkl = pickle.load(fh)
    if hasattr(coarse_pkl, "numpy"):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    try:
        data = np.load(os.path.join(RAW_DIR, name + ".npz"), allow_pickle=True)
    except:
        continue
    masks = list(data["masks"])
    scores = list(data["scores"])
    
    best_s = -float("inf")
    best_info = None
    for i, m in enumerate(masks):
        mb = m.astype(bool) if m.dtype != bool else m
        if mb.shape[:2] != (h, w):
            mb = cv2.resize(mb.astype(np.uint8)*255, (w, h), cv2.INTER_NEAREST) >= 128
        iou_val = compute_iou(mb, coarse_bool)
        edge_norm = compute_edge_align(mb, image)  # now [0,1]
        s = 0.3 * scores[i] + 0.4 * iou_val + 0.3 * edge_norm
        if s > best_s:
            best_s = s
            best_info = (scores[i], iou_val, edge_norm)
    
    if best_info is not None:
        all_best_infos.append(best_info)
        count += 1
        if count >= 200:
            break

print("=" * 60)
print("Gamma Sweep Calibration (n=%d)" % len(all_best_infos))
print("=" * 60)
print()

# Component stats
alpha_vals = np.array([0.3 * info[0] for info in all_best_infos])
beta_vals = np.array([0.4 * info[1] for info in all_best_infos])
edge_vals = np.array([info[2] for info in all_best_infos])

s_low = 0.20
s_high = 0.80

print("Component stats (n=%d):" % len(all_best_infos))
print("  alpha*score: mean=%.4f, min=%.4f, max=%.4f" % (alpha_vals.mean(), alpha_vals.min(), alpha_vals.max()))
print("  beta*IoU:    mean=%.4f, min=%.4f, max=%.4f" % (beta_vals.mean(), beta_vals.min(), beta_vals.max()))
print("  edge_norm:   mean=%.4f, min=%.4f, max=%.4f" % (edge_vals.mean(), edge_vals.min(), edge_vals.max()))
print()
print("Testing gamma values with thresholds [%.2f, %.2f]:" % (s_low, s_high))
print()

print("%-8s %-10s %-10s %-10s %-12s %-12s %-12s" % ("gamma", "S_mean", "S_median", "S_max", "FULL%", "FUSION%", "FALLBACK%"))
print("-" * 70)

gamma_range = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
for gamma in gamma_range:
    S_vals = alpha_vals + beta_vals + gamma * edge_vals
    full_pct = 100.0 * (S_vals > s_high).sum() / len(S_vals)
    fusion_pct = 100.0 * ((S_vals >= s_low) & (S_vals <= s_high)).sum() / len(S_vals)
    fallback_pct = 100.0 * (S_vals < s_low).sum() / len(S_vals)
    
    print("%-8d %-10.4f %-10.4f %-10.4f %-12.1f %-12.1f %-12.1f" % (
        gamma, S_vals.mean(), np.median(S_vals), S_vals.max(),
        full_pct, fusion_pct, fallback_pct))

print()
print("Target: ~85-95% FULL, ~3-10% FUSION, ~2-5% FALLBACK")
print("(matches original behavior: EdgeAlign allows confident adoption)")
