# === _d3 FIXED v2: 直接用 index 匹配 pkl ===
import sys, os
sys.path.insert(0, "scripts")
import offline_sam2_refine as osm
import numpy as np, cv2, json, pickle
from PIL import Image

CFG = osm.CFG
compute_iou = osm.compute_iou
compute_edge_align = osm.compute_edge_align

BACKUP_DIR = "datasets/cache/refined_pseudo_labels"
VLOOSE_DIR = "datasets/cache/refined_pseudo_labels_vloose"

coarse_dir = CFG["coarse_dir"]
with open(os.path.join(coarse_dir, "index.json")) as f:
    index_map = json.load(f)

# Build sorted image list matching UCOD-DPL order
all_images = []
for d in CFG["image_dirs"]:
    for f in sorted(os.listdir(d)):
        if f.endswith((".jpg", ".png")):
            all_images.append((os.path.join(d, f), os.path.splitext(f)[0]))
# Sort by name to match pkl index order
all_images.sort(key=lambda x: x[1])

# Filter to CAMO images with backup+vloose labels
backup_files = set(f.replace(".png", "") for f in os.listdir(BACKUP_DIR) if f.endswith(".png"))
vloose_files = set(f.replace(".png", "") for f in os.listdir(VLOOSE_DIR) if f.endswith(".png"))
camo_names = set(n for n in backup_files & vloose_files if n.startswith("camourflage_"))

result_pairs = []
for idx, (img_path, name) in enumerate(all_images):
    if name not in camo_names:
        continue
    pkl_key = str(idx)
    if pkl_key not in index_map:
        continue
    result_pairs.append((idx, img_path, name, os.path.join(coarse_dir, index_map[pkl_key])))

camo_pairs = result_pairs[:30]
print("=" * 60)
print("TEST A: backup vs vloose vs coarse (30 CAMO images)")
print("=" * 60)

results = []
for idx, img_path, name, pkl_path in camo_pairs:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    with open(pkl_path, "rb") as fh:
        coarse_pkl = pickle.load(fh)
    if hasattr(coarse_pkl, "numpy"):
        coarse = coarse_pkl.numpy()
    else:
        coarse = coarse_pkl
    coarse = coarse.squeeze()
    coarse = (coarse * 255).astype(np.uint8)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    coarse_bool = coarse >= 128
    
    bak = np.array(Image.open(os.path.join(BACKUP_DIR, name + ".png")))
    bak_bool = bak >= 128
    
    vls = np.array(Image.open(os.path.join(VLOOSE_DIR, name + ".png")))
    vls_bool = vls >= 128
    
    iou_bak = compute_iou(bak_bool, coarse_bool)
    iou_vls = compute_iou(vls_bool, coarse_bool)
    ea_bak = compute_edge_align(bak_bool, image)
    ea_vls = compute_edge_align(vls_bool, image)
    
    ratio_bak = bak_bool.sum() / bak_bool.size
    ratio_vls = vls_bool.sum() / vls_bool.size
    ratio_coarse = coarse_bool.sum() / coarse_bool.size
    eps = max(ratio_coarse, 1e-6)
    
    results.append({
        "name": name,
        "iou_bak": iou_bak, "iou_vls": iou_vls,
        "ea_bak": ea_bak, "ea_vls": ea_vls,
        "fg_bak": ratio_bak, "fg_vls": ratio_vls, "fg_coarse": ratio_coarse,
        "area_ratio_bak": ratio_bak / eps,
        "area_ratio_vls": ratio_vls / eps,
    })

n = len(results)
print("Loaded %d images\n" % n)

if n == 0:
    print("ERROR: no matching images found. Check image_dirs and pkl mapping.")
    sys.exit(1)

results.sort(key=lambda x: x["area_ratio_bak"])

iou_bak_vals = [r["iou_bak"] for r in results]
iou_vls_vals = [r["iou_vls"] for r in results]
ea_bak_vals = [r["ea_bak"] for r in results]
ea_vls_vals = [r["ea_vls"] for r in results]
area_bak = [r["area_ratio_bak"] for r in results]
area_vls = [r["area_ratio_vls"] for r in results]
fg_bak_vals = [r["fg_bak"] for r in results]
fg_vls_vals = [r["fg_vls"] for r in results]

fmt_h = "%-28s %-15s %-15s"
fmt_v = "%-28s %-15.4f %-15.4f"
fmt_i = "%-28s %-15d %-15d"
print(fmt_h % ("Metric", "backup_old", "vloose"))
print("-" * 58)
print(fmt_v % ("IoU vs coarse (mean)", np.mean(iou_bak_vals), np.mean(iou_vls_vals)))
print(fmt_v % ("IoU vs coarse (median)", np.median(iou_bak_vals), np.median(iou_vls_vals)))
print(fmt_v % ("EdgeAlign raw (mean)", np.mean(ea_bak_vals), np.mean(ea_vls_vals)))
print(fmt_v % ("EdgeAlign raw (median)", np.median(ea_bak_vals), np.median(ea_vls_vals)))
print(fmt_v % ("FG ratio (mean)", np.mean(fg_bak_vals), np.mean(fg_vls_vals)))
print(fmt_v % ("FG ratio (median)", np.median(fg_bak_vals), np.median(fg_vls_vals)))
print(fmt_v % ("Area/Coarse (mean)", np.mean(area_bak), np.mean(area_vls)))
print(fmt_v % ("Area/Coarse (median)", np.median(area_bak), np.median(area_vls)))
ac_bak_lt05 = sum(1 for a in area_bak if a < 0.5)
ac_vls_lt05 = sum(1 for a in area_vls if a < 0.5)
ac_bak_gt2 = sum(1 for a in area_bak if a > 2.0)
ac_vls_gt2 = sum(1 for a in area_vls if a > 2.0)
print(fmt_i % ("Area/Coarse < 0.5", ac_bak_lt05, ac_vls_lt05))
print(fmt_i % ("Area/Coarse > 2.0", ac_bak_gt2, ac_vls_gt2))

print("\n--- Bottom 5 (tight masks) ---")
for r in results[:5]:
    print("  %s: area/coarse=%.3f IoU=%.3f Edge=%.1f fg=%.4f" % (
        r["name"], r["area_ratio_bak"], r["iou_bak"], r["ea_bak"], r["fg_bak"]))

print("\n--- Top 5 (loose masks) ---")
for r in results[-5:]:
    print("  %s: area/coarse=%.3f IoU=%.3f Edge=%.1f fg=%.4f" % (
        r["name"], r["area_ratio_bak"], r["iou_bak"], r["ea_bak"], r["fg_bak"]))
