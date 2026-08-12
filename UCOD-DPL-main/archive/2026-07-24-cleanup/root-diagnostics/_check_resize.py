from PIL import Image
import numpy as np
names = ["camourflage_00001", "camourflage_00292", "COD10K-CAM-2-Terrestrial-26-Chameleon-1680"]
lines = []
for n in names:
    img = Image.open("datasets/cache/refined_pseudo_labels/" + n + ".png").convert("L")
    lf = (np.array(img.resize((68,68), Image.LANCZOS)) >= 128).sum()
    nf = (np.array(img.resize((68,68), Image.NEAREST)) >= 128).sum()
    lines.append(n + ": LANCZOS=" + str(lf) + " NEAREST=" + str(nf) + " diff=" + str(nf-lf))
with open("_resize_check.txt", "w") as f:
    f.write("\n".join(lines))
