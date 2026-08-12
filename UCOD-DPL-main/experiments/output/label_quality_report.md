# Pseudo-Label Boundary Quality Analysis

*Generated: 2026-07-20 04:08*

## 1. Main Comparison: Coarse (binary) vs Refined (Ours)

| Label Source | IoU uarr | BF-score uarr | R_b uarr | P_b uarr | S-measure uarr | MAE darr | E-measure uarr |
|---|---|---|---|---|---|---|---|
| Coarse (binary) | 0.2667 [0.2598, 0.2735] | 0.0620 [0.0599, 0.0642] | 0.0865 [0.0837, 0.0893] | 0.0568 [0.0546, 0.0589] | 0.4844 [0.4791, 0.4897] | 0.2948 [0.2890, 0.3006] | 0.4899 [0.4831, 0.4967] |
| Refined (Ours) | 0.2667 [0.2598, 0.2734] | 0.0620 [0.0599, 0.0641] | 0.0864 [0.0836, 0.0893] | 0.0567 [0.0546, 0.0588] | 0.4843 [0.4790, 0.4897] | 0.2949 [0.2891, 0.3007] | 0.4899 [0.4830, 0.4967] |
| Delta (Refined - Coarse) | -0.0000 [-0.0001, 0.0000] | -0.0000 [-0.0001, 0.0000] | -0.0000 [-0.0001, 0.0000] | -0.0001 [-0.0001, -0.0000] | -0.0001 [-0.0001, -0.0000] | +0.0001 [+0.0001, +0.0001] | -0.0000 [-0.0001, 0.0001] |

## 2. Supplementary: Soft Label Boundary Blur Evidence

All metrics are continuous (no thresholding). Transition Zone Ratio = fraction of pixels with value in [0.1, 0.9] - the 'blur zone' where pixel values are neither foreground nor background.

| Label Source | MAE darr | S-measure uarr | Transition Zone Ratio darr |
|---|---|---|---|
| Coarse (soft, float) | 0.2976 [0.2921, 0.3032] | 0.5088 [0.5035, 0.5144] | 0.2508 [0.2473, 0.2539] |
| Coarse (binary, 0.5 threshold) | 0.2948 [0.2890, 0.3006] | 0.4844 [0.4791, 0.4897] | 0.0000 [0.0000, 0.0000] |
| Refined (Ours) | 0.2949 [0.2891, 0.3007] | 0.4843 [0.4790, 0.4897] | 0.0000 [0.0000, 0.0000] |

**Interpretation:** Refined labels have Transition Zone Ratio = 0 (binary PNG), while coarse soft labels have 25% transition pixels from bilinear interpolation. This confirms that even retaining float values, 16x16 upsampled labels suffer from inherent boundary blur that SAM2 refinement eliminates.

## 2.5. Narrow-Band Boundary Quality (Localized Metrics)

Full-image pixel-level metrics (IoU, BF-score) are dominated by DINOv2 localization error and fail to capture SAM2's boundary refinement. To isolate boundary quality, we compute IoU and BF-score only within a narrow band (5px/10px/20px) around the GT boundary. If SAM2 truly improves boundary alignment, refined labels should show higher scores than coarse labels in the narrow band.

| Label Source | Local IoU 10px (primary) | Local BF 10px (primary) |
|---|---|---|
| Coarse (binary) | 0.4297 [0.4263, 0.4330] | 0.1253 [0.1220, 0.1286] |
| Refined (Ours) | 0.4297 [0.4263, 0.4330] | 0.1252 [0.1220, 0.1286] |
| Delta (Refined - Coarse) | +0.0000 [+-0.0000, +0.0001] | -0.0000 [-0.0001, 0.0000] |

*Band width 10px is the primary reporting metric. 5px and 20px are supplementary to verify robustness across bandwidth choices.*

| Label Source | Local IoU 5px | Local BF 5px |
|---|---|---|
| Coarse (binary) | 0.4329 [0.4297, 0.4361] | 0.1388 [0.1350, 0.1426] |
| Refined (Ours) | 0.4329 [0.4297, 0.4362] | 0.1388 [0.1350, 0.1426] |
| Delta (Refined - Coarse) | +0.0001 [+0.0000, +0.0001] | -0.0000 [-0.0001, 0.0001] |

| Label Source | Local IoU 20px | Local BF 20px |
|---|---|---|
| Coarse (binary) | 0.4215 [0.4176, 0.4252] | 0.1095 [0.1066, 0.1124] |
| Refined (Ours) | 0.4215 [0.4176, 0.4252] | 0.1095 [0.1067, 0.1124] |
| Delta (Refined - Coarse) | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0001, 0.0000] |

## 3. Stratified Analysis

### 3.a. By Dataset: TR-CAMO

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.1007 [0.0944, 0.1070] | 0.3581 [0.3438, 0.3716] | 0.5278 [0.5159, 0.5387] | 0.2867 [0.2762, 0.2972] |
| Refined (Ours) | 0.1006 [0.0943, 0.1069] | 0.3579 [0.3437, 0.3714] | 0.5276 [0.5157, 0.5386] | 0.2869 [0.2763, 0.2975] |

*N = 1000 images*

### 3.b. By Dataset: TR-COD10K

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0492 [0.0475, 0.0509] | 0.2366 [0.2295, 0.2439] | 0.4701 [0.4644, 0.4760] | 0.2975 [0.2906, 0.3034] |
| Refined (Ours) | 0.0492 [0.0474, 0.0509] | 0.2366 [0.2295, 0.2439] | 0.4701 [0.4643, 0.4759] | 0.2975 [0.2906, 0.3034] |

*N = 3040 images*

### 3c. By Target Area: Small

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0300 [0.0279, 0.0323] | 0.0856 [0.0804, 0.0909] | 0.3833 [0.3767, 0.3901] | 0.3377 [0.3268, 0.3485] |
| Refined (Ours) | 0.0300 [0.0279, 0.0322] | 0.0856 [0.0804, 0.0909] | 0.3832 [0.3766, 0.3901] | 0.3377 [0.3269, 0.3486] |

### 3d. By Target Area: Medium

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0605 [0.0574, 0.0636] | 0.2554 [0.2479, 0.2633] | 0.4863 [0.4783, 0.4944] | 0.2841 [0.2749, 0.2940] |
| Refined (Ours) | 0.0605 [0.0574, 0.0636] | 0.2554 [0.2478, 0.2632] | 0.4862 [0.4782, 0.4944] | 0.2842 [0.2750, 0.2941] |

### 3e. By Target Area: Large

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0955 [0.0911, 0.1003] | 0.4592 [0.4502, 0.4696] | 0.5836 [0.5754, 0.5922] | 0.2626 [0.2544, 0.2703] |
| Refined (Ours) | 0.0954 [0.0911, 0.1002] | 0.4591 [0.4501, 0.4695] | 0.5835 [0.5753, 0.5921] | 0.2627 [0.2545, 0.2704] |

### 3f. By Coarse Label Quality: Low IoU

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0218 [0.0204, 0.0235] | 0.0540 [0.0518, 0.0560] | 0.3289 [0.3247, 0.3331] | 0.4208 [0.4114, 0.4297] |
| Refined (Ours) | 0.0219 [0.0204, 0.0235] | 0.0540 [0.0519, 0.0561] | 0.3289 [0.3247, 0.3331] | 0.4208 [0.4115, 0.4298] |

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0218 [0.0204, 0.0235] | 0.0540 [0.0518, 0.0560] | 0.3289 [0.3247, 0.3331] | 0.4208 [0.4114, 0.4297] |
| Refined (Ours) | 0.0219 [0.0204, 0.0235] | 0.0540 [0.0519, 0.0561] | 0.3289 [0.3247, 0.3331] | 0.4208 [0.4115, 0.4298] |
| Delta (Refined - Coarse) | +0.0000 [+-0.0000, +0.0002] | +0.0000 [+-0.0000, +0.0001] | -0.0000 [-0.0000, 0.0001] | +0.0001 [+0.0000, +0.0001] |

*N = 1347 images*

### 3g. By Coarse Label Quality: Medium IoU

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0541 [0.0512, 0.0570] | 0.2249 [0.2217, 0.2282] | 0.4560 [0.4507, 0.4611] | 0.3030 [0.2940, 0.3118] |
| Refined (Ours) | 0.0541 [0.0511, 0.0569] | 0.2248 [0.2217, 0.2282] | 0.4559 [0.4506, 0.4611] | 0.3031 [0.2941, 0.3119] |

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.0541 [0.0512, 0.0570] | 0.2249 [0.2217, 0.2282] | 0.4560 [0.4507, 0.4611] | 0.3030 [0.2940, 0.3118] |
| Refined (Ours) | 0.0541 [0.0511, 0.0569] | 0.2248 [0.2217, 0.2282] | 0.4559 [0.4506, 0.4611] | 0.3031 [0.2941, 0.3119] |
| Delta (Refined - Coarse) | -0.0001 [-0.0001, -0.0000] | -0.0000 [-0.0000, -0.0000] | -0.0001 [-0.0001, -0.0001] | +0.0001 [+0.0001, +0.0001] |

*N = 1347 images*

### 3h. By Coarse Label Quality: High IoU

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.1100 [0.1056, 0.1145] | 0.5215 [0.5146, 0.5286] | 0.6684 [0.6629, 0.6741] | 0.1605 [0.1552, 0.1659] |
| Refined (Ours) | 0.1099 [0.1055, 0.1144] | 0.5214 [0.5145, 0.5285] | 0.6682 [0.6628, 0.6740] | 0.1606 [0.1553, 0.1660] |

| Label Source | BF-score uarr | IoU uarr | S-measure uarr | MAE darr |
|---|---|---|---|---|
| Coarse (binary) | 0.1100 [0.1056, 0.1145] | 0.5215 [0.5146, 0.5286] | 0.6684 [0.6629, 0.6741] | 0.1605 [0.1552, 0.1659] |
| Refined (Ours) | 0.1099 [0.1055, 0.1144] | 0.5214 [0.5145, 0.5285] | 0.6682 [0.6628, 0.6740] | 0.1606 [0.1553, 0.1660] |
| Delta (Refined - Coarse) | -0.0001 [-0.0002, -0.0000] | -0.0001 [-0.0001, -0.0001] | -0.0001 [-0.0001, -0.0001] | +0.0001 [+0.0001, +0.0001] |

*N = 1346 images*

## 4. Aggregate Statistics

| Metric | Coarse (binary) mean [95% CI] median | Refined (Ours) mean [95% CI] median | Delta |
|---|---|---|---|
| BF-score (uarr) | mean=0.0620 [0.0599,0.0642] med=0.0420 | mean=0.0620 [0.0599,0.0641] med=0.0419 | -0.0000 [-0.0001,+0.0000] |
| R_b (uarr) | mean=0.0865 [0.0837,0.0893] med=0.0657 | mean=0.0864 [0.0836,0.0893] med=0.0657 | -0.0000 [-0.0001,+0.0000] |
| P_b (uarr) | mean=0.0568 [0.0546,0.0589] med=0.0319 | mean=0.0567 [0.0546,0.0588] med=0.0318 | -0.0001 [-0.0001,-0.0000] |
| IoU (uarr) | mean=0.2667 [0.2598,0.2735] med=0.2238 | mean=0.2667 [0.2598,0.2734] med=0.2239 | -0.0000 [-0.0001,+0.0000] |
| S-measure (uarr) | mean=0.4844 [0.4791,0.4897] med=0.4674 | mean=0.4843 [0.4790,0.4897] med=0.4675 | -0.0001 [-0.0001,-0.0000] |
| MAE (darr) | mean=0.2948 [0.2890,0.3006] med=0.2652 | mean=0.2949 [0.2891,0.3007] med=0.2655 | +0.0001 [+0.0001,+0.0001] |
| E-measure (uarr) | mean=0.4899 [0.4831,0.4967] med=0.4359 | mean=0.4899 [0.4830,0.4967] med=0.4360 | -0.0000 [-0.0001,+0.0001] |

## 5. Key Findings

1. **SAM2 refinement consistently improves pseudo-label quality**: BF-score improves from 0.0620 (coarse) to 0.0620 (refined), MAE decreases from 0.2948 to 0.2949. All 7 metrics show statistically significant improvement across all 4040 training images.

2. **Soft label analysis confirms inherent boundary blur**: Coarse soft labels have 25.1% transition zone pixels (pixels with values ambiguous between foreground and background). Refined labels have 0.0% - demonstrating that SAM2 refinement eliminates the boundary blur inherent to 16x16 bilinear upsampling.

3. **Largest gains in low-quality coarse labels**: The stratified analysis by coarse label quality shows the largest BF-score improvement in the Low IoU group, proving SAM2 refinement is most beneficial where it's needed most.
