# AEEM v2：自适应边缘感知增强机制设计方案

> 日期：2026-07-24  
> 项目：UCOD-DPL + 冻结 SAM2 离线伪标签精修  
> 状态：设计审查完成，可从 Milestone 0 开始实现  
> 名称：自适应边缘感知增强机制（Adaptive Edge-Aware Enhancement Mechanism, AEEM）

## 1. 设计结论

保留原有两项创新叙事：

1. **离线零样本边界精修范式**：将完全冻结的 SAM2 作为训练前的非侵入式边界专家，改变 UCOD-DPL 的固定伪标签输入源。
2. **自适应边缘感知增强机制（AEEM）**：针对粗伪标签空间偏移、提示误导和边界幻觉，使用语义定位校正、提示路由、边界不确定带和像素级置信融合，使 SAM2 只在不可靠边界区域执行受控残差精修。

AEEM v2 不再等同于旧版的“Canny 三因子分数 + 全局阈值门控”。Canny、粗标签 IoU 和 SAM2 自评分都只能作为候选质量的辅助信号，不能单独决定整张标签的采纳、回退或融合。

## 2. 问题定义

### 2.1 原始输入

- RGB 图像 `I ∈ R^(H×W×3)`。
- DINOv2 产生的 `16×16` 固定策略粗伪标签 `C_16`。
- 上采样后的软粗标签 `C ∈ [0,1]^(H×W)`。
- 冻结 DINOv2 的特征图 `F`，优先复用训练所需的 DINOv2 特征提取逻辑。
- 冻结 SAM2 的多提示、多掩码候选集合。

### 2.2 需要解决的三类失效

1. **定位失效**：粗标签质心、覆盖区域或连通结构偏离真实目标。
2. **提示失效**：强提示迫使 SAM2 复制粗标签，松提示导致候选漂移或过分割。
3. **融合失效**：一个图像级标量同时承担候选评分、采纳决策和融合权重，无法表达像素级不确定性。

### 2.3 核心约束

- SAM2 不负责重新发现整张图中的目标，只负责在可靠定位基础上提出边界候选。
- 低定位可靠性样本不能继续用同一套粗标签提示强行调用 SAM2。
- 精修结果必须保留软值，离线阶段不得立即阈值化为整图硬选择。
- 最终算法及阈值不能读取训练 GT；GT 只用于冻结配置后的诊断和论文分析。
- 任何新实验不得覆盖历史标签、raw SAM2 输出、特征缓存或 checkpoint。

## 3. 目标与非目标

### 3.1 目标

- 提升粗伪标签的边界质量，同时不破坏目标定位、区域完整性和拓扑结构。
- 在高、中、低定位可靠性样本上采用不同的提示和回退策略。
- 将候选质量、路由决策和像素融合权重拆分为三个明确变量。
- 保持首个 MVP 与当前 UCOD-DPL 训练入口兼容。
- 为每个创新子模块建立可独立验证的标签级和训练级消融。

### 3.2 非目标

- 不微调 SAM2，不向 SAM2 反向传播梯度。
- MVP 不修改 UCOD-DPL 网络结构、DBA 解码器或 Look-Twice。
- MVP 不引入 MLLM、Stable Diffusion、跨数据集检索库或额外人工标注。
- 不使用训练 GT 选择提示、候选或阈值。
- 不以单次、单数据集、千分位提升作为机制有效的证据。

## 4. 总体架构

```text
RGB image I + 16×16 coarse label C16
                  │
                  ▼
      [A] Semantic Localization Calibration
      coarse core/background + DINO feature prototypes
                  │
          P_sem, R_loc, disagreement map
                  │
                  ▼
      [B] Adaptive Prompt Routing
      high / medium / low localization reliability
                  │
                  ▼
      [C] Multi-Prompt SAM2 Candidate Bank
      point-only / weak-box / consensus-points / local-crop
                  │
                  ▼
      [D] Candidate Quality Estimation
      semantic margin + prompt stability + boundary evidence
                  │
                  ▼
      [E] Boundary Uncertainty Band
      protect foreground core and far background
                  │
                  ▼
      [F] Pixel-wise Confidence Fusion
      Y = (1-Q)·C + Q·M_sam inside uncertainty band
                  │
                  ▼
      soft pseudo-label PNG + confidence sidecar + audit JSON
                  │
                  ▼
      original UCOD-DPL APM → DBA → Look-Twice → EMA
```

## 5. 模块设计

论文叙事保持四个核心子模块，工程实现拆成六个可测试阶段：

| 论文子模块 | 英文名称 | 工程阶段 |
|------------|----------|----------|
| 自适应语义定位校正 | Semantic Localization Calibration, SLC | 模块 A |
| 宽松自适应提示 | Adaptive Prompt Routing, APR | 模块 B、C、D |
| 边界不确定带约束 | Boundary Uncertainty Band, BUB | 模块 E |
| 像素级边缘置信融合 | Pixel-wise Confidence Fusion, PCF | 模块 F |

Local-SAM 属于 APR 的小目标辅助路由，不作为第五个核心子模块。

### 5.1 模块 A：自适应语义定位校正

#### 5.1.1 目的

为 SAM2 提供独立于粗标签边界的语义定位证据，并判断当前粗标签是否足以支持边界精修。

#### 5.1.2 可靠核心与背景种子

从软粗标签 `C` 构造：

- 可靠前景核心 `K_fg = erode(C ≥ τ_fg)`。
- 可靠背景核心 `K_bg = outside(dilate(C ≥ 0.5))`。
- 不确定区域 `U_c = Ω \ (K_fg ∪ K_bg)`。

MVP 初始配置使用固定的无标签先验：

- `τ_fg = 0.8`。
- 前景腐蚀半径随目标面积缩放，但限制在合理区间。
- 背景种子只来自粗标签膨胀区域之外，不在目标附近生成谨慎负点。

#### 5.1.3 DINO 语义原型

在 DINOv2 特征图 `F` 上计算：

```text
p_fg = mean(F[x] | x ∈ K_fg)
p_bg = mean(F[x] | x ∈ K_bg)
m_fg(x) = cos(F[x], p_fg)
m_bg(x) = cos(F[x], p_bg)
P_sem(x) = sigmoid((m_fg(x) - m_bg(x)) / T)
```

`P_sem` 是语义定位图，不直接作为最终伪标签。它只用于提示生成、定位可靠性评估和候选质量估计。

#### 5.1.4 定位可靠性

定位可靠性 `R_loc ∈ [0,1]` 由以下归一化分量构成：

- 粗标签与 `P_sem` 的区域一致性。
- 两者质心距离。
- 两者前景面积比。
- 粗标签连通域复杂度。
- DINO 语义内外边际强度。

所有分量先单独映射到 `[0,1]`，再计算平均值。MVP 不使用人工设置的大权重，也不使用 GT 校准权重。

定位质量分为三档，阈值由整套无标签训练样本的分位数确定，而不是根据 GT 指标调节：

- **High**：定位一致、允许窄边界精修。
- **Medium**：定位存在不确定性，生成多提示候选并扩大不确定带。
- **Low**：定位证据冲突，不允许 SAM2 整图改写，直接回退到粗标签或语义校正软标签。

### 5.2 模块 B：宽松自适应提示路由

提示强度由 `R_loc`、目标面积和连通结构共同决定。

#### High 路由

- 一个距离变换最大点作为正点。
- 一个弱扩张框，框不贴合粗标签边界。
- 不使用谨慎负点；最多使用一个远端安全负点。
- 目标：让 SAM2 在定位不变的情况下调整边界。

#### Medium 路由

- 正点来自 `K_fg ∩ (P_sem ≥ 0.5)` 的 1–3 个距离变换峰值。
- 同时生成 `point-only`、`weak-box` 和 `consensus-points` 三种提示变体。
- 负点只来自 `C` 与 `P_sem` 都判定为背景的远端区域。
- 目标：通过候选多样性降低单一提示误导风险。

#### Low 路由

- 不使用原粗标签质心作为唯一正点。
- 若不存在可靠语义核心，则不调用 SAM2，直接输出安全回退。
- 若存在稳定语义核心，只生成 point-only 候选，且后续只允许在受限区域内使用。
- 目标：避免把错误粗标签变成更锐利的错误标签。

#### Small 路由

- Local-SAM 只有在目标面积小且语义核心稳定时触发。
- crop 范围由粗标签和语义图的联合包围框决定。
- Local-SAM 始终是提示路由的一种候选，不直接覆盖全图输出。

### 5.3 模块 C：多提示 SAM2 候选库

每个样本的候选集合包含：

- 原始软粗标签 `C`。
- 粗标签二值版本 `C_bin`。
- 每种提示变体对应的 3 个 SAM2 multimask 输出。
- 可选 Local-SAM 的 3 个候选。

每个候选必须保存：

- `prompt_variant`。
- `sam_mask_index`。
- `sam_iou_pred`。
- 正点、负点与框坐标。
- 原始分辨率、候选面积、质心、连通域数量。
- 与粗标签、语义图和其他候选的关系指标。

候选缓存路径：

```text
artifacts/aeem_v2/<experiment_id>/candidates/<image_name>.npz
```

候选 mask 使用布尔数组 `np.packbits` 后再写入压缩 NPZ，并保存原始 shape，避免多提示全分辨率候选造成不必要的磁盘膨胀。

禁止写入历史 `datasets/cache/raw_sam2_outputs`。

### 5.4 模块 D：候选质量估计

候选质量 `Q_cand` 与最终融合权重 `Q(x)` 分离。

#### 5.4.1 语义质量 `q_sem`

度量候选内部与外部的 DINO 语义原型边际。正确候选应覆盖前景语义响应，同时排除背景语义响应。

#### 5.4.2 提示稳定性 `q_stab`

度量候选与其他提示变体输出之间的一致性。只在单一松提示下出现、且与其余候选完全冲突的掩码视为低稳定性。

#### 5.4.3 边界证据 `q_edge`

由多尺度局部梯度和边界内外特征差异构成。Canny 对齐度只占其中一项，且必须归一化到 `[0,1]`。

#### 5.4.4 结构安全 `q_safe`

使用面积比、质心位移、连通域变化检测灾难性候选。该项是软惩罚，不把“与粗标签越相似”直接等同于“越正确”。

#### 5.4.5 组合方式

```text
Q_cand = mean(q_sem, q_stab, q_edge, q_safe)
```

首版采用等权平均，避免再次出现未校准分量被大权重放大的问题。所有输入在组合前必须通过范围断言。

候选选择规则：

1. 排除空掩码、全图掩码和违反结构安全下限的候选。
2. 选择 `Q_cand` 最大的候选。
3. 若最佳候选低于无标签分位数门槛，则回退，不强行使用 SAM2。

### 5.5 模块 E：边界不确定带约束

#### 5.5.1 不确定带定义

基础边界带：

```text
B_c = dilate(C_bin, r_out) \ erode(C_bin, r_in)
```

语义分歧带：

```text
B_sem = dilate(C_bin XOR (P_sem ≥ 0.5), r_sem)
```

最终不确定区域：

```text
B_unc = B_c ∪ B_sem
```

带宽由定位路由决定：

- High：窄带。
- Medium：中等带宽并包含语义分歧区。
- Low：不允许 SAM2 跨越可靠区域，通常直接回退。

#### 5.5.2 保护区

- `K_fg` 内保持粗标签前景，不允许 SAM2 删除可靠核心。
- `K_bg` 内保持背景，不允许 SAM2 引入远端前景。
- SAM2 只能在 `B_unc` 中提出残差修改。

### 5.6 模块 F：像素级边缘置信融合

#### 5.6.1 像素置信图

在 `B_unc` 内定义：

```text
Q(x) = Q_cand · S_prompt(x) · S_sem(x) · S_edge(x)
```

- `S_prompt(x)`：候选在多提示输出中的像素共识率。
- `S_sem(x)`：候选像素与 DINO 语义图的一致性。
- `S_edge(x)`：候选边界处的局部边缘证据。

所有分量均位于 `[0,1]`。在 `B_unc` 外令 `Q(x)=0`，表示保持粗标签。

#### 5.6.2 软残差融合

```text
Y_refined(x) = (1 - Q(x)) · C(x) + Q(x) · M_sam(x)
```

关键规则：

- 不在离线阶段对 `Y_refined` 做 `0.5` 阈值化。
- `Y_refined` 以 8-bit 灰度 PNG 保存，训练时恢复到 `[0,1]`。
- 置信图另存为灰度 PNG，MVP 只用于分析，不进入训练循环。
- 回退样本输出与控制组一致的软粗标签，而不是走另一套插值路径。

## 6. MVP 与后续版本边界

### 6.1 AEEM v2 MVP

MVP 以离线生成与实验工具为主，只增加一个用于隔离精修标签路径的最小数据加载接口：

- 新增 Hard-Coarse PNG 和 Soft-Coarse PNG 控制标签生成。
- 新增语义定位图、定位可靠性和三档提示路由。
- 新增多提示候选缓存。
- 新增边界不确定带和软残差融合。
- 输出 `refined_pseudo_labels/`、`confidence/` 和 `audit.jsonl`。
- 软标签写入 `<artifact_root>/refined_pseudo_labels/*.png`。
- 在 `dataset_cfg` 增加可选的 `refined_pseudo_label_dir`，只把精修 PNG 指向当前实验目录。
- `dataset_cfg.cache_dir` 继续保持 `./datasets/cache`，确保 DINO 特征缓存、原始伪标签缓存和回退路径完全不变。
- `base_dataset.py` 优先使用显式 `refined_pseudo_label_dir`；未配置时仍回退到现有的 `<cache_dir>/refined_pseudo_labels`，保持历史配置兼容。
- `confidence/` 只作为 sidecar 保存，不进入 MVP 训练循环。
- 当前 UCOD-DPL 主训练循环和 APM 不变。

### 6.2 AEEM v2.1

只有 MVP 标签级和训练级实验通过后才实现：

- 数据集显式返回 `pseudo_confidence`。
- APM 将图像级融合权重升级为像素级融合权重。
- 损失函数支持 confidence-weighted BCE 或 ignore map。
- 判别器决定是否继续使用二值标签，或改为接收软标签与置信图。

v2.1 必须作为独立消融，不能与 MVP 同时上线后只报告总结果。

## 7. 输出目录与数据契约

### 7.1 目录结构

```text
artifacts/aeem_v2/<experiment_id>/
├── config.json
├── manifest.json
├── git_diff.patch
├── input_hashes.json
├── candidates/
│   └── <image_name>.npz
├── refined_pseudo_labels/
│   └── <image_name>.png
├── confidence/
│   └── <image_name>.png
├── controls/
│   ├── hard_coarse/refined_pseudo_labels/
│   └── soft_coarse/refined_pseudo_labels/
├── audit.jsonl
└── summary.json
```

### 7.2 加载路径契约

训练配置使用两个互不替代的路径：

```python
dataset_cfg = dict(
    cache_dir='./datasets/cache',
    refined_pseudo_label_dir=(
        './artifacts/aeem_v2/<experiment_id>/refined_pseudo_labels'
    ),
)
```

最小代码链路为：

```text
dataset_cfg.refined_pseudo_label_dir
        → DataLoaderFactory
        → USCODDataset / LRDataset
        → BaseCODDataset.refined_pseudo_label_dir
```

`BaseCODDataset` 的兼容逻辑：

```text
显式 refined_pseudo_label_dir 非空
        → 只从该目录查找精修 PNG
未配置 refined_pseudo_label_dir
        → 回退到 <cache_dir>/refined_pseudo_labels
PNG 不存在
        → 回退到原 MultiCacheManager 的 pseudo_label pkl
```

不得通过修改 `cache_dir` 切换 AEEM 实验，因为 `MultiCacheManager` 还使用它定位特征缓存和原始伪标签缓存。

### 7.3 `audit.jsonl` 必需字段

```text
image_name
dataset
route
coarse_area_ratio
semantic_area_ratio
localization_reliability
prompt_variants
candidate_count
selected_candidate_index
selected_prompt_variant
q_sem
q_stab
q_edge
q_safe
q_candidate
uncertainty_band_ratio
mean_pixel_confidence
fallback_reason
local_sam_triggered
```

### 7.4 保护规则

- 如果 `<experiment_id>` 已存在，程序直接失败，不允许覆盖。
- 每次运行记录输入文件数量、大小与 hash。
- 标签生成、训练和评估分别使用独立 experiment ID。
- 配置中不得默认指向历史 `datasets/cache/refined_pseudo_labels` 目录。
- 训练时的 `dataset_cfg.cache_dir` 必须保持官方缓存根目录；只允许通过 `refined_pseudo_label_dir` 切换本次标签 artifact。

## 8. 实验设计

### 8.1 Phase A：表示方式控制实验

目的：判断已有提升是否来自 SAM2，还是来自二值化、PNG、插值或加载路径。

| ID | 固定标签来源 | 格式与加载路径 | SAM2 |
|----|--------------|----------------|------|
| A0 | 原始 `16×16 pkl` | 原 UCOD-DPL 路径 | 否 |
| A1 | 原始粗标签硬化 | 通过独立 `refined_pseudo_label_dir` 加载 | 否 |
| A2 | 原始粗标签软上采样 | 通过独立 `refined_pseudo_label_dir` 加载 | 否 |
| A3 | 旧完整精修标签 | 当前历史 PNG 路径的只读快照 | 是 |

验收：先生成标签并核对像素统计；训练时使用完全相同的 seed、配置和评估脚本。

### 8.2 Phase B：提示与候选标签级实验

| ID | 语义定位 | 提示 | 候选选择 | 边界带 | 输出 |
|----|----------|------|----------|--------|------|
| B0 | 否 | V_current | best SAM score | 否 | hard |
| B1 | 否 | V_loose | best coarse IoU | 否 | hard |
| B2 | 否 | 多提示 | 多提示稳定性 | 否 | hard |
| B3 | 是 | 自适应路由 | `Q_cand` | 否 | hard |
| B4 | 是 | 自适应路由 | `Q_cand` | 是 | soft |

这一阶段不训练模型，只比较标签级指标和失败案例。

### 8.3 Phase C：AEEM 累积消融

| Row | 配置 | 目的 |
|-----|------|------|
| C0 | A0 原始 UCOD-DPL | 官方复现基线 |
| C1 | A1 Hard-Coarse PNG | 排除表示方式混杂 |
| C2 | Naive SAM2 全图替换 | 证明无保护 SAM2 的风险 |
| C3 | + 语义定位校正 | 验证空间偏移处理 |
| C4 | + 自适应提示与候选质量 | 验证提示和选择作用 |
| C5 | + 边界不确定带与像素融合 | AEEM 核心完整版本 |
| C6 | + Local-SAM | 小目标辅助消融 |

### 8.4 标签级指标

- 全图 IoU、S-measure、MAE。
- BF-score 与边界 precision/recall。
- 5/10/20px 窄带 IoU 与 BF-score。
- 质心偏移。
- 面积比与连通域变化。
- 灾难性失败率：相对粗标签 IoU 下降超过 `0.2` 的样本比例。
- 按 TR-CAMO/TR-COD10K、目标面积、粗标签质量和中心偏移分层。

GT 使用规则：

- 生成算法不读取 GT。
- 配置和无标签阈值先冻结，再运行 GT 诊断。
- GT 只用于证明或否证设计，不用于逐轮调参。

### 8.5 标签级进入训练的门槛

AEEM v2 只有同时满足以下条件才进入完整训练：

1. TR-CAMO 与 TR-COD10K 的 BF-score 平均增益，其 bootstrap 95% CI 下界均大于 0。
2. 两个训练集的全图 IoU 不出现超过 `0.005` 的平均下降。
3. 平均质心偏移不增加。
4. 灾难性失败率低于 `1%`。
5. High 与 Medium 路由均有正向边界收益，不能只靠单一小分组拉高平均值。

### 8.6 训练级验收

第一轮只做单 seed 筛选：

- A0、A1、C2、C5 四组。
- 相同训练配置、相同 seed、相同 epoch、相同评估脚本。
- 最终统一评估 CHAMELEON、CAMO、COD10K、NC4K。

通过单 seed 后再做三 seed 正式实验：

- 报告均值与标准差。
- 主要指标：S-measure、MAE、E-mean、WFM。
- 次要指标：F-max、F-mean、mIoU、ACC。
- 任何小于基线随机波动范围的提升不得表述为稳定贡献。

## 9. 失败与回退策略

| 失败条件 | 回退行为 |
|----------|----------|
| 粗标签为空 | 保存空标签，记录原因，不调用 SAM2 |
| 语义前景核心为空 | 回退 Soft-Coarse PNG |
| 所有 SAM2 候选为空或全图 | 回退 Soft-Coarse PNG |
| 候选质量低于门槛 | 回退 Soft-Coarse PNG |
| 候选破坏可靠前景核心 | 只保留边界带内修改 |
| 候选引入远端连通域 | 删除保护背景内的新增区域 |
| 单图异常 | 记录异常并回退，不跳过文件 |

所有回退必须产出与正常样本同名的 mask 和 confidence 文件，保证训练集数量始终为 4040。

## 10. 测试要求

### 10.1 单元测试

- 所有质量分量范围均为 `[0,1]`。
- 空掩码、全图掩码、单像素目标不会崩溃。
- 不确定带外输出严格等于粗标签。
- `Q=0` 时输出严格等于粗标签。
- `Q=1` 时只在不确定带内等于 SAM2 候选。
- 实验目录已存在时拒绝覆盖。
- selected candidate 日志与真实返回索引一致。
- 显式切换 `refined_pseudo_label_dir` 不改变 feature cache 与 pseudo-label pkl cache 路径。
- 未设置 `refined_pseudo_label_dir` 时，历史配置的加载行为保持不变。

### 10.2 小样本集成测试

- 固定 12 张样本，覆盖空标签、小目标、多连通域、高/中/低定位可靠性。
- 相同 seed 重复运行得到相同提示、候选选择和输出 hash。
- 每个输入均产生 soft label、confidence、candidate cache 和 audit 记录。

### 10.3 数据审计

- 输出数量必须为 4040。
- 不允许缺失、重复文件名或全图 mask。
- 空标签数量必须与输入粗标签一致并有明确记录。
- 每个实验保存前景面积分布、置信度分布、路由比例和回退原因分布。

## 11. 实现顺序

### Milestone 0：实验隔离与控制组

1. 新增 experiment ID 与不可覆盖输出管理。
2. 新增可选 `refined_pseudo_label_dir`，并验证其不改变现有缓存路径。
3. 新增 Hard-Coarse PNG、Soft-Coarse PNG 生成器。
4. 参数化标签级评估脚本，移除历史目录硬编码。
5. 生成 manifest 与输入 hash。

当前实现命令：

```powershell
python scripts/prepare_aeem_controls.py `
  --experiment-id m0_controls_20260724_v1
```

生成后，训练配置保持 `cache_dir='./datasets/cache'`，并分别把
`refined_pseudo_label_dir` 指向 artifact 中的 Hard-Coarse 或 Soft-Coarse
`refined_pseudo_labels/` 目录。

当前已生成：

- `configs/uscod/UCOD-DPL_dinov2_m0_hard.py`
- `configs/uscod/UCOD-DPL_dinov2_m0_soft.py`
- `artifacts/aeem_v2/m0_controls_20260724_v1/`
- `artifacts/aeem_v2/evaluations/m0_controls_20260724_v1_gt_diag/`

冻结标签配置后再运行 GT 诊断：

```powershell
python experiments/evaluate_aeem_labels.py `
  --gt-set TR-CAMO=datasets/RefCOD/TR-CAMO/gt `
  --gt-set TR-COD10K=datasets/RefCOD/TR-COD10K/gt `
  --prediction hard=<hard_refined_pseudo_labels_dir> `
  --prediction soft=<soft_refined_pseudo_labels_dir> `
  --baseline soft `
  --output-dir artifacts/aeem_v2/evaluations/<evaluation_id>
```

### Milestone 1：边界安全 MVP

1. 实现 DINO 语义原型与定位可靠性。
2. 实现 High/Medium/Low 提示路由。
3. 实现多提示候选缓存与候选元数据。
4. 实现候选质量估计。
5. 实现边界不确定带与软残差融合。
6. 完成单元测试和 12 张集成测试。

### Milestone 2：标签级验收

1. 在 4040 张训练样本上生成 AEEM v2 标签。
2. 冻结配置后运行 GT 诊断。
3. 检查进入训练的五项门槛。
4. 若失败，按模块定位原因，不进入完整训练。

### Milestone 3：训练级验收

1. 运行 A0、A1、C2、C5 单 seed。
2. 统一四数据集评估。
3. 通过后运行三 seed 正式消融。

### Milestone 4：可选 APM v2.1

只有 C5 已稳定优于 A1 和 A0 后，才评估像素 confidence 显式接入 APM 与 loss。

## 12. 论文表述边界

在完成实验前，只能写：

> 本文设计了一种由语义定位校正、自适应提示路由、边界不确定带约束和像素级置信融合组成的 AEEM，用于约束冻结 SAM2 在粗伪标签不可靠边界区域执行残差精修。

在没有相应消融前，不能写：

- EdgeAlign 是核心贡献者。
- 97% 以上候选被准确识别为高质量。
- Local-SAM 显著改善小目标。
- 所有数据集稳定超过 UCOD-DPL。
- SAM2 精修本身带来全部下游增益。

## 13. 启动条件

实现阶段从 Milestone 0 开始。开始前确认：

- 当前工作区不做 reset 或覆盖。
- 历史标签目录只读。
- 新代码默认输出到 `artifacts/aeem_v2/<experiment_id>`。
- `cache_dir` 保持 `./datasets/cache`，实验标签只通过 `refined_pseudo_label_dir` 注入。
- 首批工作不运行 SAM2 全量推理，不运行完整训练。
- Milestone 0 的控制组与审计工具通过后，再开始 Milestone 1。
