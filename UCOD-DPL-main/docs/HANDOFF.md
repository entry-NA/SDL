# UCOD-DPL + SAM2 / AEEM v2 当前交接文档

> 最后核对：2026-07-25
> 项目根目录：`C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main`
> Conda 环境：`test01`，解释器：`C:\Anaconda\envs\test01\python.exe`
> 当前阶段：AEEM v2、基线/朴素 SAM2 消融、完整方案 m4、创新点标签级证据和可靠替换控制均已完成单 seed 验证；统一 Markdown/Word 操作手册已交付。

## 新会话必须先知道的六件事

1. 当前 AEEM 已经是 **v2 新机制**，不是旧版“全局三因子门控 + Local-SAM”。
2. 用户面前统一使用三个名称：**基线、朴素 SAM2、完整方案**。`a0/a1` 只是脚本内部组名，不要把完整方案随口叫“A2”。
3. 当前完整方案是 `m4_camo_all_cod10k_qsemantic25_20260724_v1`，由 1760 张 AEEM 标签和 2280 张 Soft-Coarse 标签组成。
4. 当前完整方案按最新 Excel 的论文行达到 18 项严格更优、2 项显示精度持平、0 项更差；但它只有一个 seed，且多数差值只有千分位。
5. 额外可靠替换控制已完成：1760 AEEM + 2271 朴素 SAM2 + 9 Soft-Coarse，结论按预注册规则为“混合结果”，只能称单 seed 方向性证据。
6. 不要清理、覆盖或重新生成现有标签、artifact、checkpoint；后续优先复用统一操作手册中的只读评估入口。

---

## 1. 我们做了什么

### 1.1 研究目标与两项创新

项目基于 UCOD-DPL，在 APM 之前增加冻结 SAM2 的离线伪标签精修：

```text
图像
  -> 冻结 DINOv2 特征 / 16x16 粗伪标签
  -> 冻结 SAM2 + AEEM v2 离线精修
  -> APM
  -> DBA 解码器
  -> Look-Twice
  -> EMA 教师更新
```

当前论文叙事中的两项创新是：

1. **离线零样本边界精修范式**：把完全冻结的 SAM2 作为离线、非侵入式边界专家，改变 APM 的固定伪标签输入源。
2. **自适应边缘感知增强机制（AEEM）**：用语义定位可靠性、提示路由、边界区域像素融合和结构安全机制约束 SAM2，缓解粗标签空间偏移、提示误导、边界幻觉和碎片增长。

### 1.2 旧版 AEEM 的问题已经查清

旧版位于 `scripts/offline_sam2_refine.py`，历史设计包括框/正负点、多掩码截断选择、全局三因子门控和 Local-SAM。已确认的问题：

- `EdgeAlign` 曾少除以 `255`，早期高采纳率不可信。
- `s_upper=999` 实际几乎关闭 FULL 分支，不是“全部 FULL”。
- `gamma=50` 可令 `S>1`，使 `1-S` 为负，融合不再是凸组合。
- 强框、多个正点和负点会迫使 SAM2 复制粗标签。
- 单个图像级分数不能表达“前景核心可靠、局部边界不可靠”。
- Local-SAM 触发率太低，不能承担主要创新贡献。

结论：旧脚本只保留为历史对照与朴素标签来源，不再代表当前 AEEM。

### 1.3 AEEM v2 已经实现

核心源码在 `aeem_v2/`：

- `semantic.py`：DINOv2 前景/背景语义原型、语义概率和定位可靠性。
- `refinement.py`：提示路由、多候选质量和边界像素融合。
- `sam2_adapter.py`：SAM2 适配；同图只编码一次，框坐标使用正确的 XYXY。
- `structure.py`、`topology.py`：碎片清理、连通骨架保护和结构风险回退。
- `pipeline.py`：GPU 推理前后的 CPU 准备/后处理重叠流水线。
- `composition.py`：不同训练来源的标签组合及 Top-fraction 选择。
- `artifacts.py`、`evaluation.py`、`controls.py`、`dataset.py`：不可覆盖工件、评估和数据支持。

当前真实的 AEEM v2 流程是：

```text
DINOv2语义定位可靠性
  -> High / Medium / Low 路由
  -> 自适应点提示、弱框和安全背景点
  -> SAM2多提示、多候选
  -> q_semantic / q_stability / q_edge / q_safety
  -> 只在边界不确定带内做像素级残差融合
  -> 可靠前景核心、远端背景和连通骨架保护
  -> 结构风险时回退 Soft-Coarse
  -> 训练源剂量控制
```

关键冻结规则：

- Low 路由不调用 SAM2，直接回退。
- SAM2 只修改边界不确定带，不能整图替换可靠区域。
- 候选质量与像素融合权重分离，不再使用旧版全局 `S`。
- 最大有效连通域增长为 `1`；最大额外结构质量比例为 `0.05`。
- GT 只用于参数冻结后的诊断，不参与候选选择或阈值调参。
- Local-SAM 目前只算辅助方向，不是 v2 核心模块。

### 1.4 工程与运行能力已经完成

- 增加独立 `dataset_cfg.refined_pseudo_label_dir`，只切换实验标签，不改变 DINO 特征和原始伪标签缓存。
- 数据链路为 `DataLoaderFactory -> USCODDataset/LRDataset -> BaseCODDataset`。
- 增加 Hard-Coarse / Soft-Coarse PNG 控制组，排除了 PNG、硬化和 resize 路径的主要混杂。
- 增加不可覆盖 experiment ID、manifest、输入/输出 hash、Git diff 和 audit。
- 增加 `tqdm` 进度条。
- SAM2 推理保持 GPU；CPU 准备和后处理通过 staged pipeline 与 GPU 推理重叠，不是交替停顿式执行。
- 当前测试：`32/32 passed`。

### 1.5 已完成的主要实验

1. Milestone 0：Hard/Soft 控制组和标签加载路径隔离。
2. Milestone 1/2：语义定位、提示路由、边界带融合和结构安全校准。
3. m2：全量 4040 张 AEEM v2 标签及完整训练。
4. m3：TR-CAMO / TR-COD10K 2x2 来源隔离，确认两来源存在非加性交互。
5. m4：CAMO 全量 AEEM + COD10K `q_semantic` Top-25% AEEM 的最终单 seed 候选。
6. 正式核心消融：重新训练隔离后的基线与朴素 SAM2，旧污染 checkpoint 不再使用。

---

## 2. 目前进度如何

### 2.1 当前完整方案

实验 ID：

```text
m4_camo_all_cod10k_qsemantic25_20260724_v1
```

标签组成：

- TR-CAMO：1000 张全部使用 AEEM v2。
- TR-COD10K：760 张 `selected.q_semantic` 最高的样本使用 AEEM v2。
- TR-COD10K 其余 2280 张使用 Soft-Coarse。
- 合计：AEEM 1760，Soft-Coarse 2280，共 4040。
- 选择过程不读取 GT。

训练配置：DINOv2-base、518x518、batch 16、25 epoch、seed 42、fp16、EMA 0.99、APM `merge_method=dis`、Look-Twice 开启且阈值 0.15。

### 2.2 三组正式结果

最新人工统计文件：

```text
C:\Users\23991\Desktop\实验数据 (2).xlsx
```

以下顺序均为 `E_MEAN / SMeasure / MAE / WFM / F_MAX`，MAE 越低越好，其余越高越好。

| 方案 | 数据集 | E_MEAN | SMeasure | MAE | WFM | F_MAX |
|---|---|---:|---:|---:|---:|---:|
| 基线 | CHAMELEON | 0.9302 | 0.8630 | 0.0314 | 0.8233 | 0.8373 |
| 基线 | TE-CAMO | 0.8621 | 0.7932 | 0.0762 | 0.7475 | 0.7805 |
| 基线 | TE-COD10K | 0.9162 | 0.8344 | 0.0299 | 0.7634 | 0.7805 |
| 基线 | NC4K | 0.9234 | 0.8507 | 0.0416 | 0.8183 | 0.8370 |
| 朴素 SAM2 | CHAMELEON | 0.9323 | 0.8661 | 0.0306 | 0.8300 | 0.8490 |
| 朴素 SAM2 | TE-CAMO | 0.8529 | 0.7859 | 0.0786 | 0.7425 | 0.7822 |
| 朴素 SAM2 | TE-COD10K | 0.9179 | 0.8367 | 0.0294 | 0.7696 | 0.7926 |
| 朴素 SAM2 | NC4K | 0.9207 | 0.8498 | 0.0423 | 0.8208 | 0.8459 |
| 完整方案 | CHAMELEON | 0.9316 | 0.8648 | 0.0310 | 0.8259 | 0.8400 |
| 完整方案 | TE-CAMO | 0.8639 | 0.7939 | 0.0760 | 0.7482 | 0.7811 |
| 完整方案 | TE-COD10K | 0.9160 | 0.8344 | 0.0302 | 0.7633 | 0.7805 |
| 完整方案 | NC4K | 0.9240 | 0.8513 | 0.0415 | 0.8190 | 0.8376 |

论文参考行：

| 数据集 | E_MEAN | SMeasure | MAE | WFM | F_MAX |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | 0.931 | 0.864 | 0.031 | 0.825 | 0.838 |
| TE-CAMO | 0.862 | 0.793 | 0.077 | 0.747 | 0.779 |
| TE-COD10K | 0.916 | 0.834 | 0.031 | 0.763 | 0.779 |
| NC4K | 0.923 | 0.850 | 0.043 | 0.818 | 0.835 |

### 2.3 当前可以得出的结论

- 朴素 SAM2 相对基线：13 项提升、7 项下降；四个数据集 `F_MAX` 全部提升，平均约 `+0.0086`。
- 完整方案相对基线：15 项提升、2 项持平、3 项下降；CHAMELEON、TE-CAMO、NC4K 五项全升，TE-COD10K 基本持平/轻微下降。
- 完整方案相对论文：18 项严格更优、2 项显示精度持平、0 项下降。
- 完整方案相对朴素 SAM2：7 项提升、13 项下降；完整方案恢复了 TE-CAMO 和 NC4K 的区域/结构指标，但 CHAMELEON、TE-COD10K 和四个数据集的 `F_MAX` 不如朴素 SAM2。

因此：

- 创新点1有初步下游证据，尤其是四数据集一致的 `F_MAX` 提升；但仍需直接标签级 BF-score/Boundary IoU 才能严谨证明“边界精修”。
- 完整方案整体优于本地基线并达到论文参考行。
- 创新点2目前不能仅凭“完整方案 vs 朴素 SAM2”宣称全面成立，归因问题仍待解决。

### 2.4 标签级 AEEM v2 证据

全量 AEEM v2 相对 Soft-Coarse 的训练标签 GT 诊断：

| 训练来源 | Delta IoU | Delta BF | Delta MAE |
|---|---:|---:|---:|
| TR-CAMO | +0.031716 | +0.035000 | -0.011575 |
| TR-COD10K | +0.025613 | +0.021879 | -0.008334 |

这证明 AEEM v2 相对 Soft-Coarse 改善了训练标签本身，但不能自动推出它相对朴素 SAM2也更好，更不能替代下游消融。

### 2.5 已冻结的重要工件

```text
原始粗标签：
datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K
4040 个 .pkl（目录中另有非 pkl 元数据文件）

朴素 SAM2：
datasets/cache/naive_sam2_labels
4028 张 PNG + 12 个原始 pkl 回退

全量 AEEM v2：
artifacts/aeem_v2/m2_full4040_structure_20260724_v1
output hash: 2e3a081f55d806b3530d00626c231c0e221dd33ac0a0515e308e8fc6e2473850

最终完整方案：
artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1
output hash: 073abc4dcd13eaa24eb050a7dc063a88dda1a5a644750eeefd8b7270cb92895e
```

### 2.6 三个正式 checkpoint 与日志

基线：

```text
work_dir/uscod/UCOD-DPL_dinov2_ablation_a0_baseline/
  UCOD-DPL_dinov2_ablation_a0_baseline_20260725_v1/
```

朴素 SAM2：

```text
work_dir/uscod/UCOD-DPL_dinov2_ablation_a1_naive_sam2/
  UCOD-DPL_dinov2_ablation_a1_naive_sam2_20260725_v2/
```

完整方案：

```text
work_dir/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/
  UCOD-DPL_dinov2_aeem_v2_m4_camo_all_cod10k_qsemantic25_20260724_v1/
```

三个目录中的 `ckp/epoch25.pth/model.safetensors` 均已核实存在，`eval0.log` 是正式结果来源。

---

## 3. 有什么注意的点

### 3.1 不要再次混用旧版和新版 AEEM

当前创新点2应描述为：

```text
语义定位可靠性
+ 自适应提示路由
+ 多候选质量评估
+ 边界不确定带像素级融合
+ 结构安全回退
+ 训练源剂量控制
```

旧版“自适应框扩张、截断式多掩码、三因子全局门控、Local-SAM”只能放在历史方案/失败分析里。多候选仍存在，但其评价已升级为 `q_semantic/q_stability/q_edge/q_safety`，不能再写成旧 IoU 截断机制。

### 3.2 用户面对的命名

- 写作和交流只用：基线、朴素 SAM2、完整方案。
- `a0/a1` 只用于 PowerShell 参数和目录名。
- 不要把完整方案简称为“A2”；用户已经明确指出这种叫法会混淆。

### 3.3 当前完整方案不是朴素 SAM2 的简单叠加

朴素 SAM2 使用 4028 张历史 naive PNG；完整方案使用 1760 张 AEEM + 2280 张 Soft-Coarse。因此两组间同时变化了精修算法、样本选择和回退来源。

这不是数据错误，而是 m3 来源隔离后形成的 m4 剂量控制设计。但它会影响“创新点2的独立贡献”如何证明。下一会话必须先和用户确认实验论证方式，不能直接把某一种新对照当成既定事实。

### 3.4 论文比较口径

- 当前最新人工汇总是 `C:\Users\23991\Desktop\实验数据 (2).xlsx`。
- 旧 `实验数据.xlsx`、旧报告和讨论大纲中曾出现不同论文基准，不可混用。
- 论文行只有三位小数，千分位差异不能写成“显著提升”。
- 当前只有 seed 42。正式结论应写“达到或略超论文报告值”，多 seed 后才能讨论稳定性。
- 本地复现基线本身在多个指标上已高于论文，因此“超过论文”不能代替“超过本地基线”。

### 3.5 标签和缓存路径

- `cache_dir` 必须保持 `./datasets/cache`。
- 实验只通过 `dataset_cfg.refined_pseudo_label_dir` 切换标签。
- 不要把 artifact 目录赋给 `cache_dir`，否则会改变 DINO 特征和原始伪标签缓存链路。
- A0 配置中的 `NO_REFINED_LABELS` 路径必须保持不存在；一旦创建，基线可能误读 PNG。
- 每次新实验使用新 experiment ID、新目录和新 manifest，禁止覆盖。

### 3.6 GT 使用边界

- GT 可用于最终标签质量诊断、BF-score、Boundary IoU、MAE、质心偏移和统计显著性分析。
- GT 不得用于挑选 `q_semantic` 阈值、选择单张候选、调提示规则或决定训练标签。
- 当前 q_semantic Top-25% 是读取冻结的无 GT 审计分数。

### 3.7 GPU、CPU与进度条

- SAM2 推理使用 GPU；图像读取、提示构造、语义/结构后处理和保存主要在 CPU。
- `aeem_v2/pipeline.py` 已让一个 CPU prepare worker 和多个 finish workers 与 GPU 推理重叠。
- 全量 m2 本机耗时约 2 小时35分钟；GPU占用不持续满载不等于没有用GPU。
- `run_aeem_v2_mvp.py` 已有 `tqdm`；不要为显示效果破坏有序流水线。

### 3.8 Git 工作区

工作区当前很脏，包含大量未提交的正式源码、文档、数据入口和实验脚本。严禁 `git reset --hard`、`git checkout --` 或批量清理。artifact生成时的 commit 不足以复现，必须同时保留 artifact中的 `git_diff.patch`。

---

## 4. 我们遇到了什么挫折

### 4.1 早期实验数据和训练状态被污染

历史上曾在训练期间更改默认精修标签目录，导致旧 checkpoint 无法确定读取的是哪一版标签。曾出现 CAMO SMeasure 约 0.41 的近全黑模型，清理错误状态并恢复标签后才回到正常水平。因此旧 checkpoint 不再用于正式消融，2026-07-25 已重新训练隔离后的基线和朴素 SAM2。

### 4.2 旧门控方向误诊

曾长时间扫描 `s_lower/s_upper/gamma`，但核心问题包括 EdgeAlign 量纲、非法融合、提示复制和候选碎片。继续扫三个参数没有可解释性，AEEM v2 已放弃这条主线。

### 4.3 SAM2提示的两难

- 强提示：输出接近粗标签，几乎没有新增信息。
- 极松提示：能产生新轮廓，但可能漂移、过分割或增加连通域。

AEEM v2 因此采用定位可靠性路由、多提示候选、边界带限制和结构回退，而不是寻找一个全局固定提示。

### 4.4 COD10K碎片和跨数据集权衡

m2全量 AEEM/AEEM 的标签级指标在 CAMO、COD10K 都改善，但下游主要提升 TE-CAMO，CHAMELEON和TE-COD10K轻微下降。m3来源隔离进一步证明 TR-CAMO 与 TR-COD10K 贡献存在非加性交互。m4因此引入COD10K `q_semantic` Top-25%剂量控制，而不是继续修改门控。

### 4.5 A1朴素SAM2训练崩溃

首次 A1 训练报错：

```text
AttributeError: 'list' object has no attribute 'dim'
```

根因：4028张PNG加载为 `[1,68,68]`，12张缺失PNG的pkl回退为 `[1,16,16]`；混合batch无法stack，collate返回list。

修复：`BaseCODDataset` 仅在存在精修PNG目录且当前样本回退pkl时，把回退张量双线性缩放到68x68。纯A0 pkl路径仍保持16x16，基线不受影响。A1正式有效版本是 `20260725_v2`；失败的v1保留作为事故记录。

### 4.6 进度显示和GPU利用率疑问

早期全量处理只有逐图文本，没有标准进度条；CPU预处理和GPU推理也没有充分重叠。后来增加 `tqdm` 和有界 staged pipeline，并用测试验证 finish 阶段与下一次推理可重叠、结果仍保持输入顺序。

### 4.7 叙事混淆

会话中曾再次把旧版“自适应提示、多掩码门控、Local-SAM”写成创新点2的内部消融，也曾把完整方案简称为A2。两者均已被用户纠正。新会话必须以本文件、`docs/AEEM_V2_QSEMANTIC25_FINAL_CONFIG.md` 和实际 `aeem_v2/` 代码为准，不能凭对话记忆补写模块。

---

## 5. 什么代码和数据不能动

“不能动”指：未经用户明确同意，不得原地修改、删除、覆盖、移动或用新内容冒充原实验来源。

### 5.1 绝对保护的数据与工件

| 路径 | 原因 |
|---|---|
| `datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K` | A0原始4040粗伪标签来源 |
| `datasets/cache/naive_sam2_labels` | A1正式朴素SAM2来源，4028 PNG |
| `artifacts/aeem_v2/m2_full4040_structure_20260724_v1` | 全量AEEM v2候选、audit、manifest和hash |
| `artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1` | 当前完整方案标签和来源清单 |
| `work_dir/uscod/UCOD-DPL_dinov2_ablation_a0_baseline/...20260725_v1` | 正式基线checkpoint和日志 |
| `work_dir/uscod/UCOD-DPL_dinov2_ablation_a1_naive_sam2/...20260725_v2` | 正式朴素SAM2 checkpoint和日志 |
| `work_dir/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/...m4...` | 当前完整方案checkpoint和日志 |
| `datasets/cache/raw_sam2_outputs` | 历史候选来源，不可覆盖 |

不需要为了新实验“清理旧标签”。新配置用独立路径即可。

### 5.2 不能回退的关键数据加载代码

| 文件 | 必须保留的行为 |
|---|---|
| `data/datasets/base_dataset.py` | 独立精修标签路径；PNG优先；部分PNG时pkl回退统一为68x68；纯基线仍走原始pkl |
| `data/datasets/dataloader_utils.py` | 传递 `refined_pseudo_label_dir` |
| `data/datasets/uscod_dataset.py` | 传递可选精修目录 |
| `data/datasets/lr_dataset.py` | 传递可选精修目录 |

这些行为有单元测试保护。修改前必须先解释为什么，修改后至少运行全部32项测试。

### 5.3 冻结的实验入口

```text
configs/uscod/UCOD-DPL_dinov2_ablation_a0_baseline.py
configs/uscod/UCOD-DPL_dinov2_ablation_a1_naive_sam2.py
scripts/run_core_ablation_train.ps1
scripts/run_core_ablation_eval.ps1
scripts/run_aeem_v2_train.ps1
scripts/run_aeem_v2_eval.ps1
scripts/prepare_aeem_v2_qsemantic25.ps1
```

这些文件定义了当前结果的复现口径。若确需修改，必须复制成新配置/新脚本并使用新experiment ID，不能原地改后继续引用旧结果。

### 5.4 历史旧脚本

`scripts/offline_sam2_refine.py` 含旧门控实验修改，是 naive 标签的历史来源。不要在原文件上“顺手修复”为AEEM v2，也不要用它重新覆盖 `naive_sam2_labels`。新的算法改动进入 `aeem_v2/` 或新版本脚本。

### 5.5 AEEM v2 源码的版本规则

`aeem_v2/` 可以为下一版本扩展，但不能修改后仍声称旧 m2/m4 artifact由新代码生成。任何算法变化必须：

1. 使用新 experiment ID。
2. 输出新 artifact 目录。
3. 保存 manifest、Git diff、输入/输出 hash。
4. 保留旧 m2/m4 不动。
5. 先通过单元测试和标签级诊断，再训练。

### 5.6 明确禁止

- 不执行 `git reset --hard` 或 `git checkout --`。
- 不删除失败的A1 v1、旧日志或旧checkpoint。
- 不创建A0配置中的 `NO_REFINED_LABELS` 目录。
- 不把 `cache_dir` 指向artifact。
- 不使用 `--mode native`；旧脚本实际参数是 `naive`，拼成`native`会静默生成零标签。
- 不覆盖同名experiment目录。
- 不再扫描旧版 `s_lower/s_upper/gamma` 作为AEEM v2优化。
- 不用GT选择阈值或训练样本。
- 不因单seed、0.001量级变化宣称统计显著。

---

## 6. 接下来的目标是什么

### 6.1 创新点1直接证据已完成

`innovation1_naive_vs_softcoarse_20260725_v1` 已完成 4040 张标签级比较。朴素 SAM2 相对 Soft-Coarse 的 Boundary IoU 与 BF-score 明确提升，但区域 IoU 没有稳定提升，MAE 和少量灾难失败需要诚实报告。GT 只用于冻结后诊断。

### 6.2 创新点2额外控制已完成

- 工件：`artifacts/aeem_v2/innovation2_reliable_on_naive_20260725_v1`。
- 来源：1760 AEEM、2271 朴素 SAM2、9 Soft-Coarse fallback，共 4040。
- 训练：独立 `work_dir_validation_20260725`，25 epoch，五个 checkpoint 完整。
- 评估：CHAMELEON、TE-CAMO、TE-COD10K、NC4K 均完成。
- 相对朴素 SAM2 宏平均：SMeasure `+0.002575`、MAE `-0.000825`、E_MEAN `+0.004050`、F_MAX `-0.005600`、WFM `+0.000050`。
- 预注册判定：**混合结果**。SMeasure 和 MAE 都只有 2/4 数据集方向改善，且 F_MAX 下降；不能声称全面提升或统计显著。

完整方案 m4 仍是论文当前完整方案；额外控制不替代 m4，也不命名为 A2。

### 6.3 第三目标：统计稳定性

在创新证据表获得用户确认后：

- 最终候选至少补2个seed，形成3-seed均值和标准差。
- 需要时增加COD10K随机25%对照，区分 `q_semantic` 排序效果与单纯少用AEEM的效果。
- 剂量消融0/25/50/100只在用户确认且有明确判定标准时运行，不再边看GT边调。

### 6.4 当前不要做的事

- 不立刻重跑基线、朴素SAM2或m4；三组checkpoint和日志都在。
- 不重新运行2.5小时的全量SAM2。
- 不先写漂亮结论再找实验支撑。
- 不修改AEEM模块定义来迎合已有结果。

---

## 7. 新会话接手后的直接操作顺序

1. 优先读取 `docs/AEEM_V2_UNIFIED_EXPERIMENT_MANUAL.md`，再按需读取本文件和两个冻结协议。
2. 运行只读检查：`git status --short`，确认四组 `eval0.log`、epoch25 和三个关键标签 hash 仍存在。
3. 复核旧结果时只执行统一手册第 4 节评估命令，不重新训练，不运行 SAM2。
4. 若用户要求新复现，必须使用新实验 ID 或新 `WorkDir`，不得覆盖旧标签、checkpoint 和日志。
5. 后续研究重点是多 seed 稳定性或经批准的 q_semantic 随机 25% 对照，不再调旧版全局门控。

测试命令：

```powershell
conda activate test01
Set-Location "C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main"
& "C:\Anaconda\envs\test01\python.exe" -m unittest discover -s tests -v
```

现有评估入口（需要时可复查，不需要重新训练）：

```powershell
& .\scripts\run_core_ablation_eval.ps1 -Group a0
& .\scripts\run_core_ablation_eval.ps1 -Group a1

& .\scripts\run_aeem_v2_eval.ps1 `
  -ExperimentId m4_camo_all_cod10k_qsemantic25_20260724_v1 `
  -Checkpoint "work_dir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_m4_camo_all_cod10k_qsemantic25_20260724_v1\ckp\epoch25.pth" `
  -Port 11152
```

## 最终交接判断

AEEM v2 源码、4040 全量标签、来源隔离、完整方案 m4、基线/朴素 SAM2 消融、两项创新的标签级证据、可靠替换额外控制和统一操作手册均已完成。可靠替换结果是“混合结果”，这一定性不能在后续文稿中改写为全面提升。

统一手册是当前最高优先级操作入口。后续任何训练必须隔离新 ID/目录；若要声称稳健或统计显著，需要另行批准多 seed，而不是用当前单 seed 的千分位差值代替不确定性分析。
