# AEEM v2 q_semantic Top-25% 最终候选配置记录

更新日期：2026-07-25  
状态：单随机种子最终候选，已完成四个测试集评估；尚需多随机种子复验。

## 1. 项目目标与创新定位

项目目标是在 UCOD-DPL 的无监督伪装目标检测流程中，引入完全冻结的 SAM2，对 DINOv2 生成的低分辨率粗伪标签进行离线边界精修，同时不修改 UCOD-DPL 的网络主体。

两项创新保持不变：

1. **离线零样本边界精修范式**：将冻结的 SAM2 作为离线边界专家，放在 APM 之前，改变 APM 的伪标签输入源。
2. **自适应边缘感知增强机制（Adaptive Edge-Aware Enhancement Mechanism, AEEM）**：解决粗标签空间偏移、SAM2 提示误导、边界幻觉和结构碎片问题。

当前推荐的论文主线名称仍为“自适应边缘感知增强机制（AEEM）”。变化的是机制内部技术内涵：从旧版全图级门控，升级为语义定位校正、提示路由、多候选质量评估、边界带像素级融合、结构安全回退和训练源剂量控制。

## 2. 旧版创新方案及问题

旧版 AEEM 主要包含：面积自适应框扩张、多层级正负点提示、截断式多掩码选择、SAM2/粗标签/图像边缘三因子图像级评分、`s_lower`/`s_upper`/`gamma` 门控，以及 Local-SAM 小目标精修。

排查后确认的主要问题：

- `EdgeAlign` 曾存在少除以 255 的量纲错误，导致早期采纳率不能作为质量证据。
- `gamma=50` 会使融合权重超过 1，使粗标签权重 `1-S` 变为负数，不再是合法凸组合。
- `s_upper=999` 实际关闭 FULL 分支，而不是“关闭融合、全部 FULL”。
- 强框、多个正点和多个负点会让 SAM2 近似复制粗标签，无法提供独立边界信息。
- 单个图像级分数 `S` 无法描述“内部可靠、局部边界不可靠”的空间差异。
- COD10K 的 SAM2 候选更容易产生碎片和额外连通域，影响 CHAMELEON、COD10K 的跨数据集泛化。
- Local-SAM 触发率很低，不再作为当前主机制的核心贡献。

因此，当前版本不再继续以扫描 `s_lower`、`s_upper`、`gamma` 为主要优化方向。

## 3. AEEM v2 当前技术结构

### 3.1 语义定位校正

- 使用冻结的 DINOv2 特征缓存构建前景、背景原型。
- 计算像素对前景/背景原型的相似度差，得到语义概率图。
- 可靠性由面积一致性、质心一致性、连通域一致性、区域 IoU 和语义间隔五项平均得到。
- 路由阈值：`low < 0.33`、`0.33 <= medium < 0.67`、`high >= 0.67`。

### 3.2 自适应提示路由

- High：单个高置信正点；生成 `point_only` 和 `weak_box` 两类提示。
- Medium：最多 3 个正点；增加 `consensus_points`，负点只从远离粗标签且语义置信低的安全背景中选择。
- Low：不调用 SAM2，直接回退到粗标签。
- SAM2 每类提示返回多掩码候选，因此 High 通常有 6 个候选，Medium 通常有 9 个候选。

### 3.3 多候选质量评估

每个候选拆分为四项质量：

- `q_semantic`：候选内部前景语义和外部背景语义的一致性。
- `q_stability`：不同提示生成的掩码之间是否稳定一致。
- `q_edge`：候选边界是否得到真实图像梯度支持。
- `q_safety`：面积、质心、可靠前景覆盖和可靠背景排除的综合安全性。

候选总质量为四项等权平均；`q_safety >= 0.25` 才视为有效候选；最终最低候选质量为 `0.35`。

### 3.4 边界不确定带像素级融合

SAM2 不再修改整幅标签，只在边界不确定带内执行受控残差精修：

- High 路由半径：等效目标半径的 5%，限制在 2–12 像素。
- Medium 路由半径：等效目标半径的 10%，限制在 4–20 像素。
- Low 路由半径为 0，直接回退。
- 可靠前景核心和远端可靠背景被保护，不允许 SAM2 覆盖。
- 像素置信图：`Q(x) = 候选质量 × 提示共识 × 语义一致性 × 边缘支持`。

### 3.5 结构安全校准

- 最大有效连通域增长：`1`。
- 最大额外结构质量比例：`0.05`。
- 超过阈值、无提示、无可靠候选或结构风险时回退到 Soft-Coarse。

全量 AEEM v2 工件统计：

- High：1897 张。
- Medium：2126 张。
- Low：17 张。
- 结构增长回退：979 张。
- 无提示回退：17 张。
- SAM2 实际编码图像：4023 张。
- 生成耗时：约 2 小时 35 分钟（本机记录，硬件相关）。

## 4. 最终训练标签配置

最终实验 ID：`m4_camo_all_cod10k_qsemantic25_20260724_v1`

标签组成：

- TR-CAMO：1000 张全部使用 AEEM v2 标签。
- TR-COD10K：3040 张按冻结审计字段 `selected.q_semantic` 降序排序，最高 25%，即 760 张使用 AEEM v2；其余 2280 张使用 Soft-Coarse。
- 总计：AEEM 1760 张，Soft-Coarse 2280 张，共 4040 张。
- 3027 张 COD10K 样本具有有效评分，13 张无有效候选并自动留在 Soft-Coarse 组。
- 入选的 `q_semantic` 范围：`0.8362326473–0.9331809469`。
- 选择过程不读取 GT；同分时按文件名稳定排序。

这一剂量控制来自训练源隔离实验：TR-CAMO 的 AEEM 收益较稳定，而 TR-COD10K 全量 AEEM 容易引入碎片和跨数据集权衡。因此最终方案不是“所有标签都交给 SAM2”，而是“CAMO 全量 + COD10K 高语义质量子集”。

## 5. 训练配置

| 配置项 | 当前值 |
|---|---|
| DINOv2 | `facebook/dinov2-base` |
| 输入尺寸 | `518 × 518` |
| 解码特征尺寸 | `68` |
| Batch size | `16` |
| Epoch | `25` |
| 学生学习率 | `2e-4` |
| 判别器学习率 | `1e-3` |
| EMA | `0.99` |
| APM merge | `dis` |
| Look-Twice | 开启 |
| `look_twice_th` | `0.15` |
| `expand_type` | `dynamic` |
| 随机种子 | `42` |
| 混合精度 | Accelerate 启动参数 `fp16` |
| 训练集 | `TR-CAMO + TR-COD10K` |
| 当前验证集配置 | `NC4K` |
| DataLoader workers | `0` |

训练网络仍是原 UCOD-DPL；AEEM v2 只改变离线伪标签输入。

## 6. 工件与检查点

- 最终标签工件：`artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1`
- 最终标签目录：`artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1/refined_pseudo_labels`
- 工件输出哈希：`073abc4dcd13eaa24eb050a7dc063a88dda1a5a644750eeefd8b7270cb92895e`
- 全量 AEEM v2 工件：`artifacts/aeem_v2/m2_full4040_structure_20260724_v1`
- 全量 AEEM 输出哈希：`2e3a081f55d806b3530d00626c231c0e221dd33ac0a0515e308e8fc6e2473850`
- 最终检查点：`work_dir/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/UCOD-DPL_dinov2_aeem_v2_m4_camo_all_cod10k_qsemantic25_20260724_v1/ckp/epoch25.pth`
- 实际权重文件位于上述目录中的 `model.safetensors`。
- 工件记录的 Git commit：`7b7ca16e05bc34ee4fd7057541ce5f15b6ec8ae3`。
- 工件生成时工作区存在未提交修改；对应差异已保存到工件中的 `git_diff.patch`，不能只凭 commit 重建。

## 7. 当前评估结果

| 数据集 | E_MEAN | SMeasure | MAE | WFM | F_MAX |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | 0.9316 | 0.8648 | 0.0310 | 0.8259 | 0.8400 |
| TE-CAMO | 0.8639 | 0.7939 | 0.0760 | 0.7482 | 0.7811 |
| TE-COD10K | 0.9160 | 0.8344 | 0.0302 | 0.7633 | 0.7805 |
| NC4K | 0.9240 | 0.8513 | 0.0415 | 0.8190 | 0.8376 |

论文参考值使用 `C:\Users\23991\Desktop\实验数据.xlsx` 中 Sheet1 第 40–45 行记录：

| 数据集 | E_MEAN | SMeasure | MAE | WFM | F_MAX |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | 0.931 | 0.864 | 0.031 | 0.825 | 0.838 |
| TE-CAMO | 0.862 | 0.793 | 0.077 | 0.747 | 0.779 |
| TE-COD10K | 0.916 | 0.834 | 0.031 | 0.763 | 0.779 |
| NC4K | 0.923 | 0.850 | 0.043 | 0.818 | 0.835 |

按这套论文口径，20 个可比指标中 18 项严格优于论文、2 项在显示精度下持平、0 项下降。四个 SMeasure 均高于论文参考值。

与 Excel 中“基线”行比较，当前四个数据集的 SMeasure 均不低于本地基线：CHAMELEON `+0.0018`、TE-CAMO `+0.0007`、TE-COD10K 四位小数下持平、NC4K `+0.0006`。

需要同时保留的限制：当前结果低于 Excel 中历史“完整”行的四个 SMeasure，说明它是当前可复现、结构安全并超过论文参考值的候选配置，但不是历史记录中的最高单次分数。正式论文结论仍需统一协议并进行多随机种子复验。

## 8. 手动复现命令

项目目录：`C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main`

```powershell
conda activate test01
Set-Location "C:\Users\23991\Desktop\新建文件夹\UCOD-DPL-main\UCOD-DPL-main"

& .\scripts\prepare_aeem_v2_qsemantic25.ps1

& .\scripts\run_aeem_v2_train.ps1 `
  -ExperimentId m4_camo_all_cod10k_qsemantic25_20260724_v1 `
  -Port 11151

& .\scripts\run_aeem_v2_eval.ps1 `
  -ExperimentId m4_camo_all_cod10k_qsemantic25_20260724_v1 `
  -Checkpoint "work_dir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_m4_camo_all_cod10k_qsemantic25_20260724_v1\ckp\epoch25.pth" `
  -Port 11152
```

已有完整工件时，准备脚本会检查 `manifest.json` 后跳过，不会覆盖。训练和评估均按实验 ID 定位标签与输出，不需要清理旧实验。

## 9. 下一阶段

1. 固定当前配置，增加至少 3 个随机种子，报告均值和标准差。
2. 增加 COD10K 随机 25% 对照，证明收益来自 `q_semantic` 排序而不是“少用 75% AEEM”本身。
3. 增加 COD10K 0%、25%、50%、100% 剂量消融，验证剂量与跨数据集泛化的关系。
4. 在统一 checkpoint、评估脚本、输入尺寸和 Look-Twice 设置下复跑本地基线及历史“完整”方案。
5. 论文表述暂用“在当前单 seed、统一 Excel 论文口径下全面不低于 UCOD-DPL”，不要提前写成统计显著超越。
