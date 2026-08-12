# 创新点2：归因证据与额外增量控制协议

> 日期：2026-07-25  
> 状态：标签级诊断、可靠替换标签工件、单 seed 25 epoch 训练与四测试集评估均已完成  
> 保护边界：不修改训练主干、数据加载链路、冻结标签生成代码、既有标签、checkpoint 或日志

## 1. 要回答的问题

创新点2的核心主张不是“完整方案在所有边界指标上超过朴素 SAM2”，而是：AEEM 的语义定位、提示路由、边界带约束与结构安全回退，能否在保留正边界收益的同时，减少朴素 SAM2 的区域漂移、碎片和严重失败。

证据分为三层，不能混用：

1. 标签级机制证据：完整方案标签与朴素 SAM2 在相同 4040 张训练样本上的直接比较。
2. 既有同底座下游控制：全量 AEEM 标签训练与朴素 SAM2 标签训练的比较。
3. 已完成的额外增量控制：以朴素 SAM2 为底座，只在冻结的 1760 个可靠样本上换用 AEEM 标签。

## 2. 已完成：标签级机制证据

输出：`artifacts/aeem_v2/innovation2_m2_vs_naive_20260725_v1`

- 完整方案标签：`m2_full4040_structure_20260724_v1`，4040 张。
- 朴素 SAM2：4028 张现有 PNG；缺失 12 张显式回退 Soft-Coarse。
- GT：TR-CAMO 1000 张、TR-COD10K 3040 张。GT 只用于冻结后诊断，不参与选择或调参。
- Manifest：`status=complete`、`gt_count=4040`、`row_count=8080`、fallback 12。

下表为“完整方案标签 - 朴素 SAM2 标签”：

| 指标 | 平均差值 | 95% bootstrap CI | 解释 |
|---|---:|---:|---|
| IoU | +0.028078 | [+0.025808, +0.030247] | 区域一致性改善 |
| MAE | -0.013335 | [-0.016113, -0.010863] | 软像素误差降低 |
| 最大连通域质心偏移 | -0.004527 | 区间严格小于 0 | 定位漂移缓解 |
| raw 连通域数量误差 | -65.6025 | 区间严格小于 0 | 微小碎片显著减少 |
| Boundary IoU | -0.017168 | 区间严格小于 0 | 牺牲部分朴素 SAM2 的极锐边界收益 |
| BF-score | -0.046053 | 区间严格小于 0 | 同上 |

完整方案标签的绝对 Boundary IoU/BF-score 为 `0.111681/0.086987`，仍高于 Soft-Coarse 的 `0.083851/0.061860`。因此可辩护结论是“用部分边界锐度换取区域与结构安全，同时保留相对粗标签的正边界收益”，不能写成“全面超过朴素 SAM2”。

## 3. 已完成：既有同训练底座控制

比较对象：

- 全量 AEEM：`UCOD-DPL_dinov2_aeem_v2_m2_full4040_structure_20260724_v1`
- 朴素 SAM2：`UCOD-DPL_dinov2_ablation_a1_naive_sam2_20260725_v2`

两份保存的 `config.yaml` 已逐字段核对。除 `exp_name`、`work_dir/log_path`、`checkpoint` 和 `refined_pseudo_label_dir` 这些实验身份/输入输出路径外，数据集、DINOv2-base、518 输入、batch 16、25 epoch、优化参数、EMA、APM、Look-Twice 与评估设置一致。因此无需重复训练 m2。

下表为“全量 AEEM - 朴素 SAM2”；MAE 越低越好，其余指标越高越好：

| 数据集 | Delta E_MEAN | Delta F_MAX | Delta SMeasure | Delta MAE | Delta WFM |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | -0.0023 | -0.0129 | -0.0032 | +0.0009 | -0.0076 |
| TE-CAMO | +0.0156 | +0.0023 | +0.0124 | -0.0038 | +0.0107 |
| TE-COD10K | -0.0028 | -0.0156 | -0.0035 | +0.0013 | -0.0087 |
| NC4K | +0.0037 | -0.0104 | +0.0012 | -0.0007 | -0.0028 |
| 四数据集宏平均 | +0.00355 | -0.00915 | +0.00173 | -0.00058 | -0.00210 |

共 8/20 项改善、12/20 项下降。宏平均 E_MEAN、SMeasure、MAE 改善，F_MAX、WFM 下降，与标签级的“区域/结构安全和边界锐度之间存在权衡”一致。由于只有单 seed，以上只能称为方向性证据，不能称统计显著。

该控制仍不是最终完整方案 m4 的严格归因，因为 m2 使用 4040 张 AEEM 标签，而 m4 使用 1760 张 AEEM + 2280 张 Soft-Coarse。

## 4. 已完成：以朴素 SAM2 为底座的可靠替换控制

### 4.1 唯一研究变量

保持正式朴素 SAM2 训练的模型、数据顺序、seed、优化器、学习率、batch、epoch、EMA、APM、Look-Twice、评估脚本与四个测试集不变。唯一实质变量是训练样本的 `refined_pseudo_label_dir`：在冻结的 m4 可靠样本集合上换用现有 AEEM 标签，其余样本继续使用现有朴素 SAM2 标签。

该控制只能称为“额外增量控制”，不能替代当前完整方案，也不能命名为 A2。

### 4.2 冻结组合规则

可靠集合直接读取现有 m4 `audit.jsonl`，不得重新计算阈值，不得读取 GT：

- 1760 个 `source_type=aeem` 样本：使用 `m2_full4040_structure_20260724_v1` 中已有 AEEM 标签。
- 其余 2280 个样本：优先使用 `datasets/cache/naive_sam2_labels` 中已有朴素 SAM2 PNG。
- 上述 2280 个样本中只有 9 个 TR-COD10K 样本缺少 naive PNG：回退 `m0_controls_20260724_v1` 的 Soft-Coarse。

12 个 naive 缺失样本的交叉核对结果是：3 个 TR-CAMO 样本已包含在 1760 个 AEEM 可靠集合内，另外 9 个 TR-COD10K 样本位于非可靠集合。因此预期最终构成为：

| 来源 | 数量 |
|---|---:|
| AEEM | 1760 |
| 朴素 SAM2 | 2271 |
| Soft-Coarse fallback | 9 |
| 合计 | 4040 |

### 4.3 实际实验身份与工件核验

- 标签工件 ID：`innovation2_reliable_on_naive_20260725_v1`
- 训练实验名：`UCOD-DPL_dinov2_aeem_v2_innovation2_reliable_on_naive_20260725_v1`
- 标签工件必须使用全新目录，并保存 `manifest.json`、`audit.jsonl`、输入/输出逐文件 hash、组合配置和生成时 Git diff。
- 训练输出必须使用全新 `work_dir`；不得覆盖 m2、m4、朴素 SAM2 或其他正式消融的目录。

实际核验结果：

- 标签目录：`artifacts/aeem_v2/innovation2_reliable_on_naive_20260725_v1`
- Manifest：`status=complete`，输入 4040、输出 4040，TR-CAMO 1000、TR-COD10K 3040。
- 来源：AEEM 1760、朴素 SAM2 2271、Soft-Coarse fallback 9。
- 逐文件复制前后 hash 不一致数：0。
- `output_hashes.json` SHA256：`a2e1d827fd75ffc3c69c4207d403b7e4970ed0957f453d06ee701cb55730d116`。
- 训练目录：`work_dir_validation_20260725/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/UCOD-DPL_dinov2_aeem_v2_innovation2_reliable_on_naive_20260725_v1`。
- epoch5、10、15、20、25 的 checkpoint 均存在；各 `model.safetensors` 为 790832 字节。
- 评估日志：上述训练目录下的 `eval0.log`。

## 5. 预注册判定标准

主要比较是“可靠替换控制 - 正式朴素 SAM2”，m4 只作次要描述性参照。

主要终点：四数据集宏平均 SMeasure 与 MAE。

- **机制一致的正向证据**：宏平均 SMeasure 上升且宏平均 MAE 下降，并且两项各自在至少 3/4 个数据集上方向一致。
- **完整净增益**：满足上条，同时宏平均 F_MAX 不下降且宏平均 WFM 不下降。
- **混合结果**：SMeasure/MAE 的宏平均方向正确，但跨数据集一致性不足，或 F_MAX/WFM 仍下降。只能报告区域/结构收益与边界代价，不能声称全面提升。
- **不支持独立下游增益**：宏平均 SMeasure 未上升且宏平均 MAE 未下降，或两项仅由单一数据集驱动。

E_MEAN、F_MAX、WFM 和 20 项逐数据集胜负均完整报告，但不在看到结果后更换主要终点。单 seed 通过上述规则也只能称“方向性支持”；若要声称稳健或统计显著，必须另行批准多 seed 复现，不能用千分位差值直接代替不确定性分析。

## 6. 实际结果与预注册判定

| 数据集 | E_MEAN | F_MAX | SMeasure | MAE | WFM |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | 0.9322 | 0.8405 | 0.8650 | 0.0310 | 0.8261 |
| TE-CAMO | 0.8675 | 0.7869 | 0.7981 | 0.0747 | 0.7540 |
| TE-COD10K | 0.9158 | 0.7810 | 0.8341 | 0.0304 | 0.7632 |
| NC4K | 0.9245 | 0.8389 | 0.8516 | 0.0415 | 0.8198 |
| 四数据集宏平均 | 0.910000 | 0.811825 | 0.837200 | 0.044400 | 0.790775 |

相对正式朴素 SAM2 的宏平均变化为：SMeasure `+0.002575`、MAE `-0.000825`、E_MEAN `+0.004050`、F_MAX `-0.005600`、WFM `+0.000050`。SMeasure 和 MAE 均只有 2/4 个数据集方向改善。

按第 5 节预注册规则，必须判定为 **混合结果**：主要终点的宏平均方向正确，但两项跨数据集一致性均未达到至少 3/4，且 F_MAX 仍下降。该单 seed 结果只能称为方向性证据，不能称为全面提升、稳健提升或统计显著。

## 7. 复核入口

- 标签级比较：`artifacts/aeem_v2/innovation2_m2_vs_naive_20260725_v1`
- m4 可靠集合：`artifacts/aeem_v2/m4_camo_all_cod10k_qsemantic25_20260724_v1/audit.jsonl`
- 全量 AEEM 评估日志：`work_dir/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/UCOD-DPL_dinov2_aeem_v2_m2_full4040_structure_20260724_v1/eval0.log`
- 朴素 SAM2 评估日志：`work_dir/uscod/UCOD-DPL_dinov2_ablation_a1_naive_sam2/UCOD-DPL_dinov2_ablation_a1_naive_sam2_20260725_v2/eval0.log`
- 可靠替换工件：`artifacts/aeem_v2/innovation2_reliable_on_naive_20260725_v1`
- 可靠替换训练与评估：`work_dir_validation_20260725/uscod/UCOD-DPL_dinov2_aeem_v2_full4040/UCOD-DPL_dinov2_aeem_v2_innovation2_reliable_on_naive_20260725_v1`
