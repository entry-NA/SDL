# AEEM v2 Milestone 3：训练来源隔离方案

## 1. 本轮统一评估结果

所有结果使用同一训练配置、同一评估脚本和同一复现基线。

| 数据集 | 复现基线 SMeasure | AEEM/AEEM SMeasure | ΔS | 复现基线 MAE | AEEM/AEEM MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|
| CHAMELEON | 0.8642 | 0.8629 | -0.0013 | 0.0311 | 0.0315 | +0.0004 |
| TE-CAMO | 0.7921 | 0.7983 | +0.0062 | 0.0766 | 0.0748 | -0.0018 |
| TE-COD10K | 0.8347 | 0.8332 | -0.0015 | 0.0300 | 0.0307 | +0.0007 |
| NC4K | 0.8504 | 0.8510 | +0.0006 | 0.0419 | 0.0416 | -0.0003 |

四数据集宏平均：SMeasure 提升 `0.0010`，MAE 改善 `0.00025`。当前完整方案不是整体失败，但只在 TE-CAMO 上形成了清晰增益，其余三个数据集的变化较小。

## 2. 已知证据

标签级 GT 诊断显示：

| 训练来源 | ΔIoU | ΔBF | ΔMAE |
|---|---:|---:|---:|
| TR-CAMO | +0.031716 | +0.035000 | -0.011575 |
| TR-COD10K | +0.025613 | +0.021879 | -0.008334 |

两套训练标签的像素与边界指标均改善，但 TR-COD10K 的下游测试指标没有同步改善。因此，下一步不修改 AEEM 参数，而是隔离两套训练来源的贡献及其交互。

## 3. 2×2 来源隔离矩阵

| 组别 | TR-CAMO 标签 | TR-COD10K 标签 | 状态 |
|---|---|---|---|
| Soft/Soft | Soft-Coarse | Soft-Coarse | 已有复现基线 |
| AEEM/Soft | AEEM v2 | Soft-Coarse | 下一组，优先运行 |
| Soft/AEEM | Soft-Coarse | AEEM v2 | 第二组，用于因果确认 |
| AEEM/AEEM | AEEM v2 | AEEM v2 | 已完成 |

混合标签仅复制现有 PNG，不重新运行 SAM2。每个工件固定为4040张标签，并记录来源、文件哈希、Git状态和不可覆盖 manifest。

### 3.1 AEEM/Soft 实测结果

| 数据集 | Soft/Soft S | AEEM/Soft S | AEEM/AEEM S | AEEM/Soft 相对基线 | AEEM/Soft MAE |
|---|---:|---:|---:|---:|---:|
| CHAMELEON | 0.8642 | 0.8648 | 0.8629 | +0.0006 | 0.0310 |
| TE-CAMO | 0.7921 | 0.7905 | 0.7983 | -0.0016 | 0.0772 |
| TE-COD10K | 0.8347 | 0.8356 | 0.8332 | +0.0009 | 0.0296 |
| NC4K | 0.8504 | 0.8505 | 0.8510 | +0.0001 | 0.0420 |

从 AEEM/Soft 切换到 AEEM/AEEM，唯一新增变量是 TR-COD10K 使用 AEEM 标签。该变量对应：TE-CAMO `+0.0078`，CHAMELEON `-0.0019`，TE-COD10K `-0.0024`，NC4K `+0.0005`。因此当前首要假设是 TR-COD10K AEEM 标签制造了明显的跨数据集权衡，而不是门控参数导致所有数据集同步变化。

下一项可证伪预测：若上述效应主要由 TR-COD10K AEEM 标签独立产生，则 Soft/AEEM 应接近 CHAMELEON `0.8623`、TE-CAMO `0.7999`、TE-COD10K `0.8323`、NC4K `0.8509`。若实测明显偏离，则说明两套精修来源之间存在非加性交互。

### 3.2 Soft/AEEM 实测结果

实际结果为：CHAMELEON `0.8640`、TE-CAMO `0.7959`、TE-COD10K `0.8338`、NC4K `0.8508`。与上面的加性预测相比，四个数据集分别偏离 `+0.0017`、`-0.0040`、`+0.0015`、`-0.0001`，因此确认两套训练来源存在非加性交互。完整 AEEM/AEEM 仍是宏平均最优，但它牺牲了 CHAMELEON 和 TE-COD10K 的少量性能换取 TE-CAMO 的较大增益。

## 3.3 下一轮：无GT语义一致性剂量控制

TR-COD10K 审计中，冻结的 `selected.q_semantic` 与标签级 ΔIoU 的 Spearman 相关为 `0.644`。取该分数最高的25%（3040张中760张）时，标签级平均 ΔIoU 为 `+0.0707`，高于全量平均 `+0.0256`。下一轮固定使用：

- TR-CAMO：全部 AEEM v2；
- TR-COD10K：仅 `selected.q_semantic` 最高25%使用 AEEM，其余使用 Soft-Coarse；
- 该选择只读取冻结审计信号，不读取GT；同分按文件名稳定排序。

这是剂量控制诊断，不在本轮结果出来后继续调分位数。只有该组完成四数据集评估后，才决定是否做随机25%对照和第二个seed。

## 4. 实验顺序

### 4.1 生成两个混合工件

```powershell
& .\scripts\prepare_aeem_v2_source_isolation.ps1
```

生成：

- `m3_isolate_camo_20260724_v1`：AEEM/Soft
- `m3_isolate_cod10k_20260724_v1`：Soft/AEEM

### 4.2 优先训练 AEEM/Soft

```powershell
& .\scripts\run_aeem_v2_train.ps1 `
  -ExperimentId m3_isolate_camo_20260724_v1 `
  -Port 11147
```

### 4.3 统一评估 AEEM/Soft

```powershell
& .\scripts\run_aeem_v2_eval.ps1 `
  -ExperimentId m3_isolate_camo_20260724_v1 `
  -Checkpoint "work_dir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_m3_isolate_camo_20260724_v1\ckp\epoch25.pth" `
  -Port 11148
```

## 5. 判定逻辑

- 若 AEEM/Soft 保留 TE-CAMO 增益，同时恢复 CHAMELEON 和 TE-COD10K，则 TR-COD10K 的 AEEM 标签或两来源交互是主要退化源。
- 若 AEEM/Soft 与 AEEM/AEEM 表现接近，则优先运行 Soft/AEEM，检查 TR-CAMO 标签是否主导当前变化。
- 若两组单来源实验均优于 AEEM/AEEM，则存在负向来源交互，不应继续使用简单拼接训练。
- 若所有差值仍在约 `0.001` 内波动，则先重复候选最优组的第二个 seed，再决定是否修改算法。

在完成来源隔离前，不扫描 `s_lower`、`s_upper`、`gamma`，也不修改 SAM2 提示与边界带参数。
