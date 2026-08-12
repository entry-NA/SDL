# AEEM v2 GitHub 首次复现、消融、完整方案、验证与多随机种子统一操作手册

> 版本：2026-07-30  
> 项目：UCOD-DPL + 冻结 SAM2 / AEEM v2  
> 适用终端：Windows PowerShell  
> 参考 Python：`C:\Anaconda\envs\test01\python.exe`  
> 当前仓库状态：已清理缓存、生成标签、artifact 和训练 checkpoint；必须按本手册从输入恢复开始运行。  
> 保护原则：不修改模型主体；每次训练使用新 `WorkDir`，每次标签生成使用新实验 ID。

## 目录

1. [先看结论与实验边界](#1-先看结论与实验边界)
2. [GitHub 克隆后的环境和输入恢复](#2-github-克隆后的环境和输入恢复)
3. [公共预检与缓存说明](#3-公共预检与缓存说明)
4. [消融实验：基线与朴素-SAM2](#4-消融实验基线与朴素-sam2)
5. [完整方案-m4](#5-完整方案-m4)
6. [可靠替换验证实验](#6-可靠替换验证实验)
7. [多随机种子实验](#7-多随机种子实验)
8. [结果汇总与论文写法](#8-结果汇总与论文写法)
9. [完整性与复现核验](#9-完整性与复现核验)
10. [常见错误与处理](#10-常见错误与处理)
11. [历史单-seed-结果](#11-历史单-seed-结果)
12. [关键文件索引](#12-关键文件索引)

## 1. 先看结论与实验边界

### 1.1 本手册覆盖什么

- 消融实验：原始粗标签基线与朴素 SAM2。
- 完整方案：TR-CAMO 全量 AEEM + TR-COD10K `q_semantic` Top-25% AEEM，其余 Soft-Coarse。
- 可靠替换验证：以朴素 SAM2 为底座，只替换冻结可靠集合。
- 多随机种子：固定同一套标签，使用 seed 42、3407、2025 重复训练和评估。
- GitHub 清理后的数据恢复、缓存重建、结果汇总和常见排错。

### 1.2 当前哪些内容已不在仓库

以下均是有意删除的大型生成文件，不属于 GitHub 普通源码仓库：

- `datasets/cache/`：粗伪标签、朴素 SAM2 PNG 和 DINOv2 特征缓存。
- `artifacts/`：Soft-Coarse、全量 AEEM、m4 与可靠替换组合标签。
- `work_dir*`：训练 checkpoint、预测图和运行日志。
- 本地 DINOv2-base 大权重副本；程序首次运行时会从 Hugging Face 下载。

因此，旧手册中的“直接复用已有 checkpoint/工件”命令只具有历史说明作用，在干净仓库中不能直接执行。

### 1.3 绝对不能改变的实验规则

- 不修改网络结构、损失函数、优化器、batch size、输入尺寸、epoch 数或评估脚本来迎合结果。
- 不用 GT 选择样本、阈值、候选或随机种子；GT 只用于冻结方案后的诊断。
- 不创建基线配置中的 `NO_REFINED_LABELS` 目录。
- 不把 `cache_dir` 指向 artifact；精修标签只通过 `refined_pseudo_label_dir` 注入。
- 多 seed 时只改变训练随机种子，标签工件必须固定并由所有 seed 共用。
- 所有方案统一评估 epoch25，不允许为不同数据集挑选各自最优 epoch。
- 预先固定 seed 列表，所有 seed 都要报告；不能看到结果后删除不理想 seed。

## 2. GitHub 克隆后的环境和输入恢复

### 2.1 创建环境

```powershell
conda create -n test01 python=3.9 -y
conda activate test01
Set-Location "<你的项目目录>"
pip install -r requirement.txt
Set-ExecutionPolicy -Scope Process Bypass
```

检查 CUDA：

```powershell
$python = "C:\Anaconda\envs\test01\python.exe"
& $python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

### 2.2 恢复 RefCOD 数据集

按项目 `README.md` 的 Dataset Preparation 链接下载数据集，最终目录至少包含：

```text
datasets/RefCOD/TR-CAMO/im
datasets/RefCOD/TR-COD10K/im
datasets/RefCOD/CHAMELEON
datasets/RefCOD/TE-CAMO
datasets/RefCOD/TE-COD10K
datasets/RefCOD/NC4K
```

只读检查训练图像数量：

```powershell
(Get-ChildItem ".\datasets\RefCOD\TR-CAMO\im" -File).Count
(Get-ChildItem ".\datasets\RefCOD\TR-COD10K\im" -File).Count
```

两项应分别为 1000 和 3040。

### 2.3 恢复 4040 个原始粗伪标签

按 `README.md` 的 pseudo labels 链接下载并解压到：

```text
datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K
```

检查：

```powershell
$coarse = ".\datasets\cache\pseudo_label_cache\TR-CAMO+TR-COD10K"
Test-Path "$coarse\index.json"
(Get-ChildItem $coarse -Filter *.pkl -File).Count
```

必须得到 `True` 和 4040。没有这一步，基线、朴素 SAM2、Soft-Coarse 和 AEEM 都不能重建。

### 2.4 DINOv2 和 SAM2 权重

- DINOv2：训练或评估首次运行时自动下载 `facebook/dinov2-base`，随后自动建立 `datasets/cache/features_cache/dinov2`。缓存可删除，但下次会重新提取并增加等待时间。
- SAM2：朴素 SAM2 脚本会自动下载 `facebook/sam2.1-hiera-tiny`；完整 AEEM 脚本需要显式传入 `sam2.1_hiera_tiny.pt`。

查找本机 SAM2 checkpoint：

```powershell
$sam2Checkpoint = Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" `
  -Filter "sam2.1_hiera_tiny.pt" -File -Recurse | Select-Object -First 1 -ExpandProperty FullName
$sam2Checkpoint
Test-Path $sam2Checkpoint
```

## 3. 公共预检与缓存说明

### 3.1 运行测试

```powershell
& $python -m unittest discover -s tests -v
```

只有全部测试通过后再启动训练。

### 3.2 检查固定协议

所有可比较组必须共同使用：DINOv2-base、518x518、batch 16、25 epoch、FP16、EMA 0.99、APM `merge_method=dis`、Look-Twice 开启且阈值 0.15。

训练入口支持 `--seed`，两个 PowerShell 训练脚本支持 `-Seed`。默认值仍为 42；不传 seed 的旧命令行为不变。实际 seed 会写入每次运行保存的 `config.yaml`。

### 3.3 首次缓存重建

删除缓存后首次构建数据集会显示 DINOv2 特征提取进度。程序使用冻结 DINOv2 和固定预处理重建派生特征；这不修改图像、伪标签或模型主体。不要只删除缓存 `.pkl` 而保留旧 `index.json`；需要清理时应删除整个对应缓存目录。

## 4. 消融实验：基线与朴素 SAM2

### 4.1 生成朴素 SAM2 标签

此步骤只生成一次，后续所有 seed 共用同一目录：

```powershell
& $python -u scripts\offline_sam2_refine.py `
  --mode naive `
  --dataset_dir "datasets\RefCOD" `
  --coarse_dir "datasets\cache\pseudo_label_cache\TR-CAMO+TR-COD10K" `
  --output_dir "datasets\cache\refined_pseudo_labels"
```

检查：

```powershell
(Get-ChildItem ".\datasets\cache\naive_sam2_labels" -Filter *.png -File).Count
```

正式语义是 4028 张 PNG，另外 12 个空粗掩码样本回退原始 `.pkl`。

### 4.2 seed 42 消融训练与评估

```powershell
$singleWorkDir = "work_dir_single_seed_42"

& .\scripts\run_core_ablation_train.ps1 `
  -Group a0 -Seed 42 -WorkDir $singleWorkDir -Port 11211
& .\scripts\run_core_ablation_eval.ps1 `
  -Group a0 -WorkDir $singleWorkDir -Port 11212

& .\scripts\run_core_ablation_train.ps1 `
  -Group a1 -Seed 42 -WorkDir $singleWorkDir -Port 11213
& .\scripts\run_core_ablation_eval.ps1 `
  -Group a1 -WorkDir $singleWorkDir -Port 11214
```

`a0` 只使用 4040 个原始粗标签；`a1` 使用 4028 张朴素 SAM2 PNG和 12 个 `.pkl` fallback。二者训练设置必须相同。

## 5. 完整方案 m4

### 5.1 标签生成顺序

以下 ID 是本次干净重建示例。再次运行时应换新日期或版本后缀，不能覆盖已有目录。

```powershell
$m0Id = "m0_controls_20260730_rebuild_v1"
$m2Id = "m2_full4040_structure_20260730_rebuild_v1"
$m4Id = "m4_camo_all_cod10k_qsemantic25_20260730_rebuild_v1"

# 1. 4040 张 Hard/Soft-Coarse 控制标签
& $python -u scripts\prepare_aeem_controls.py --experiment-id $m0Id

# 2. 4040 张全量 AEEM v2 标签；会运行冻结 SAM2
& .\scripts\run_aeem_v2_full4040.ps1 `
  -ExperimentId $m2Id `
  -Checkpoint $sam2Checkpoint `
  -PostprocessWorkers 2 `
  -PipelineBuffer 4

# 3. TR-CAMO 全量 + TR-COD10K q_semantic Top-25%
& .\scripts\prepare_aeem_v2_qsemantic25.ps1 `
  -AeemExperimentId $m2Id `
  -ControlExperimentId $m0Id `
  -CamoExperimentId $m4Id
```

检查 m4：

```powershell
$m4Artifact = "artifacts\aeem_v2\$m4Id"
$m4Manifest = Get-Content "$m4Artifact\manifest.json" -Raw | ConvertFrom-Json
$m4Manifest.status
$m4Manifest.output_count
$m4Manifest.source_counts
(Get-ChildItem "$m4Artifact\refined_pseudo_labels" -Filter *.png -File).Count
```

通过标准：`status=complete`、输出 4040 张，其中 AEEM 1760、Soft-Coarse 2280。选择过程不读取 GT。

### 5.2 seed 42 训练与评估

```powershell
$m4WorkDir = "work_dir_m4_single_seed_42"

& .\scripts\run_aeem_v2_train.ps1 `
  -ExperimentId $m4Id -Seed 42 -WorkDir $m4WorkDir -Port 11221

$m4Checkpoint = "$m4WorkDir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_$m4Id\ckp\epoch25.pth"

& .\scripts\run_aeem_v2_eval.ps1 `
  -ExperimentId $m4Id -WorkDir $m4WorkDir `
  -Checkpoint $m4Checkpoint -Port 11222
```

## 6. 可靠替换验证实验

### 6.1 实验定义

该验证以朴素 SAM2 为底座，将 m4 冻结可靠集合中的 1760 个样本替换为 AEEM；其余样本优先使用朴素 SAM2，缺失 PNG 的 9 个样本使用 Soft-Coarse。GT 不参与选择。历史单 seed 结果属于“混合结果”，不能写成全面提升。

### 6.2 生成固定验证标签

```powershell
$validationId = "innovation2_reliable_on_naive_20260730_rebuild_v1"

& $python -u experiments\prepare_reliable_replacement_control.py `
  --experiment-id $validationId `
  --selection-audit "artifacts\aeem_v2\$m4Id\audit.jsonl" `
  --aeem-dir "artifacts\aeem_v2\$m2Id\refined_pseudo_labels" `
  --naive-dir "datasets\cache\naive_sam2_labels" `
  --soft-dir "artifacts\aeem_v2\$m0Id\controls\soft_coarse\refined_pseudo_labels"
```

检查：

```powershell
$validationArtifact = "artifacts\aeem_v2\$validationId"
$validationManifest = Get-Content "$validationArtifact\manifest.json" -Raw | ConvertFrom-Json
$validationManifest.status
$validationManifest.output_count
$validationManifest.source_counts
```

通过标准：4040 张，`aeem=1760`、`naive_sam2=2271`、`soft_fallback=9`。

### 6.3 seed 42 训练与评估

```powershell
$validationWorkDir = "work_dir_validation_single_seed_42"

& .\scripts\run_aeem_v2_train.ps1 `
  -ExperimentId $validationId -Seed 42 `
  -WorkDir $validationWorkDir -Port 11231

$validationCheckpoint = "$validationWorkDir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_$validationId\ckp\epoch25.pth"

& .\scripts\run_aeem_v2_eval.ps1 `
  -ExperimentId $validationId -WorkDir $validationWorkDir `
  -Checkpoint $validationCheckpoint -Port 11232
```

## 7. 多随机种子实验

### 7.1 为什么必须做多 seed

单 seed 的千分位差异可能来自初始化、DataLoader shuffle、CUDA 算子和并行归约。多 seed 的目标是估计训练不确定性，不是寻找最有利的 seed。预先固定：

```text
42, 3407, 2025
```

seed 42 保持与历史协议连续；3407 和 2025 在运行前固定。所有组使用相同三组 seed，形成配对比较。

### 7.2 多 seed 前冻结标签

只执行一次第 4.1、5.1 和 6.2 节。随后记录以下目录 hash，并让三个 seed 共用：

```text
datasets/cache/pseudo_label_cache/TR-CAMO+TR-COD10K
datasets/cache/naive_sam2_labels
artifacts/aeem_v2/<m4Id>
artifacts/aeem_v2/<validationId>
```

多 seed 期间禁止重新生成或替换标签，否则“训练随机性”与“输入变化”会混在一起。

### 7.3 基线与朴素 SAM2：三 seed

```powershell
$seedRuns = @(
  @{ Seed = 42;   TrainA0 = 12101; EvalA0 = 12102; TrainA1 = 12103; EvalA1 = 12104 },
  @{ Seed = 3407; TrainA0 = 12111; EvalA0 = 12112; TrainA1 = 12113; EvalA1 = 12114 },
  @{ Seed = 2025; TrainA0 = 12121; EvalA0 = 12122; TrainA1 = 12123; EvalA1 = 12124 }
)

foreach ($run in $seedRuns) {
  $seed = $run.Seed
  $workDir = "work_dir_multiseed\seed_$seed"

  & .\scripts\run_core_ablation_train.ps1 `
    -Group a0 -Seed $seed -WorkDir $workDir -Port $run.TrainA0
  & .\scripts\run_core_ablation_eval.ps1 `
    -Group a0 -WorkDir $workDir -Port $run.EvalA0

  & .\scripts\run_core_ablation_train.ps1 `
    -Group a1 -Seed $seed -WorkDir $workDir -Port $run.TrainA1
  & .\scripts\run_core_ablation_eval.ps1 `
    -Group a1 -WorkDir $workDir -Port $run.EvalA1
}
```

### 7.4 完整方案与可靠替换：三 seed

```powershell
$aeemSeedRuns = @(
  @{ Seed = 42;   TrainM4 = 12201; EvalM4 = 12202; TrainVal = 12203; EvalVal = 12204 },
  @{ Seed = 3407; TrainM4 = 12211; EvalM4 = 12212; TrainVal = 12213; EvalVal = 12214 },
  @{ Seed = 2025; TrainM4 = 12221; EvalM4 = 12222; TrainVal = 12223; EvalVal = 12224 }
)

foreach ($run in $aeemSeedRuns) {
  $seed = $run.Seed
  $workDir = "work_dir_multiseed\seed_$seed"

  & .\scripts\run_aeem_v2_train.ps1 `
    -ExperimentId $m4Id -Seed $seed -WorkDir $workDir -Port $run.TrainM4
  $m4Checkpoint = "$workDir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_$m4Id\ckp\epoch25.pth"
  & .\scripts\run_aeem_v2_eval.ps1 `
    -ExperimentId $m4Id -WorkDir $workDir `
    -Checkpoint $m4Checkpoint -Port $run.EvalM4

  & .\scripts\run_aeem_v2_train.ps1 `
    -ExperimentId $validationId -Seed $seed -WorkDir $workDir -Port $run.TrainVal
  $validationCheckpoint = "$workDir\uscod\UCOD-DPL_dinov2_aeem_v2_full4040\UCOD-DPL_dinov2_aeem_v2_$validationId\ckp\epoch25.pth"
  & .\scripts\run_aeem_v2_eval.ps1 `
    -ExperimentId $validationId -WorkDir $workDir `
    -Checkpoint $validationCheckpoint -Port $run.EvalVal
}
```

完整四组需要 12 次训练。若计算预算不足，最低优先级是先完成“基线 vs 完整方案”同三 seed 的 6 次训练；但没有“朴素 SAM2 vs 可靠替换”的同 seed 结果时，不能声称验证实验稳定成立。

### 7.5 每个 seed 的验收

```powershell
$seeds = 42,3407,2025
foreach ($seed in $seeds) {
  Get-ChildItem "work_dir_multiseed\seed_$seed" -Filter config.yaml -File -Recurse |
    ForEach-Object { Select-String -Path $_.FullName -Pattern '^seed:' }
  Get-ChildItem "work_dir_multiseed\seed_$seed" -Filter model.safetensors -File -Recurse |
    Where-Object { $_.FullName -match 'epoch25\.pth' }
}
```

必须看到每次运行保存的 seed、epoch25 权重以及四测试集评估日志。

## 8. 结果汇总与论文写法

### 8.1 建立结果模板

```powershell
& $python scripts\summarize_multiseed_results.py `
  --write-template experiments\output\multiseed_results.csv
```

用 Excel 打开 CSV。每个评估日志固定按 CHAMELEON、TE-CAMO、TE-COD10K、NC4K 顺序输出，把 `E_MEAN/F_MAX/SMeasure/MAE/WFM` 填入对应行。组名不可修改：

```text
baseline
naive_sam2
full_m4
reliable_replacement
```

### 8.2 计算均值和标准差

填满 4 组 x 3 seed x 4 数据集共 48 行后运行：

```powershell
& $python scripts\summarize_multiseed_results.py `
  --input experiments\output\multiseed_results.csv `
  --output experiments\output\multiseed_summary.csv
```

输出同时包含每个数据集和四数据集宏平均的均值、sample standard deviation。论文表格建议写成 `mean ± std`。

### 8.3 正确比较方式

- 完整方案与基线：按相同 seed 配对求差，再报告三组差值的均值和标准差。
- 可靠替换与朴素 SAM2：同样按相同 seed 配对。
- MAE 越低越好；其他四项越高越好。
- `n=3` 主要用于稳定性描述，统计功效有限；不要只凭三个 seed 宣称强统计显著。
- 报告全部预注册 seed、均值、标准差和失败运行；不得只挑最优 seed 或最优数据集。

推荐表述：完整方案在三个预注册 seed 上相对基线的宏平均变化为 `均值 ± 标准差`，并说明方向一致的 seed 数和数据集数。

禁止表述：结果完全一致、每次必然提升、删除缓存不产生任何数值变化、或单 seed 千分位差异具有统计显著性。

## 9. 完整性与复现核验

### 9.1 标签与配置

每次标签工件必须保留 `manifest.json`、`audit.jsonl`、输入/输出 hash 和源代码状态。训练前后不要改标签目录。每次训练保存的 `config.yaml` 应核对：seed、数据路径、batch、epoch、学习率、EMA、APM 和 Look-Twice。

### 9.2 GitHub 仓库与大文件

GitHub 仓库保存源码、配置、脚本、测试、Markdown/Word 手册和小型官方权重。DINOv2 大权重、数据集、缓存、artifact、checkpoint 和预测图应通过下载链接、GitHub Release 或外部存储提供，不要提交到普通 Git 历史。

### 9.3 复现含义

- 同一标签 hash + 同一配置 + 同一 seed + 同一环境：目标是尽量接近原结果，但 GPU 算子仍可能不是字节级确定。
- 同一协议 + 不同 seed：用于估计训练波动，不应要求每一位小数相同。
- 重新生成标签后 hash 不同：必须作为新实验版本报告，不能冒充历史 m4。
- 删除缓存本身不属于造假；隐瞒 seed、删掉不利 seed、用 GT 选配置或把新结果冒充旧 checkpoint 才会破坏实验诚信。

## 10. 常见错误与处理

### 10.1 `Expected 4040 original pkl labels`

粗伪标签没有下载到正确目录。检查 `index.json` 和 4040 个 `.pkl`，不要用空目录继续训练。

### 10.2 DINOv2 或 SAM2 下载失败

确认网络可访问 Hugging Face；也可提前下载对应官方权重。不要替换成不同模型规模或不同 checkpoint 后仍与原实验直接比较。

### 10.3 `Artifact already exists`

不可变工件保护生效。更换新实验 ID，不要覆盖已有目录。

### 10.4 `Experiment directory already exists`

该 `WorkDir` 已经运行过同一实验。更换新 `WorkDir`；不要删除已有结果后原地重跑。

### 10.5 `Checkpoint not found`

`epoch25.pth` 是目录式 checkpoint，实际权重是其中的 `model.safetensors`。检查 `WorkDir`、配置目录层级、实验 ID 和 `ckp` 层。

### 10.6 混合 batch 尺寸不一致

朴素 SAM2 的 4028 张 PNG 为 68x68，12 个 fallback `.pkl` 为 16x16。当前数据加载器会在混合目录存在时统一 fallback 尺寸。确认使用当前代码，不要修改那 12 个标签。

### 10.7 CUDA OOM

先用 `nvidia-smi` 关闭不相关任务。不要为了通过而擅自修改 batch、输入尺寸或模型；若必须改变，应作为新协议单独报告。

### 10.8 Accelerate 端口冲突

更换未占用的 `-Port` 即可。端口不是实验变量。

### 10.9 同一个 seed 仍有轻微差异

GPU、驱动、CUDA、cuDNN、PyTorch 与并行归约可能造成轻微差异。核对标签 hash 和保存的 `config.yaml`；不能把“设置 seed”写成“保证位级完全一致”。

## 11. 历史单 seed 结果

以下是清理前保存到文档的 seed 42 历史记录，不代表当前仓库仍含对应 checkpoint 或日志：

| 组别 | E_MEAN | F_MAX | SMeasure | MAE | WFM |
|---|---:|---:|---:|---:|---:|
| 基线 | 0.907975 | 0.808825 | 0.835325 | 0.044775 | 0.788125 |
| 朴素 SAM2 | 0.905950 | 0.817425 | 0.834625 | 0.045225 | 0.790725 |
| 完整方案 m4 | 0.908875 | 0.809800 | 0.836100 | 0.044675 | 0.789100 |
| 可靠替换控制 | 0.910000 | 0.811825 | 0.837200 | 0.044400 | 0.790775 |

可靠替换相对朴素 SAM2 的历史结论是“混合结果”：SMeasure 与 MAE 只在 2/4 数据集方向改善，F_MAX 下降。多 seed 完成前不能把它改写为稳定提升或统计显著。

历史 hash 仅供重新生成后的对照：

| 工件 | SHA256 |
|---|---|
| 全量 AEEM `m2_full4040_structure_20260724_v1` 输出清单 | `2e3a081f55d806b3530d00626c231c0e221dd33ac0a0515e308e8fc6e2473850` |
| 完整方案 m4 输出清单 | `073abc4dcd13eaa24eb050a7dc063a88dda1a5a644750eeefd8b7270cb92895e` |
| 可靠替换 `output_hashes.json` | `a2e1d827fd75ffc3c69c4207d403b7e4970ed0957f453d06ee701cb55730d116` |

## 12. 关键文件索引

| 文件 | 作用 |
|---|---|
| `scripts/train.py` | 训练入口，读取 `--seed`，默认 42 |
| `scripts/run_core_ablation_train.ps1` | 基线/朴素 SAM2 训练，支持 `-Seed` |
| `scripts/run_core_ablation_eval.ps1` | 基线/朴素 SAM2 四测试集评估 |
| `scripts/offline_sam2_refine.py` | 朴素 SAM2 标签生成，支持仓库相对数据路径 |
| `scripts/prepare_aeem_controls.py` | Hard/Soft-Coarse 控制标签 |
| `scripts/run_aeem_v2_full4040.ps1` | 全量 4040 张 AEEM v2 标签生成 |
| `scripts/prepare_aeem_v2_qsemantic25.ps1` | 完整方案 m4 标签组合 |
| `scripts/run_aeem_v2_train.ps1` | 完整方案/验证训练，支持 `-Seed` |
| `scripts/run_aeem_v2_eval.ps1` | 完整方案/验证四测试集评估 |
| `experiments/prepare_reliable_replacement_control.py` | 可靠替换组合标签 |
| `scripts/summarize_multiseed_results.py` | 三 seed 结果模板与均值/标准差汇总 |
| `docs/AEEM_CORE_ABLATION_PROTOCOL.md` | 核心消融协议 |
| `docs/AEEM_V2_QSEMANTIC25_FINAL_CONFIG.md` | 完整方案冻结配置记录 |
| `docs/INNOVATION2_ATTRIBUTION_PROTOCOL.md` | 可靠替换预注册与历史判定 |

最终提交应包含本手册、源码、配置、测试和依赖说明。大型数据、模型与生成结果若不放 GitHub，必须提供明确下载或重建方法。
