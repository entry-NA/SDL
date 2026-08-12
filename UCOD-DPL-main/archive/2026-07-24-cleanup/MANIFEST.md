# 2026-07-24 可恢复清理清单

本目录保存从项目根目录移出的临时诊断产物。此次操作只移动文件，没有删除文件，也没有修改模型代码、数据集、伪标签、权重或 checkpoint。

## 归档原因

- 根目录 `_*.py`：一次性诊断、校准、扫描或修复脚本，正式代码与文档未引用。
- 根目录 `_*.txt`：上述诊断脚本产生的 stdout、stderr 和结果记录。
- 根目录 `debug_*.png`：粗标签与 SAM2 输出的临时可视化。
- `Users23991AppDataLocalTempsam2_check/`：误落在项目根目录的临时检查目录。

## 数量

| 类型 | 数量 | 大小 |
|---|---:|---:|
| 根目录诊断文件 | 39 | 约 210 KB |
| 临时目录内文件 | 20 | 3,296,025 bytes |

## 路径映射

所有根目录诊断文件：

```text
原路径：<project-root>/<filename>
归档：  <project-root>/archive/2026-07-24-cleanup/root-diagnostics/<filename>
```

临时目录：

```text
原路径：<project-root>/Users23991AppDataLocalTempsam2_check/
归档：  <project-root>/archive/2026-07-24-cleanup/temp-directory/Users23991AppDataLocalTempsam2_check/
```

## 根目录诊断文件

```text
_area_func.py
_calibrate.py
_calibrate_result.txt
_check_labels.py
_check_out.txt
_check_quality.py
_check_resize.py
_d1_fixed.py
_d1_result.txt
_d2_maskselect.py
_d2_result.txt
_d3_fixed.py
_d3_result.txt
_dg_stderr.txt
_dg_stdout.txt
_diagnose.py
_f_stderr.txt
_f_stdout.txt
_fix_syntax.py
_gamma_sweep.py
_gamma_sweep2.py
_gamma_sweep2_result.txt
_gamma_sweep_result.txt
_p0_err.txt
_p0_out.txt
_p1_stderr.txt
_p1_stdout.txt
_q_err.txt
_q_out.txt
_serr.txt
_sok.txt
_t_stderr.txt
_t_stdout.txt
debug_coarse_0.png
debug_coarse_1500.png
debug_coarse_500.png
debug_sam2_0.png
debug_sam2_1500.png
debug_sam2_500.png
```

## 恢复方法

在项目根目录运行以下 PowerShell 命令。恢复前应确认根目录不存在同名文件，避免覆盖后来生成的新文件。

```powershell
$archive = 'archive\2026-07-24-cleanup'
Get-ChildItem "$archive\root-diagnostics" -File | ForEach-Object {
    if (Test-Path -LiteralPath $_.Name) {
        throw "目标已存在，停止恢复：$($_.Name)"
    }
    Move-Item -LiteralPath $_.FullName -Destination '.'
}

$tempSource = "$archive\temp-directory\Users23991AppDataLocalTempsam2_check"
if (Test-Path -LiteralPath 'Users23991AppDataLocalTempsam2_check') {
    throw '目标临时目录已存在，停止恢复。'
}
Move-Item -LiteralPath $tempSource -Destination '.'
```

## 保留未动的内容

- `datasets/` 及其 cache、粗标签、精修标签和 raw SAM2 输出。
- `weights/`、`work_dir/`、checkpoint 和预测产物。
- `scripts/`、`data/`、`models/`、`engine/`、`configs/`。
- `experiments/` 中正式实验脚本、配置、报告和输出。
- `EXPERIMENT_LOG.md`、`docs/HANDOFF.md` 和研究设计文档。

标签目录的数量、大小和 SHA-256 基线另见 `LABEL_BASELINE.csv` 与 `BASELINE_SUMMARY.md`。
