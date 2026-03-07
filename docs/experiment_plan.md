# MTN 实验计划（基于 `docs/requirement.md` 与 `README.md`）

## 1. 实验目标与交付物

### 1.1 目标

- 使用文本提示词完成一次 Text-to-3D 训练并导出 360 度视频。
- 记录训练期间 GPU 显存变化并分析资源开销。
- 识别至少 2 种失败模式（几何失真、部件缺失、纹理伪影等）。
- 进行至少 1 项超参数改动实验并比较结果差异。

### 1.2 最终交付物

- 训练命令与关键参数记录。
- 结果视频与（可选）网格模型导出结果记录。
- GPU 显存监控日志。
- 失败案例图文分析。
- 超参数对比分析。
- 可直接写入实验报告的结构化素材。

## 2. 实验前准备

### 2.1 路径与环境约定

- 项目根目录：`d:\Projects\UM\MTN`
- 建议 Conda 环境名：`MTN`
- 训练工作目录（workspace 示例）：`trial_baseline`

### 2.2 建议建立实验记录目录

在项目根目录执行：

```powershell
New-Item -ItemType Directory -Path .\docs\report\exp3 -Force
New-Item -ItemType Directory -Path .\docs\report\exp3\logs -Force
New-Item -ItemType Directory -Path .\docs\report\exp3\screenshots -Force
New-Item -ItemType Directory -Path .\docs\report\exp3\artifacts -Force
```

记录要求：

- `logs/`：保存环境信息、命令、终端输出、显存日志。
- `screenshots/`：保存训练过程与结果截图（失败案例必须有图）。
- `artifacts/`：保存导出视频/网格的索引说明。

## 3. 阶段一：环境检查与安装

### 3.1 CUDA 与基础环境检查

可复制执行命令：

```powershell
nvcc -V
nvidia-smi
```

记录内容：

- 按 README 要求确认 `cuda-toolkit` 已正确导出，并能直接执行 `nvcc -V`。
- GPU 型号、显存总量、驱动版本。
- 将输出复制到 `docs/report/exp3/logs/00_env_check.txt`。

结果记录模板：

- [ ] 已执行 `nvcc -V`
- [x] 已执行 `nvidia-smi`
- [x] 已保存输出到 `docs/report/exp3/logs/00_env_check.txt`
- 关键结果记录：
  - CUDA 版本：
  - Driver 版本：
  - GPU 型号：
  - 显存总量：

### 3.2 创建并激活环境（若已完成可跳过）

本节严格按 README 的环境配置整理。若只以 README 为准，则当前 plan 里之前补充的 `torchvision`、`ninja`、`psutil`、`torchmetrics`、`packaging`、`Pillow`、`gpustat` 都不属于 README 原始安装步骤。

- README 测试环境为：`python 3.9 & torch 1.13 & CUDA 11.5/11.7`。
- README 要求先确保 `nvcc -V` 可正常执行。
- README 安装顺序只包含：创建环境、安装 `pytorch==1.13.1`、安装 `gcc/gxx`、进入项目目录、安装 `requirements.txt`。

可复制执行命令（项目环境安装）：

```powershell
conda create --name MTN python=3.9 -y
conda activate MTN
pip install pytorch==1.13.1
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y
git clone https://github.com/Texaser/MTN.git
cd MTN
pip install -r requirements.txt --no-build-isolation
```

记录内容：

- 安装是否成功（成功/失败）。
- 如失败，记录报错关键词与解决方式。
- 保存到 `docs/report/exp3/logs/01_install_log.txt`。

结果记录模板：

- [ ] 已执行 `conda create --name MTN python=3.9 -y`
- [ ] 已执行 `conda activate MTN`
- [ ] 已执行 `pip install pytorch==1.13.1`
- [ ] 已执行 `conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y`
- [ ] 已执行 `git clone https://github.com/Texaser/MTN.git`
- [ ] 已执行 `cd MTN`
- [ ] 已执行 `pip install -r requirements.txt --no-build-isolation`
- [ ] 已保存安装日志到 `docs/report/exp3/logs/01_install_log.txt`
- 安装结论：
  - 是否成功（是/否）：
  - 若失败，报错关键词：
  - 解决方法：

说明：

- 若严格按 README，以上命令应原样记录。
- 若按当前 Windows 机器实际落地，`pip install pytorch==1.13.1` 与 `conda install gcc/gxx` 可能需要后续再按平台修正，但这已超出 README 原始写法。

Windows 补充依赖（非 README 原始步骤，建议在实验记录中单独标注为 Windows-specific）：

- [Windows-specific] Visual Studio 2022 C++ toolchain（确保 `cl.exe` 可用）。
- [Windows-specific] CUDA Toolkit 11.7（确保 `nvcc -V` 可用，并尽量与 README 的 CUDA 11.7 测试环境对齐）。
- [Windows-specific] 若 `pip install pytorch==1.13.1` 无法得到可用 CUDA 版本，改为安装与 CUDA 11.7 对应的 PyTorch 轮子。
- [Windows-specific] 若执行 `pip show gpustat`，则需额外安装 `gpustat`；它不在 README 原始安装步骤中，但在本实验计划后续显存监控步骤中会用到。

可复制执行命令（Windows 补充项，仅在 Windows 实际落地时使用）：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;" + $env:Path
where cl.exe
nvcc -V
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
pip install gpustat
```

Windows 补充记录模板：

- [ ] [Windows-specific] 已确认 `cl.exe` 可正常执行
- [ ] [Windows-specific] 已确认 `nvcc -V` 可正常执行
- [ ] [Windows-specific] 已执行 `pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117`
- [ ] [Windows-specific] 已执行 `pip install gpustat`

### 3.3 Python 与关键包版本确认

可复制执行命令：

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip show gpustat
```

记录内容：

- Python 版本、Torch 版本、CUDA 可用状态。
- `gpustat` 版本。
- 保存到 `docs/report/exp3/logs/02_versions.txt`。

结果记录模板：

- [ ] 已执行 `python --version`
- [ ] 已执行 `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
- [ ] 已执行 `pip show gpustat`
- [ ] 已保存版本信息到 `docs/report/exp3/logs/02_versions.txt`
- 关键结果记录：
  - Python 版本：
  - Torch 版本：
  - CUDA 可用（True/False）：
  - gpustat 版本：

## 4. 阶段二：基线实验（必做）

### 4.1 启动显存监控（训练前）

可复制执行命令：

```powershell
gpustat --color -i 5 | Tee-Object -FilePath .\docs\report\exp3\logs\10_gpustat_baseline.txt
```

记录内容：

- 该终端保持运行到训练结束。
- 监控间隔建议 5 秒。
- 若中断，记录中断时间与原因。

结果记录模板：

- [ ] 已在独立终端执行 `gpustat --color -i 5 | Tee-Object -FilePath .\docs\report\exp3\logs\10_gpustat_baseline.txt`
- [ ] 监控终端保持到训练结束
- [ ] 显存日志文件已生成
- 记录：
  - 开始时间：
  - 结束时间：
  - 是否中断（是/否）：
  - 中断原因（如有）：

### 4.2 基线训练命令

可复制执行命令：

```powershell
python main.py -O --text "a tiger dressed as a doctor" --workspace trial_baseline --iters 6000 --batch_size 1 --IF --perpneg
```

可选（显存不足时）：

```powershell
python main.py -O --text "a tiger dressed as a doctor" --workspace trial_baseline --iters 6000 --batch_size 1 --sd_version 2.1
```

记录内容：

- 完整命令行。
- 开始/结束时间、总时长。
- 是否出现 OOM 或其他错误。
- 关键日志片段（loss 变化、异常信息）。
- 保存到 `docs/report/exp3/logs/11_train_baseline.txt`。

结果记录模板：

- [ ] 已执行基线训练命令
- [ ] 如显存不足，已改用 SD2.1 命令
- [ ] 已保存训练日志到 `docs/report/exp3/logs/11_train_baseline.txt`
- 训练记录：
  - 实际执行命令：
  - 开始时间：
  - 结束时间：
  - 总时长：
  - 是否 OOM（是/否）：
  - 其他报错：
  - 关键 loss 片段：

### 4.3 导出视频与网格

可复制执行命令：

```powershell
python main.py --workspace trial_baseline -O --test
python main.py --workspace trial_baseline -O --test --save_mesh
```

记录内容：

- 生成文件路径与文件大小。
- 视频主观质量描述（几何完整度、纹理清晰度、多视角一致性）。
- 网格质量描述（孔洞、噪声面、纹理错位）。
- 保存到 `docs/report/exp3/logs/12_export_baseline.txt`。

结果记录模板：

- [ ] 已执行 `python main.py --workspace trial_baseline -O --test`
- [ ] 已执行 `python main.py --workspace trial_baseline -O --test --save_mesh`
- [ ] 已保存导出记录到 `docs/report/exp3/logs/12_export_baseline.txt`
- 文件记录：
  - 视频路径：
  - 视频大小：
  - 网格路径：
  - 网格大小：
- 质量评分（1-5）：
  - Geometry：
  - Texture：
  - Multi-view consistency：
- 简要结论：

## 5. 阶段三：失败模式分析（至少两个案例）

### 5.1 失败案例采集方法

可复制执行命令（失败案例示例）：

- 从基线结果中截取失败视角。
- 如基线质量较好，可改提示词重训一轮制造失败案例，例如：

```powershell
python main.py -O --text "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" --workspace trial_failure_case --iters 6000 --IF --batch_size 1 --perpneg
```

记录内容（每个案例都要记录）：

- 案例编号：`F1`、`F2`。
- 使用命令与参数。
- 现象描述：几何塌陷/部件缺失/纹理拉伸/多视角不一致。
- 可能原因：提示词歧义、学习率不合适、视角采样不足、扩散引导偏差。
- 截图路径（至少 2 张/案例）保存于 `docs/report/exp3/screenshots/`。

结果记录模板：

- F1：
  - Command：
  - Symptom：
  - Suspected cause：
  - Evidence images：
  - Improvement idea：
- F2：
  - Command：
  - Symptom：
  - Suspected cause：
  - Evidence images：
  - Improvement idea：

### 5.2 失败分析记录模板

建议在 `docs/report/exp3/logs/20_failure_analysis.md` 写入：

```markdown
## F1
- Command:
- Symptom:
- Suspected cause:
- Evidence images:
- Improvement idea:

## F2
- Command:
- Symptom:
- Suspected cause:
- Evidence images:
- Improvement idea:
```

## 6. 阶段四：超参数敏感性实验（至少 1 项改动）

### 6.1 推荐改动项

优先选择其中 1 项：

- 学习率（`--lr`）：例如从默认改为 `3e-4` 或 `5e-4`。
- 训练步数（`--iters`）：例如 6000 改 8000。
- 渲染分辨率相关参数（若代码支持）。

### 6.2 对比实验命令示例（学习率）

可复制执行命令：

```powershell
python main.py -O --text "a tiger dressed as a doctor" --workspace trial_lr3e4 --iters 6000 --IF --batch_size 1 --perpneg --lr 3e-4
python main.py -O --text "a tiger dressed as a doctor" --workspace trial_lr5e4 --iters 6000 --IF --batch_size 1 --perpneg --lr 5e-4
```

记录内容：

- 每次实验完整命令、训练时长、显存峰值。
- 输出质量评分（可用 1-5 分）：
  - 几何完整度
  - 纹理真实性
  - 多视角一致性
  - 收敛稳定性
- 保存到 `docs/report/exp3/logs/30_hparam_study.md`。

结果记录模板：

- [ ] 已执行 Exp-1 训练
- [ ] 已执行 Exp-2 训练
- [ ] 已保存日志到 `docs/report/exp3/logs/30_hparam_study.md`
- 对比命令记录：
  - Exp-1 Command：
  - Exp-2 Command：

### 6.3 对比表模板

建议记录为：

```markdown
| Workspace     | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
|---------------|-----------|------------|-----------|----------|---------|-------------|-----------|-------|
| trial_baseline| default   |            |           |          |         |             |           |       |
| trial_lr3e4   | lr=3e-4   |            |           |          |         |             |           |       |
| trial_lr5e4   | lr=5e-4   |            |           |          |         |             |           |       |
```

## 7. 阶段五：显存与效率总结

### 7.1 显存日志提炼

可复制执行命令：

从 `10_gpustat_baseline.txt` 与对比实验日志提取：

- 峰值显存。
- 平均显存（可粗略按关键时段估计）。
- 显存波动区间。

可用命令辅助筛选：

```powershell
Select-String -Path .\docs\report\exp3\logs\10_gpustat_baseline.txt -Pattern "MiB"
```

记录内容：

- 不同实验设置下的显存变化对比。
- 显存与质量、稳定性的关系判断。

结果记录模板：

- [ ] 已从 `10_gpustat_baseline.txt` 提取峰值显存
- [ ] 已估算平均显存
- [ ] 已记录显存波动区间
- [ ] 已完成不同参数设置的显存对比说明
- 结果记录：
  - Baseline Peak VRAM：
  - Exp-1 Peak VRAM：
  - Exp-2 Peak VRAM：
  - 结论：

## 8. 报告撰写映射（对应课程要求）

### 8.1 报告章节建议

- Introduction：实验背景与目标。
- Method：环境、命令、参数设置。
- Results：视频/网格结果与截图。
- Failure Analysis：至少两个失败案例。
- Hyper-parameter Study：参数改动与对比表。
- Discussion：改进建议（至少两条）。
- Conclusion：结论与反思。

### 8.2 改进建议示例（可写入报告）

- 采用分阶段训练与分辨率递进策略提升稳定性。
- 对困难提示词引入更强先验或多视角约束。
- 针对纹理伪影加入后处理或正则化约束。

## 9. 建议执行日程

### Day 1

- 完成环境检查、安装验证、基线训练启动。

### Day 2

- 完成基线导出、失败案例采集与记录。

### Day 3

- 完成超参数对比实验、显存总结与报告素材整理。

## 10. 最终自检清单

- [ ] 已记录完整基线训练命令与结果。
- [ ] 已保存显存监控日志。
- [ ] 已分析至少 2 个失败案例并附截图。
- [ ] 已完成至少 1 项超参数改动并进行对比。
- [ ] 已形成可直接用于报告的表格与结论素材。

