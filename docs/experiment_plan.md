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

- 项目根目录：`/home/ubuntu-user/temp/MTN`
- 建议 Conda 环境名：`MTN`
- 训练工作目录（workspace 示例）：`trial_baseline`

### 2.2 建议建立实验记录目录

在项目根目录执行：

```bash
mkdir -p ./logs ./docs/report/exp3/{screenshots,artifacts}
```

记录要求：

- `./logs/`：保存环境信息、训练日志、CSV 指标日志、显存日志。
- `screenshots/`：保存训练过程与结果截图（失败案例必须有图）。
- `artifacts/`：保存导出视频/网格的索引说明。

## 3. 阶段一：环境检查与安装

### 3.1 CUDA 与基础环境检查

可复制执行命令：

```bash
nvcc -V
nvidia-smi
```

记录内容：

- 本实验统一以本机环境 `CUDA 12.8 + torch 2.8.0` 为目标环境，先确认当前机器的 `cuda-toolkit` 已正确导出，并能直接执行 `nvcc -V`。
- GPU 型号、显存总量、驱动版本。

结果记录模板：

- [x] 已执行 `nvcc -V`
- [x] 已执行 `nvidia-smi`
- 关键结果记录：
  - CUDA 版本：
  - Driver 版本：
  - GPU 型号：
  - 显存总量：

### 3.2 创建并激活环境（若已完成可跳过）

本节仅保留 Ubuntu 环境安装说明，统一使用本机环境 `Python 3.9 + CUDA 12.8 + torch 2.8.0`。

- 目标环境：`python 3.9 & torch 2.8.0 & CUDA 12.8`
- 本机 CUDA 路径：`/usr/local/cuda-12.8`
- 说明：`torch 1.13.1+cu117` 与本机 CUDA 12.8 不匹配，需改为 `cu128` 版本。

可复制执行命令（项目环境安装）：

```bash
conda create --name MTN python=3.9 -y
conda activate MTN
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$CUDA_HOME/lib64:$PATH
nvcc -V
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y
python -m pip install ninja
pip install gpustat
grep -v '^torch\s*==' requirements.txt | pip install -r /dev/stdin --no-build-isolation
```

记录内容：

- 安装是否成功（成功/失败）。
- 如失败，记录报错关键词与解决方式。

结果记录模板：

- [x] 已执行 `conda create --name MTN python=3.9 -y`
- [x] 已执行 `conda activate MTN`
- [x] 已设置 `CUDA_HOME/CUDA_PATH` 指向本机 CUDA 路径（当前为 `/usr/local/cuda-12.8`）
- [x] 已执行 `nvcc -V` 并确认输出与本机 CUDA 路径一致
- [x] 已执行 `pip uninstall -y torch torchvision torchaudio`
- [x] 已执行 `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`
- [x] 已执行 `conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y`
- [x] 已执行 `python -m pip install ninja`
- [x] 已执行 `pip install gpustat`
- [x] 已执行 `grep -v '^torch\s*==' requirements.txt | pip install -r /dev/stdin --no-build-isolation`
- 安装结论：
  - 是否成功（是/否）：
  - 若失败，报错关键词：
  - 解决方法：

说明：

- 本实验统一使用 Ubuntu，本机 `nvcc` 路径为 `/usr/local/cuda-12.8/bin/nvcc`。
- PyTorch 安装命令使用 `cu128` 轮子。
- `requirements.txt` 已更新为 `torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0`。
- `gpustat` 已安装，用于后续显存监控与实验记录。

### 3.3 Python 与关键包版本确认

可复制执行命令：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip show gpustat
```

记录内容：

- Python 版本、Torch 版本、CUDA 可用状态。
- `gpustat` 版本。

结果记录模板：

- [x] 已执行 `python --version`
- [x] 已执行 `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
- [x] 已执行 `pip show gpustat`
- 关键结果记录：
  - Python 版本：
  - Torch 版本：
  - CUDA 可用（True/False）：
  - gpustat 版本：

### 3.4 Hugging Face 权限开通（使用 `--IF` 前必须完成）

若要运行带 `--IF` 的命令，必须先在浏览器中接受 DeepFloyd IF 的使用条款，并在当前 Conda 环境 `MTN` 中完成 Hugging Face 登录。否则会出现 `401 Unauthorized` 或 `GatedRepoError`。

#### 3.4.1 浏览器端接受模型条款

按顺序操作：

1. 打开模型页面：<https://huggingface.co/DeepFloyd/IF-I-XL-v1.0>
2. 登录你的 Hugging Face 账号。
3. 在模型页面点击类似 `Access repository`、`Agree and access` 或 `Accept` 的按钮。
4. 阅读并接受页面中的使用条款。
5. 确认页面已显示你具有该仓库访问权限。

补充：不要只登录而不点“接受条款”，否则命令行仍会报 `401`。

#### 3.4.2 在现有 `MTN` 环境中登录 Hugging Face

可复制执行命令：

```bash
conda activate MTN
python -m pip install --force-reinstall --no-deps "huggingface_hub==0.25.0"
python -m huggingface_hub.commands.huggingface_cli login
```

说明：

- 不要执行 `python -m pip install -U "huggingface_hub[cli]"`，否则会把 `huggingface_hub` 升到与当前项目不兼容的版本。
- Access Token 获取页面：<https://huggingface.co/settings/tokens>
- 使用 `read` 权限 token 即可。

#### 3.4.3 登录后验证权限

可复制执行命令：

```bash
conda activate MTN
python -m huggingface_hub.commands.huggingface_cli whoami
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="DeepFloyd/IF-I-XL-v1.0", filename="model_index.json")
print(path)
PY
```

验证标准：

- `python -m huggingface_hub.commands.huggingface_cli whoami` 能返回当前用户名。
- `hf_hub_download(...)` 能打印本地缓存路径。

若仍失败：重新确认条款已接受、账号一致，再重新登录。

结果记录模板：

- [x] 已打开 `https://huggingface.co/DeepFloyd/IF-I-XL-v1.0`
- [x] 已接受 DeepFloyd IF 使用条款
- [x] 已在 `MTN` 环境执行 `python -m huggingface_hub.commands.huggingface_cli login`
- [x] 已执行 `python -m huggingface_hub.commands.huggingface_cli whoami`
- [x] 已执行 `hf_hub_download` 验证下载权限
- 关键结果记录：
  - Hugging Face 用户名：
  - 是否可访问 `DeepFloyd/IF-I-XL-v1.0`（是/否）：
  - 若失败，报错关键词：

## 4. 阶段二：基线实验（必做）

### 4.1 启动显存监控（训练前）

可复制执行命令：

```bash
gpustat --color -i 5 | tee ./logs/10_gpustat_baseline.txt
```

记录内容：

- 该终端保持运行到训练结束。
- 监控间隔建议 5 秒。
- 若中断，记录中断时间与原因。

结果记录模板：

- [x] 已在独立终端执行 `gpustat --color -i 5 | tee ./logs/10_gpustat_baseline.txt`
- [ ] 监控终端完整保持到训练结束
- [x] 显存日志文件已生成
- 记录：
  - 开始时间：已执行
  - 结束时间：手动结束
  - 是否中断（是/否）：是
  - 中断原因（如有）：当前 `./logs/10_gpustat_baseline.txt` 不是本次 `trial_if_lowmem` 的完整监控日志；后续超参数实验前建议重新启动一次独立监控。

### 4.2 基线训练命令

先决条件：

- 若使用 `--IF`，必须先完成上面的 3.4，否则训练会因 Hugging Face gated repo 权限不足而直接失败。
- 若机器曾因高显存占用自动关机，先用下面的低显存 IF 命令，不要直接运行 `--IF --perpneg`。

#### 4.2.1 推荐低显存 IF 命令（优先使用）

可复制执行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lowmem --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48
```

说明：

- 保留 `--IF`，但去掉 `--perpneg`，显存压力会明显低于原命令。
- `--vram_O` 用空间换速度，可进一步降低显存峰值。
- `--w 48 --h 48` 会比默认 `64x64` 更省显存。
- 若该命令稳定，再尝试更高分辨率或加入 `--perpneg`。

自动日志保存：

- 文本日志：`./logs/trial_if_lowmem_log_df.txt`
- 结构化日志：`./logs/trial_if_lowmem_train_metrics_df.csv`
- TensorBoard：`./logs/tensorboard/trial_if_lowmem/df/`

#### 4.2.2 原始高开销 IF 命令（仅在显存稳定时使用）

建议先单独开启显存监控，再运行训练命令。

可复制执行命令（显存监控）：

```bash
gpustat --color -i 5 | tee ./logs/13_gpustat_if_perpneg.txt
```

说明：

- 该命令应在独立终端中先启动，并保持到训练结束。
- 建议把本轮高开销实验的 `gpustat` 日志单独保存，避免和前面的低显存实验混淆。

可复制执行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg
```

建议输出位置：

- 显存日志：`./logs/13_gpustat_if_perpneg.txt`
- 文本日志：`./logs/trial_if_perpneg_log_df.txt`
- 结构化日志：`./logs/trial_if_perpneg_train_metrics_df.csv`
- 结果目录：`./trial_if_perpneg/`

记录内容：

- `gpustat` 监控命令与日志路径。
- 完整命令行。
- 开始/结束时间、总时长。
- 是否出现 OOM 或其他错误。
- 关键日志片段（loss 变化、异常信息）。
- 自动保存的日志文件路径。

结果记录：

- [x] 已执行高开销 IF 命令
- [x] 已生成 `./logs/13_gpustat_if_perpneg.txt`
- [x] 已生成自动文本日志与 CSV 日志
- [x] 已确认本轮实验发生 OOM
- 实验记录：
  - 实际执行命令：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg`
  - 开始时间：`2026-03-07 22:10:34`
  - 结束时间：`2026-03-07 22:10:36` 左右
  - 是否 OOM（是/否）：是
  - 报错关键词：`torch.OutOfMemoryError: CUDA out of memory`
  - 触发位置：`guidance/if_utils.py` 中 `train_step_perpneg()` 调用 IF `unet` 前向时显存不足
  - 直接原因：`--IF --perpneg` 使用默认 `64x64` 且未开启 `--vram_O`，显存占用快速升至约 `22.86 GiB`
  - `gpustat` 观测：训练启动后显存一度达到约 `14013 MiB / 24564 MiB`，随后训练进程在第 1 个 epoch 的第 4 step 因 OOM 退出；退出后显存回落到系统空闲水平
  - 文本日志路径：`./logs/trial_if_perpneg_log_df.txt`
  - 结构化日志路径：`./logs/trial_if_perpneg_train_metrics_df.csv`
  - 显存监控日志：`./logs/13_gpustat_if_perpneg.txt`
  - 结论：当前机器不适合直接运行原始 `--IF --perpneg` 配置，应改用降显存版本。

#### 4.2.3 保留 IF + perpneg 的降显存实验（下一步建议）

若仍希望测试 `IF + perpneg`，建议先使用下面的降显存版本，而不要直接回到 4.2.2 的默认设置。

可复制执行命令（显存监控）：

```bash
gpustat --color -i 5 | tee ./logs/14_gpustat_if_perpneg_lowmem.txt
```

可复制执行命令（降显存训练）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg_lowmem --iters 6000 --batch_size 1 --IF --perpneg --vram_O --w 48 --h 48
```

说明：

- 保留 `--IF --perpneg`，但加入 `--vram_O --w 48 --h 48`，尽量降低显存峰值。
- 该命令仍然可能比 `trial_if_lowmem` 更吃显存，但比 4.2.2 更有机会跑通。
- 若仍 OOM，可进一步尝试更保守配置，例如 `--w 40 --h 40`。

建议输出位置：

- 显存日志：`./logs/14_gpustat_if_perpneg_lowmem.txt`
- 文本日志：`./logs/trial_if_perpneg_lowmem_log_df.txt`
- 结构化日志：`./logs/trial_if_perpneg_lowmem_train_metrics_df.csv`
- 结果目录：`./trial_if_perpneg_lowmem/`

结果记录：

- [x] 已执行降显存 `IF + perpneg` 训练
- [x] 已生成 `./logs/14_gpustat_if_perpneg_lowmem.txt`
- [x] 已生成自动文本日志与 CSV 日志
- [x] 已完成测试视频导出
- [x] 本轮未发生 OOM
- 训练记录：
  - 实际执行命令：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg_lowmem --iters 6000 --batch_size 1 --IF --perpneg --vram_O --w 48 --h 48`
  - 开始时间：`2026-03-07 22:15:39`
  - 结束时间：`2026-03-07 23:24:30` 左右（测试导出结束更晚）
  - 总时长：约 `103.9772` 分钟
  - 是否 OOM（是/否）：否
  - 其他报错：无致命报错；仅有 AMP 弃用警告与 `TORCH_CUDA_ARCH_LIST` 提示
  - 关键 loss 片段：`avg_loss` 基本在 `1.0000 ~ 1.0002`
  - 文本日志路径：`./logs/trial_if_perpneg_lowmem_log_df.txt`
  - 结构化日志路径：`./logs/trial_if_perpneg_lowmem_train_metrics_df.csv`
  - 显存监控日志：`./logs/14_gpustat_if_perpneg_lowmem.txt`
  - 结果视频路径：`./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4`
  - 深度视频路径：`./trial_if_perpneg_lowmem/results/df_ep0060_depth.mp4`
  - 法线视频路径：`./trial_if_perpneg_lowmem/results/df_ep0060_normal.mp4`
  - 显存结论：训练日志显示 GPU 常驻约 `14.0 ~ 15.0GB`，峰值约 `13.8GB`；明显高于不带 `perpneg` 的低显存 IF 基线，但已成功跑通
  - 结论：`--vram_O --w 48 --h 48` 足以让 `IF + perpneg` 在当前 RTX 4090 上完成训练，但代价是训练和测试时间显著增加。

### 4.3 导出视频与网格

可复制执行命令：

```bash
python main.py --workspace trial_if_lowmem -O --test
python main.py --workspace trial_if_lowmem -O --test --save_mesh
```

记录内容：

- 生成文件路径与文件大小。
- 视频主观质量描述（几何完整度、纹理清晰度、多视角一致性）。
- 网格质量描述（孔洞、噪声面、纹理错位）。
- 保存到 `./logs/12_export_baseline.txt`。

结果记录模板：

- [x] 已执行测试导出（训练结束后自动完成）
- [x] 已执行 `python main.py --workspace trial_if_lowmem -O --test --save_mesh`
- [ ] 已保存导出记录到 `./logs/12_export_baseline.txt`
- 文件记录：
  - 视频路径：`./trial_if_lowmem/results/df_ep0060_rgb.mp4`
  - 深度视频路径：`./trial_if_lowmem/results/df_ep0060_depth.mp4`
  - 法线视频路径：`./trial_if_lowmem/results/df_ep0060_normal.mp4`
  - 网格路径：`./trial_if_lowmem/mesh/mesh.obj`
  - 材质路径：`./trial_if_lowmem/mesh/mesh.mtl`
  - 贴图路径：`./trial_if_lowmem/mesh/albedo.png`
  - 网格规模：`15022` vertices, `30056` faces
- 质量评分（1-5）：
  - Geometry：
  - Texture：
  - Multi-view consistency：
- 简要结论：已完成视频与网格导出，可继续进行主观质量评估与失败案例截图。

视频截图要求与辅助脚本：

- 报告中建议加入成功结果截图、失败案例截图和超参数对比截图。
- 不需要把完整 `trial*/` 目录提交到仓库，但建议保留视频截图作为视觉证据。
- 可使用 `./scripts/extract_video_screenshots.py` 从导出视频中自动抽取多张代表帧，统一保存到 `./logs/video_screenshots/`。

可复制执行命令：

```bash
python ./scripts/extract_video_screenshots.py ./trial_if_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lowmem_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_lr3e4/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lr3e4_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_perpneg_lowmem_rgb
```

输出说明：

- `shot_*.png`：自动导出的截图文件。
- `_summary.txt`：记录帧号与时间戳，便于在报告中注明截图来源。

### 4.4 学习率对比实验（已完成）

本轮已完成 1 组超参数对比实验，在低显存 IF 配置基础上仅修改学习率为 `3e-4`。

实际执行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr3e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 3e-4
```

结果记录：

- [x] 已执行 `trial_if_lr3e4` 训练
- [x] 已自动生成文本日志
- [x] 已自动生成 CSV 指标日志
- [x] 已生成测试结果视频
- [x] 已生成 `./logs/11_gpustat_lr3e4.txt`
- 训练记录：
  - 开始时间：`2026-03-07 19:48:47`
  - 结束时间：`2026-03-07 20:06:06` 左右
  - 总时长：约 `17.4350` 分钟
  - 是否 OOM（是/否）：否
  - 其他报错：无致命报错；仅有 AMP 弃用警告与 `TORCH_CUDA_ARCH_LIST` 提示
  - 关键 loss 片段：`avg_loss` 基本稳定在 `1.0000 ~ 1.0001`
  - 文本日志路径：`./logs/trial_if_lr3e4_log_df.txt`
  - 结构化日志路径：`./logs/trial_if_lr3e4_train_metrics_df.csv`
  - 显存监控日志：`./logs/11_gpustat_lr3e4.txt`
  - 结果视频路径：`./trial_if_lr3e4/results/df_ep0060_rgb.mp4`
  - 深度视频路径：`./trial_if_lr3e4/results/df_ep0060_depth.mp4`
  - 法线视频路径：`./trial_if_lr3e4/results/df_ep0060_normal.mp4`
  - 显存结论：训练日志记录 GPU 约 `11.0GB`，峰值约 `12.0GB`；`gpustat` 训练时段最高约 `13.3GB / 24.6GB`

与 `trial_if_lowmem` 的初步对比：

- 两组实验都未出现 OOM。
- `trial_if_lr3e4` 总时长约 `17.44` 分钟，和 `trial_if_lowmem` 的 `17.77` 分钟接近。
- 两组实验显存占用接近，说明本轮仅修改学习率，没有明显增加显存压力。
- 当前仍需人工查看视频主观质量，再判断 `--lr 3e-4` 是否优于基线。

### 4.5 下一步

1. 查看 `./trial_if_lowmem/results/df_ep0060_rgb.mp4` 与 `./trial_if_lr3e4/results/df_ep0060_rgb.mp4`，做主观质量对比。
2. 若 `trial_if_lr3e4` 质量更好，可继续执行 `python main.py --workspace trial_if_lr3e4 -O --test --save_mesh` 导出网格。
3. 补写 `./logs/30_hparam_study.md`，把两组实验整理成对比表。
4. 使用 `python ./scripts/extract_video_screenshots.py ...` 从结果视频中抽取代表帧，统一保存到 `./logs/video_screenshots/`，再挑选插入报告。

## 5. 阶段三：失败模式分析（至少两个案例）

### 5.1 失败案例采集方法

可复制执行命令（失败案例示例）：

- 从基线结果中截取失败视角。
- 如基线质量较好，可改提示词重训一轮制造失败案例，例如：

```bash
python main.py -O --text "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" --workspace trial_failure_case --iters 6000 --IF --batch_size 1 --perpneg
```

记录内容（每个案例都要记录）：

- 案例编号：`F1`、`F2`。
- 使用命令与参数。
- 现象描述：几何塌陷/部件缺失/纹理拉伸/多视角不一致。
- 可能原因：提示词歧义、学习率不合适、视角采样不足、扩散引导偏差。
- 截图路径（至少 2 张/案例）保存于 `docs/report/exp3/screenshots/`。

可复制执行命令（为失败分析准备截图）：

```bash
python ./scripts/extract_video_screenshots.py ./trial_if_lowmem/results/df_ep0060_rgb.mp4 --count 6 --skip-first-last --output-dir ./logs/video_screenshots/failure_trial_if_lowmem
python ./scripts/extract_video_screenshots.py ./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4 --count 6 --skip-first-last --output-dir ./logs/video_screenshots/failure_trial_if_perpneg_lowmem
```

说明：

- 先在 `./logs/video_screenshots/` 中筛选有缺陷的视角。
- 再把最终要放进报告的 PNG 复制到 `./docs/report/exp3/screenshots/`。

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

建议在 `./logs/20_failure_analysis.md` 写入：

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

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr3e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 3e-4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr5e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 5e-4
```

记录内容：

- 每次实验完整命令、训练时长、显存峰值。
- 输出质量评分（可用 1-5 分）：
  - 几何完整度
  - 纹理真实性
  - 多视角一致性
  - 收敛稳定性
- 保存到 `./logs/30_hparam_study.md`。

结果记录模板：

- [x] 已执行 Exp-1 训练
- [ ] 已执行 Exp-2 训练
- [ ] 已保存日志到 `./logs/30_hparam_study.md`
- 对比命令记录：
  - Exp-1 Command：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr3e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 3e-4`
  - Exp-2 Command：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr5e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 5e-4`

### 6.3 对比表模板

建议记录为：

```markdown
| Workspace        | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
|------------------|-----------|------------|-----------|----------|---------|-------------|-----------|-------|
| trial_if_lowmem  | default   | 17.77 min  | 12.0 GB   |          |         |             | stable    | low-memory IF baseline |
| trial_if_lr3e4   | lr=3e-4   | 17.44 min  | 12.0 GB   |          |         |             | stable    | result videos exported |
| trial_if_lr5e4   | lr=5e-4   |            |           |          |         |             |           | pending |
```

## 7. 阶段五：显存与效率总结

### 7.1 显存日志提炼

可复制执行命令：

从 `./logs/10_gpustat_baseline.txt` 与对比实验日志提取：

- 峰值显存。
- 平均显存（可粗略按关键时段估计）。
- 显存波动区间。

可用命令辅助筛选：

```bash
grep "MiB" ./logs/10_gpustat_baseline.txt
```

记录内容：

- 不同实验设置下的显存变化对比。
- 显存与质量、稳定性的关系判断。

结果记录模板：

- [x] 已从训练日志与 `gpustat` 日志提取本轮峰值显存
- [ ] 已估算平均显存
- [ ] 已记录显存波动区间
- [x] 已完成一版不同参数设置的显存对比说明
- 结果记录：
  - Baseline Peak VRAM：`12.0GB`（`trial_if_lowmem` 训练日志）
  - Exp-1 Peak VRAM：`12.0GB`（`trial_if_lr3e4` 训练日志；`gpustat` 训练时段最高约 `13.3GB / 24.6GB`）
  - Exp-2 Peak VRAM：待补充
  - 结论：当前低显存 IF 基线与 `lr=3e-4` 对比实验都能稳定控制在 RTX 4090 可承受范围内；单独提高学习率到 `3e-4` 没有带来明显额外显存压力。

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

- [x] 已记录完整基线训练命令与结果。
- [x] 已保存显存监控日志。
- [ ] 已分析至少 2 个失败案例并附截图。
- [x] 已完成至少 1 项超参数改动并进行对比。
- [ ] 已形成可直接用于报告的表格与结论素材。

