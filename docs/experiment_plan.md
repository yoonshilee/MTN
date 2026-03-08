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
- [x] 监控终端完整保持到训练结束
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

#### 4.2.3 保留 IF + perpneg 的降显存实验

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

#### 4.2.4 新增实验：IF tiger prompt + perpneg 基线（6000 iter / 约 60 epoch）

计划目的：

- 将提示词改为更接近自然图像表述的 `a DSLR photo of a tiger dressed as a doctor`，观察在 `IF + perpneg` 条件下是否比现有结果更稳定。
- 保留 `6000 iter`，便于和前面已完成实验直接对比；按当前 `dataset_size_train = 100` 估算，`6000 iter` 约对应 `60 epoch`。

用户提出的命令：

```bash
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --num_steps 32 --upsample_steps 16
```

合理性分析：

- `--IF --perpneg --negative_w -3.0` 的组合是合理的，`-3.0` 仍位于代码注释建议的 `0 ~ -4` 范围内，比默认 `-2` 更强地抑制 Janus 问题，但也更容易带来平面化、噪声或纹理退化。
- `--vram_O` 是合理的。前文已经证明 `IF + perpneg` 属于高显存配置，保留低显存优化是必要的。
- `--batch_size 1` 也是合理的，符合当前显存约束。
- 主要问题在于：`-O` 会自动启用 `--cuda_ray`，而 `--num_steps` 和 `--upsample_steps` 仅在不使用 `--cuda_ray` 时才生效，因此这两个参数在该命令下基本不会起作用。
- 另一个风险在于该命令没有显式设置 `--w 48 --h 48`，会回到默认训练分辨率 `64x64`。即使开启了 `--vram_O`，相比已经跑通的 `48x48` 配置，仍会明显提高显存压力和失败风险。

建议采用的修正版命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48
```

说明：

- 若坚持保留 `-O`，建议删除 `--num_steps 32 --upsample_steps 16`，因为它们不会带来预期效果。
- 若你的目标是专门测试 `--num_steps` / `--upsample_steps` 对结果的影响，则不应使用 `-O`，而应单独设计一组不用 `--cuda_ray` 的实验。
- 该实验建议单独开启显存监控，并使用独立 workspace，避免与现有 `trial_if_perpneg_lowmem` 混淆。

建议输出位置：

- 显存日志：`./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt`
- 文本日志：`./logs/trial_perpneg_if_tiger_baseline_6000_log_df.txt`
- 结构化日志：`./logs/trial_perpneg_if_tiger_baseline_6000_train_metrics_df.csv`
- 结果目录：`./trial_perpneg_if_tiger_baseline_6000/`

建议执行顺序：

```bash
gpustat --color -i 5 | tee ./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48
```

记录内容：

- 是否成功完成 6000 iter / 约 60 epoch。
- 是否出现 OOM、输出噪声化或几何过度扁平化。
- 与 `trial_if_perpneg_lowmem` 相比，提示词改为 `DSLR photo` 后，几何质量、纹理质量、多视角一致性是否改善。

结果记录：

- [x] 已执行 `trial_perpneg_if_tiger_baseline_6000` 训练
- [x] 已自动生成文本日志与 CSV 日志
- [x] 已完成测试视频导出
- [ ] 已生成 `./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt`
- [x] 本轮未发生 OOM
- 训练记录：
  - 实际执行命令：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48`
  - 开始时间：`2026-03-08 14:37:13`
  - 结束时间：`2026-03-08 15:45` 左右（测试导出完成）
  - 总时长：约 `69.6215` 分钟
  - 是否完成 6000 iter / 60 epoch（是/否）：是
  - 是否 OOM（是/否）：否
  - 其他报错：无致命报错；仅有 AMP 弃用警告、`TORCH_CUDA_ARCH_LIST` 提示，以及 `huggingface_hub` 的 `resume_download` FutureWarning
  - 关键 loss 片段：`avg_loss` 基本稳定在 `1.0000 ~ 1.0001`
  - 文本日志路径：`./logs/trial_perpneg_if_tiger_baseline_6000_log_df.txt`
  - 结构化日志路径：`./logs/trial_perpneg_if_tiger_baseline_6000_train_metrics_df.csv`
  - TensorBoard 路径：`./logs/tensorboard/trial_perpneg_if_tiger_baseline_6000/df/`
  - 结果视频路径：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
  - 深度视频路径：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_depth.mp4`
  - 法线视频路径：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_normal.mp4`
  - 显存结论：训练日志显示 GPU 常驻约 `14.4 ~ 14.5GB`；按训练日志中的 `Peak_GPU` 字段统计，最高约 `13.3GB`，整体与 `trial_if_perpneg_lowmem` 同属高显存但可运行配置
  - 稳定性结论：本轮在 `--negative_w -3.0` 与 `DSLR photo` 提示词设定下完整跑通，无 OOM，中后期各 epoch loss 基本无明显波动
  - 质量结论：已完成人工查看结果视频，生成结果整体接近一个球体，未能形成可辨认的“老虎医生”三维结构，可判定为本轮训练失败；该结果表现为严重几何塌缩/过度平滑，主体语义未正确成形
  - 主观判断：相较 `trial_if_perpneg_lowmem`，本轮 `DSLR photo + negative_w=-3.0` 设定没有带来可用质量提升，反而仍然落入明显失败模式，不适合作为最终展示结果
  - 备注：本轮未附带独立 `gpustat` 输出，因此显存分析当前以训练日志为准；若后续要写入最终显存对比表，建议补充一次独立显存监控实验或在报告中注明数据来源差异

#### 4.2.5 新增实验：IF tiger 正常提示词基线（去掉 `perpneg`）

计划目的：

- 使用更接近常规文本到三维基线的设置，重新测试提示词 `a DSLR photo of a tiger dressed as a doctor`。
- 与上一轮失败的 `IF + perpneg + negative_w=-3.0` 实验形成直接对照，验证“球体化失败”是否主要由 `perpneg` 强约束引起。
- 保留 `IF`、`6000 iter`、`seed=3407` 与 `64x64` 分辨率，便于后续将该组结果纳入最终报告中的同主题对比。

用户拟采用的新命令：

```bash
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace exp017_if_tiger --iters 6000 --IF --batch_size 1 --h 64 --w 64 --seed 3407 --vram_O --eval_interval 10 --test_interval 100 --dataset_size_test 100
```

实验设计说明：

- 本轮实验的核心变化是：去掉 `--perpneg` 与 `--negative_w -3.0`，改为更标准的 IF 文本提示训练配置。
- 本轮显式指定 `--h 64 --w 64`，即使用 `64x64` 训练分辨率；这会比此前跑通的 `48x48` 低显存配置更吃显存。
- 本轮仍保留 `-O` 与 `--vram_O`，因此仍属于当前机器可尝试的中高显存配置，但理论上应明显比 `IF + perpneg` 更稳定。
- 本轮命令改为与本计划其他实验一致的相对写法，默认在项目根目录下执行即可。

建议执行前补充显存监控：

```bash
gpustat --color -i 5 | tee ./logs/17_gpustat_exp017_if_tiger.txt
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace exp017_if_tiger --iters 6000 --IF --batch_size 1 --h 64 --w 64 --seed 3407 --vram_O --eval_interval 10 --test_interval 100 --dataset_size_test 100
```

建议输出位置：

- 显存日志：`./logs/17_gpustat_exp017_if_tiger.txt`
- 结果目录：`./exp017_if_tiger/`
- 文本日志：通常由程序自动写入对应 central logs 目录，建议训练后检查 `exp017_if_tiger` 对应的文本日志与 CSV 日志是否已生成

重点观察项：

- 是否出现 OOM。
- 是否摆脱上一轮“生成球体”的失败模式。
- 几何主体是否开始形成清晰的“虎头/躯干/医生服饰”轮廓。
- 在 `64x64` 分辨率下，纹理是否优于此前 `48x48` 低显存实验。
- 训练时长与峰值显存是否明显高于 `trial_if_lowmem` 与 `trial_if_lr3e4`。

与上一轮实验命令的差异说明：

- 上一轮命令为：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48`
- 本轮新命令移除了 `--perpneg`，因此不会再启用 perpendicular negative guidance。
- 本轮新命令移除了 `--negative_w -3.0`，因此不再对负向约束施加强抑制。
- 本轮新命令把训练分辨率明确设为 `--h 64 --w 64`，高于上一轮的 `48x48`，因此显存压力通常会更高。
- 本轮新命令显式固定 `--seed 3407`，有利于后续复现实验。
- 本轮新命令新增 `--eval_interval 10`、`--test_interval 100`、`--dataset_size_test 100`，使评估与测试频率设置更明确，但它们不是决定成败的主要变化。
- 本轮新命令改回相对路径写法，便于和本计划中的其他实验命令保持一致。

预期结论：

- 如果本轮结果明显好于上一轮“球体化失败”结果，则可以初步判断：`perpneg + negative_w=-3.0` 是导致该主题生成失败的重要因素之一。
- 如果本轮虽然更稳定，但仍然失败，则更可能说明：问题不仅来自 `perpneg`，还与 `IF` 对该提示词的三维几何先验不足有关。
- 若本轮能够生成可辨认主体，则这组实验将成为“同一提示词下去掉 `perpneg` 后质量改善”的关键证据，可直接写入最终报告的对比分析。

结果记录：

- [x] 已执行 `exp017_if_tiger` 训练
- [x] 已生成 `./logs/17_gpustat_exp017_if_tiger.txt`
- [x] 已自动生成文本日志与 CSV 日志
- [x] 已完成测试视频导出
- 训练记录：
  - 实际执行命令：`python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace exp017_if_tiger --iters 6000 --IF --batch_size 1 --h 64 --w 64 --seed 3407 --vram_O --eval_interval 10 --test_interval 100 --dataset_size_test 100`
  - 开始时间：`2026-03-08 16:14:44`
  - 结束时间：`2026-03-08 16:36:20` 左右（测试导出完成）
  - 总时长：约 `21.7783` 分钟
  - 是否完成 6000 iter / 60 epoch（是/否）：是
  - 是否 OOM（是/否）：否
  - 其他报错：无致命报错；主要为 AMP 弃用警告、`TORCH_CUDA_ARCH_LIST` 提示，以及 `huggingface_hub` 的 `resume_download` FutureWarning
  - 关键 loss 片段：`avg_loss` 基本稳定在 `1.0000 ~ 1.0002`，后期个别 epoch 到 `1.0003`
  - 文本日志路径：`./logs/exp017_if_tiger_log_df.txt`
  - 结构化日志路径：`./logs/exp017_if_tiger_train_metrics_df.csv`
  - TensorBoard 路径：`./logs/tensorboard/exp017_if_tiger/df/`
  - 显存监控日志：`./logs/17_gpustat_exp017_if_tiger.txt`
  - 结果视频路径：`./exp017_if_tiger/results/df_ep0060_rgb.mp4`
  - 深度视频路径：`./exp017_if_tiger/results/df_ep0060_depth.mp4`
  - 法线视频路径：`./exp017_if_tiger/results/df_ep0060_normal.mp4`
  - 显存结论：训练日志显示 GPU 常驻约 `17.4 ~ 17.6GB`；`Peak_GPU` 在首个 epoch 约 `13.0GB`，后续大多在 `9.0 ~ 9.4GB`；整体显存占用明显高于 `48x48` 的低显存 IF 基线
  - 速度结论：在去掉 `perpneg` 后，总训练+测试时长约 `21.78` 分钟，显著短于上一轮 `trial_perpneg_if_tiger_baseline_6000` 的约 `69.62` 分钟
  - 质量结论：本轮已成功产出视频与验证图，但当前文档尚未补入人工主观观察；仍需查看 `df_ep0060_rgb.mp4` 后再判断几何是否成形、是否优于上一轮球体化失败结果
  - 与 `trial_perpneg_if_tiger_baseline_6000` 的对比结论：本轮在移除 `perpneg` 与 `negative_w=-3.0` 后训练稳定性明显更好、耗时显著下降，且未出现训练侧失败；后续若视频主观质量也优于上一轮，则可进一步支持“强负向约束是上一轮失败重要原因之一”的判断

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
  - Geometry：2
  - Texture：2
  - Multi-view consistency：3
- 评分说明：
  - Geometry：关注三维形状是否合理，重点看整体轮廓、结构完整性，以及是否存在塌陷、缺失、穿插、局部变形等几何问题。
  - Texture：关注表面颜色与纹理质量，重点看纹理是否清晰、是否符合提示词，以及是否存在模糊、拉伸、错位、噪声伪影等问题。
  - Multi-view consistency：关注不同视角下结果是否一致，重点看物体在旋转过程中是否出现结构跳变、纹理突变、身份变化、闪烁或忽隐忽现等现象。
- 评分参考：
  - 1 分：质量很差，缺陷非常明显，难以作为有效结果展示。
  - 2 分：质量较差，能看出目标但存在较多明显问题。
  - 3 分：质量一般，主体基本可辨认，但仍有较明显瑕疵。
  - 4 分：质量较好，整体结果可信，只有少量局部问题。
  - 5 分：质量很好，结果稳定自然，缺陷很少。
- 简要结论：已完成视频与网格导出与主观质量评分。`trial_if_lowmem` 主体可辨认，但几何结构和纹理质量偏弱，多视角一致性一般，可作为低显存基线保留。

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
- 当前主观质量评分记录为：`Geometry = 4`、`Texture = 3`、`Multi-view consistency = 3`。
- 主观结论：`trial_if_lr3e4` 的几何质量明显优于 `trial_if_lowmem`，纹理略有提升，但多视角一致性没有明显改善；在不增加显存压力的前提下，是当前更适合作为最终展示候选的结果。

### 4.5 下一步

当前最优先只做 4 件事：

1. 将三组主观质量评分整理到最终报告表格与正文描述中。
2. 从 `./logs/video_screenshots/` 中挑选成功图、失败图、对比图，复制到 `./docs/report/exp3/screenshots/`。
3. 在 `./logs/20_failure_analysis.md` 中补齐 `F1`、`F2`。
4. 在 `./logs/30_hparam_study.md` 中整理三组实验对比结论。

若你认为某一组结果最好，再额外执行该组的 mesh 导出：

```bash
python main.py --workspace <best_workspace> -O --test --save_mesh
```

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

### 6.2 本次实际对比的 3 组实验

本次不再新增命令，直接比较已经完成的 3 组结果：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lowmem --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr3e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 3e-4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg_lowmem --iters 6000 --batch_size 1 --IF --perpneg --vram_O --w 48 --h 48
```

需要补写的只有：

- 三组的优缺点总结。
- 哪一组最适合当最终展示结果。

### 6.3 对比表模板

建议记录为：

| Workspace | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
| ------------------ | ----------- | ------------ | ----------- | ---------- | --------- | ------------- | ----------- | ------- |
| trial_if_lowmem | default | 17.77 min | 12.0 GB | 2 | 2 | 3 | stable | low-memory IF baseline; recognizable result but weak geometry and texture |
| trial_if_lr3e4 | lr=3e-4 | 17.44 min | 12.0 GB | 4 | 3 | 3 | stable | best geometry among current runs; texture improved slightly |
| trial_if_perpneg_lowmem | perpneg + vram_O | 103.98 min | 13.8 GB | 1 | 1 | 1 | stable but slow | output is mostly noise; not suitable as final result |

主观质量补充记录：

- `trial_if_lowmem`：`Geometry = 2`、`Texture = 2`、`Multi-view consistency = 3`。结果主体可辨认，但几何结构较弱、表面纹理较粗糙。
- `trial_if_lr3e4`：`Geometry = 4`、`Texture = 3`、`Multi-view consistency = 3`。几何完整度最好，纹理较基线有改善，但多视角稳定性仍然一般。
- `trial_if_perpneg_lowmem`：结果基本表现为噪声，虽然训练流程跑通，但不适合作为有效展示结果；在对比表中按最低分记录以反映其实用价值较低。

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
  - Exp-2 Peak VRAM：`13.8GB`（`trial_if_perpneg_lowmem` 训练日志）
  - 结论：`trial_if_lowmem` 与 `trial_if_lr3e4` 显存和时长接近；`trial_if_perpneg_lowmem` 能跑通，但显存更高、时间显著更长。

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
- [ ] 已形成可直接用于报告的表格、截图与结论素材。

