# MTN 实验计划与当前进展

本文档基于 `docs/requirement.md`、`README.md` 和当前工作区已有产出整理，保留重要实验命令、关键参数和结果，删除重复模板与过长说明，便于直接转写实验报告。

## 1. 课程要求与当前结论

课程要求包括：

- 完成至少 1 次 Text-to-3D 训练并导出结果视频。
- 记录训练期间 GPU 显存变化。
- 分析至少 2 种失败模式。
- 至少完成 1 项超参数改动实验并比较效果。
- 在报告中给出实验观察、失败分析、改进建议。

当前判断：

- 已完成：训练、视频导出、显存日志记录、至少 1 组超参数实验、至少 2 个可分析的失败案例来源。
- 尚未补齐：失败分析文档、超参数对比文档、报告截图归档、导出结果摘要文件。
- 是否还需要更多实验：**不需要新增“必做”实验**。现有实验已经足够满足课程要求并支撑报告。若还想增强说服力，只建议补 1 个定向小实验，而不是再开新的大规模训练。

## 2. 还需要补充的产出

根据当前工作区检查，下面这些内容仍需补齐：

1. `./logs/12_export_baseline.txt`
   - 当前缺失。
   - 应补：基线导出文件路径、网格规模、主观质量结论。

2. `./logs/20_failure_analysis.md`
   - 当前缺失。
   - 应补：至少 2 个失败案例，包含命令、现象、可能原因、截图路径、改进建议。

3. `./logs/30_hparam_study.md`
   - 当前缺失。
   - 应补：`trial_if_lowmem`、`trial_if_lr3e4`、`trial_if_perpneg_lowmem` 三组对比结论，以及最终推荐结果。

4. `./docs/report/exp3/screenshots/`
   - 当前目录不存在。
   - 应补：成功案例图、失败案例图、对比图。

5. `./docs/report/exp3/artifacts/`
   - 当前目录不存在。
   - 应补：视频/网格索引说明，便于报告引用。

6. 最终报告中的 2 条改进建议
   - 计划中给了方向，但还没有整理成最终可提交表述。

补充说明：

- `./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt` 实际已经存在，原文档中该项未勾选是过时状态。
- `./logs/video_screenshots/` 下已经有 3 组截图：`trial_if_lowmem_rgb`、`trial_if_lr3e4_rgb`、`trial_if_perpneg_lowmem_rgb`，但还没有复制到最终报告目录。

## 3. 环境与前置条件

### 3.1 环境结论

- 目标环境：`Python 3.9 + CUDA 12.8 + torch 2.8.0`
- 当前实验统一基于 IF 路线开展。
- 使用 `--IF` 前必须先在 Hugging Face 接受 `DeepFloyd/IF-I-XL-v1.0` 使用条款并完成登录。

### 3.2 关键环境命令

```bash
nvcc -V
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip show gpustat
```

### 3.3 安装命令（保留最终有效版本）

```bash
conda create --name MTN python=3.9 -y
conda activate MTN
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$CUDA_HOME/lib64:$PATH
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y
python -m pip install ninja
pip install gpustat
grep -v '^torch\s*==' requirements.txt | pip install -r /dev/stdin --no-build-isolation
```

### 3.4 Hugging Face 登录与权限验证

```bash
conda activate MTN
python -m pip install --force-reinstall --no-deps "huggingface_hub==0.25.0"
python -m huggingface_hub.commands.huggingface_cli login
python -m huggingface_hub.commands.huggingface_cli whoami
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="DeepFloyd/IF-I-XL-v1.0", filename="model_index.json")
print(path)
PY
```

## 4. 已完成实验与核心结果

### 4.1 实验总览

| Workspace | 关键参数 | 时长 | 显存/稳定性 | 结果判断 |
| --- | --- | --- | --- | --- |
| `trial_if_lowmem` | `--IF --vram_O --w 48 --h 48` | 约 17.77 min | 峰值约 `12.0GB`，稳定 | 可作为低显存基线，质量一般 |
| `trial_if_lr3e4` | 在基线上加 `--lr 3e-4` | 约 17.44 min | 峰值约 `12.0GB`，稳定 | 当前三组对比里最适合展示 |
| `trial_if_perpneg` | `--IF --perpneg` 默认 `64x64` | 数秒内失败 | OOM | 失败案例，证明默认 IF+perpneg 过重 |
| `trial_if_perpneg_lowmem` | `--IF --perpneg --vram_O --w 48 --h 48` | 约 103.98 min | 峰值约 `13.8GB`，稳定但很慢 | 跑通但质量差，不适合展示 |
| `trial_perpneg_if_tiger_baseline_6000` | `DSLR prompt + --perpneg --negative_w -3.0` | 约 69.62 min | 高显存但未 OOM | 严重球体化失败 |
| `exp017_if_tiger` | `DSLR prompt`，去掉 `perpneg`，`64x64` | 约 21.78 min | 显存更高但稳定 | 成功率明显提升，可作为专题对照证据 |

### 4.2 基线：低显存 IF

命令：

```bash
gpustat --color -i 5 | tee ./logs/10_gpustat_baseline.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lowmem --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48
```

结果：

- 文本日志：`./logs/trial_if_lowmem_log_df.txt`
- CSV 日志：`./logs/trial_if_lowmem_train_metrics_df.csv`
- 结果视频：`./trial_if_lowmem/results/df_ep0060_rgb.mp4`
- 深度视频：`./trial_if_lowmem/results/df_ep0060_depth.mp4`
- 法线视频：`./trial_if_lowmem/results/df_ep0060_normal.mp4`
- 结论：主体可辨认，但几何和纹理偏弱，适合作为低显存基线，不适合作为最终最佳结果。

主观评分：

- Geometry：2
- Texture：2
- Multi-view consistency：3

### 4.3 学习率实验：`lr=3e-4`

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_lr3e4 --iters 6000 --batch_size 1 --IF --vram_O --w 48 --h 48 --lr 3e-4
```

结果：

- 文本日志：`./logs/trial_if_lr3e4_log_df.txt`
- CSV 日志：`./logs/trial_if_lr3e4_train_metrics_df.csv`
- 显存日志：`./logs/11_gpustat_lr3e4.txt`
- 结果视频：`./trial_if_lr3e4/results/df_ep0060_rgb.mp4`
- 深度视频：`./trial_if_lr3e4/results/df_ep0060_depth.mp4`
- 法线视频：`./trial_if_lr3e4/results/df_ep0060_normal.mp4`
- 训练时长：约 `17.4350` 分钟
- 峰值显存：训练日志约 `12.0GB`，`gpustat` 训练时段最高约 `13.3GB / 24.6GB`
- 结论：在不明显增加显存与时长的前提下，几何质量优于 `trial_if_lowmem`，是当前最适合作为最终展示候选的结果。

主观评分：

- Geometry：3
- Texture：3
- Multi-view consistency：3

### 4.4 高开销失败案例：默认 `IF + perpneg`

命令：

```bash
gpustat --color -i 5 | tee ./logs/13_gpustat_if_perpneg.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg
```

结果：

- 文本日志：`./logs/trial_if_perpneg_log_df.txt`
- CSV 日志：`./logs/trial_if_perpneg_train_metrics_df.csv`
- 显存日志：`./logs/13_gpustat_if_perpneg.txt`
- 报错：`torch.OutOfMemoryError: CUDA out of memory`
- 触发位置：`guidance/if_utils.py` 的 `train_step_perpneg()`
- 结论：默认 `64x64` 下直接使用 `--IF --perpneg` 在当前机器上不可行，是清晰的 OOM 失败案例。

### 4.5 降显存 `IF + perpneg`

命令：

```bash
gpustat --color -i 5 | tee ./logs/14_gpustat_if_perpneg_lowmem.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg_lowmem --iters 6000 --batch_size 1 --IF --perpneg --vram_O --w 48 --h 48
```

结果：

- 文本日志：`./logs/trial_if_perpneg_lowmem_log_df.txt`
- CSV 日志：`./logs/trial_if_perpneg_lowmem_train_metrics_df.csv`
- 显存日志：`./logs/14_gpustat_if_perpneg_lowmem.txt`
- 结果视频：`./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4`
- 结果判断：未 OOM，但耗时约 `103.9772` 分钟，显存峰值约 `13.8GB`，最终结果接近噪声，不适合作为展示结果。

### 4.6 `DSLR tiger` + `perpneg` 对照实验

原始用户想法：

```bash
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --num_steps 32 --upsample_steps 16
```

修正后实际执行命令：

```bash
gpustat --color -i 5 | tee ./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48
```

保留修正理由：

- `-O` 会启用 `--cuda_ray`，因此 `--num_steps` 和 `--upsample_steps` 基本不会生效。
- 若不显式设置 `--w 48 --h 48`，会回到默认 `64x64`，失败风险更高。

结果：

- 文本日志：`./logs/trial_perpneg_if_tiger_baseline_6000_log_df.txt`
- CSV 日志：`./logs/trial_perpneg_if_tiger_baseline_6000_train_metrics_df.csv`
- 显存日志：`./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt`
- 结果视频：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
- 训练时长：约 `69.6215` 分钟
- 结果判断：虽然完整跑通且未 OOM，但结果整体接近球体，属于严重几何塌缩/过度平滑失败，不适合作为最终结果。

### 4.7 `DSLR tiger` 去掉 `perpneg` 的专题实验

命令：

```bash
gpustat --color -i 5 | tee ./logs/17_gpustat_exp017_if_tiger.txt
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace exp017_if_tiger --iters 6000 --IF --batch_size 1 --h 64 --w 64 --seed 3407 --vram_O --eval_interval 10 --test_interval 100 --dataset_size_test 100
```

结果：

- 文本日志：`./logs/exp017_if_tiger_log_df.txt`
- CSV 日志：`./logs/exp017_if_tiger_train_metrics_df.csv`
- 显存日志：`./logs/17_gpustat_exp017_if_tiger.txt`
- 结果视频：`./exp017_if_tiger/results/df_ep0060_rgb.mp4`
- 深度视频：`./exp017_if_tiger/results/df_ep0060_depth.mp4`
- 法线视频：`./exp017_if_tiger/results/df_ep0060_normal.mp4`
- 训练时长：约 `21.7783` 分钟
- 显存结论：GPU 常驻约 `17.4 ~ 17.6GB`，明显高于 `48x48` 低显存基线
- 质量结论：主体可辨认，多视角连贯性较好，但仍有服装不自然和多头/Janus 伪影
- 对比结论：移除 `perpneg` 与 `negative_w=-3.0` 后，质量和稳定性都明显优于 `trial_perpneg_if_tiger_baseline_6000`

## 5. 导出结果与现有素材

### 5.1 基线导出命令

```bash
python main.py --workspace trial_if_lowmem -O --test
python main.py --workspace trial_if_lowmem -O --test --save_mesh
```

已确认存在的导出文件：

- 视频：`./trial_if_lowmem/results/df_ep0060_rgb.mp4`
- 深度视频：`./trial_if_lowmem/results/df_ep0060_depth.mp4`
- 法线视频：`./trial_if_lowmem/results/df_ep0060_normal.mp4`
- 网格：`./trial_if_lowmem/mesh/mesh.obj`
- 材质：`./trial_if_lowmem/mesh/mesh.mtl`
- 贴图：`./trial_if_lowmem/mesh/albedo.png`
- 网格规模：`15022` vertices，`30056` faces

### 5.2 已有截图素材

现有截图目录：

- `./logs/video_screenshots/trial_if_lowmem_rgb/`
- `./logs/video_screenshots/trial_if_lr3e4_rgb/`
- `./logs/video_screenshots/trial_if_perpneg_lowmem_rgb/`

截图导出命令：

```bash
python ./scripts/extract_video_screenshots.py ./trial_if_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lowmem_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_lr3e4/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lr3e4_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_perpneg_lowmem_rgb
```

## 6. 失败分析与超参数研究应如何补写

### 6.1 建议作为报告中的两个失败案例

`F1`：`trial_if_perpneg`

- Command：`python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg`
- Symptom：训练早期直接 OOM
- Suspected cause：`IF + perpneg + 64x64` 显存负担过高，且未开启 `--vram_O`
- Evidence：`./logs/13_gpustat_if_perpneg.txt` 与 `./logs/trial_if_perpneg_log_df.txt`
- Improvement idea：使用 `--vram_O --w 48 --h 48`，或直接去掉 `perpneg`

`F2`：`trial_perpneg_if_tiger_baseline_6000`

- Command：`python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48`
- Symptom：训练跑通但结果球体化，主体语义未成形
- Suspected cause：`perpneg` 与较强的 `negative_w=-3.0` 对该提示词约束过强，导致几何塌缩/过度平滑
- Evidence：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
- Improvement idea：移除 `perpneg`，或降低负向权重并改用更稳定提示词/种子

### 6.2 三组超参数对比结论

建议写入 `./logs/30_hparam_study.md` 的核心结论：

| Workspace | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `trial_if_lowmem` | default | `17.77 min` | `12.0 GB` | 2 | 2 | 3 | stable | low-memory IF baseline |
| `trial_if_lr3e4` | `lr=3e-4` | `17.44 min` | `12.0 GB` | 3 | 3 | 3 | stable | 当前最佳展示候选 |
| `trial_if_perpneg_lowmem` | `perpneg + vram_O` | `103.98 min` | `13.8 GB` | 1 | 1 | 1 | stable but slow | 可跑通但质量很差 |

推荐结论：

- 若报告只选 1 组主展示结果，优先选 `trial_if_lr3e4`。
- 若报告要做机制对照，可再加入 `exp017_if_tiger` 与 `trial_perpneg_if_tiger_baseline_6000`，用来说明 `perpneg` 强约束对某些提示词可能造成显著退化。

## 7. 是否还需要更多实验

结论：**不需要再做新的必做实验。**

原因：

1. 课程要求的 4 个核心任务都已经有实验支撑。
2. 现有实验已经覆盖了：稳定可运行基线、学习率改动、显存/OOM 失败、`perpneg` 强约束失败、去掉 `perpneg` 后的改善对照。
3. 当前最大的缺口在“整理交付物”，不在“缺少训练样本”。

若还有时间，只建议做下面 1 项中的任意 1 项：

1. 为当前最佳结果导出 mesh：

    ```bash
    python main.py --workspace trial_if_lr3e4 -O --test --save_mesh
    ```

2. 为 `exp017_if_tiger` 补导出截图，作为专题对照图：

    ```bash
    python ./scripts/extract_video_screenshots.py ./exp017_if_tiger/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/exp017_if_tiger_rgb
    ```

3. 若想增强结论稳健性，可补 1 次轻量重复实验，例如固定 `exp017_if_tiger` 配置仅更换 `seed`。这属于加分项，不是必需项。

## 8. 建议立即完成的收尾动作

1. 建立 `./docs/report/exp3/screenshots/` 与 `./docs/report/exp3/artifacts/`。
2. 从 `./logs/video_screenshots/` 复制成功图、失败图、对比图到报告目录。
3. 补写 `./logs/20_failure_analysis.md`。
4. 补写 `./logs/30_hparam_study.md`。
5. 补写 `./logs/12_export_baseline.txt`。
6. 报告正文优先采用：`trial_if_lr3e4` 作为主结果，`trial_if_perpneg` 与 `trial_perpneg_if_tiger_baseline_6000` 作为失败分析，`exp017_if_tiger` 作为机制对照。

## 9. 最终自检清单

- [x] 已记录完整训练命令与关键参数
- [x] 已保存多组显存监控日志
- [x] 已完成至少 1 项超参数改动并具备对比依据
- [x] 已具备至少 2 个失败案例素材来源
- [ ] 已补齐失败分析文档
- [ ] 已补齐超参数对比文档
- [ ] 已整理报告截图目录
- [ ] 已整理 artifacts 索引目录
- [ ] 已形成最终报告可直接引用的表格与图注
