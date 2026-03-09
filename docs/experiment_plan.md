# MTN 实验计划与当前进展

本文档基于 `docs/requirement.md`、`README.md` 以及当前工作区的已有产出整理，按照实验的实际执行顺序自上而下组织。所有步骤均整理为可勾选清单；已完成内容直接标记，未完成内容保留空白勾选框。

## 1. 实验目标与最终交付

- [x] 完成至少 1 次 Text-to-3D 训练并导出结果视频
- [x] 记录训练期间 GPU 显存变化
- [x] 完成至少 1 项超参数改动实验并比较效果
- [x] 获得至少 2 个可用于失败分析的案例
- [x] 整理最终报告截图
- [x] 建立 artifacts 索引
- [ ] 补齐失败分析与超参数研究成稿

最终报告建议如下：

- 最佳结果采用 4.7 实验，文档内统一命名为 `trial_if_tiger_no_perpneg_64`。

## 2. 环境准备与前置条件

### 2.1 环境目标

- [x] 目标环境确定为 `Python 3.9 + CUDA 12.8 + torch 2.8.0`
- [x] 确认本轮实验统一走 IF 路线
- [x] 明确使用 `--IF` 前需要完成 Hugging Face 权限开通

### 2.2 环境检查命令

- [x] 已准备环境检查命令

```bash
nvcc -V
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pip show gpustat
```

### 2.3 安装命令

- [x] 已确定最终有效安装命令

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

### 2.4 Hugging Face 权限验证

- [x] 已确认 IF 需要先接受 `DeepFloyd/IF-I-XL-v1.0` 使用条款
- [x] 已整理登录与验证命令

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

## 3. 实验总览

- [x] 已完成实验总览整理

| 文档命名 | 实际 workspace | 关键参数 | 时长 | 显存/稳定性 | 结果判断 |
| --- | --- | --- | --- | --- | --- |
| `trial_if_lowmem` | `trial_if_lowmem` | `--IF --vram_O --w 48 --h 48` | 约 17.77 min | 峰值约 `12.0GB`，稳定 | 图像模糊，可辨认度低，作为低显存基线保留 |
| `trial_if_lr3e4` | `trial_if_lr3e4` | 在基线上加 `--lr 3e-4` | 约 17.44 min | 峰值约 `12.0GB`，稳定 | 比基线稍清晰，但仍有明显伪影 |
| `trial_if_perpneg` | `trial_if_perpneg` | `--IF --perpneg` 默认 `64x64` | 数秒内失败 | OOM | 失败案例，证明默认 IF+perpneg 过重 |
| `trial_if_perpneg_lowmem` | `trial_if_perpneg_lowmem` | `--IF --perpneg --vram_O --w 48 --h 48` | 约 103.98 min | 峰值约 `13.8GB`，稳定但很慢 | 呈现立体噪声，不适合作为展示结果 |
| `trial_perpneg_if_tiger_baseline_6000` | `trial_perpneg_if_tiger_baseline_6000` | `DSLR prompt + --perpneg --negative_w -3.0` | 约 69.62 min | 高显存但未 OOM | 呈现大型立体光球，严重失败 |
| `trial_if_tiger_no_perpneg_64` | `trial_if_tiger_no_perpneg_64` | `DSLR prompt`，去掉 `perpneg`，`64x64` | 约 21.78 min | 显存更高但稳定 | 当前结果最清晰，最适合作为最佳结果 |

## 4. 实验执行与结果记录

### 4.1 基线：低显存 IF

- [x] 已开启基线显存监控
- [x] 已完成训练
- [x] 已自动生成日志与测试视频

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
- 质量描述：图像整体较为模糊，仅能勉强辨认为老虎；头部与胸前可隐约看出蓝白色衣物，但整体存在大面积不连续色块，结构稳定性较差。

主观评分：

- Geometry：2
- Texture：1
- Multi-view consistency：2

结论：

- 作为低显存基线可保留。
- 可辨认主体较弱，不适合作为最终最佳结果。

### 4.2 学习率实验：`lr=3e-4`

- [x] 已完成学习率对比实验
- [x] 已生成日志与测试视频

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
- 质量描述：相比 `trial_if_lowmem` 略清晰，但边缘有明显噪点；大褂呈橙色，语义偏移明显；出现两个尾巴，头部还有明显伪影。

主观评分：

- Geometry：2
- Texture：2
- Multi-view consistency：2

结论：

- 相比低显存基线略有提升，但仍存在明显的结构错误与语义偏移。
- 不再作为最佳结果，改为超参数对比中的中间结果。

### 4.3 高开销失败案例：默认 `IF + perpneg`

- [x] 已执行高开销 IF+perpneg 实验
- [x] 已记录 OOM 失败日志

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
- 结论：默认 `64x64` 下直接使用 `--IF --perpneg` 在当前机器上不可行，是典型 OOM 失败案例。

### 4.4 降显存 `IF + perpneg`

- [x] 已完成降显存 IF+perpneg 实验
- [x] 已生成日志与测试视频

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
- 训练时长：约 `103.9772` 分钟
- 峰值显存：约 `13.8GB`
- 质量描述：结果主要表现为立体噪声，虽然保留了一定三维体积感，但未能形成有效主体。

主观评分：

- Geometry：1
- Texture：1
- Multi-view consistency：1

结论：

- 可作为失败案例使用。
- 训练代价高且结果不可用，不适合作为展示结果。

### 4.5 `DSLR tiger` + `perpneg` 对照实验

- [x] 已完成该对照实验
- [x] 已生成日志、显存记录与测试视频

原始用户想法：

```bash
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --num_steps 32 --upsample_steps 16
```

修正后实际执行命令：

```bash
gpustat --color -i 5 | tee ./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48
```

修正理由：

- `-O` 会启用 `--cuda_ray`，因此 `--num_steps` 和 `--upsample_steps` 基本不会生效。
- 若不显式设置 `--w 48 --h 48`，会回到默认 `64x64`，失败风险更高。

结果：

- 文本日志：`./logs/trial_perpneg_if_tiger_baseline_6000_log_df.txt`
- CSV 日志：`./logs/trial_perpneg_if_tiger_baseline_6000_train_metrics_df.csv`
- 显存日志：`./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt`
- 结果视频：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
- 训练时长：约 `69.6215` 分钟
- 质量描述：结果整体呈现为大型立体光球，虽然具有一定体积感和发光感，但未形成可辨认的“老虎医生”结构。

主观评分：

- Geometry：1
- Texture：1
- Multi-view consistency：1

结论：

- 属于典型几何塌缩失败案例。
- 可直接与 4.6 的无 `perpneg` 结果构成强对照。

### 4.6 `DSLR tiger` 去掉 `perpneg` 的最佳实验

- [x] 已完成该实验
- [x] 已生成日志、显存记录与测试视频
- [x] 已确定为本次实验最佳结果
- [ ] 如需统一截图素材，后续可补充导出并整理到报告目录

文档统一命名：`trial_if_tiger_no_perpneg_64`

命令：

```bash
gpustat --color -i 5 | tee ./logs/17_gpustat_trial_if_tiger_no_perpneg_64.txt
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_if_tiger_no_perpneg_64 --iters 6000 --IF --batch_size 1 --h 64 --w 64 --seed 3407 --vram_O --eval_interval 10 --test_interval 100 --dataset_size_test 100
```

结果：

- 文本日志：`./logs/trial_if_tiger_no_perpneg_64_log_df.txt`
- CSV 日志：`./logs/trial_if_tiger_no_perpneg_64_train_metrics_df.csv`
- 显存日志：`./logs/17_gpustat_trial_if_tiger_no_perpneg_64.txt`
- 结果视频：`./trial_if_tiger_no_perpneg_64/results/df_ep0060_rgb.mp4`
- 深度视频：`./trial_if_tiger_no_perpneg_64/results/df_ep0060_depth.mp4`
- 法线视频：`./trial_if_tiger_no_perpneg_64/results/df_ep0060_normal.mp4`
- 训练时长：约 `21.7783` 分钟
- 显存结论：GPU 常驻约 `17.4 ~ 17.6GB`，明显高于 `48x48` 低显存基线
- 质量描述：这是当前唯一生成结果较为清晰的实验。主体能够明确识别为老虎，多视角连贯性明显优于前面几组；虽然仍存在服装不够自然、局部轻微模糊以及头部 Janus 伪影，但整体已形成稳定且可展示的三维主体。

主观评分：

- Geometry：4
- Texture：4
- Multi-view consistency：4

结论：

- 本次实验的最佳结果采用 4.6。
- 与 4.5 相比，去掉 `perpneg` 后质量提升非常明显，是报告中最关键的正向结果。

## 5. 导出结果与素材整理

### 5.1 基线导出与网格

- [x] 已执行基线测试导出
- [x] 已执行基线 mesh 导出
- [x] 已把导出摘要补写到 `./logs/12_export_baseline.txt`

导出命令：

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

### 5.2 截图素材

- [x] 已生成 3 组基础截图素材
- [ ] 需要将最终用于报告的截图复制到 `./docs/report/exp3/screenshots/`
- [ ] 如需突出最佳结果，建议补充导出 4.6 的截图素材

现有截图目录：

- `./logs/video_screenshots/trial_if_lowmem_rgb/`
- `./logs/video_screenshots/trial_if_lr3e4_rgb/`
- `./logs/video_screenshots/trial_if_perpneg_lowmem_rgb/`

截图导出命令：

```bash
python ./scripts/extract_video_screenshots.py ./trial_if_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lowmem_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_lr3e4/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_lr3e4_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_perpneg_lowmem/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_perpneg_lowmem_rgb
python ./scripts/extract_video_screenshots.py ./trial_if_tiger_no_perpneg_64/results/df_ep0060_rgb.mp4 --count 4 --skip-first-last --output-dir ./logs/video_screenshots/trial_if_tiger_no_perpneg_64_rgb
```

## 6. 失败案例分析

### 6.1 失败案例 F1：默认 `IF + perpneg` OOM

- [x] 已确定 F1 案例
- [x] 需要补 2 张失败截图或日志截图到报告目录

案例信息：

- Command：`python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg`
- Symptom：训练早期直接 OOM，未能完成有效训练
- Suspected cause：`IF + perpneg + 64x64` 显存开销过高，且未开启 `--vram_O`
- Evidence：`./logs/13_gpustat_if_perpneg.txt`、`./logs/trial_if_perpneg_log_df.txt`
- Improvement idea：使用 `--vram_O --w 48 --h 48` 降低显存压力，或直接取消 `perpneg`

### 6.2 失败案例 F2：`DSLR tiger + perpneg` 几何塌缩

- [x] 已确定 F2 案例
- [x] 需要补 2 张失败截图到报告目录

案例信息：

- Command：`python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48`
- Symptom：训练跑通但结果表现为大型立体光球，主体语义完全未成形
- Suspected cause：`perpneg` 与较强的 `negative_w=-3.0` 对该提示词约束过强，导致几何塌缩并过度平滑
- Evidence：`./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
- Improvement idea：移除 `perpneg`，或减弱负向权重，再配合更稳定提示词和固定 seed

### 6.3 失败案例补充说明

- [x] 已形成两个失败案例的文字分析框架
- [x] 需要把最终截图放到 `./docs/report/exp3/screenshots/`

## 7. 超参数与方法对比

### 7.1 三组基础对比

- [x] 已完成三组基础实验对比结论整理
- [x] 已把正式版对比表补写到 `./logs/30_hparam_study.md`

| Workspace | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `trial_if_lowmem` | default | `17.77 min` | `12.0 GB` | 2 | 1 | 2 | stable | 模糊且色块化明显 |
| `trial_if_lr3e4` | `lr=3e-4` | `17.44 min` | `12.0 GB` | 2 | 2 | 2 | stable | 略清晰，但有双尾和头部伪影 |
| `trial_if_perpneg_lowmem` | `perpneg + vram_O` | `103.98 min` | `13.8 GB` | 1 | 1 | 1 | stable but slow | 立体噪声，几乎不可用 |

### 7.2 最佳结果与对照结论

- [x] 已确定最佳结果为 4.6
- [x] 已确定 4.5 与 4.6 构成最强机制对照

推荐结论：

- 基础三组中，`trial_if_lr3e4` 相比低显存基线略有改进，但提升有限。
- 最强的正向结果来自 4.6，即文档命名 `trial_if_tiger_no_perpneg_64`。
- 4.5 与 4.6 的直接对比说明：对于该提示词，移除 `perpneg` 比继续增强负向约束更有效。

## 8. 后续仅需补充的素材与文档

- [x] 已建立 `./docs/report/exp3/screenshots/`
- [x] 已建立 `./docs/report/exp3/artifacts/`
- [x] 已补写 `./logs/12_export_baseline.txt`
- [x] 已补写 `./logs/20_failure_analysis.md`
- [x] 已补写 `./logs/30_hparam_study.md`
- [x] 复制成功图、失败图、对比图到报告目录
- [x] 若要突出最佳结果，补导出 4.6 的截图素材

## 9. 最终自检清单

- [x] 已记录完整训练命令与关键参数
- [x] 已保存多组显存监控日志
- [x] 已完成至少 1 项超参数改动并具备对比依据
- [x] 已具备至少 2 个失败案例素材来源
- [x] 已确定本实验最佳结果为 4.6
- [x] 已补齐失败分析文档
- [x] 已补齐超参数对比文档
- [x] 已整理报告截图目录
- [x] 已整理 artifacts 索引目录
- [ ] 已形成最终报告可直接引用的表格与图注
