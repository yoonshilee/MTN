# Failure Analysis

## F1. Default IF + perpneg causes early OOM

- Command: `python main.py -O --text "a tiger dressed as a doctor" --workspace trial_if_perpneg --iters 6000 --batch_size 1 --IF --perpneg`
- Symptom: Training terminates almost immediately with `torch.OutOfMemoryError: CUDA out of memory` before producing usable results.
- Suspected cause: The combination of `--IF`, `--perpneg`, default `64x64` resolution, and no `--vram_O` creates a memory load that exceeds the available GPU budget.
- Evidence logs:
  - `./logs/13_gpustat_if_perpneg.txt`
  - `./logs/trial_if_perpneg_log_df.txt`
  - `./logs/trial_if_perpneg_train_metrics_df.csv`
- Evidence images: No final render was produced. The report can instead cite the OOM log and the gpustat record as evidence.
- Improvement idea:
  - Add `--vram_O`
  - Reduce resolution to `--w 48 --h 48`
  - Remove `--perpneg` when memory is the primary constraint

## F2. Strong perpneg guidance collapses the geometry

- Command: `python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger_baseline_6000 --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0 --vram_O --w 48 --h 48`
- Symptom: Training completes without OOM, but the final render degenerates into a large glowing 3D sphere instead of a recognizable tiger doctor.
- Suspected cause: The prompt is more semantically specific, but `perpneg` together with the strong `negative_w=-3.0` likely over-constrains view generation and suppresses stable object formation, leading to geometry collapse and over-smoothing.
- Evidence files:
  - `./trial_perpneg_if_tiger_baseline_6000/results/df_ep0060_rgb.mp4`
  - `./logs/15_gpustat_perpneg_if_tiger_baseline_6000.txt`
  - `./logs/trial_perpneg_if_tiger_baseline_6000_log_df.txt`
- Evidence images: Screenshots should be selected from the result video and copied into `./docs/report/exp3/screenshots/`.
- Improvement idea:
  - Remove `--perpneg`
  - Reduce the absolute value of `negative_w`
  - Keep the DSLR-style prompt but use the no-perpneg setting as the main configuration

## Comparative note

These two failures are useful because they correspond to different failure modes:

- F1 is a resource failure: the training pipeline cannot even run stably.
- F2 is a quality failure: training runs to completion, but the geometry collapses and the semantics are lost.

Together they show that better final quality in this project depends not only on enough GPU memory, but also on choosing a guidance setting that does not over-constrain the optimization.
