# Experiment Report-3: 3D Generation

## Objectives

The goal of this experiment is to explore state-of-the-art 3D content generation techniques using diffusion-based models. You will:

- Generate a 3D object from a text prompt (e.g., “a tiger dressed as a doctor” ).
- Analyze common failure modes and limitations.
- Investigate the impact of hyper-parameters on output quality and training stability.
- Reflect on memory usage and computational efficiency.

## Preparation

Before starting, ensure your environment is properly set up:
CUDA out of memory, reduce resolution from 64 ￫48 or switch to a lighter Stable Diffusion version.

- Codebase: <https://github.com/Texaser/MTN>
- Video demo: <https://www.youtube.com/watch?v=LH6-wKg30FQ>
- Check PyTorch compatibility: <https://pytorch.org/get-started/previous-versions>
- Install monitoring tool: pip install gpustat; gpustat

## Tasks

Complete the following tasks during the lab session:

1. 3D Generation: Use a text prompt like “a tiger dressed as a doctor” to generate a 3D video. Record your command and results.
2. Memory Monitoring: Use gpustat to log GPU memory usage throughout training.
3. Failure Analysis: Identify at least two common failure cases (e.g., distorted geometry, missing parts, texture artifacts).
4. Hyper-parameter Study: Read the config/hyper-parameters in <https://github.com/Texaser/> MTN/blob/main/main.py#L22-L173. Change one parameter (e.g., learning rate, number of views, resolution) and observe its effect.

## Lab Report Requirements

- Submit your report before 23:59 on 15 March via UM Moodle.
- Minimum length: 2 pages using the provided LaTeX template.
- Your report must include:
  - A brief introduction and objective summary.
7 of 9
  - Detailed description of your experiments and observations.
  - Analysis of flaws/failures with visual examples (screenshots encouraged)
  - At least two concrete improvement suggestions (e.g., better initialization, multi-stage refinement, data augmentation).
  - Reflection on hyper-parameter sensitivity.

## Notes

- If your previous training completed quickly and you want to resume or retrain, make sure to not delete or rename the checkpoint file (e.g., trial/checkpoints/df_ep0060.pth) unless you intend to start fresh.
- To avoid accidental overwrite: mv trial/checkpoints/df_ep0060.pth df_ep0060_backup.pth
