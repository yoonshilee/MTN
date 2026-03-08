# Artifacts Index

## Best Result

- Plan name: `trial_if_tiger_no_perpneg_64`
- Actual workspace: `exp017_if_tiger`
- RGB video: `./exp017_if_tiger/results/df_ep0060_rgb.mp4`
- Depth video: `./exp017_if_tiger/results/df_ep0060_depth.mp4`
- Normal video: `./exp017_if_tiger/results/df_ep0060_normal.mp4`

## Baseline Result

- Workspace: `trial_if_lowmem`
- RGB video: `./trial_if_lowmem/results/df_ep0060_rgb.mp4`
- Depth video: `./trial_if_lowmem/results/df_ep0060_depth.mp4`
- Normal video: `./trial_if_lowmem/results/df_ep0060_normal.mp4`
- Mesh: `./trial_if_lowmem/mesh/mesh.obj`
- Material: `./trial_if_lowmem/mesh/mesh.mtl`
- Texture: `./trial_if_lowmem/mesh/albedo.png`

## Hyper-Parameter Comparison

- `trial_if_lowmem`
- `trial_if_lr3e4`
- `trial_if_perpneg_lowmem`

## Failure Cases

- OOM failure: `trial_if_perpneg`
- Geometry collapse failure: `trial_perpneg_if_tiger_baseline_6000`

## Note

The final report should reference screenshots from `./docs/report/exp3/screenshots/`. The current index records the canonical source files in the workspace.
