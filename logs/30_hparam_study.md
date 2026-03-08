# Hyper-Parameter Study

## Compared runs

This section compares three baseline-style runs using the prompt `a tiger dressed as a doctor`.

| Workspace | Key Param | Train Time | Peak VRAM | Geometry | Texture | Consistency | Stability | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `trial_if_lowmem` | default low-memory IF | `17.77 min` | `12.0 GB` | 2 | 1 | 2 | stable | blurry result with large disconnected color blocks |
| `trial_if_lr3e4` | `lr=3e-4` | `17.44 min` | `12.0 GB` | 2 | 2 | 2 | stable | slightly clearer, but still has strong artifacts |
| `trial_if_perpneg_lowmem` | `perpneg + vram_O` | `103.98 min` | `13.8 GB` | 1 | 1 | 1 | stable but slow | output becomes 3D noise and is visually unusable |

## Qualitative observations

### 1. `trial_if_lowmem`

- The result is blurry and only barely recognizable as a tiger.
- The head and chest area weakly suggest blue-white clothing.
- Large color patches appear disconnected, so the object lacks coherent structure.

### 2. `trial_if_lr3e4`

- Increasing the learning rate to `3e-4` makes the object slightly clearer than `trial_if_lowmem`.
- However, the result still contains obvious edge noise.
- The coat becomes orange rather than the intended doctor outfit.
- The model produces two tails and visible artifacts around the head.

### 3. `trial_if_perpneg_lowmem`

- This run completes without OOM, but the final output resembles volumetric noise.
- The result has a 3D appearance, yet it does not form a usable tiger doctor object.
- It is substantially slower than the other two runs while giving the worst visual quality.

## Conclusion

- `trial_if_lr3e4` is only a modest improvement over `trial_if_lowmem`.
- `trial_if_perpneg_lowmem` is not worth using as a final result because the quality collapses despite the much longer runtime.
- Among these three runs, `trial_if_lr3e4` is the strongest baseline-style comparison result.

## Best overall result outside the three-run baseline comparison

The clearest result in the whole experiment is not one of the three baseline comparison runs above. It is the no-perpneg DSLR-prompt experiment recorded in the plan as `trial_if_tiger_no_perpneg_64`, whose actual workspace is `exp017_if_tiger`.

This comparison supports the following interpretation:

- A small learning-rate increase helps slightly.
- Strong `perpneg` guidance is much riskier for this prompt than simply removing it.
- For the final report, the best showcase result should therefore be the no-perpneg DSLR-prompt run rather than any of the three baseline-style runs.
