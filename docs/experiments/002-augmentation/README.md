# Experiment 002: Data Augmentation and Normalization

GitHub issue: [#9](https://github.com/samuelstevens/beetle-traits/issues/9)

## Setup

Three conditions compared across 6 learning rates (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):

- **baseline**: no normalization, no augmentation (`DecodeRGB -> Resize`)
- **norm-only**: ImageNet normalization, no augmentation (`DecodeRGB -> Resize -> Normalize`)
- **full-aug**: ImageNet normalization + spatial/color augmentation (`DecodeRGB -> RandomResizedCrop -> RandomFlip -> RandomRotation -> ColorJitter -> Normalize`)

Conservative crop settings for full-aug: `crop_scale=(0.9, 1.0)`, `crop_ratio=(0.95, 1.067)`.

Training data: Hawaii + BeetlePalooza only (biorepo was not included in this experiment). All runs used `n_workers=8`, `batch_size=256`, frozen DINOv3 ViT-S/16 backbone.

## Results

### Normalization is a clear win

Best val loss per condition (across all LRs):

| Condition | Best LR | Best val loss | Best val length err (cm) |
|-----------|---------|---------------|--------------------------|
| baseline  | 3e-4    | 189.55        | 0.061                    |
| norm-only | 3e-4    | 176.29        | 0.058                    |
| full-aug  | 1e-2    | 309.98        | 0.074                    |

norm-only beats baseline by ~7% on val loss and ~5% on val length error. This makes sense: DINOv3 was pretrained with ImageNet normalization, so feeding it normalized inputs better matches its expected distribution.

### Full augmentation does not help (yet)

full-aug fails the acceptance criteria from the spec: its best val loss (309.98) is 76% worse than norm-only (176.29), far from the required `<= 0.98 * norm-only`.

Likely reasons:
- Conservative crop settings (scale 0.9-1.0) provide minimal diversity.
- full-aug's best LR (1e-2) is much higher than norm-only's (3e-4), suggesting the augmented training signal is noisier and the model struggles to converge at lower LRs.
- Runs were limited to ~25k-41k steps; augmentation benefits may need longer training.

The generalization gap (val - train) is smaller for full-aug (0.027 cm) than norm-only (0.030 cm), consistent with augmentation reducing overfitting, but the absolute performance is still worse.

### LR sensitivity

norm-only is robust across LRs (val loss 176-303), while full-aug is much more LR-sensitive (val loss 310-904). Baseline falls in between (190-351). See `artifacts/lr_vs_final_val_loss.png`.

### Throughput

| Condition | Mean steps/sec |
|-----------|----------------|
| baseline  | 5.34           |
| norm-only | 5.09           |
| full-aug  | 2.93           |

full-aug is ~45% slower than baseline due to CPU-side transform cost. norm-only has negligible overhead. This was measured with `n_workers=8`; earlier runs with `n_workers=4` showed even worse full-aug throughput (see below).

## Decision

Merge **norm-only** as the new default pipeline. Do not enable full augmentation yet.

## Open questions

- Would more aggressive crop settings (scale 0.5-1.0) help, or would OOB fraction become too high? The `demo.py` notebook now includes biorepo in the OOB sweep to inform `aug_biorepo` defaults.
- Would longer training (100k+ steps) let full-aug catch up?
- Per-dataset augmentation configs (`aug_hawaii`, `aug_beetlepalooza`, `aug_biorepo`) are now explicit with no global fallback. biorepo defaults still need tuning based on OOB analysis.

## Log

### 2026-02-09: full-aug runs are dataloader-bound at `n_workers=4`

Run: Slurm array `3618837` (9 quick runs).

Observed:
- `full-aug` jobs (`2`, `5`, `8`) were consistently the slowest.
- `full-aug` throughput was about `85-92s` per 200 train steps.
- `baseline` throughput was about `39-48s` per 200 train steps.
- `norm-only` throughput was about `43-61s` per 200 train steps.

Evidence that this is augmentation cost (not just node variance):
- On node `a0306`: `norm-only` job `1` took `00:09:31`, `full-aug` job `2` took `00:16:35`.
- On node `a0337`: `baseline` job `6` took `00:09:27`, `full-aug` job `5` took `00:16:12`.

Resolution: increased `n_workers` from 4 to 8. full-aug throughput improved to ~2.93 steps/sec (from ~2.2-2.4), closing the gap somewhat but still ~45% slower than baseline.
