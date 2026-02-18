# Gaussian Heatmap Keypoint Regression Spec

GitHub issue: [#1](https://github.com/samuelstevens/beetle-traits/issues/1)

References:
- [ViTPose (Xu et al., NeurIPS 2022)](https://arxiv.org/abs/2204.12484)
- [ViTPose++ (Xu et al., TPAMI 2023)](https://arxiv.org/abs/2212.04246)

## Motivation

The current pipeline predicts keypoint coordinates by passing the CLS token through a 1-layer MLP (`768 -> 768 -> 8`), regressing (x, y) directly with MSE loss. This discards the spatial information in the ViT's patch tokens.

Heatmap-based regression is the standard approach in pose estimation (SimpleBaseline, ViTPose). Instead of regressing coordinates from a global feature, the model predicts a per-keypoint probability map from spatially arranged patch features. Benefits:

1. Patch tokens retain spatial layout -- a natural fit for localization.
2. Richer supervision signal: every heatmap pixel contributes to the loss, not just 2 scalar coordinates.
3. Uncertainty for free: heatmap spread reflects localization confidence.
4. Proven at scale in human pose estimation (ViTPose: 81.1 mAP on COCO).

## Final Decisions

| Topic | Decision |
|---|---|
| Backbone | Frozen DINOv3 ViT-S/16 (384-dim, 16x16 patch grid on 256x256 input). Same as exp 002. |
| Patch tokens | Reshape 256 patch tokens into 16x16 x 384 spatial feature map. Discard CLS token. |
| Decoder | 2 transpose-conv layers (4x upsample total: 16x16 -> 64x64) + 1x1 conv to 4 heatmaps. |
| Decoder channels | Deconv1: 384 -> 256. Deconv2: 256 -> 128. Final 1x1 conv: 128 -> 4. |
| Decoder normalization | GroupNorm + ReLU between deconv layers. No activation after final 1x1 conv. |
| Deconv kernel | 4x4, stride=2, padding=1 (clean 2x upsampling, no checkerboard artifacts). |
| Heatmap resolution | 64x64 (1/4 of 256x256 input). Matches ViTPose convention. |
| Number of heatmaps | 4 independent channels (one per keypoint endpoint: width_p0, width_p1, length_p0, length_p1). |
| Gaussian target | Unnormalized Gaussian (peak=1.0) centered on ground truth keypoint in 64x64 heatmap space. |
| Sigma | Sweep variable: {1, 2, 3} in heatmap pixels (= {4, 8, 12} in 256x256 image pixels). |
| Loss | MSE between predicted heatmaps (raw logits, no activation) and Gaussian targets. Per-channel masking for unobserved keypoints. Permutation-invariant per line: compute loss for both endpoint orderings, take the minimum. Normalize by active elements only. |
| Coord extraction | Soft-argmax on predicted heatmaps at validation. Scale coordinates from 64x64 back to 256x256 using UDPP convention. |
| Coord convention | UDPP pixel-center alignment: `hx = (x + 0.5) / 4 - 0.5`. Inverse: `x = (hx + 0.5) * 4 - 0.5`. Soft-argmax grid spans `[-0.5, 63.5]` (not `[0, 63]`) so that edge pixels `x=0` and `x=255` are reachable. |
| Endpoint order | Permutation-invariant loss per line: `min(loss(ch0->gt_a, ch1->gt_b), loss(ch0->gt_b, ch1->gt_a))`. No canonicalization needed. |
| Optimizer | AdamW, weight_decay=0.1 on conv weights only. No weight decay on GroupNorm params or biases. |
| LR sweep | {1e-3, 3e-3, 1e-2, 3e-2}. |
| Augmentation | Full augmentation (same as exp 002 full-aug condition). |
| Training steps | 100,000. |
| Wall time | 12 hours. |
| Seeds | 1 per (sigma, LR) combo. |
| Datasets | Train: Hawaii + BeetlePalooza + BioRepo. Val: Hawaii + BioRepo. Same as exp 002. |
| Total runs | 3 sigma x 4 LR = 12. |

## Architecture

```
Input image (256x256x3)
    |
    v
Frozen DINOv3 ViT-S/16
    |
    v
Patch tokens (256 tokens, 384-dim each)
    |  reshape to spatial grid
    v
Feature map (16x16 x 384)
    |
    v
ConvTranspose2d(384, 256, kernel=4, stride=2, pad=1)  ->  32x32 x 256
GroupNorm(32 groups, 256 channels) + ReLU
    |
    v
ConvTranspose2d(256, 128, kernel=4, stride=2, pad=1)  ->  64x64 x 128
GroupNorm(32 groups, 128 channels) + ReLU
    |
    v
Conv2d(128, 4, kernel=1)  ->  64x64 x 4
    |
    v
4 heatmaps (64x64 each), raw logits
```

Trainable parameters: ~700K (decoder only). Backbone is frozen (~22M params, ViT-S/16).

## Gaussian Heatmap Target

For each keypoint at position (x, y) in 256x256 image space:

1. Convert to heatmap coordinates using UDPP convention: `(hx, hy) = ((x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5)`.
2. Generate a 64x64 grid of Gaussian values: `G[i, j] = exp(-((j - hx)^2 + (i - hy)^2) / (2 * sigma^2))`.
3. Peak value is 1.0 (unnormalized).
4. Background is ~0.0 (Gaussian tails).

The target tensor per sample has shape `(4, 64, 64)`.

### Out-of-bounds keypoints

If a keypoint is outside the 256x256 image (from augmentation), its Gaussian center falls outside the 64x64 grid. The Gaussian will be partially visible or fully zero. The existing `oob_policy="supervise_oob"` applies -- the loss is still computed on these heatmaps. The model learns to predict background (all zeros) for off-frame keypoints.

### Loss masking and normalization

BeetlePalooza lacks width annotations. The existing `loss_mask` (shape `(2,)` for width/length lines) extends to heatmap channels:

- `loss_mask[0] = 0` (width unobserved): zero loss on heatmap channels 0 and 1 (width endpoints).
- `loss_mask[1] = 1` (length observed): compute loss normally on channels 2 and 3.

The mask broadcasts over the `(64, 64)` spatial dimensions.

**Normalization:** Divide by the number of active heatmap elements, not total elements. `loss = sum(squared_error * channel_mask) / (sum(channel_mask) * H * W + eps)`. This prevents the effective gradient scale from changing when channels are masked.

### Permutation-invariant loss

For each line (width or length), the two endpoints can be assigned to heatmap channels in either order. Compute loss for both assignments and take the minimum:

```
loss_direct = mse(ch_a, gt_p0) + mse(ch_b, gt_p1)
loss_swapped = mse(ch_a, gt_p1) + mse(ch_b, gt_p0)
line_loss = min(loss_direct, loss_swapped)
```

This mirrors the order-invariant matching already used in validation metrics and prevents contradictory supervision when annotation endpoint order is inconsistent.

## Soft-Argmax Coordinate Extraction

At validation, extract coordinates from predicted heatmaps:

1. For each keypoint's 64x64 heatmap, stabilize logits: `h = h - max(h)`.
2. Apply spatial softmax: `p[i,j] = exp(h[i,j]) / sum(exp(h))`.
3. Compute expected coordinates using shifted grids that span `[-0.5, 63.5]`: `x_grid[j] = j - 0.5 + 0.5 = j`, wait -- more precisely, use `x_grid[j] = j` and `y_grid[i] = i` (standard 0-indexed), then: `hx = sum(p * x_grid)`, `hy = sum(p * y_grid)`.
4. Scale to image space using inverse UDPP: `x = (hx + 0.5) * 4 - 0.5`, `y = (hy + 0.5) * 4 - 0.5`.

Note: with this mapping, `hx=0 -> x=1.5` and `hx=63 -> x=253.5`. To reach `x=0`, the model would need `hx=-0.375`, which is outside the grid. In practice, border keypoints are rare after augmentation (crops center content), and the ~1.5px border bias is acceptable. If it matters, the grid can be extended to `[-0.5, 63.5]` in a follow-up.

These coordinates replace the MLP's direct (x, y) predictions. The existing metric pipeline (affine back-mapping to original space, order-invariant matching, cm conversion) remains unchanged.

## Sweep Design

12 runs total: 3 sigma values x 4 learning rates.

| Sigma (heatmap px) | LR=1e-3 | LR=3e-3 | LR=1e-2 | LR=3e-2 |
|---|---|---|---|---|
| 1 | run 1 | run 2 | run 3 | run 4 |
| 2 | run 5 | run 6 | run 7 | run 8 |
| 3 | run 9 | run 10 | run 11 | run 12 |

All runs use:
- Full augmentation (crop, flip, rotation, color jitter, normalize)
- Seed: deterministic per (sigma, LR) pair
- 100k steps, 12h wall time
- AdamW, weight_decay=0.1
- wandb tag: `exp-004-heatmap`

Comparison baseline: exp 002 full-aug results (coordinate MSE with same augmentation).

## Implementation Plan

### 1. Gaussian heatmap transform

New grain transform in `src/btx/data/transforms.py`:
- Input: sample with `tgt` field (shape `(2, 2, 2)`, keypoint coords in 256x256 space).
- Output: adds `heatmap_tgt` field (shape `(4, 64, 64)`, Gaussian heatmaps).
- Configurable `sigma` and `heatmap_size` (default 64).
- Applied after `FinalizeTargets`, before `Normalize`.

### 2. Heatmap decoder model

New model file `src/btx/modeling/heatmap.py`:
- Takes patch tokens from frozen DINOv3, reshapes to 16x16 spatial grid.
- 2 transpose-conv layers with GroupNorm + ReLU.
- 1x1 conv to 4 output channels.
- Output shape: `(4, 64, 64)`.

### 3. Training updates in `train.py`

- New model config option (`model_type: "heatmap"` or similar).
- Heatmap MSE loss with permutation-invariant matching and active-element normalization.
- Soft-argmax coordinate extraction for validation metrics.
- Same metric pipeline (cm metrics in original space) as exp 002.
- Diagnostic logging: per-channel max logit, heatmap entropy, fraction of near-uniform maps. Logged to wandb for early collapse detection.

### 4. Sweep script

`docs/experiments/004-heatmap/sweep.py`: generates 12 run configs.

### Implementation note (current + follow-up)

- Current implementation path for this experiment generates Gaussian targets in CPU data transforms (`GaussianHeatmap` in `src/btx/data/transforms.py`).
- Follow-up plan: move Gaussian target generation to on-device JAX in the training loss path (from `batch["tgt"]`) for better throughput and cleaner objective ownership.
- For now, run the sweep with the CPU-target path, then compare throughput and metric parity against a later GPU-target implementation.

## Metric Definitions

Same as exp 002. All physical metrics computed in original image space after affine back-mapping.

- `point_err_cm`: Euclidean distance between predicted and GT endpoints (order-invariant).
- `line_err_cm`: Absolute difference in line lengths.
- `length_line_err_cm`, `width_line_err_cm`: Per-trait line errors.
- `oob_points_frac`: Fraction of target keypoints outside image bounds.

Primary metric: `val/hawaii/length_line_err_cm` (lower is better).
Guardrail: `val/hawaii/point_err_cm`.

## Testing Strategy

1. Unit test: Gaussian heatmap generation (correct center, sigma, peak value, shape).
2. Unit test: soft-argmax round-trip (generate Gaussian at known coords, soft-argmax recovers coords within tolerance).
3. Unit test: loss masking (BeetlePalooza width channels get zero loss).
4. Unit test: decoder output shape (16x16 input -> 64x64 x 4 output).
5. Unit test: UDPP coordinate round-trip (image -> heatmap -> image within tolerance).
6. Unit test: permutation-invariant loss (swapped endpoints give same or lower loss).
7. Unit test: token reshape spatial correctness (patch at grid position (i,j) corresponds to correct image region).
8. Unit test: border-coordinate reachability (x=0 and x=255 round-trip behavior).
9. Unit test: permutation + mask combined (masked width channels contribute exactly zero with permutation-invariant loss).
10. Quick smoke run: 500 steps, verify loss decreases and metrics are finite.

## Risks

1. **Heatmap MSE class imbalance.** ~99% of heatmap pixels are background (near zero). The model can achieve low loss by predicting all zeros. Mitigation: unnormalized Gaussian with peak=1.0 ensures the keypoint signal is strong. ViTPose uses this successfully.
2. **Train/val mismatch.** We optimize per-pixel MSE but evaluate coordinate error via soft-argmax. The model never receives gradient about coordinate accuracy. Mitigation: this is standard in pose estimation and works in practice. Can add coordinate auxiliary loss in a follow-up if needed.
3. **Throughput.** The conv decoder adds compute on top of the frozen backbone. Expected to be faster than exp 002's full-aug overhead since the decoder is lightweight (~700K params), but needs verification.
4. **Token reshape correctness.** DINOv3 may include extra tokens (CLS, register tokens) or use a non-obvious ordering. Assert token count equals 256 (16x16) after stripping CLS, and add a spatial sanity test (known patch position maps to correct image region).
5. **Soft-argmax instability.** Raw logits can drift in scale, making softmax very peaky or very flat. Mitigation: subtract max logit before softmax (`h -= max(h)`) for numerical stability. Already specified in coord extraction section.
6. **Sigma confounds loss scale.** Larger sigma has ~sigma^2 more positive mass in the Gaussian target, changing effective gradient magnitude across sweep conditions. A "better" sigma might just be a better-scaled optimization. Mitigation: monitor both heatmap MSE loss and coordinate-level metrics. If sigma effect is strong, consider normalizing target energy per keypoint in a follow-up.
7. **All-zero collapse.** Model can minimize loss by predicting near-zero everywhere. Mitigation: log per-channel max logit, heatmap entropy, and fraction of near-uniform maps per validation step. If max logit stays near zero after 1k steps, the model is collapsing.
