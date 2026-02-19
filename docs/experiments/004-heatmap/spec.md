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
| Patch tokens | Reshape 256 patch tokens into 16x16 x 384 spatial feature map. |
| Decoder | 2 transpose-conv layers (4x upsample total: 16x16 -> 64x64) + 1x1 conv to 4 heatmaps. |
| Decoder channels | Deconv1: 384 -> 256. Deconv2: 256 -> 128. Final 1x1 conv: 128 -> 4. |
| Decoder normalization | GroupNorm + ReLU between deconv layers. No activation after final 1x1 conv. |
| Deconv kernel | 4x4, stride=2, padding=1 (clean 2x upsampling, no checkerboard artifacts). |
| Heatmap resolution | 64x64 (1/4 of 256x256 input). Matches ViTPose convention. |
| Number of heatmaps | 4 independent channels (one per keypoint endpoint: width_p0, width_p1, length_p0, length_p1). |
| Gaussian target | Unnormalized Gaussian (peak=1.0) centered on ground truth keypoint in 64x64 heatmap space. |
| Sigma | Sweep variable: {1, 2, 3} in heatmap pixels (= {4, 8, 12} in 256x256 image pixels). |
| Loss | Permutation-invariant cross-entropy (CE): for each endpoint channel, normalize Gaussian target to a spatial probability distribution, compute CE against `log_softmax` of predicted logits, then do order-invariant min across endpoint assignments per line. Apply per-line `loss_mask`. |
| Coord extraction | Soft-argmax on predicted heatmaps at validation. Scale coordinates from 64x64 back to 256x256 using UDPP convention. |
| Coord convention | UDPP pixel-center alignment: `hx = (x + 0.5) / 4 - 0.5`. Inverse: `x = (hx + 0.5) * 4 - 0.5`. Current soft-argmax axis is `[0, 63]`, so exact borders (`x=0`, `x=255`) are not exactly reachable. |
| Endpoint order | Permutation-invariant loss per line: `min(loss(ch0->gt_a, ch1->gt_b), loss(ch0->gt_b, ch1->gt_a))`. No canonicalization needed. |
| Optimizer | AdamW (`optax.adamw`) on trainable decoder parameters. |
| LR sweep | {1e-3, 3e-3, 1e-2, 3e-2}. |
| Augmentation | Current sweep runs normalize-only preprocessing (`go=True`, `normalize=True`, `crop=False`) for all datasets. |
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

The mask is applied at line level (`width`, `length`) after permutation-invariant matching.

For channel CE, each target channel is normalized to a probability distribution:

`tgt_prob = tgt / max(sum(tgt), eps)`.

Then channel CE is:

`ce = -sum(tgt_prob * log_softmax(pred_logits))`.

Per sample, we average masked line losses by number of active lines (`sum(loss_mask)`), and for the batch loss we weight each sample by its active-line count so partially supervised samples do not dominate.

### Permutation-invariant loss

For each line (width or length), the two endpoints can be assigned to heatmap channels in either order. Compute loss for both assignments and take the minimum:

```
loss_direct = ce(ch_a, gt_p0) + ce(ch_b, gt_p1)
loss_swapped = ce(ch_a, gt_p1) + ce(ch_b, gt_p0)
line_loss = min(loss_direct, loss_swapped)
```

This mirrors the order-invariant matching already used in validation metrics and prevents contradictory supervision when annotation endpoint order is inconsistent.

## Soft-Argmax Coordinate Extraction

At validation, extract coordinates from predicted heatmaps:

1. For each keypoint's 64x64 heatmap, stabilize logits: `h = h - max(h)`.
2. Apply spatial softmax: `p[i,j] = exp(h[i,j]) / sum(exp(h))`.
3. Compute expected coordinates with 0-indexed heatmap axes: `hx = sum(p * x_grid)`, `hy = sum(p * y_grid)`, where `x_grid, y_grid in [0, 63]`.
4. Scale to image space using inverse UDPP: `x = (hx + 0.5) * 4 - 0.5`, `y = (hy + 0.5) * 4 - 0.5`.

Note: with this mapping, `hx=0 -> x=1.5` and `hx=63 -> x=253.5`. To reach `x=0`, the model would need `hx=-0.375`, which is outside the grid. This is the observed boundary bias from soft-argmax with the current axis choice.

These coordinates replace the MLP's direct (x, y) predictions. The existing metric pipeline (affine back-mapping to original space, order-invariant matching, cm conversion) remains unchanged.

## Sweep Design

12 runs total: 3 sigma values x 4 learning rates.

| Sigma (heatmap px) | LR=1e-3 | LR=3e-3 | LR=1e-2 | LR=3e-2 |
|---|---|---|---|---|
| 1 | run 1 | run 2 | run 3 | run 4 |
| 2 | run 5 | run 6 | run 7 | run 8 |
| 3 | run 9 | run 10 | run 11 | run 12 |

All runs use:
- Normalize-only preprocessing (no crop/flip/rotation/color jitter in this sweep script)
- Seed: deterministic per (sigma, LR) pair
- 100k steps, 12h wall time
- AdamW
- wandb tags: `exp-004-heatmap`, `ce-fix`

Comparison baseline: exp 002 full-aug results (coordinate MSE with same augmentation).

## Implementation Status

1. Heatmap target generation is objective-owned and on-device in `src/btx/objectives/heatmap.py` (`HeatmapObj.get_loss_aux` calls `make_targets` from `batch["tgt"]`).
2. Data transforms no longer provide `heatmap_tgt`; the heatmap objective asserts `heatmap_tgt` is absent.
3. Runtime objective dispatch is polymorphic: `cfg.objective.get_obj()` returns an `Obj`, and training calls `obj.get_loss_aux(...)` with no objective-type branching in `loss_and_aux`.
4. Heatmap objective diagnostics (max logit, entropy, near-uniform fraction) are logged under `heatmap/*` metric keys and namespaced in training as `train/heatmap/*` and `val/*/heatmap/*`.

## Metric Definitions

Same as exp 002. All physical metrics computed in original image space after affine back-mapping.

- `point_err_cm`: Euclidean distance between predicted and GT endpoints (order-invariant).
- `line_err_cm`: Absolute difference in line lengths.
- `length_line_err_cm`, `width_line_err_cm`: Per-trait line errors.
- `oob_points_frac`: Fraction of target keypoints outside image bounds.

Primary metric: `val/hawaii/length_line_err_cm` (lower is better).
Guardrail: `val/hawaii/point_err_cm`.

## Testing Strategy

1. Unit tests for UDP coordinate transforms (image->heatmap->image and heatmap->image->heatmap round-trips).
2. Unit tests for target generation shape/channel ordering and Gaussian peak behavior.
3. Unit tests for CE objective properties: permutation invariance, masking, all-masked behavior, far-OOB behavior, and finite gradients.
4. Hypothesis-based randomized tests validating permutation invariance and agreement with a manual CE formula.
5. Unit tests for objective dispatch (`cfg.get_obj()`) and loss-path contract (`heatmap_tgt` absent from data pipeline).
6. Unit tests for train-step masking/aggregation behavior and order-invariant endpoint matching in metrics.

## Risks

1. **Train/val objective mismatch.** We optimize CE over spatial maps but evaluate geometric coordinate error after soft-argmax. Mitigation: track both objective loss and cm metrics; add coordinate-side auxiliary loss only if needed.
2. **Soft-argmax boundary bias.** Current axis `[0, 63]` cannot represent exact image borders under UDP inverse. Mitigation: accept as known tradeoff for now and monitor edge-heavy subsets.
3. **Far-OOB targets can become effectively zero-mass.** For sufficiently far centers, Gaussian channels may numerically underflow to all zeros, yielding zero CE and zero gradient for that channel. Mitigation: accepted behavior for far-OOB cases; monitor OOB frequency.
4. **Heatmap collapse remains possible.** Uniform/low-contrast logits can hurt decode quality even when loss appears stable. Mitigation: log per-channel max logit, entropy, and near-uniform fraction each validation cycle.
5. **Throughput.** Decoder and objective computations add overhead; verify wall-time and throughput against baseline runs.
6. **Sigma confounds optimization behavior.** Sigma changes target entropy and gradient distribution. Mitigation: compare coordinate metrics, not just objective loss, across sigma settings.
