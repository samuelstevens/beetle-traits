# Data Augmentation and Normalization Spec

GitHub issue: [#9](https://github.com/samuelstevens/beetle-traits/issues/9)

## Motivation

The current pipeline applies `DecodeRGB -> Resize(256)` and then trains a head on frozen DINOv3 features. We want two changes:

1. Add input normalization expected by DINOv3.
2. Add train-time augmentation to reduce overfitting.

The highest-risk failure mode is wrong physical metrics (cm) due to coordinate-space bugs under crop/flip/rotation. This spec makes metric correctness explicit and testable.

## Final Decisions

| Topic | Decision |
|---|---|
| Input size | Fixed `256x256` for now. |
| Train pipeline | `DecodeRGB -> RandomResizedCrop -> RandomFlip -> RandomRotation -> ColorJitter -> Normalize`. |
| Eval pipeline | `DecodeRGB -> Resize(256) -> Normalize` (no augmentation). |
| Normalization | ImageNet mean/std: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`. |
| Crop params | `scale=(0.5, 1.0)`, `ratio=(3/4, 4/3)` (shape distortion allowed). |
| Rotation sampling | With probability `rotation_prob`, sample angle uniformly from `[0, 360)`; otherwise no rotation. |
| OOB supervision policy | Configurable literal with options `"mask_any_oob"`, `"mask_all_oob"`, `"supervise_oob"`. Default: `"supervise_oob"`. |
| OOB coordinates | No clipping. |
| OOB logging | Log `oob_points_frac` for each batch. |
| Metric coordinate contract | Keep raw annotations in original space and track a full affine transform per sample. Compute cm metrics in original space only. |
| Point-wise metric matching | Order-invariant endpoint matching per line (`direct` vs `swapped`, take lower total endpoint error). |
| Pixel metrics | Remove pixel-space metrics from reported training/validation metrics. |
| Missing scalebar | Keep sample for training loss, but exclude from cm metrics in both train and validation via `scalebar_valid=False`. |
| Scalebar physical length | Assume scalebar length is always `1.0 cm`. |
| Acceptance checks | Use fixed-seed thresholds (defined below) on `2000`-step runs before any `2 hour` run. |

## Data Contract

Per-sample fields after dataset transforms:

- `points_px`: raw endpoints in original image space, shape `(2, 2, 2)`. Immutable.
- `scalebar_px`: raw scalebar endpoints in original image space, shape `(2, 2)`. Immutable.
- `scalebar_valid`: scalar/boolean mask indicating whether cm metrics are valid for this sample.
- `tgt`: transformed endpoints in augmented image space (model target), shape `(2, 2, 2)`.
- `t_aug_from_orig`: `3x3` affine matrix in homogeneous coordinates.
- `t_orig_from_aug`: inverse of `t_aug_from_orig` (`np.linalg.inv`).
- `loss_mask`: per-line supervision mask, shape `(2,)`, still used for width/length training availability.

`tgt` derivation timing:

1. Spatial transforms do not mutate `points_px`.
2. Each spatial transform composes its affine into a running `t_aug_from_orig`.
3. `tgt` is materialized once, after the full spatial transform chain, by applying the final composed affine to `points_px`.

Invariant assertions:

1. `t_orig_from_aug @ t_aug_from_orig` is close to identity.
2. `tgt` equals applying `t_aug_from_orig` to `points_px` (within tolerance).
3. All matrix values are finite.
4. In-bounds convention is `0 <= x < 256` and `0 <= y < 256` when computing OOB counters.
5. If `px_per_cm <= min_px_per_cm` or non-finite, set `scalebar_valid=False`.

## Affine Composition Rules

We compose geometric transforms into one matrix instead of storing partial metadata like only `scale_x/scale_y`.

Use homogeneous point vectors `[x, y, 1]^T`.

### Crop + Resize

For crop origin `(x0, y0)`, crop size `(crop_w, crop_h)`, output `s=256`:

- `sx = s / crop_w`
- `sy = s / crop_h`
- matrix:

```text
[ sx   0   -sx*x0 ]
[  0  sy   -sy*y0 ]
[  0   0      1   ]
```

### Horizontal Flip

On `s=256` with in-bounds `[0, s)`:

- `x' = (s - 1) - x`
- matrix:

```text
[ -1   0  s-1 ]
[  0   1   0  ]
[  0   0   1  ]
```

### Vertical Flip

- `y' = (s - 1) - y`

```text
[ 1   0   0   ]
[ 0  -1  s-1  ]
[ 0   0   1   ]
```

### Rotation by angle theta (continuous)

Center of rotation is `(c, c)` where `c = (s - 1) / 2`.

```text
[ cos(t)  -sin(t)  c*(1-cos(t)) + c*sin(t) ]
[ sin(t)   cos(t)  c*(1-cos(t)) - c*sin(t) ]
[   0        0              1               ]
```

Sampling rule: if `rng.random() < rotation_prob`, sample `theta` uniformly from `[0, 360)`; else `theta = 0`.

## AugmentConfig

```python
@beartype.beartype
@dataclasses.dataclass(frozen=True)
class AugmentConfig:
    go: bool = True
    size: int = 256

    crop_scale_min: float = 0.5
    crop_scale_max: float = 1.0
    crop_ratio_min: float = 0.75
    crop_ratio_max: float = 1.333

    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotation_prob: float = 0.75
    """Probability of applying a non-identity rotation (continuous [0, 360))."""

    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1

    oob_policy: tp.Literal["mask_any_oob", "mask_all_oob", "supervise_oob"] = "supervise_oob"
    min_px_per_cm: float = 1e-6
    """If px_per_cm <= min_px_per_cm, cm metrics are masked out (scalebar_valid=False)."""
```

Notes:

- `size` is fixed at `256` for this experiment and should be asserted.
- `hflip_prob` and `vflip_prob` are sampled independently.
- With default `oob_policy="supervise_oob"`, `loss_mask` is not modified by OOB status.

## Metric Definitions

Physical metrics are computed in original image coordinates only.

Given predictions `pred_aug` and targets `tgt` in augmented space:

1. Map predictions to original space:
   - `pred_orig = apply_affine(t_orig_from_aug, pred_aug)`
2. Ground truth in original space is `points_px`.
3. Per-sample `px_per_cm`:
   - `px_per_cm = ||scalebar_px[1] - scalebar_px[0]||` (scalebar is `1 cm`).
4. Validate cm metric denominator:
   - if `px_per_cm` is non-finite or `px_per_cm <= min_px_per_cm`, set `scalebar_valid=False`.
5. Convert point/line errors:
   - `err_cm = err_px_orig / px_per_cm` only when `scalebar_valid=True`.
6. If `scalebar_valid == False`, cm metrics for that sample are set to `nan` and excluded from aggregate means/medians.

Point metric endpoint matching is order-invariant per line:

- Compute direct pairing cost and swapped pairing cost.
- Use the lower-cost pairing for point-wise error reporting.

## Missing Scalebar Policy

If a sample has no valid scalebar annotation, or a degenerate/invalid scalebar (`px_per_cm <= min_px_per_cm`):

1. Keep the sample in training.
2. Keep training loss active on keypoints (subject to `loss_mask` from dataset).
3. Set `scalebar_valid=False` so cm metrics are excluded in train and validation.

## Logging

Add `oob_points_frac` to batch metrics:

- numerator: count of transformed endpoints outside `[0, 256)` in either x or y.
- denominator: total number of endpoints in batch.

This is diagnostic only; it does not alter supervision with default `supervise_oob`.

## Implementation Plan

1. Add `Normalize` transform in `src/btx/data/utils.py`.
2. Add spatial/color augmentation transforms in `src/btx/data/transforms.py`.
3. Track `t_aug_from_orig`/`t_orig_from_aug` through spatial transforms.
4. Compose per-transform affines through the spatial chain and then build `tgt` once by applying the final `t_aug_from_orig` to immutable `points_px`.
5. Update `make_dataset` in `train.py` to support train/eval pipeline split with `AugmentConfig`.
6. Update metrics in `train.py`:
   - remove pixel-space metrics from reported outputs,
   - compute point/line cm metrics in original space,
   - apply `scalebar_valid`,
   - use order-invariant matching for point metrics.
7. Add `oob_points_frac` logging.

## Testing Strategy

1. Unit tests for affine math:
   - crop, flips, 90-degree rotations, composition, inverse round-trip.
2. Unit tests for metric correctness:
   - cm metrics unchanged by augmentations when predictions equal transformed targets.
3. Unit tests for order-invariant endpoint matching.
4. Unit tests for `scalebar_valid` behavior on missing scalebar samples.
5. Visual sanity check notebook for transformed image + transformed target overlay.
6. Training comparisons:
   - baseline,
   - normalization-only,
   - full augmentation,
   with `2000`-step quick runs and optional `2 hour` run.

## Acceptance Criteria

Primary metric: `val/line_err_cm` (lower is better).
Guardrail metric: `val/point_err_cm`.

Quick runs:

1. Run baseline, normalization-only, and full augmentation for `2000` steps using fixed seeds `[17, 23, 47]`.
2. At step `2000`, compute the mean across seeds for `val/line_err_cm` and `val/point_err_cm`.
3. Pass/fail thresholds:
   - Normalization-only must satisfy `mean(val/line_err_cm)_norm <= 1.02 * mean(val/line_err_cm)_base`.
   - Full augmentation is eligible for `2 hour` run only if:
     - `mean(val/line_err_cm)_aug <= 0.98 * mean(val/line_err_cm)_norm`, and
     - `mean(val/point_err_cm)_aug <= 1.02 * mean(val/point_err_cm)_norm`.

Two-hour run:

1. Run normalization-only and full augmentation for up to `2 hours` (same seed/hardware budget).
2. Promote full augmentation only if final validation metrics satisfy:
   - `val/line_err_cm_aug <= 0.98 * val/line_err_cm_norm`, and
   - `val/point_err_cm_aug <= 1.02 * val/point_err_cm_norm`.
