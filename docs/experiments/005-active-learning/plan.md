# Experiment 005: Iterative Active Learning Pipeline

## Context

We just merged Gaussian heatmap regression (PR #14, experiment 004). The model trains on Hawaii + BeetlePalooza and evaluates on BioRepo (Arizona beetles). The next step is an iterative annotation cycle: train model, evaluate, identify failure modes, select informative samples for annotation, annotate, retrain. This plan implements the minimal infrastructure to run that loop practically.

Currently there is no way to: (a) save/load trained model checkpoints, (b) run inference on a dataset and collect per-sample results, or (c) systematically identify which samples to annotate next.

## Active Learning Strategy

### Core Loop

```
train model on L
  -> run inference on L + U (unlabeled pool)
  -> identify failure strata (per-species, per-group-image)
  -> select next annotation batch (uncertain + diverse)
  -> annotators correct model predictions (SLEAP-style pre-label + correct)
  -> merge corrections into L, retrain
```

### Acquisition: Uncertainty + Diversity

We already have the signals we need:

- **Uncertainty**: heatmap entropy from `get_diagnostics()` in `src/btx/objectives/heatmap.py`. High entropy = model is unsure where the keypoint is. Mean entropy across 4 channels gives a per-sample uncertainty score.
- **Diversity**: DINOv2 CLS token (384-dim) from frozen backbone. K-means clustering on CLS embeddings, then pick the most uncertain sample per cluster. This avoids redundant near-duplicate selections.
- **Error-stratified**: group errors by `scientific_name` and `group_img_basename` to find systematic failure modes (bad species, bad group images).

Don't use: sophisticated deep AL methods (BADGE, VAAL, LL4AL). Benchmarks show they frequently underperform random selection and add complexity.

### Domain Shift Consideration

Model trained on Hawaiian beetles may be confidently wrong on Arizona (BioRepo) beetles. Heatmap entropy alone is unreliable under domain shift. Complement with:
- **Domain distance**: cosine distance of CLS embedding from training data centroid. High distance = more OOD.
- **TTA variance** (future): run augmented copies, measure prediction variance.

### Stopping Criterion

Stop when per-stratum `line_err_cm` on BioRepo is within inter-annotator disagreement. Need a human baseline: have 2-3 annotators label the same 30-50 images and compute disagreement distribution.

### Key References

- [Settles - Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf) (canonical; pool-based sampling, batch-mode selection)
- [Thaler et al. - Uncertainty Estimation for Heatmap-Based Landmark Localization](https://arxiv.org/abs/2203.02351) (directly applicable; quantile binning for heatmap predictions)
- DeepLabCut protocol (Nath et al. 2019): iterative refinement using confidence-derived failure cases
- [SLEAP docs](https://docs.sleap.ai/dev/learnings/prediction-assisted-labeling/): label -> train -> predict -> correct -> repeat

## Implementation Plan

### 1. Add checkpoint saving to training

**File:** `train.py`

Add checkpoint saving with three triggers:
- **Periodic**: every N validation steps (configurable, default every 10K steps), save `model_step{step}.eqx`
- **Best validation**: whenever validation loss improves, save `model_best.eqx` (overwrite previous best)
- **End of training**: save `model_final.eqx`

Save to `logs/<wandb_run_id>/`. Use `eqx.tree_serialise_leaves` following the existing pattern in `src/btx/modeling/dinov3.py:519`.

Also add a `load_model(cfg, ckpt_fpath)` function that reconstructs a model from config + checkpoint file using `eqx.tree_deserialise_leaves`.

### 2. Add `extract_features` to the heatmap model

**File:** `src/btx/modeling/heatmap.py`

Add a method `extract_features(x_hwc) -> (logits_chw, cls_d)` that returns both heatmap logits and the DINOv2 CLS token (384-dim for ViT-S16). Reuses the `__call__` logic but also captures `vit_out["cls"]`. Keep `__call__` unchanged so training code is unaffected.

### 3. Create inference script

**New file:** `inference.py`

A standalone script that:
1. Loads a trained model checkpoint via `load_ckpt`
2. Builds a dataset (hawaii, beetlepalooza, biorepo) with eval transforms (no augmentation)
3. Runs the model over every sample, collecting per-sample: predictions, errors (point_err_cm, line_err_cm), heatmap diagnostics (entropy, max_logit, near_uniform), CLS embeddings, and metadata
4. Saves results as a Parquet file via Polars

Reuses `loss_and_aux` from `train.py` for metrics consistency. Uses `extract_features` for CLS tokens. Config is a frozen dataclass parsed by tyro.

Output schema (one row per sample):

| Column | Type | Source |
|---|---|---|
| `beetle_id` | str | metadata |
| `scientific_name` | str | metadata |
| `group_img_basename` | str | metadata |
| `img_fpath` | str | metadata |
| `dataset` | str | which dataset |
| `sample_loss` | f32 | TrainAux.sample_loss |
| `width_line_err_cm` | f32 | TrainAux |
| `length_line_err_cm` | f32 | TrainAux |
| `mean_entropy` | f32 | mean heatmap entropy across 4 channels |
| `pred_coords_px` | list[f32] | flat 8-element predicted coords |
| `gt_coords_px` | list[f32] | flat 8-element GT coords |
| `cls_embedding` | list[f32] | 384-element CLS token |

### 4. Create analysis notebook

**New file:** `notebooks/active_learning.py` (marimo)

Loads inference Parquet and produces:

1. **Per-species error table**: group by species, show median/mean/max line_err_cm. Sort by median error. Flag species above a configurable threshold.
2. **Uncertainty vs error scatter**: mean_entropy vs length_line_err_cm. Shows whether uncertainty predicts error.
3. **Embedding visualization**: UMAP of CLS embeddings colored by (a) species, (b) error magnitude, (c) dataset source. Reveals whether failures cluster in feature space and whether the domain gap is visible.
4. **Sample selection**: combine uncertainty (top-K by mean_entropy) + diversity (K-means on CLS embeddings, pick most uncertain per cluster). Output a ranked list of beetle_ids to annotate next, saved to CSV.
5. **Per-group-image analysis**: flag group images where all beetles have high error (annotation quality issue vs model failure).

### 5. Create annotation export script

**New file:** `scripts/export_for_annotation.py`

Given inference Parquet + a list of beetle_ids (or top-K selection criteria):
1. Load original images
2. Overlay model predictions using the `plot_preds` pattern from `train.py:412-469`
3. Save annotated images + a CSV manifest (beetle_id, img_fpath, species, pred_coords, confidence)

Output structure:
```
logs/annotation_export/
  manifest.csv
  images/
    beetle_A001.png  # original image with predictions overlaid
    ...
```

Annotators view exported images, correct endpoint positions, record corrections in CSV. This is the SLEAP-style "predict then correct" loop.

### 6. Create annotation import script

**New file:** `scripts/import_corrections.py`

Reads a corrections CSV (beetle_id, corrected endpoint coords), validates coordinates are within image bounds (assertion-based), merges corrections into the dataset's `annotations.json`. Reports how many samples were added/corrected and mean correction magnitude.

## Sequencing

All 6 steps, implemented incrementally. Each step compiles and passes tests before moving to the next.

1. Checkpoint save/load -- periodic + best + end
2. `extract_features` method + test
3. Inference script
4. Analysis notebook
5. Export script
6. Import script

## What We're NOT Building

- No automated retraining pipeline (manual loop, 3-5 iterations)
- No web UI for annotation (exported images + CSV)
- No complex acquisition functions (just entropy + K-means diversity)
- No wandb artifact tracking (local Parquet files)
- No checkpoint manager beyond periodic/best/end saves

## Verification

- Unit test: `extract_features` produces identical logits as `__call__`, CLS shape is `(embed_dim,)`
- Unit test: checkpoint round-trip (save then load, verify predictions match)
- Integration test: run `inference.py` on a tiny synthetic dataset, verify Parquet output schema
- Manual: run inference on a real checkpoint, open analysis notebook, verify plots render
- Manual: export annotations for a few samples, verify images have predictions overlaid

## Key Files

| File | Action |
|------|--------|
| `train.py` | Add checkpoint saving (periodic, best, end) + `load_model` |
| `src/btx/modeling/heatmap.py` | Add `extract_features` method |
| `inference.py` | New: inference script |
| `notebooks/active_learning.py` | New: marimo analysis notebook |
| `scripts/export_for_annotation.py` | New: annotation export |
| `scripts/import_corrections.py` | New: annotation import |
| `tests/test_modeling_heatmap.py` | Add `extract_features` test |
| `tests/test_inference.py` | Add e2e test |
