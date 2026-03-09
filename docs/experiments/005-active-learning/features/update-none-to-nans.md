# Run inference on unlabeled BioRepo samples

## Context

We have ~58K unlabeled BioRepo beetle images (accessible via `BioRepoConfig(split="unlabeled")`). We need to run inference on them to get CLS embeddings, predictions, and heatmap entropy for active learning. Currently the unlabeled `__getitem__` returns `None` for annotation fields, which crashes `InitAugState` transforms (shape assertions on lines 236-244 of `transforms.py`).

## Approach: NaN/zero arrays instead of None

Change `biorepo.py`'s unlabeled `__getitem__` to return properly-shaped arrays:
- `points_px`: `np.full((2, 2, 2), np.nan)`
- `scalebar_px`: `np.full((2, 2), np.nan)`
- `loss_mask`: `np.zeros(2)` (zero = skip both measurements)
- `scalebar_valid`: `np.bool_(False)`

The entire pipeline (transforms, dataloader, forward_batch) works unchanged. `loss_mask=[0, 0]` causes all error metrics to come out NaN -- correct since there's no ground truth. CLS embeddings, predictions, and entropy remain valid.

## Changes

### 1. `src/btx/data/biorepo.py` -- unlabeled __getitem__ (line ~207)
Replace `None` values with shaped NaN/zero arrays.

### 2. `src/btx/data/utils.py` -- revert optional fields (lines 32-39)
Change `Float[...] | None` back to `Float[...]` for all four fields. None is no longer needed since we use NaN/zero sentinels.

### 3. `tests/test_splits.py` -- update unlabeled test (line ~115)
Change assertions from `is None` to shape/value checks (NaN for points, zero for loss_mask, etc.).

### 4. `docs/experiments/005-active-learning/sweeps/inference.py` -- add unlabeled
Add a config per run_id that includes BioRepo `split="unlabeled"`. Output to `{run_id}_unlabeled.parquet` (separate file since these have no useful error metrics).

### 5. `docs/experiments/005-active-learning/notebook.py` -- update demo
Update `demo_unlabeled_biorepo_dataset()` to check NaN arrays instead of None.

## Open question: inference Config shape

The current `inference.Config` has one field per dataset (`hawaii`, `beetlepalooza`, `biorepo`). To run inference on unlabeled BioRepo alongside labeled data, we need a way to specify a second BioRepo config. Options being considered:

1. Add `biorepo_unlabeled: BioRepoConfig` field (clunky but simple)
2. Change `biorepo` to `list[BioRepoConfig]` (may have tyro issues with lists of dataclasses)
3. Add a generic `unlabeled: Union[BioRepoConfig, ...]` field
4. Use the sweep mechanism (separate sweep configs for labeled vs unlabeled runs)

## Verification

1. `uv run python -m pytest tests/test_splits.py -k "not slow" -v`
2. `uv run python -m pytest -k "not slow"` (all existing tests still pass)
3. Launch inference sweep, verify unlabeled Parquet has valid embeddings + NaN metrics
