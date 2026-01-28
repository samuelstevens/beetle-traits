# 001 - Extra data (BeetlePalooza) vs Hawaii-only

Goal: measure whether adding BeetlePalooza training data improves validation on the Hawaii val split, with a focus on elytra length.

Hypothesis: length-only extra data helps reduce Hawaii validation error for elytra length without hurting overall validation loss.

Model: frozen DINOv3 checkpoint with the MLP head. This is the default in `train.Config` and `train.py` already freezes the ViT and trains only the head.

Data and evaluation notes:
- Validation is always on Hawaii val (`train.py` builds the val loader from `hawaii` only).
- Hawaii points are ordered as [width, length]. Length is line index 1.
- BeetlePalooza width is masked in the dataset (`loss_mask = [0.0, 1.0]`), so extra data is length-only by default.
- `hawaii.include_polylines` must be False for now (`hawaii.py` raises NotImplementedError otherwise).
- Current wandb metrics aggregate width and length together. For length-only comparisons, we should compute per-line metrics in a notebook later (see `notebooks/`).

Sweep file:
- `sweep.py`: six runs (three seeds x two data configurations).

How to run:
- Update the paths at the top of `sweep.py` (Hawaii/BeetlePalooza roots, annotations, DINOv3 checkpoint).
- Use `--sweep` with `launch.py`.
- Add `--slurm-acct` and `--slurm-partition` if you want submitit to use Slurm.
- If you need shorter runs, add `--n-steps`, `--val-every`, and `--save-every` overrides.

Example:

```sh
uv run launch.py \
  --sweep docs/experiments/001-extra-data/sweep.py \
  model:frozen
```

Outputs to compare:
- `val/loss`, `val/line_err_cm`, `val/median_line_err_cm`, `val/max_line_err_cm` on Hawaii val.
- For length-only conclusions, use a notebook to compute line index 1 metrics explicitly.
