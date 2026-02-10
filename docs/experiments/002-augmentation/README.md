# Findings

## 2026-02-09: full-aug runs are dataloader-bound at `n_workers=4`

Run: Slurm array `3618837` (9 quick runs).

Observed:
- `full-aug` jobs (`2`, `5`, `8`) were consistently the slowest.
- `full-aug` throughput was about `85-92s` per 200 train steps.
- `baseline` throughput was about `39-48s` per 200 train steps.
- `norm-only` throughput was about `43-61s` per 200 train steps.

Evidence that this is augmentation cost (not just node variance):
- On node `a0306`: `norm-only` job `1` took `00:09:31`, `full-aug` job `2` took `00:16:35`.
- On node `a0337`: `baseline` job `6` took `00:09:27`, `full-aug` job `5` took `00:16:12`.

Interpretation:
- The train-time random spatial/color transforms in `full-aug` add substantial CPU-side dataloader overhead relative to resize-only pipelines.

Action item:
- Rerun `exp-002` with higher dataloader worker count (increase `n_workers` above `4`; start with `8`) and compare throughput.
