# Validation Visualizations for Active Learning Selection

## 1. Entropy calibration: actual heatmaps at different entropy levels

Show unlabeled beetle images at entropy percentiles (p5, p25, p50, p75, p95, p99) with the **real model heatmap** overlaid. This requires running the model forward pass on selected images to get the 4-channel logits, converting to probabilities, and overlaying the max-channel heatmap with the magma colormap (same as demo.py).

**What it catches**: Whether high entropy actually means the model is confused (diffuse, multi-modal heatmaps) vs. noise or artifacts. If p99 images look obviously easy, the entropy signal is garbage and everything downstream is wrong.

**Requires**: Loading a checkpoint and running ~24 forward passes. Too heavy for the notebook on a login node (1GB RSS limit). Should be a small script or a Slurm job that saves the figures to disk, then the notebook just displays them.

**Pipeline change**: New script `visualize_heatmaps.py` (or a `--save_heatmaps` flag on inference.py). Loads one checkpoint, picks ~24 images at entropy percentiles from the parquet, runs forward passes, overlays the max-channel heatmap with magma colormap, saves PNGs. Runs as a Slurm job; notebook just displays the saved images. **Complexity: medium.** Need to extract the heatmap rendering logic from demo.py into something reusable, wire up image selection from parquet entropy percentiles, and add a sweep/Slurm entry point.

## 2. Cross-run entropy correlation

Three pairwise scatter plots: run A's normalized entropy vs run B's for all 58K beetles. Points should cluster along the diagonal if runs agree.

**What it catches**: If one run is decorrelated (scattered off-diagonal), our normalize-then-min aggregation is dominated by that run's noise. Would explain why certain beetles get low min-scores despite being uncertain in 2 of 3 runs.

**Requires**: Only the existing parquets. Lightweight, can run in notebook.

**Pipeline change**: None needed. Already implemented in notebook.

## 3. Embedding UMAP with selection overlay

2D UMAP of all 58K embeddings. Color points in a 2x2 grid: priority yes/no x in-selected-group yes/no. Gray = neither, orange = priority but not selected, blue = selected but not priority (free riders), red = priority and selected.

**What it catches**: Whether diversity is working. If red dots are clumped in one region, the greedy loop is chasing a single genus's entropy and the clustering isn't providing diversity. Red should be spread across the space. Blue clumps reveal free riders -- non-priority beetles dragged into annotation because they share a group image with a priority beetle.

**Requires**: The exact priority set and cluster labels. Currently the notebook uses a proxy (top 1857 by min-entropy), which doesn't match rank.py's per-cluster top-N selection. UMAP itself is fine in the notebook (~35s for 10K points).

**Pipeline change**: rank.py should output `{out_fpath}_all_unlabeled.csv` with per-beetle metadata for ALL unlabeled beetles: `beetle_id`, `group_img_basename`, `is_priority`, `min_norm_entropy`, `cluster_ids`. This replaces the proxy with the exact priority set. **Complexity: low.** The data is already computed in rank.py; just needs to be written out as an additional CSV.

## 4. Cluster coverage bar chart

For each of the 73 clusters: what fraction of that cluster's priority beetles ended up in selected groups? Should be fairly uniform (10-30% per cluster).

**What it catches**: Whether the diversity mechanism actually spreads selections across clusters. If a few clusters have 100% coverage and most have 0%, the greedy loop is concentrating despite our mechanism.

**Requires**: K-means cluster labels per beetle. Running K-means (58K x 384) in the notebook crashes the login node (1GB RSS limit). rank.py already runs K-means but doesn't save the per-beetle labels.

**Pipeline change**: Same as #3 -- the `_all_unlabeled.csv` from rank.py includes `cluster_ids`. Once that file exists, the notebook just reads it and groups by cluster. **Complexity: low** (bundled with #3).

## 5. Group size vs. priority density scatter

x = total beetles in group, y = n_priority / n_total (priority density). Highlight selected groups. Optionally compare alpha=0.8 vs alpha=1.0.

**What it catches**: Whether cost_alpha is doing its job. Selected groups should have high priority density (upper region), not just be the biggest groups (right side). Without cost_alpha, we'd see selections clustering on the right (large groups).

**Requires**: The round1 CSVs. Lightweight, can run in notebook.

**Pipeline change**: None needed. The groups CSV already has `n_priority_covered` and `n_total_beetles`. The notebook just needs to compute the ratio and plot.

## Summary of pipeline changes

| Change | Fixes | Complexity |
|---|---|---|
| rank.py: add `_all_unlabeled.csv` output | #3, #4 | Low |
| New script: `visualize_heatmaps.py` | #1 | Medium |
| (none needed) | #2, #5 | - |
