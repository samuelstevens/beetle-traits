# Validation Visualizations for Active Learning Selection

## 1. Entropy calibration: actual heatmaps at different entropy levels

Show unlabeled beetle images at entropy percentiles (p5, p25, p50, p75, p95, p99) with the **real model heatmap** overlaid. This requires running the model forward pass on selected images to get the 4-channel logits, converting to probabilities, and overlaying the max-channel heatmap with the magma colormap (same as demo.py).

**What it catches**: Whether high entropy actually means the model is confused (diffuse, multi-modal heatmaps) vs. noise or artifacts. If p99 images look obviously easy, the entropy signal is garbage and everything downstream is wrong.

**Requires**: Loading a checkpoint and running ~24 forward passes. Too heavy for the notebook on a login node (1GB RSS limit). Should be a small script or a Slurm job that saves the figures to disk, then the notebook just displays them.

## 2. Cross-run entropy correlation

Three pairwise scatter plots: run A's normalized entropy vs run B's for all 58K beetles. Points should cluster along the diagonal if runs agree.

**What it catches**: If one run is decorrelated (scattered off-diagonal), our normalize-then-min aggregation is dominated by that run's noise. Would explain why certain beetles get low min-scores despite being uncertain in 2 of 3 runs.

**Requires**: Only the existing parquets. Lightweight, can run in notebook.

## 3. Embedding PCA with selection overlay

2D PCA of all 58K embeddings. Color points: gray = not priority, orange = priority but not in a selected group, red = in a selected group.

**What it catches**: Whether diversity is working. If red dots are clumped in one region, the greedy loop is chasing a single genus's entropy and the clustering isn't providing diversity. Red should be spread across the space.

**Requires**: Existing parquets + the round1 CSVs. Lightweight.

## 4. Cluster coverage bar chart

For each of the 73 clusters: what fraction of that cluster's priority beetles ended up in selected groups? Should be fairly uniform (10-30% per cluster).

**What it catches**: Whether the diversity mechanism actually spreads selections across clusters. If a few clusters have 100% coverage and most have 0%, the greedy loop is concentrating despite our mechanism.

**Requires**: Need to re-run K-means (or save cluster assignments in rank.py output). Could add cluster labels to the beetles CSV.

## 5. Group size vs. priority density scatter

x = total beetles in group, y = n_priority / n_total (priority density). Highlight selected groups. Optionally compare alpha=0.8 vs alpha=1.0.

**What it catches**: Whether cost_alpha is doing its job. Selected groups should have high priority density (upper region), not just be the biggest groups (right side). Without cost_alpha, we'd see selections clustering on the right (large groups).

**Requires**: The round1 CSVs. Lightweight.
