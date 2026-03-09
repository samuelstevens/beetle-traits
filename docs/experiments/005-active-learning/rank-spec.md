# rank.py: Active Learning Sample Selection

## Purpose

Given inference results from multiple training runs on unlabeled BioRepo beetles, select the most informative group images for human annotation. Balances uncertainty (heatmap entropy), diversity (K-means clustering on CLS embeddings), and annotation efficiency (greedy set cover over group images).

## Input

- Glob pattern matching one or more `*_unlabeled.parquet` files (one per training run).
- Each parquet has columns: `beetle_id`, `scientific_name`, `group_img_basename`, `img_fpath`, `mean_entropy`, `cls_embedding` (384-dim list), plus others.
- All parquets must contain the same set of `beetle_id` values (asserted).

## Algorithm

### 1. Load and validate

- Load all parquets matching the glob pattern.
- Assert identical `beetle_id` sets across all runs.
- Warn and exclude beetles with empty `group_img_basename` (these have no group image metadata in the source CSV and cannot be annotated).

### 2. Normalize entropy per run

Convert each run's `mean_entropy` to percentile ranks (0-1) within that run. This handles calibration differences between heatmap heads trained with different learning rates. A beetle at the 95th percentile in run A and the 90th percentile in run B is comparable, even if raw entropy scales differ.

### 3. Per-run clustering and priority selection

For each run independently:

1. Extract CLS embeddings (58K x 384 numpy array).
2. K-means with K clusters (default K=73, matching number of genera).
3. Within each cluster, pick proportionally: each cluster gets `max(1, round(cluster_size / total_size * K * N_avg))` priority slots, where `N_avg` is the target average picks per cluster (default 10). Larger clusters get more picks; every cluster gets at least 1.
4. Pick the highest normalized-entropy individuals to fill each cluster's quota.

**Why proportional quotas**: Fixed N per cluster overrepresents small clusters (10 out of 50 = 20%) and underrepresents large ones (10 out of 2000 = 0.5%). Proportional allocation ensures sampling density is uniform across the embedding space.

**Why per-run clustering**: Currently the DINOv2 backbone is frozen, so embeddings are identical across runs and clustering results will be the same. However, if we fine-tune the backbone in the future, embeddings will differ across runs. Per-run clustering is forward-compatible with that scenario at minimal extra cost (~2x K-means time).

### 4. Union priority sets

A beetle is a "priority individual" if it appears in the top for ANY run's clustering. This is generous inclusion: if any run's clustering puts a beetle in a high-entropy region, we consider it.

With 3 runs and K=73, the priority set size is at most 3 * sum(quotas) beetles, but typically smaller due to overlap.

### 5. Aggregate entropy: normalized min across runs

For each priority beetle, its score = min(normalized_entropy) across all runs.

**Why normalized min**: Conservative, but calibration-robust. Normalizing to percentile ranks first ensures one miscalibrated (overconfident) run can't suppress genuinely hard samples just because its raw entropy scale is lower. After normalization, min means: "this beetle is uncertain even in the run that's most confident about it." This pairs with the generous union in step 4: cast a wide net for candidates, then score conservatively.

### 6. Greedy set cover (entropy-weighted)

Select group images to annotate:

1. Initialize uncovered = all priority individuals.
2. For each candidate group image, compute score = sum of normalized-min-entropy of its uncovered priority individuals.
3. Pick the group image with the highest score.
4. Remove its priority individuals from uncovered.
5. Repeat until `--n-groups` budget is exhausted or all priority individuals are covered.

**Why entropy-weighted**: A group image containing one extremely uncertain beetle (normalized score 0.99) can outrank one with two moderately uncertain beetles (0.5 each). This ensures the greediest selections target the most informative samples, not just the most numerous.

**Why group images**: Annotators work on group images (pinned specimen photographs containing multiple beetles). Selecting at the group level maximizes annotation throughput: one image opened = all beetles in it annotated.

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `k` | 73 | Number of K-means clusters. Matches number of genera in unlabeled pool. |
| `n_avg` | 10 | Target average picks per cluster (actual picks are proportional to cluster size). |
| `n_groups` | 50 | Number of group images to select (annotation budget). |

## Output

Two CSV files, derived from a single `--out-fpath` (e.g., `results/round1`):

### `{out_fpath}_groups.csv`

One row per selected group image, ordered by selection rank.

| Column | Type | Description |
|---|---|---|
| `rank` | int | Selection order (1 = first picked) |
| `group_img_basename` | str | Group image identifier |
| `abs_group_img_path` | str | Absolute path to group image file |
| `n_priority_covered` | int | Number of priority individuals in this group |
| `entropy_covered` | float | Sum of normalized-min-entropy of priority individuals covered |
| `n_total_beetles` | int | Total beetles in this group image (not just priority) |
| `n_clusters_covered` | int | Number of distinct clusters covered by this pick |
| `cumulative_priority_covered` | int | Running total of priority individuals covered |

### `{out_fpath}_beetles.csv`

One row per priority beetle that falls in a selected group image.

| Column | Type | Description |
|---|---|---|
| `beetle_id` | str | Individual beetle identifier |
| `group_img_basename` | str | Group image this beetle belongs to |
| `abs_group_img_path` | str | Absolute path to group image file |
| `scientific_name` | str | Species name |
| `norm_min_entropy` | float | Min normalized entropy across runs (the selection score) |
| `norm_max_entropy` | float | Max normalized entropy across runs |
| `raw_min_entropy` | float | Min raw heatmap entropy across runs |
| `raw_max_entropy` | float | Max raw heatmap entropy across runs |
| `cluster_ids` | str | Comma-separated cluster IDs from each run |
| `group_rank` | int | Rank of this beetle's group image |

## Integration

- Launched via `uv run launch.py rank --sweep <sweep_file>` as a Slurm job.
- Config is a frozen dataclass parsed by tyro, consistent with `inference.py`.
- Sweep file defines the glob pattern, output path, and hyperparameters.

## Validation

- Assert all input parquets have identical `beetle_id` sets.
- Assert K <= number of unique beetles (after excluding empty group_img).
- Assert n_avg >= 1 and n_groups >= 1.
- Assert output directory exists (or create it).
- Log: number of runs loaded, number of beetles, number excluded (empty group_img), per-cluster quota distribution, priority set size, number of group images selected, cluster coverage summary.

## Edge cases (from peer review)

- **Cluster size < quota**: If a cluster has fewer beetles than its allocated quota, take all of them.
- **Entropy ties at cutoff**: Break ties by beetle_id (deterministic).
- **Single run**: Algorithm degenerates gracefully: no normalization needed (percentile of one run = percentile), min = only value, union = single set.
- **All priority beetles in one group image**: That group gets picked first; remaining budget fills from other groups. Logged as a warning.
