# rank.py
"""Active learning sample selection: pick the most informative group images for annotation.

Balances uncertainty (heatmap entropy), diversity (K-means on CLS embeddings), and annotation efficiency (greedy set cover over group images). See docs/experiments/005-active-learning/rank-spec.md for the full algorithm specification.

Run with:
    uv run launch.py rank --sweep docs/experiments/005-active-learning/sweeps/rank.py
"""

import dataclasses
import glob
import logging
import pathlib
import typing as tp

import beartype
import numpy as np
import polars as pl
import tyro
from scipy.stats import rankdata
from sklearn.cluster import KMeans

import btx.configs

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("rank")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    glob_pattern: str = ""
    """Glob pattern matching one or more *_unlabeled.parquet files."""
    out_fpath: str = "results/round1"
    """Output path prefix. Produces {out_fpath}_groups.csv, {out_fpath}_beetles.csv, and {out_fpath}_all_unlabeled.csv."""
    k: int = 73
    """Number of K-means clusters (default matches number of genera)."""
    n_avg: int = 10
    """Target average picks per cluster (actual picks proportional to cluster size)."""
    n_groups: int = 50
    """Number of group images to select (annotation budget)."""
    cost_alpha: float = 1.0
    """Cost exponent for greedy selection. 0 = ignore group size (raw entropy sum), 1 = fully normalize by group size. Interpolates between the two."""
    slurm_acct: str = ""
    """Slurm account. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 0.5
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to write Slurm logs."""


REQUIRED_COLS = {
    "beetle_id",
    "scientific_name",
    "group_img_basename",
    "img_fpath",
    "mean_entropy",
    "cls_embedding",
}


@beartype.beartype
def load_runs(pattern: str) -> list[pl.DataFrame]:
    """Load all parquets matching glob pattern. Validates identical row sets across runs."""
    # SAM: can we add a schema to this function so that readers know what's in the dataframes without running the code? See the schema in inference.py? Something like that?
    fpaths = sorted(glob.glob(pattern))
    assert fpaths, f"No files match glob pattern: '{pattern}'"
    logger.info("Found %d parquet files matching '%s'.", len(fpaths), pattern)

    runs = []
    for fpath in fpaths:
        df = pl.read_parquet(fpath)
        missing = REQUIRED_COLS - set(df.columns)
        assert not missing, f"Missing columns in {fpath}: {missing}"
        runs.append(df)
        logger.info("  %s: %d rows", fpath, len(df))

    # All runs must have identical beetle_id sequences (same dataset, same order).
    ref_ids = runs[0]["beetle_id"].to_list()
    for i, df in enumerate(runs[1:], 1):
        assert df["beetle_id"].to_list() == ref_ids, (
            f"Run {i} has different beetle_id sequence than run 0."
        )

    return runs


@beartype.beartype
def normalize_entropy(runs: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Convert each run's mean_entropy to percentile ranks (0-1)."""
    out = []
    for df in runs:
        entropy = df["mean_entropy"].to_numpy()
        # rankdata gives 1-based ranks; divide by n to get percentile in (0, 1].
        pct = rankdata(entropy, method="average") / len(entropy)
        out.append(df.with_columns(pl.Series("norm_entropy", pct, dtype=pl.Float64)))
    return out


@beartype.beartype
def select_priority(
    runs: list[pl.DataFrame], *, k: int, n_avg: int
) -> tuple[set[int], list[np.ndarray]]:
    """Per-run K-means clustering with proportional quotas. Returns union of priority row indices and per-run cluster labels."""
    n = len(runs[0])
    assert k <= n, f"k={k} exceeds number of beetles ({n})."
    assert n_avg >= 1, f"n_avg must be >= 1, got {n_avg}."

    priority_i: set[int] = set()
    all_labels: list[np.ndarray] = []

    for r, df in enumerate(runs):
        emb = np.array(df["cls_embedding"].to_list(), dtype=np.float32)
        # SAM: what about when we use something other than DINOv3 ViT-S? We'll have 768, 1024, etc for the embeddings.
        assert emb.shape == (n, 384), f"Run {r}: expected ({n}, 384), got {emb.shape}"

        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(emb)
        all_labels.append(labels)

        norm_ent = df["norm_entropy"].to_numpy()

        # Proportional quotas per cluster.
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        run_picks = 0
        for cid, csize in zip(cluster_ids, cluster_sizes):
            quota = max(1, round(csize / n * k * n_avg))
            mask = labels == cid
            indices = np.where(mask)[0]
            # Sort by norm_entropy descending, break ties by index (deterministic).
            order = np.argsort(-norm_ent[indices], kind="stable")
            picks = indices[order[: min(quota, len(indices))]]
            priority_i.update(picks.tolist())
            run_picks += len(picks)

        logger.info(
            "Run %d: K-means done. %d picks from proportional quotas.", r, run_picks
        )

    logger.info(
        "Union priority set: %d individuals (from %d runs).", len(priority_i), len(runs)
    )
    return priority_i, all_labels


@beartype.beartype
def compute_scores(runs: list[pl.DataFrame], priority_i: set[int]) -> dict[int, float]:
    """For each priority beetle, score = min(norm_entropy) across runs."""
    scores: dict[int, float] = {}
    for i in priority_i:
        min_ent = min(float(df["norm_entropy"][i]) for df in runs)
        scores[i] = min_ent
    return scores


@beartype.beartype
def greedy_set_cover(
    runs: list[pl.DataFrame],
    priority_i: set[int],
    scores: dict[int, float],
    all_labels: list[np.ndarray],
    *,
    n_groups: int,
    cost_alpha: float,
) -> tuple[list[dict], list[dict]]:
    """Entropy-weighted greedy set cover over group images. Returns (group_rows, beetle_rows)."""
    ref = runs[0]
    assert n_groups >= 1, f"n_groups must be >= 1, got {n_groups}."

    # Exclude beetles with empty group_img_basename.
    group_basenames = ref["group_img_basename"].to_list()
    empty_count = sum(1 for i in priority_i if group_basenames[i] == "")
    if empty_count:
        logger.warning(
            "Excluding %d priority beetles with empty group_img_basename.", empty_count
        )
    priority_i = {i for i in priority_i if group_basenames[i] != ""}

    # Build group -> row indices mapping (all beetles, not just priority).
    group_to_rows: dict[str, list[int]] = {}
    for i in range(len(ref)):
        g = group_basenames[i]
        if g == "":
            continue
        group_to_rows.setdefault(g, []).append(i)

    # Build group -> priority row indices.
    group_to_priority: dict[str, set[int]] = {}
    for i in priority_i:
        g = group_basenames[i]
        group_to_priority.setdefault(g, set()).add(i)

    logger.info(
        "Group images with priority beetles: %d (of %d total groups).",
        len(group_to_priority),
        len(group_to_rows),
    )

    # Derive absolute group image path from beetle img_fpath.
    # Beetle images are at .../cropped_images/{group_name}/{beetle_file}.png
    # Group images are at .../group_images/{group_name}.jpg (common convention).
    # Since we don't know the exact group image path scheme, store the basename's parent from img_fpath.
    img_fpaths = ref["img_fpath"].to_list()

    def get_abs_group_img_fpath(group_basename: str) -> str:
        """Best-effort absolute path for group image. Uses parent-of-parent of any beetle's img_fpath."""
        rows = group_to_rows[group_basename]
        parent = str(pathlib.Path(img_fpaths[rows[0]]).parent.parent)
        return str(pathlib.Path(parent) / group_basename)

    remaining = set(priority_i)
    selected_groups: list[dict] = []
    cumulative_covered = 0

    for rank in range(1, n_groups + 1):
        if not remaining:
            logger.info(
                "All %d priority beetles covered after %d groups.",
                len(priority_i),
                rank - 1,
            )
            break

        best_group: str | None = None
        best_score = -1.0
        best_raw = 0.0
        best_uncovered: set[int] = set()

        for g, pri in group_to_priority.items():
            uncovered = pri & remaining
            if not uncovered:
                continue
            raw = sum(scores[i] for i in uncovered)
            cost = len(group_to_rows[g]) ** cost_alpha
            score = raw / cost if cost > 0 else raw
            if score > best_score:
                best_group = g
                best_score = score
                best_raw = raw
                best_uncovered = uncovered

        if best_group is None:
            break

        remaining -= best_uncovered
        cumulative_covered += len(best_uncovered)

        # Count distinct clusters covered by this pick.
        clusters_covered: set[int] = set()
        for i in best_uncovered:
            for labels in all_labels:
                clusters_covered.add(int(labels[i]))

        selected_groups.append({
            "rank": rank,
            "group_img_basename": best_group,
            "abs_group_img_path": get_abs_group_img_fpath(best_group),
            "n_priority_covered": len(best_uncovered),
            "entropy_covered": round(best_raw, 4),
            "n_total_beetles": len(group_to_rows[best_group]),
            "n_clusters_covered": len(clusters_covered),
            "cumulative_priority_covered": cumulative_covered,
        })

    logger.info(
        "Selected %d group images covering %d/%d priority beetles.",
        len(selected_groups),
        cumulative_covered,
        len(priority_i),
    )

    # Build beetle detail rows for priority beetles in selected groups.
    selected_group_set = {g["group_img_basename"] for g in selected_groups}
    group_rank_map = {g["group_img_basename"]: g["rank"] for g in selected_groups}

    beetle_rows: list[dict] = []
    for i in priority_i:
        g = group_basenames[i]
        if g not in selected_group_set:
            continue

        raw_entropies = [float(df["mean_entropy"][i]) for df in runs]
        norm_entropies = [float(df["norm_entropy"][i]) for df in runs]
        cluster_ids = ",".join(str(int(labels[i])) for labels in all_labels)

        beetle_rows.append({
            "beetle_id": ref["beetle_id"][i],
            "group_img_basename": g,
            "abs_group_img_path": get_abs_group_img_fpath(g),
            "scientific_name": ref["scientific_name"][i],
            "norm_min_entropy": round(min(norm_entropies), 6),
            "norm_max_entropy": round(max(norm_entropies), 6),
            "raw_min_entropy": round(min(raw_entropies), 4),
            "raw_max_entropy": round(max(raw_entropies), 4),
            "cluster_ids": cluster_ids,
            "group_rank": group_rank_map[g],
        })

    beetle_rows.sort(key=lambda r: (r["group_rank"], -r["norm_min_entropy"]))
    return selected_groups, beetle_rows


@beartype.beartype
def run_rank(cfg: Config):
    """Full ranking pipeline. Called directly or via submitit."""
    # SAM: what do you think about these being in __post_init__?
    assert cfg.glob_pattern, "glob_pattern must not be empty."
    assert cfg.k >= 1, f"k must be >= 1, got {cfg.k}."
    assert cfg.n_avg >= 1, f"n_avg must be >= 1, got {cfg.n_avg}."
    assert cfg.n_groups >= 1, f"n_groups must be >= 1, got {cfg.n_groups}."

    # SAM: What numbers are you referencing? 1, 2, 3-4, etc?
    # 1. Load and validate.
    runs = load_runs(cfg.glob_pattern)

    # Warn and exclude beetles with empty group_img_basename.
    # SAM: What about the other runs? or do all runs have the same group_img_basename column?
    n_empty = (runs[0]["group_img_basename"] == "").sum()
    if n_empty:
        logger.warning(
            "%d beetles have empty group_img_basename (will be excluded from group selection).",
            n_empty,
        )

    # 2. Normalize entropy per run.
    runs = normalize_entropy(runs)

    # 3-4. Per-run clustering and union priority sets.
    priority_i, all_labels = select_priority(runs, k=cfg.k, n_avg=cfg.n_avg)

    # 5. Aggregate entropy: normalized min across runs.
    scores = compute_scores(runs, priority_i)

    # 6. Greedy set cover.
    group_rows, beetle_rows = greedy_set_cover(
        runs,
        priority_i,
        scores,
        all_labels,
        n_groups=cfg.n_groups,
        cost_alpha=cfg.cost_alpha,
    )

    # Write output CSVs.
    out = pathlib.Path(cfg.out_fpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    groups_fpath = out.parent / f"{out.name}_groups.csv"
    beetles_fpath = out.parent / f"{out.name}_beetles.csv"
    all_fpath = out.parent / f"{out.name}_all_unlabeled.csv"

    pl.DataFrame(group_rows).write_csv(groups_fpath)
    logger.info("Wrote %d groups to '%s'.", len(group_rows), groups_fpath)

    pl.DataFrame(beetle_rows).write_csv(beetles_fpath)
    logger.info("Wrote %d beetles to '%s'.", len(beetle_rows), beetles_fpath)

    # All-unlabeled CSV: per-beetle metadata for notebook visualizations (#3, #4).
    ref = runs[0]
    n = len(ref)

    norm_ent_stack = np.stack([df["norm_entropy"].to_numpy() for df in runs])
    min_norm_ent = np.round(norm_ent_stack.min(axis=0), 6)

    is_pri = np.zeros(n, dtype=bool)
    for i in priority_i:
        is_pri[i] = True

    # Per-run cluster labels as comma-separated string.
    cluster_strs = [
        ",".join(str(int(labels[i])) for labels in all_labels) for i in range(n)
    ]

    all_df = pl.DataFrame({
        "beetle_id": ref["beetle_id"],
        "group_img_basename": ref["group_img_basename"],
        "is_priority": is_pri,
        "min_norm_entropy": min_norm_ent,
        "cluster_ids": cluster_strs,
    })
    all_df.write_csv(all_fpath)
    logger.info("Wrote %d beetles to '%s'.", len(all_df), all_fpath)


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
):
    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = btx.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return

        cfgs, errs = btx.configs.load_cfgs(cfg, default=Config(), sweep_dcts=sweep_dcts)
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    base = cfgs[0]
    for c in cfgs[1:]:
        msg = (
            "Sweep configs must share slurm_acct, slurm_partition, n_hours, and log_to."
        )
        assert c.slurm_acct == base.slurm_acct, msg
        assert c.slurm_partition == base.slurm_partition, msg
        assert c.n_hours == base.n_hours, msg
        assert c.log_to == base.log_to, msg

    if base.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=base.log_to)
        executor.update_parameters(
            job_name="beetle-rank",
            time=int(base.n_hours * 60),
            partition=base.slurm_partition,
            gpus_per_node=0,
            ntasks_per_node=1,
            mem="16GB",
            stderr_to_stdout=True,
            account=base.slurm_acct,
        )
    else:
        import submitit

        executor = submitit.DebugExecutor(folder=base.log_to)

    with executor.batch():
        jobs = [executor.submit(run_rank, c) for c in cfgs]

    for job in jobs:
        logger.info("Running job %s.", job.job_id)

    for job in jobs:
        job.result()


if __name__ == "__main__":
    tyro.cli(main)
