"""Core training logic for scalebar localization.

Separated from train_scalebar.py (the launcher) because submitit and jaxtyping
cannot coexist in the same file.
"""

import dataclasses
import heapq
import logging
import pathlib
import typing as tp
from collections.abc import Iterable

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree, jaxtyped
from PIL import Image, ImageDraw

import btx.data
import btx.helpers
import btx.metrics
import btx.modeling
import btx.objectives
import wandb

logger = logging.getLogger("scalebar_trainer")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ValConfig:
    every: int = 500
    n_fixed: int = 5
    n_worst: int = 1
    n_random: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 17
    model: btx.modeling.Config = btx.modeling.heatmap.Heatmap()
    scalebar: btx.data.ScalebarClipConfig = btx.data.ScalebarClipConfig(go=False)
    aug: btx.data.AugmentConfig = btx.data.AugmentConfig()
    objective: btx.objectives.Config = btx.objectives.Heatmap()
    batch_size: int = 16
    """Small batch size; only ~75 clip images total."""
    n_workers: int = 4
    val: ValConfig = ValConfig()
    tags: list[str] = dataclasses.field(default_factory=list)
    save_every: int = 100
    log_every: int = 100
    n_steps: int = 10_000
    log_to: pathlib.Path = pathlib.Path("./logs")
    learning_rate: float = 3e-4
    ckpt_dpath: pathlib.Path = pathlib.Path("./checkpoints")
    schedule: tp.Literal["cosine", "wsd", "none"] = "none"
    warmup_steps: int = 0
    decay_steps: int = 0
    weight_decay: float = 0.05
    wandb_project: str = "beetle-traits"
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 1.0


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    batch: dict[str, Array],
    *,
    obj: btx.objectives.Obj,
) -> tuple[Float[Array, ""], dict]:
    model = eqx.combine(diff_model, static_model)
    forward = tp.cast(tp.Callable[[Array], Array], model)
    preds_raw = jax.vmap(forward)(batch["img"])

    mask_line = batch["loss_mask"]
    obj_aux = obj.get_loss_aux(preds_raw=preds_raw, batch=batch, mask_line=mask_line)
    obj_metrics = {f"{obj.key}/{k}": v for k, v in obj_aux.metrics.items()}

    return obj_aux.loss, {
        "loss": obj_aux.loss,
        "sample_loss": obj_aux.sample_loss,
        "preds": obj_aux.preds,
        **obj_metrics,
    }


@eqx.filter_jit()
@jaxtyped(typechecker=beartype.beartype)
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    batch: dict[str, Array],
    filter_spec: PyTree[bool],
    *,
    obj: btx.objectives.Obj,
) -> tuple[eqx.Module, tp.Any, dict]:
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss_fn = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)
    (_, aux), grads = loss_fn(diff_model, static_model, batch, obj=obj)
    updates, new_state = optim.update(grads, state, diff_model)
    diff_model = eqx.apply_updates(diff_model, updates)
    return eqx.combine(diff_model, static_model), new_state, aux


@beartype.beartype
def get_trainable_filter_spec(model: eqx.Module) -> PyTree[bool]:
    filter_spec = jax.tree_util.tree_map(eqx.is_array, model)
    if not hasattr(model, "vit"):
        return filter_spec
    vit = tp.cast(eqx.Module, getattr(model, "vit"))
    frozen_vit = jax.tree_util.tree_map(lambda _: False, vit)
    return eqx.tree_at(lambda tree: tree.vit, filter_spec, frozen_vit)


@jaxtyped(typechecker=beartype.beartype)
def plot_preds(
    batch: dict[str, Array],
    metadata: dict[str, object],
    preds: Float[Array, "batch 2 2 2"],
    sample_idx: int,
    *,
    cache_size: int | None,
) -> tuple[str, Image.Image]:
    img_fpath = tp.cast(list[str], metadata["img_fpath"])[sample_idx]
    img = Image.open(img_fpath)
    orig_w, orig_h = img.size

    # Only channel 0 is supervised (scalebar); ignore channel 1.
    # points_px and pred_orig are in cache_size coords when caching is active;
    # scale to original image coords before drawing.
    gt_scalebar = np.asarray(batch["points_px"][sample_idx, 0])

    i = sample_idx
    pred_orig = btx.metrics.apply_affine(
        batch["t_orig_from_aug"][i : i + 1], preds[i : i + 1]
    )[0]
    pred_scalebar = np.asarray(pred_orig[0])

    px_scale = 1.0
    if cache_size is not None:
        scale = np.array([orig_w / cache_size, orig_h / cache_size], dtype=np.float64)
        gt_scalebar = gt_scalebar * scale
        pred_scalebar = pred_scalebar * scale
        px_scale = float(np.sqrt(scale[0] * scale[1]))

    r = max(1, round(4 * px_scale))
    line_w = max(1, round(3 * px_scale))
    draw = ImageDraw.Draw(img)
    draw.line(
        [tuple(gt_scalebar[0]), tuple(gt_scalebar[1])], fill=(0, 255, 0), width=line_w
    )
    draw.line(
        [tuple(pred_scalebar[0]), tuple(pred_scalebar[1])],
        fill=(255, 0, 0),
        width=line_w,
    )
    for pt in [*gt_scalebar, *pred_scalebar]:
        x, y = pt
        col = (
            (0, 255, 0) if any(np.allclose(pt, q) for q in gt_scalebar) else (255, 0, 0)
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    return tp.cast(list[str], metadata["group_img_basename"])[sample_idx], img


@beartype.beartype
def validate(
    cfg: Config,
    model: eqx.Module,
    filter_spec: PyTree[bool],
    ds: btx.data.Dataset,
    dl: Iterable[dict[str, object]],
    fixed_indices: Int[np.ndarray, " n"],
    *,
    obj: btx.objectives.Obj,
    cache_size: int | None,
) -> dict[str, float | wandb.Image]:
    diff_model, static_model = eqx.partition(model, filter_spec)
    fixed_batch_idxs = fixed_indices // cfg.batch_size
    fixed_sample_idxs = fixed_indices % cfg.batch_size

    rng = np.random.default_rng()
    random_indices = rng.choice(
        len(ds), size=min(cfg.val.n_random, len(ds)), replace=False
    )
    random_batch_idxs = random_indices // cfg.batch_size
    random_sample_idxs = random_indices % cfg.batch_size

    all_losses = []
    all_err_cm = []
    images = {}
    worst_candidates: list[tuple] = []
    n_seen = 0

    for i, batch in enumerate(dl):
        batch, metadata = btx.helpers.to_device(batch)
        _, aux = loss_and_aux(diff_model, static_model, batch, obj=obj)

        all_losses.append(aux["sample_loss"])

        # Endpoint error in cm. GT scalebar = 1cm, so gt_length_px == px_per_cm.
        preds_orig = btx.metrics.apply_affine(batch["t_orig_from_aug"], aux["preds"])
        gt = batch["points_px"][:, 0:1, :, :]  # scalebar channel only: (batch, 1, 2, 2)
        pred = preds_orig[:, 0:1, :, :]
        gt_aligned = btx.metrics.choose_endpoint_matching(pred, gt)
        gt_length_px = jnp.linalg.norm(
            gt_aligned[:, 0, 1, :] - gt_aligned[:, 0, 0, :], axis=-1
        )
        err_px = (
            jnp.linalg.norm(pred[:, 0, 0, :] - gt_aligned[:, 0, 0, :], axis=-1)
            + jnp.linalg.norm(pred[:, 0, 1, :] - gt_aligned[:, 0, 1, :], axis=-1)
        ) / 2.0
        all_err_cm.append(err_px / jnp.maximum(gt_length_px, btx.metrics.MIN_PX_PER_CM))

        actual_bs = aux["preds"].shape[0]

        for bi, si in zip(fixed_batch_idxs, fixed_sample_idxs):
            if i == bi and si < actual_bs:
                name, img = plot_preds(
                    batch, metadata, aux["preds"], int(si), cache_size=cache_size
                )
                images[f"images/fixed/{name}"] = wandb.Image(img)

        for bi, si in zip(random_batch_idxs, random_sample_idxs):
            if i == bi and si < actual_bs:
                name, img = plot_preds(
                    batch, metadata, aux["preds"], int(si), cache_size=cache_size
                )
                images[f"images/random/{name}"] = wandb.Image(img)

        for j, err in enumerate(aux["sample_loss"]):
            err_val = float(err)
            sample_batch = {
                "points_px": jnp.asarray(batch["points_px"][j])[jnp.newaxis],
                "t_orig_from_aug": jnp.asarray(batch["t_orig_from_aug"][j])[
                    jnp.newaxis
                ],
            }
            sample_meta = {
                "img_fpath": [metadata["img_fpath"][j]],
                "group_img_basename": [metadata["group_img_basename"][j]],
            }
            # n_seen tiebreaker prevents dict comparison if two losses are equal.
            candidate = (
                err_val,
                n_seen,
                sample_batch,
                sample_meta,
                jnp.asarray(aux["preds"][j])[jnp.newaxis],
            )
            n_seen += 1
            if len(worst_candidates) < cfg.val.n_worst:
                heapq.heappush(worst_candidates, candidate)
            elif err_val > worst_candidates[0][0]:
                heapq.heapreplace(worst_candidates, candidate)

    for _, _n, sb, sm, sp in worst_candidates:
        name, img = plot_preds(sb, sm, sp, 0, cache_size=cache_size)
        images[f"images/worst/{name}"] = wandb.Image(img)

    losses = jnp.concatenate([v.reshape(-1) for v in all_losses])
    err_cm = jnp.concatenate([v.reshape(-1) for v in all_err_cm])
    return {
        "loss": jnp.nanmean(losses).item(),
        "max_loss": jnp.nanmax(losses).item(),
        "err_cm": jnp.nanmean(err_cm).item(),
        "median_err_cm": jnp.nanmedian(err_cm).item(),
        "max_err_cm": jnp.nanmax(err_cm).item(),
        **images,
    }


def wsd_schedule(peak: float, total: int, warmup: int, decay: int) -> optax.Schedule:
    stable = total - warmup - decay
    assert stable >= 0, f"Negative stable steps: {warmup=} + {decay=} > {total=}"
    segments = []
    if warmup > 0:
        segments.append((warmup, optax.linear_schedule(0.0, peak, warmup)))
    if stable > 0:
        segments.append((stable, optax.constant_schedule(peak)))
    if decay > 0:
        segments.append((decay, optax.linear_schedule(peak, 0.0, decay)))
    if not segments:
        return optax.constant_schedule(peak)
    if len(segments) == 1:
        return segments[0][1]
    boundaries, n = [], 0
    for steps, _ in segments[:-1]:
        n += steps
        boundaries.append(n)
    return optax.join_schedules([s for _, s in segments], boundaries)


@beartype.beartype
def train(cfg: Config) -> None:
    key = jax.random.key(seed=cfg.seed)

    cache_size = cfg.aug.size if cfg.scalebar.cache else None
    train_cfg = dataclasses.replace(cfg.scalebar, split="train", cache_size=cache_size)
    val_cfg = dataclasses.replace(cfg.scalebar, split="val", cache_size=cache_size)
    train_ds = btx.data.ScalebarClipDataset(train_cfg)
    val_ds = btx.data.ScalebarClipDataset(val_cfg)

    assert len(val_ds) >= cfg.val.n_fixed, (
        f"Val has {len(val_ds)} samples, need at least cfg.val.n_fixed={cfg.val.n_fixed}."
    )

    train_dl = btx.data.make_dataloader(
        [train_ds],
        [cfg.aug],
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=True,
        finite=False,
        is_train=True,
        img_caches=[train_ds._img_cache],
    )
    val_dl = btx.data.make_dataloader(
        [val_ds],
        [btx.data.AugmentConfig(go=False, size=cfg.aug.size)],
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        shuffle=False,
        finite=True,
        is_train=False,
        img_caches=[val_ds._img_cache],
    )

    fixed_indices = np.random.default_rng(seed=cfg.seed).choice(
        len(val_ds), size=cfg.val.n_fixed, replace=False
    )

    if cfg.schedule == "wsd":
        schedule = wsd_schedule(
            cfg.learning_rate, cfg.n_steps, cfg.warmup_steps, cfg.decay_steps
        )
    elif cfg.schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.n_steps,
            end_value=0.0,
        )
    elif cfg.schedule == "none":
        schedule = optax.constant_schedule(cfg.learning_rate)
    else:
        tp.assert_never(cfg.schedule)

    optim = optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)
    model = btx.modeling.make(cfg.model, key)
    filter_spec = get_trainable_filter_spec(model)
    state = optim.init(eqx.partition(model, filter_spec)[0])
    obj = cfg.objective.get_obj()

    run = wandb.init(
        project=cfg.wandb_project,
        config=dataclasses.asdict(cfg),
        tags=cfg.tags,
        dir=".wandb",
    )
    ckpt_dpath = cfg.ckpt_dpath / run.id
    ckpt_dpath.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint dir: %s", ckpt_dpath)

    for step, batch in enumerate(train_dl):
        batch, metadata = btx.helpers.to_device(batch)
        model, state, aux = step_model(model, optim, state, batch, filter_spec, obj=obj)

        if step % cfg.save_every == 0:
            name, img = plot_preds(
                batch, metadata, aux["preds"], 0, cache_size=cache_size
            )
            run.log({"step": step, f"images/train/{name}": wandb.Image(img)}, step=step)

        if step % cfg.log_every == 0:
            metrics = {
                f"train/{k}": jnp.nanmean(v).item()
                for k, v in aux.items()
                if k != "preds"
            }
            metrics["step"] = step
            run.log(metrics, step=step)
            logger.info("Step %d  train/loss=%.4f", step, metrics["train/loss"])

        if step % cfg.val.every == 0:
            val_metrics = validate(
                cfg,
                model,
                filter_spec,
                val_ds,
                val_dl,
                fixed_indices,
                obj=obj,
                cache_size=cache_size,
            )
            run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logger.info(
                "Step %d  val/loss=%.4f  val/err_cm=%.4f  val/median_err_cm=%.4f",
                step,
                val_metrics["loss"],
                val_metrics["err_cm"],
                val_metrics["median_err_cm"],
            )

        if step >= cfg.n_steps:
            break

    ckpt_fpath = ckpt_dpath / "model.eqx"
    btx.modeling.save_ckpt(model, cfg.model, cfg.objective, ckpt_fpath)
    logger.info("Saved checkpoint to '%s'.", ckpt_fpath)
