# train.py
import dataclasses
import beartype
import wandb
import torch
import tyro
from torch.utils.data import DataLoader
from torch import nn, optim

import beetle_traits as btr


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    model: btr.nn.Config = btr.nn.Bioclip()
    """Model config."""


@beartype.beartype
def train_one_epoch(model, loader, optim, loss_fn, device) -> float:
    model.train()
    epoch_loss = 0

    for batch in loader:
        imgs_bwh = batch["image"].to(device)
        keypoints_b22 = batch["keypoints"].to(device)  # shape (B, 2, 2)
        optim.zero_grad()
        pred_b22 = model(imgs_bwh)
        loss = loss_fn(pred_b22, keypoints_b22)
        loss.backward()
        optim.step()

        bsz, *_ = imgs_bwh.shape
        epoch_loss += loss.item() * bsz

    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_accum, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs = batch["image"].to(device)
        kp = batch["keypoints"].to(device)
        pred = model(imgs).view_as(kp)
        mae_accum += trait_mae(pred, kp).item() * imgs.size(0)
        correct += pck(pred, kp, tol="human_sd")  # bool count
        total += imgs.size(0) * kp.size(1)  # two traits per sample
    return {"MAE_mm": mae_accum / len(loader.dataset), "PCT": correct / total * 100}


@beartype.beartype
def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = beetle_traits.models.build(cfg.model)
    model.fc = nn.Linear(model.fc.in_features, 4)  # x1,y1,x2,y2
    model.to(device)

    train_ds = beetle_traits.data.HawaiiDataset(
        cfg["train_root"], cfg["train_csv"], augment=True
    )
    val_ds = beetle_traits.data.ElytraDataset(cfg["val_root"], cfg["val_csv"])
    train_ld = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False)

    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss = nn.SmoothL1Loss()

    wandb.init(project=cfg["wb_project"], config=cfg)
    best_pct = 0.0
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_ld, opt, loss, device)
        val_metrics = evaluate(model, val_ld, device)
        wandb.log({"train_loss": train_loss, **val_metrics}, step=epoch)
        if val_metrics["PCT"] > best_pct:
            best_pct = val_metrics["PCT"]
            torch.save(model.state_dict(), Path(cfg["out_dir"]) / "best.pt")
        print(
            f"Epoch {epoch:03d} | {time.time() - t0:.1f}s | "
            f"MAE {val_metrics['MAE_mm']:.3f} | PCT {val_metrics['PCT']:.2f}"
        )
        if best_pct >= cfg["target_pct"]:
            break


if __name__ == "__main__":
    tyro.cli(main)
