"""
train.py
--------
Training loop shared by all three team-member models. Select which model
to train with the --model flag:

    python train.py --model unetpp               # Member A: BCE + Dice loss
    python train.py --model attention_unet        # Member B: Tversky loss
    python train.py --model resunetpp              # Member C: Combo loss
    python train.py --all                          # train all three sequentially

Each model gets its own checkpoint file and training-history CSV (see
config.MODEL_REGISTRY), so all three can be trained independently and then
benchmarked side by side in evaluate.py / the final report.

Features:
  - Loss function automatically selected per model (utils/losses.py)
  - Deep supervision handled transparently for models that use it (UNet++)
  - Per-epoch training & validation loss + Dice coefficient tracking
  - Model checkpointing (saves best validation Dice)
  - Early stopping
"""

import os
import csv
import time
import copy
import argparse

import torch
import torch.optim as optim

import config
from dataset import get_dataloaders
from model_builder import build_model, build_loss, get_final_output, compute_loss
from utils.losses import dice_coefficient


def run_epoch(model, model_name, loader, optimizer, loss_fn, device, train: bool = True):
    """
    Runs one full pass over `loader`, either in training mode (with
    backprop) or evaluation mode (no gradient updates).

    Returns
    -------
    (avg_loss, avg_dice) : tuple of float
    """
    model.train() if train else model.eval()

    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = compute_loss(model_name, loss_fn, outputs, masks)

            if train:
                loss.backward()
                optimizer.step()

            final_out = get_final_output(outputs)
            dice = dice_coefficient(final_out, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def train_model(model_name: str = None):
    """
    Main training entry point for one of the three registered models.
    """
    model_name = model_name or config.DEFAULT_MODEL
    if model_name not in config.MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(config.MODEL_REGISTRY.keys())}")

    registry_entry = config.MODEL_REGISTRY[model_name]
    device = config.DEVICE

    print(f"=== Training {registry_entry['display_name']} (model_name='{model_name}') ===")
    print(f"Using device: {device}")
    print(f"Loss function: {registry_entry['loss']}")

    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(model_name)
    loss_fn = build_loss(model_name)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)

    history_path = os.path.join(config.OUTPUT_DIR, f"training_history_{model_name}.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice",
                          "epoch_time_sec"])

    best_val_dice = -1.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, model_name, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_dice = run_epoch(model, model_name, val_loader, optimizer, loss_fn, device, train=False)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
              f"time={epoch_time:.1f}s")

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice, epoch_time])

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, registry_entry["checkpoint_name"])
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": best_model_state,
                "val_dice": best_val_dice,
                "num_parameters": n_params,
            }, ckpt_path)
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f}) at {ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"Training complete for {model_name}. Best validation Dice: {best_val_dice:.4f}")
    print(f"Training history saved to: {history_path}")
    return model, history_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one of the three team-member segmentation models.")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                         choices=list(config.MODEL_REGISTRY.keys()),
                         help="Which model to train: unetpp (Member A), attention_unet (Member B), "
                              "resunetpp (Member C).")
    parser.add_argument("--all", action="store_true",
                         help="Train all three registered models sequentially.")
    args = parser.parse_args()

    if args.all:
        for name in config.MODEL_REGISTRY.keys():
            train_model(name)
    else:
        train_model(args.model)
