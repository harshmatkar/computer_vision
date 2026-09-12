"""
train.py
--------
Training loop for the Lightweight UNet++ actin-edge segmentation model.

Features:
  - Combined BCEWithLogits + Dice loss (config.BCE_WEIGHT / config.DICE_WEIGHT)
  - Deep supervision: loss is computed and averaged across ALL decoder
    output depths returned by the model when config.USE_DEEP_SUPERVISION=True
  - Per-epoch training & validation loss + Dice coefficient tracking
  - Model checkpointing (saves best validation Dice)
  - Early stopping
  - Saves a training_history.csv (loss/dice per epoch) for later plotting
    (e.g. the box-plot / benchmarking figures required in the final report)
"""

import os
import csv
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from models.unet_plus_plus import LightUNetPlusPlus, count_parameters


def dice_coefficient(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the Sørensen-Dice coefficient between a predicted probability
    map (after sigmoid) and a binary target mask.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw (pre-sigmoid) model output, shape (B, 1, H, W).
    target : torch.Tensor
        Binary ground-truth mask, shape (B, 1, H, W), values in {0, 1}.

    Returns
    -------
    torch.Tensor
        Scalar mean Dice coefficient over the batch.
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice loss = 1 - Dice coefficient (to be minimized)."""
    return 1.0 - dice_coefficient(pred_logits, target)


def combined_loss(outputs, target: torch.Tensor, bce_fn: nn.Module) -> torch.Tensor:
    """
    Computes the combined BCE + Dice loss, averaged across every
    deep-supervision output if the model returns a list of logits
    (one per decoder depth), or computed directly if it returns a
    single tensor.

    total_loss = BCE_WEIGHT * BCEWithLogitsLoss + DICE_WEIGHT * DiceLoss
    """
    if isinstance(outputs, (list, tuple)):
        losses = []
        for out in outputs:
            bce = bce_fn(out, target)
            d_loss = dice_loss(out, target)
            losses.append(config.BCE_WEIGHT * bce + config.DICE_WEIGHT * d_loss)
        return torch.stack(losses).mean()
    else:
        bce = bce_fn(outputs, target)
        d_loss = dice_loss(outputs, target)
        return config.BCE_WEIGHT * bce + config.DICE_WEIGHT * d_loss


def get_final_output(outputs):
    """Extracts the final (deepest / most refined) prediction for Dice metric reporting."""
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def run_epoch(model, loader, optimizer, bce_fn, device, train: bool = True):
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
            loss = combined_loss(outputs, masks, bce_fn)

            if train:
                loss.backward()
                optimizer.step()

            final_out = get_final_output(outputs)
            dice = dice_coefficient(final_out, masks)

            total_loss += loss.item()
            total_dice += dice.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def train_model():
    """
    Main training entry point. Builds dataloaders, model, optimizer, then
    runs the training loop with validation, checkpointing, early stopping,
    and CSV history logging.
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = LightUNetPlusPlus(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_filters=config.BASE_FILTERS,
        depth=config.DEPTH,
        deep_supervision=config.USE_DEEP_SUPERVISION,
    ).to(device)
    print(f"Model parameter count: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    bce_fn = nn.BCEWithLogitsLoss()

    history_path = os.path.join(config.OUTPUT_DIR, "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice",
                          "epoch_time_sec"])

    best_val_dice = -1.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, bce_fn, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, optimizer, bce_fn, device, train=False)
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
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "val_dice": best_val_dice,
                "config": {
                    "base_filters": config.BASE_FILTERS,
                    "depth": config.DEPTH,
                    "deep_supervision": config.USE_DEEP_SUPERVISION,
                },
            }, ckpt_path)
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f}) at {ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")
    print(f"Training history saved to: {history_path}")
    return model, history_path


if __name__ == "__main__":
    train_model()
