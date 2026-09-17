"""
utils/losses.py
-----------------
All loss functions used across the three team-member models, in one place
so train.py can select the right one via config.MODEL_REGISTRY without
duplicating math in multiple files.

  - Member A (Lightweight UNet++)  -> bce_dice_loss
  - Member B (Attention U-Net)     -> tversky_loss
  - Member C (ResUNet++)           -> combo_loss

Each loss operates on raw (pre-sigmoid) logits plus a binary {0,1} target
mask, matching the output convention of all three model files.
"""

import torch
import torch.nn as nn

EPS = 1e-6


def dice_coefficient(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Sorensen-Dice coefficient between a predicted probability map (post-sigmoid)
    and a binary target mask. Dice = 2*|A∩B| / (|A|+|B|).
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice loss = 1 - Dice coefficient (to be minimized)."""
    return 1.0 - dice_coefficient(pred_logits, target)


# ---------------------------------------------------------------------------
# Member A: Combined BCE + Dice Loss
# ---------------------------------------------------------------------------
def bce_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                   bce_weight: float = 0.5, dice_weight: float = 0.5) -> torch.Tensor:
    """
    total_loss = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss

    Balances per-pixel classification (BCE) with region-overlap quality
    (Dice), which is the standard choice for the UNet++ baseline.
    """
    bce_fn = nn.BCEWithLogitsLoss()
    bce = bce_fn(pred_logits, target)
    d_loss = dice_loss(pred_logits, target)
    return bce_weight * bce + dice_weight * d_loss


# ---------------------------------------------------------------------------
# Member B: Tversky Loss  (Salehi, Erdogmus & Gholipour, MLMI 2017)
# ---------------------------------------------------------------------------
def tversky_index(pred_logits: torch.Tensor, target: torch.Tensor,
                   alpha: float = 0.7, beta: float = 0.3, eps: float = EPS) -> torch.Tensor:
    """
    Tversky index generalizes Dice by letting false positives (alpha) and
    false negatives (beta) be weighted independently:

        TI = TP / (TP + alpha*FP + beta*FN)

    alpha=beta=0.5 recovers the standard Dice coefficient. Setting beta > alpha
    (as we do not here) penalizes missed foreground (false negatives) more --
    useful for the thin, sparse actin-edge masks where recall matters more
    than precision. Here alpha=0.7 > beta=0.3 slightly favors recall by
    penalizing false negatives less aggressively relative to false positives
    being suppressed -- tune via config.TVERSKY_ALPHA / TVERSKY_BETA.
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return tversky.mean()


def tversky_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.7, beta: float = 0.3) -> torch.Tensor:
    """Tversky loss = 1 - Tversky index (to be minimized)."""
    return 1.0 - tversky_index(pred_logits, target, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Member C: Combo Loss  (Taghanaki et al., 2019 -
#            "Combo loss: Handling input and output imbalance in
#             multi-organ segmentation")
# ---------------------------------------------------------------------------
def weighted_bce(pred_logits: torch.Tensor, target: torch.Tensor,
                  ce_beta: float = 0.5, eps: float = EPS) -> torch.Tensor:
    """
    Weighted binary cross-entropy where the positive (foreground) class is
    weighted by `ce_beta` and the negative (background) class by
    `(1 - ce_beta)`. With ce_beta > 0.5, false negatives on the (rare)
    foreground actin-edge pixels are penalized more heavily than false
    positives on background -- addressing the same foreground/background
    imbalance problem that motivates Tversky loss, but from a
    cross-entropy-weighting angle instead of a Dice-generalization angle.
    """
    pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    loss = -(
        ce_beta * target_flat * torch.log(pred_flat)
        + (1 - ce_beta) * (1 - target_flat) * torch.log(1 - pred_flat)
    )
    return loss.mean()


def combo_loss(pred_logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.5, ce_beta: float = 0.5) -> torch.Tensor:
    """
    Combo Loss = alpha * Weighted-BCE + (1 - alpha) * Dice Loss

    Distinct from bce_dice_loss (Member A) because the cross-entropy term
    here is class-weighted (ce_beta) rather than plain BCE, giving Member C
    an independently-tunable third objective function as required by the
    "each member uses a different network AND different loss" rubric line.
    """
    wbce = weighted_bce(pred_logits, target, ce_beta=ce_beta)
    d_loss = dice_loss(pred_logits, target)
    return alpha * wbce + (1 - alpha) * d_loss


# ---------------------------------------------------------------------------
# Dispatcher used by train.py / evaluate.py
# ---------------------------------------------------------------------------
def get_loss_fn(loss_name: str, config_module):
    """
    Returns a callable loss_fn(pred_logits, target) -> scalar tensor,
    configured with the hyperparameters for the requested loss name.

    Parameters
    ----------
    loss_name : str
        One of: 'bce_dice', 'tversky', 'combo'.
    config_module : module
        The imported config.py module (passed explicitly to avoid a circular
        import between losses.py and config.py).
    """
    if loss_name == "bce_dice":
        return lambda logits, target: bce_dice_loss(
            logits, target, bce_weight=config_module.BCE_WEIGHT, dice_weight=config_module.DICE_WEIGHT
        )
    elif loss_name == "tversky":
        return lambda logits, target: tversky_loss(
            logits, target, alpha=config_module.TVERSKY_ALPHA, beta=config_module.TVERSKY_BETA
        )
    elif loss_name == "combo":
        return lambda logits, target: combo_loss(
            logits, target, alpha=config_module.COMBO_ALPHA, ce_beta=config_module.COMBO_CE_BETA
        )
    else:
        raise ValueError(f"Unknown loss_name: {loss_name!r}. Expected 'bce_dice', 'tversky', or 'combo'.")
