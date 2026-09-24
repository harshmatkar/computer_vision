"""
utils/losses.py
-----------------
All loss functions used across the three team-member models, in one place
so train.py can select the right one via config.MODEL_REGISTRY without
duplicating math in multiple files. Implemented as tf.keras-compatible
loss functions: each takes (y_true, y_pred_logits) and returns a scalar
Tensor, matching the signature Keras expects for `model.compile(loss=...)`.

  - Member A (Lightweight UNet++)  -> bce_dice_loss
  - Member B (Attention U-Net)     -> tversky_loss
  - Member C (ResUNet++)           -> combo_loss

All three operate on raw (pre-sigmoid) logits plus a binary {0,1} target
mask, matching the output convention of all three model files (no final
activation on the output Conv2D layer).
"""

import tensorflow as tf

EPS = 1e-6


def dice_coefficient(y_true: tf.Tensor, y_pred_logits: tf.Tensor, eps: float = EPS) -> tf.Tensor:
    """
    Sorensen-Dice coefficient between a predicted probability map
    (post-sigmoid) and a binary target mask. Dice = 2*|A∩B| / (|A|+|B|).
    """
    y_pred = tf.sigmoid(y_pred_logits)
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
    """Dice loss = 1 - Dice coefficient (to be minimized)."""
    return 1.0 - dice_coefficient(y_true, y_pred_logits)


# ---------------------------------------------------------------------------
# Member A: Combined BCE + Dice Loss
# ---------------------------------------------------------------------------
def make_bce_dice_loss(bce_weight: float = 0.5, dice_weight: float = 0.5):
    """
    Returns a loss_fn(y_true, y_pred_logits) computing:
        total_loss = bce_weight * BinaryCrossentropy + dice_weight * DiceLoss

    Balances per-pixel classification (BCE) with region-overlap quality
    (Dice), which is the standard choice for the UNet++ baseline.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(y_true, y_pred_logits):
        bce = bce_fn(y_true, y_pred_logits)
        d_loss = dice_loss(y_true, y_pred_logits)
        return bce_weight * bce + dice_weight * d_loss

    return loss_fn


# ---------------------------------------------------------------------------
# Member B: Tversky Loss  (Salehi, Erdogmus & Gholipour, MLMI 2017)
# ---------------------------------------------------------------------------
def tversky_index(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
                   alpha: float = 0.7, beta: float = 0.3, eps: float = EPS) -> tf.Tensor:
    """
    Tversky index generalizes Dice by letting false positives (alpha) and
    false negatives (beta) be weighted independently:

        TI = TP / (TP + alpha*FP + beta*FN)

    alpha=beta=0.5 recovers the standard Dice coefficient. Here alpha=0.7 >
    beta=0.3 (config.TVERSKY_ALPHA / TVERSKY_BETA) slightly favors recall
    by penalizing false negatives less aggressively relative to false
    positives -- useful for the thin, sparse actin-edge masks where missing
    real edge pixels is worse than a few extra false positives.
    """
    y_pred = tf.sigmoid(y_pred_logits)
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=1)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return tf.reduce_mean(tversky)


def make_tversky_loss(alpha: float = 0.7, beta: float = 0.3):
    """Returns a loss_fn(y_true, y_pred_logits) = 1 - Tversky index."""

    def loss_fn(y_true, y_pred_logits):
        return 1.0 - tversky_index(y_true, y_pred_logits, alpha=alpha, beta=beta)

    return loss_fn


# ---------------------------------------------------------------------------
# Member C: Combo Loss  (Taghanaki et al., 2019 -
#            "Combo loss: Handling input and output imbalance in
#             multi-organ segmentation")
# ---------------------------------------------------------------------------
def weighted_bce(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
                  ce_beta: float = 0.5, eps: float = EPS) -> tf.Tensor:
    """
    Weighted binary cross-entropy where the positive (foreground) class is
    weighted by `ce_beta` and the negative (background) class by
    `(1 - ce_beta)`. With ce_beta > 0.5, false negatives on the (rare)
    foreground actin-edge pixels are penalized more heavily than false
    positives on background -- addressing the same foreground/background
    imbalance problem that motivates Tversky loss, but from a
    cross-entropy-weighting angle rather than a Dice-generalization angle.
    """
    y_pred = tf.clip_by_value(tf.sigmoid(y_pred_logits), eps, 1 - eps)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    loss = -(
        ce_beta * y_true_flat * tf.math.log(y_pred_flat)
        + (1 - ce_beta) * (1 - y_true_flat) * tf.math.log(1 - y_pred_flat)
    )
    return tf.reduce_mean(loss)


def make_combo_loss(alpha: float = 0.5, ce_beta: float = 0.5):
    """
    Returns a loss_fn(y_true, y_pred_logits) computing:
        Combo Loss = alpha * Weighted-BCE + (1 - alpha) * Dice Loss

    Distinct from bce_dice_loss (Member A) because the cross-entropy term
    here is class-weighted (ce_beta) rather than plain BCE, giving Member C
    an independently-tunable third objective function as required by the
    "each member uses a different network AND different loss" rubric line.
    """

    def loss_fn(y_true, y_pred_logits):
        wbce = weighted_bce(y_true, y_pred_logits, ce_beta=ce_beta)
        d_loss = dice_loss(y_true, y_pred_logits)
        return alpha * wbce + (1 - alpha) * d_loss

    return loss_fn


# ---------------------------------------------------------------------------
# Dispatcher used by train.py / evaluate.py
# ---------------------------------------------------------------------------
def get_loss_fn(loss_name: str, config_module):
    """
    Returns a callable loss_fn(y_true, y_pred_logits) -> scalar tensor,
    configured with the hyperparameters for the requested loss name.

    Parameters
    ----------
    loss_name : str
        One of: 'bce_dice', 'tversky', 'combo'.
    config_module : module
        The imported config.py module (passed explicitly to avoid a
        circular import between losses.py and config.py).
    """
    if loss_name == "bce_dice":
        return make_bce_dice_loss(bce_weight=config_module.BCE_WEIGHT,
                                   dice_weight=config_module.DICE_WEIGHT)
    elif loss_name == "tversky":
        return make_tversky_loss(alpha=config_module.TVERSKY_ALPHA,
                                  beta=config_module.TVERSKY_BETA)
    elif loss_name == "combo":
        return make_combo_loss(alpha=config_module.COMBO_ALPHA,
                                ce_beta=config_module.COMBO_CE_BETA)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name!r}. Expected 'bce_dice', 'tversky', or 'combo'.")


# ---------------------------------------------------------------------------
# Keras Metric wrapper so Dice can be tracked during model.fit()
# ---------------------------------------------------------------------------
def dice_metric(y_true, y_pred_logits):
    """A plain function usable directly in `model.compile(metrics=[...])`."""
    return dice_coefficient(y_true, y_pred_logits)
