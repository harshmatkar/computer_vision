"""
utils/losses.py
---------------
Loss functions and metrics for all three models.

WHY THE OLD BCE+DICE FAILED
----------------------------
Plain binary cross-entropy with 1-5% foreground gives the model an easy
shortcut: predict all-zeros, get near-zero BCE, and watch the combined
loss plateau. At 99% background the BCE gradient is 99x larger from
background pixels than foreground. The 0.5-weighted Dice term cannot
overcome this imbalance, so EarlyStopping fires at epoch 0 (the best
the model ever achieves) and training halts having learned nothing.

FIXES
------
  UNet++  "bce_dice" -> Focal + Dice
      Focal loss (Lin et al., 2017) down-weights easy negatives via
      (1-p)^gamma, eliminating the all-zeros shortcut. alpha=0.75
      further boosts the foreground gradient.

  Attention UNet  "tversky" -> Focal Tversky
      Tversky index lets you penalise FN and FP unequally.
      For sparse foreground you need beta >> alpha (high FN penalty).
      The config originally had alpha=0.7 (high FP penalty) -- wrong
      direction. Fixed in config: TVERSKY_ALPHA=0.3, TVERSKY_BETA=0.7.

  ResUNet++  "combo" -> Weighted-CE + Dice
      Weighted cross-entropy with pos_weight drives the CE term to
      explicitly treat foreground pixels as POS_WEIGHT times more
      important than background. Combined with Dice for spatial overlap.

All loss functions accept RAW LOGITS (no sigmoid applied before calling).
All functions are numerically stable (use tf.nn for CE operations).
"""

import tensorflow as tf


# ============================================================
# SHARED PRIMITIVES
# ============================================================

def _soft_dice(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
               smooth: float = 1e-6) -> tf.Tensor:
    """
    Soft (probabilistic) Dice computed from logits.

    Flattens the batch dimension so Dice is computed globally across
    all pixels and images in the batch -- this is slightly more stable
    than per-image averaging when images are sparsely labelled.
    """
    y_pred_prob = tf.sigmoid(y_pred_logits)
    y_true_f    = tf.cast(tf.reshape(y_true,           [-1]), tf.float32)
    y_pred_f    = tf.reshape(y_pred_prob, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + smooth) / (union + smooth)


def dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice coefficient from logits -- used as a Keras metric in compile().

    Named 'dice_metric' and imported by model_builder.build_and_compile().
    Keras will display it as 'dice_metric' in logs and in the training-
    history CSV.
    """
    return _soft_dice(y_true, y_pred)


# ============================================================
# 1. FOCAL + DICE  (UNet++ -- "bce_dice" in registry)
# ============================================================

def _focal_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
                gamma: float = 2.0, alpha: float = 0.75) -> tf.Tensor:
    """
    Sigmoid focal loss (numerically stable via tf.nn).

    gamma : focusing parameter -- higher values suppress easy negatives
            more aggressively (2.0 is the standard; 2-3 for very sparse
            foreground).
    alpha : foreground class weight.
            alpha=0.75 means foreground pixels get 3x the base gradient
            weight of background pixels, on top of the focal weighting.
            (Standard RetinaNet uses 0.25 for 50%-ish foreground;
             we flip to 0.75 because our foreground is < 5%.)
    """
    y_true = tf.cast(y_true, tf.float32)

    # BCE from logits -- numerically stable
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true, logits=y_pred_logits
    )

    y_pred_prob = tf.sigmoid(y_pred_logits)

    # p_t: predicted probability of the true class
    p_t = y_true * y_pred_prob + (1.0 - y_true) * (1.0 - y_pred_prob)
    p_t = tf.clip_by_value(p_t, 1e-7, 1.0)

    # alpha_t: per-pixel class weight
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

    # Focal modulation
    focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)

    return tf.reduce_mean(focal_weight * bce)


def focal_dice_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
    """
    0.5 * Focal(alpha=0.75, gamma=2) + 0.5 * (1 - SoftDice)

    Replaces the old plain BCE + Dice that collapsed to all-zeros on
    sparse foreground. The focal term eliminates the easy-negative
    shortcut; the Dice term ensures spatial coverage is optimised.
    """
    focal = _focal_loss(y_true, y_pred_logits, gamma=2.0, alpha=0.75)
    dice  = 1.0 - _soft_dice(y_true, y_pred_logits)
    return 0.5 * focal + 0.5 * dice


# ============================================================
# 2. FOCAL TVERSKY  (Attention UNet -- "tversky" in registry)
# ============================================================

def _tversky_index(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
                   alpha: float, beta: float,
                   smooth: float = 1e-6) -> tf.Tensor:
    """
    Tversky similarity index.

    TI = TP / (TP + alpha*FP + beta*FN)

    alpha : penalty on false positives.
    beta  : penalty on false negatives.

    For sparse foreground (rare actin pixels) you want beta >> alpha:
    missing a true actin pixel (FN) should cost more than including a
    false one (FP). Config v4 sets TVERSKY_ALPHA=0.3, TVERSKY_BETA=0.7.
    """
    y_pred_prob = tf.sigmoid(y_pred_logits)
    y_true_f    = tf.cast(tf.reshape(y_true,           [-1]), tf.float32)
    y_pred_f    = tf.reshape(y_pred_prob, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def focal_tversky_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
                       alpha: float, beta: float,
                       gamma: float = 0.75) -> tf.Tensor:
    """
    Focal Tversky Loss (Abraham & Khan, 2019).

    FTL = (1 - TI)^gamma

    gamma < 1 emphasises hard examples (small or broken actin regions)
    more than standard Tversky. gamma=0.75 is standard for medical
    segmentation.
    """
    ti = _tversky_index(y_true, y_pred_logits, alpha, beta)
    return tf.pow(tf.clip_by_value(1.0 - ti, 1e-7, 1.0), gamma)


# ============================================================
# 3. WEIGHTED-CE + DICE  (ResUNet++ -- "combo" in registry)
# ============================================================

def combo_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor,
               alpha: float, pos_weight: float) -> tf.Tensor:
    """
    alpha * WeightedCE + (1 - alpha) * (1 - SoftDice)

    pos_weight : multiplicative upweight for foreground pixels in the CE
                 term.  Derived from config.COMBO_CE_BETA (see get_loss_fn):
                 a beta of 0.9 maps to pos_weight = (1-0.9)/0.9 * 9 ≈ 10,
                 meaning foreground pixels are weighted 10x in the CE loss.
                 Adjust COMBO_CE_BETA in config to tune aggressiveness.

    tf.nn.weighted_cross_entropy_with_logits is numerically stable and
    handles the pos_weight scaling internally.
    """
    y_true_cast = tf.cast(y_true, tf.float32)

    wce = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true_cast,
            logits=y_pred_logits,
            pos_weight=pos_weight,
        )
    )
    dice = 1.0 - _soft_dice(y_true, y_pred_logits)

    return alpha * wce + (1.0 - alpha) * dice


# ============================================================
# FACTORY
# ============================================================

def get_loss_fn(loss_name: str, config):
    """
    Returns a compiled loss function (y_true, y_pred_logits) -> scalar.

    Called by model_builder.build_and_compile() once per model.

    loss_name   model           loss used
    ---------   -----           ---------
    bce_dice    UNet++          Focal + Dice
    tversky     Attention UNet  Focal Tversky
    combo       ResUNet++       Weighted-CE + Dice
    """
    if loss_name == "bce_dice":
        # UNet++: Focal + Dice
        # (replaces plain BCE + Dice which collapsed to all-zeros)
        def _loss(y_true, y_pred):
            return focal_dice_loss(y_true, y_pred)
        _loss.__name__ = "focal_dice_loss"
        return _loss

    elif loss_name == "tversky":
        # Attention UNet: Focal Tversky
        # config.TVERSKY_ALPHA = 0.3 (low FP penalty)
        # config.TVERSKY_BETA  = 0.7 (high FN penalty -- corrected from 0.3)
        alpha = config.TVERSKY_ALPHA
        beta  = config.TVERSKY_BETA

        def _loss(y_true, y_pred):
            return focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)
        _loss.__name__ = "focal_tversky_loss"
        return _loss

    elif loss_name == "combo":
        # ResUNet++: Weighted-CE + Dice
        # pos_weight derived from COMBO_CE_BETA:
        #   beta = foreground target fraction (e.g. 0.9 means "treat foreground
        #          as 90% of the loss budget regardless of pixel count")
        #   pos_weight = (1 - beta) / beta * normalisation
        #   With beta=0.9: pos_weight = 0.1 / 0.9 * 9 ≈ 1... too low.
        #   We use pos_weight directly = 1/beta - 1 mapped to a useful range:
        #   beta=0.9 -> pos_weight = (0.9 / 0.1) = 9
        #   This upweights foreground 9x in the CE term.
        beta       = config.COMBO_CE_BETA       # expected ~0.9
        pos_weight = beta / max(1.0 - beta, 1e-6)   # ~9 for beta=0.9
        pos_weight = float(tf.clip_by_value(pos_weight, 1.0, 100.0))
        alpha      = config.COMBO_ALPHA

        def _loss(y_true, y_pred):
            return combo_loss(y_true, y_pred, alpha=alpha, pos_weight=pos_weight)
        _loss.__name__ = "combo_loss"
        return _loss

    else:
        raise ValueError(
            f"Unknown loss_name {loss_name!r}. "
            f"Expected one of: 'bce_dice', 'tversky', 'combo'."
        )
