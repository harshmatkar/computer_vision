"""
utils/ablation_losses.py
------------------------
Loss used for every model in the ablation / final training, so the loss
is never a confounder. Models output LOGITS (no sigmoid).
"""

import tensorflow as tf


def bce_dice_loss(y_true, logits):
    """Binary cross-entropy (from logits) + soft Dice loss."""
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, logits, from_logits=True))
    prob = tf.sigmoid(logits)
    inter = tf.reduce_sum(prob * y_true, axis=[1, 2, 3])
    denom = tf.reduce_sum(prob, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3])
    dice = 1.0 - tf.reduce_mean((2.0 * inter + 1.0) / (denom + 1.0))
    return bce + dice