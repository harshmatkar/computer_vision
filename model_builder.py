"""
model_builder.py
------------------
Central place that maps a model name ('unetpp', 'attention_unet',
'resunetpp') to its Keras architecture, its associated loss function, and
a ready-to-train compiled model, so train.py / evaluate.py don't need
architecture-specific branching logic in more than one place.

Member A (UNet++) is a multi-output Keras model (deep supervision: one
logits head per supervised decoder depth); Members B and C are
single-output models. `is_multi_output()` / `num_outputs()` let train.py
adapt the tf.data pipeline (duplicating the target mask once per output)
without needing to know which architecture is currently selected.
"""

import config
from models.unet_plus_plus import build_unet_plus_plus
from models.attention_unet import build_attention_unet
from models.resunet_pp import build_resunet_pp
from utils.losses import get_loss_fn, dice_metric


def build_model(model_name: str):
    """
    Instantiates the requested (uncompiled) Keras model.

    Parameters
    ----------
    model_name : str
        One of the keys in config.MODEL_REGISTRY: 'unetpp', 'attention_unet',
        'resunetpp'.

    Returns
    -------
    tf.keras.Model
    """
    if model_name not in config.MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Valid options: {list(config.MODEL_REGISTRY.keys())}"
        )

    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.IN_CHANNELS)

    if model_name == "unetpp":
        model = build_unet_plus_plus(
            input_shape=input_shape,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=config.DEPTH,
            deep_supervision=config.USE_DEEP_SUPERVISION,
        )
    elif model_name == "attention_unet":
        model = build_attention_unet(
            input_shape=input_shape,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=config.DEPTH,
        )
    elif model_name == "resunetpp":
        model = build_resunet_pp(
            input_shape=input_shape,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=3,  # ResUNet++ uses 3 encoder stages in the original paper
        )

    return model


def is_multi_output(model_name: str) -> bool:
    """Only Member A's UNet++ uses deep supervision (multiple output heads)."""
    return model_name == "unetpp" and config.USE_DEEP_SUPERVISION


def num_outputs(model_name: str) -> int:
    """Number of output heads for this model (used to shape the tf.data targets)."""
    if is_multi_output(model_name):
        return config.DEPTH
    return 1


def build_and_compile(model_name: str):
    """
    Builds the requested model and compiles it with its registered loss
    function, the Adam optimizer, and Dice-coefficient tracking.

    For the multi-output UNet++ (deep supervision), the same loss and
    metric are applied to every output head with equal loss_weights, so
    Keras averages the loss across all supervised depths internally --
    equivalent to manually averaging a list of per-depth losses.

    Returns
    -------
    model : compiled tf.keras.Model
    """
    import tensorflow as tf

    model = build_model(model_name)
    loss_fn = get_loss_fn(config.MODEL_REGISTRY[model_name]["loss"], config)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    if is_multi_output(model_name):
        n_out = num_outputs(model_name)
        model.compile(
            optimizer=optimizer,
            loss=[loss_fn] * n_out,
            loss_weights=[1.0 / n_out] * n_out,
            metrics=[[dice_metric]] * n_out,
        )
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[dice_metric])

    return model
