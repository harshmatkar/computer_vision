"""
model_builder.py
------------------
Central place that maps a model name ('unetpp', 'attention_unet',
'resunetpp') to its constructor and its associated loss function, so
train.py / evaluate.py don't need to hard-code architecture-specific
branching logic in more than one place.
"""

import config
from models.unet_plus_plus import LightUNetPlusPlus
from models.attention_unet import AttentionUNet
from models.resunet_pp import ResUNetPlusPlus
from utils.losses import get_loss_fn


def build_model(model_name: str):
    """
    Instantiates the requested model on config.DEVICE.

    Parameters
    ----------
    model_name : str
        One of the keys in config.MODEL_REGISTRY: 'unetpp', 'attention_unet',
        'resunetpp'.

    Returns
    -------
    torch.nn.Module
    """
    if model_name not in config.MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Valid options: {list(config.MODEL_REGISTRY.keys())}"
        )

    if model_name == "unetpp":
        model = LightUNetPlusPlus(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=config.DEPTH,
            deep_supervision=config.USE_DEEP_SUPERVISION,
        )
    elif model_name == "attention_unet":
        model = AttentionUNet(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=config.DEPTH,
        )
    elif model_name == "resunetpp":
        model = ResUNetPlusPlus(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_filters=config.BASE_FILTERS,
            depth=3,  # ResUNet++ uses 3 encoder stages in the original paper
        )

    return model.to(config.DEVICE)


def build_loss(model_name: str):
    """Returns the configured loss_fn(pred_logits, target) callable for this model."""
    loss_name = config.MODEL_REGISTRY[model_name]["loss"]
    return get_loss_fn(loss_name, config)


def get_final_output(outputs):
    """
    Normalizes model output to a single logits tensor.
    Only Member A's UNet++ returns a list (deep supervision); Members B and C
    return a single tensor directly.
    """
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def compute_loss(model_name: str, loss_fn, outputs, target):
    """
    Applies `loss_fn` to model outputs, averaging across all deep-supervision
    heads if the model returns a list (UNet++), or applying it directly to
    a single logits tensor otherwise (Attention U-Net, ResUNet++).
    """
    if isinstance(outputs, (list, tuple)):
        losses = [loss_fn(out, target) for out in outputs]
        import torch
        return torch.stack(losses).mean()
    else:
        return loss_fn(outputs, target)
