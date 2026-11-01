"""
models/ablation.py
------------------
Registry for the ablation models when the model files sit DIRECTLY in models/
(flat layout):

    models/blocks.py  encdec.py  unet.py  resunet.py  resunet_se.py
    models/attunet.py  attunet_aspp.py  ablation.py   (this file)

    from models.ablation import build_ablation_model, ABLATION_VARIANTS

Models:   encdec | unet | resunet | resse | attunet | attaspp
All output LOGITS and are built with input_shape=(None, None, 1).
"""

from . import encdec, unet, resunet, resunet_se, attunet, attunet_aspp

_MODULES = {m.NAME: m for m in
            (encdec, unet, resunet, resunet_se, attunet, attunet_aspp)}

ABLATION_VARIANTS = {k: dict(label=m.LABEL, display=m.DISPLAY)
                     for k, m in _MODULES.items()}
BUILDERS = {k: m.build for k, m in _MODULES.items()}


def build_ablation_model(variant, input_shape=(None, None, 1),
                         base_filters=16, depth=4):
    """Build a model by name: encdec | unet | resunet | resse | attunet | attaspp."""
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant '{variant}'; choose from {list(BUILDERS)}")
    return BUILDERS[variant](input_shape=input_shape,
                             base_filters=base_filters, depth=depth)


def count_parameters(model):
    return int(model.count_params())