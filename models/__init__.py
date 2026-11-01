"""
models/ablation/
----------------
One file per ablation model (shared building blocks in blocks.py):

    encdec.py        encdec    encoder-decoder, no skips
    unet.py          unet      + skip connections
    resunet.py       resunet   + residual blocks
    resunet_se.py    resse     + SE in the residual blocks
    attunet.py       attunet   + attention gates
    attunet_aspp.py  attaspp   + ASPP bridge

Chains (each model adds ONE component to the previous one):
    res:  encdec -> unet -> resunet -> resse
    att:  encdec -> unet -> attunet -> attaspp

All models output LOGITS and are built with input_shape=(None, None, 1).

    from models.ablation import build_ablation_model, ABLATION_VARIANTS
"""

def build_ablation_model(variant, input_shape=(None, None, 1),
                         base_filters=16, depth=4):
    """Build a model by name: encdec | unet | resunet | resse | attunet | attaspp."""
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant '{variant}'; choose from {list(BUILDERS)}")
    return BUILDERS[variant](input_shape=input_shape,
                             base_filters=base_filters, depth=depth)


def count_parameters(model):
    return int(model.count_params())
