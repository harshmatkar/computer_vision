"""
models/ablation/attunet.py
--------------------------
C (att chain): U-Net + attention gates on the skip connections.
"""

from tensorflow.keras import layers, Model
from .blocks import conv_block, attention_gate, width

NAME = "attunet"
LABEL = "Att-UNet"
DISPLAY = "+ attention gates"


def build(input_shape=(None, None, 1), base_filters=16, depth=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = conv_block(x, width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = conv_block(x, width(base_filters, depth))

    # decoder: each skip is filtered by an attention gate before concatenation
    for d in reversed(range(depth)):
        f = width(base_filters, d)
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        skip = attention_gate(skips[d], x, max(f // 2, 1))
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")
