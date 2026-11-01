"""
models/models.py
----------------
All ablation models in one file (building blocks are in blocks.py).

    encdec   encoder-decoder, no skip connections
    unet     + skip connections
    resunet  + residual blocks
    resse    + SE in the residual blocks      (alias: resunetse)
    attunet  + attention gates
    attaspp  + ASPP bridge                    (alias: attunet_aspp)

All models output LOGITS (no sigmoid) and are built with input_shape=(None, None, 1).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from headers import *
from headers import ALL_MODELS

class blocks:
    @staticmethod
    def width(base, d):
        """Channels at level d. base may be a float (used for parameter matching)."""
        return max(2, int(round(base * (2 ** d))))

    @staticmethod
    def conv_block(x, filters):
        """Plain block: (conv3x3 - BN - ReLU) x 2."""
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        return x

    @staticmethod
    def se_block(x, ratio=8):
        """Squeeze-and-excitation: re-weight channels with a learned 0-1 gate."""
        c = int(x.shape[-1])
        s = layers.GlobalAveragePooling2D()(x)
        s = layers.Dense(max(c // ratio, 4), activation="relu")(s)
        s = layers.Dense(c, activation="sigmoid")(s)
        s = layers.Reshape((1, 1, c))(s)
        return layers.Multiply()([x, s])

    @staticmethod
    def _shortcut(x, filters):
        """Identity shortcut, or 1x1 conv + BN when the channel count changes."""
        if int(x.shape[-1]) != filters:
            x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
        return x

    @staticmethod
    def res_block(x, filters):
        """Residual block: ReLU( F(x) + shortcut(x) ), F = conv-BN-ReLU-conv-BN."""
        shortcut = blocks._shortcut(x, filters)
        y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Add()([y, shortcut])
        return layers.Activation("relu")(y)

    @staticmethod
    def res_se_block(x, filters):
        """Residual block with SE applied to the residual branch before the add."""
        shortcut = blocks._shortcut(x, filters)
        y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = blocks.se_block(y)
        y = layers.Add()([y, shortcut])
        return layers.Activation("relu")(y)

    @staticmethod
    def attention_gate(skip, gating, inter_filters):
        """Additive attention gate (Oktay et al., 2018).
        skip and gating are at the same resolution (gating = upsampled decoder
        feature). Returns skip multiplied by a 0-1 attention map."""
        theta = layers.Conv2D(inter_filters, 1, padding="same")(skip)
        phi = layers.Conv2D(inter_filters, 1, padding="same")(gating)
        a = layers.Activation("relu")(layers.Add()([theta, phi]))
        psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(a)
        return layers.Multiply()([skip, psi])

    @staticmethod
    def aspp_block(x, filters, rates=(1, 2, 4)):
        """ASPP bridge: a 1x1 conv and parallel 3x3 atrous convs (several
        dilation rates) are concatenated and fused by a 1x1 conv."""
        branches = []
        b = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        b = layers.BatchNormalization()(b)
        branches.append(layers.Activation("relu")(b))
        for r in rates:
            b = layers.Conv2D(filters, 3, padding="same", dilation_rate=r, use_bias=False)(x)
            b = layers.BatchNormalization()(b)
            branches.append(layers.Activation("relu")(b))
        x = layers.Concatenate()(branches)
        x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

def fcn(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "fcn"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder: standard conv blocks + max pooling
    for d in range(depth):
        x = blocks.conv_block(x, blocks.width(base_filters, d))
        x = layers.MaxPooling2D(2)(x)

    # bridge (ASPP or standard conv block; using conv_block here)
    x = blocks.conv_block(x, blocks.width(base_filters, depth))

    # decoder: upsample all the way back to the original input resolution
    x = layers.UpSampling2D(size=(2 ** depth), interpolation="bilinear")(x)

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")

def unet(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "unet"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = blocks.conv_block(x, blocks.width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = blocks.conv_block(x, blocks.width(base_filters, depth))

    # decoder
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = blocks.conv_block(x, blocks.width(base_filters, d))

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")

def resunet(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "resunet"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = blocks.res_block(x, blocks.width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = blocks.res_block(x, blocks.width(base_filters, depth))

    # decoder
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = blocks.res_block(x, blocks.width(base_filters, d))

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")


def resunetse(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "resse"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = blocks.res_se_block(x, blocks.width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = blocks.res_se_block(x, blocks.width(base_filters, depth))

    # decoder
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = blocks.res_se_block(x, blocks.width(base_filters, d))

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")


def attunet(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "attunet"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = blocks.conv_block(x, blocks.width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = blocks.conv_block(x, blocks.width(base_filters, depth))

    # decoder: each skip is filtered by an attention gate before concatenation
    for d in reversed(range(depth)):
        f = blocks.width(base_filters, d)
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        skip = blocks.attention_gate(skips[d], x, max(f // 2, 1))
        x = layers.Concatenate()([x, skip])
        x = blocks.conv_block(x, f)

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")

def attunetaspp(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "attaspp"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    skips = []
    for d in range(depth):
        x = blocks.conv_block(x, blocks.width(base_filters, d))
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # bridge: ASPP
    x = blocks.aspp_block(x, blocks.width(base_filters, depth))

    # decoder with attention gates
    for d in reversed(range(depth)):
        f = blocks.width(base_filters, d)
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        skip = blocks.attention_gate(skips[d], x, max(f // 2, 1))
        x = layers.Concatenate()([x, skip])
        x = blocks.conv_block(x, f)

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")

def encdec(input_shape=(None, None, 1), base_filters=16, depth=4):
    NAME = "encdec"
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # encoder
    for d in range(depth):
        x = blocks.conv_block(x, blocks.width(base_filters, d))
        x = layers.MaxPooling2D(2)(x)

    # bridge
    x = blocks.conv_block(x, blocks.width(base_filters, depth))

    # decoder (nothing to concatenate)
    for d in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = blocks.conv_block(x, blocks.width(base_filters, d))

    outputs = layers.Conv2D(1, 1, padding="same")(x)      # logits
    return Model(inputs, outputs, name=f"ablation_{NAME}")

def count_parameters(model: Model) -> int:
    """Returns the total number of trainable parameters in the model."""
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))

def get_model(model_name, input_shape=(None, None, 1), base_filters=16, depth=4):
    """Factory function to build network architectures by name."""
    name = model_name.lower()
    if name not in ALL_MODELS:
        raise ValueError(f"Unsupported model name: '{model_name}'. Choose from {ALL_MODELS}")
    
    model_func = {
        "fcn": fcn,
        "unet": unet, # type: ignore
        "resunet": resunet, # type: ignore
        "resse": resunetse,# type: ignore
        "attunet": attunet,# type: ignore
        "attaspp": attunetaspp,# type: ignore
        "encdec": encdec# type: ignore
    }

    return model_func[name](input_shape=input_shape, base_filters=base_filters, depth=depth)
