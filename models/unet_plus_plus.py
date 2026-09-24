"""
models/unet_plus_plus.py
--------------------------
Lightweight UNet++ (Zhou et al., "UNet++: A Nested U-Net Architecture for
Medical Image Segmentation," DLMIA 2018) implemented in TensorFlow/Keras
using the Functional API, with SeparableConv2D (depthwise separable
convolutions) in every conv block.

Why Depthwise Separable Convolutions?
--------------------------------------
A standard 3x3 conv with C_in -> C_out channels costs:
    3 * 3 * C_in * C_out            multiply-accumulate ops per pixel
A depthwise separable conv (depthwise 3x3 + pointwise 1x1) costs:
    3 * 3 * C_in  +  C_in * C_out   multiply-accumulate ops per pixel
Keras's `SeparableConv2D` layer implements exactly this decomposition in a
single call, which is why this file is noticeably shorter than an
equivalent from-scratch PyTorch implementation. For typical channel widths
used here (16-256), this is roughly an 8-9x reduction in FLOPs/parameters
per layer versus a standard Conv2D of the same width.

Why UNet++ (nested skip connections) instead of vanilla UNet?
---------------------------------------------------------------
Vanilla UNet skips features directly from encoder stage i to decoder stage
i, forcing the decoder to fuse features of very different semantic depth
in one step. UNet++ inserts a grid of intermediate convolutional nodes
(X^{i,j}) that progressively bridge that gap before the final fusion,
which empirically improves boundary delineation -- important here since we
are segmenting a thin actin edge band.

Deep supervision in Keras
---------------------------
The model below is a genuine multi-output Keras Model: every decoder node
along the top row (X^{0,1} ... X^{0,depth}) gets its own 1x1 output head.
train.py compiles the model with one loss (and equal loss_weights) per
output, so Keras averages the loss across all supervised depths internally
-- the same behavior as the manual list-of-tensors averaging used in a
from-scratch PyTorch version, but expressed natively through
`model.compile(loss=[...], loss_weights=[...])`.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters: int, name_prefix: str):
    """Two stacked SeparableConv2D -> BN -> ReLU layers -> one UNet++ node X^{i,j}."""
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False,
                                name=f"{name_prefix}_sepconv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False,
                                name=f"{name_prefix}_sepconv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    return x


def build_unet_plus_plus(input_shape=(256, 256, 1), out_channels=1, base_filters=16,
                          depth=4, deep_supervision=True) -> Model:
    """
    Builds the Lightweight UNet++ Keras model.

    Parameters
    ----------
    input_shape : tuple
        (H, W, C) of the input image; C=1 for single-channel fluorescence
        microscopy.
    out_channels : int
        Number of output segmentation channels (1 for binary actin-edge mask).
    base_filters : int
        Filters in the first encoder stage; doubles each stage.
    depth : int
        Number of encoder down-sampling stages (depth+1 total levels).
    deep_supervision : bool
        If True, returns a multi-output model with one logits head per
        supervised decoder depth (X^{0,1} .. X^{0,depth}); if False, a
        single-output model using only the deepest/most-refined head.

    Returns
    -------
    tf.keras.Model
    """
    filters = [base_filters * (2 ** i) for i in range(depth + 1)]

    inputs = layers.Input(shape=input_shape, name="input_image")
    nodes = {}  # nodes["i_j"] -> tensor for node X^{i,j}

    # --- Encoder column j = 0 -------------------------------------------------
    cur = inputs
    for i in range(depth + 1):
        if i > 0:
            cur = layers.MaxPooling2D(pool_size=2, name=f"pool_{i}")(nodes[f"{i-1}_0"])
        cur = conv_block(cur, filters[i], name_prefix=f"x{i}_0")
        nodes[f"{i}_0"] = cur

    # --- Nested decoder columns j = 1 .. depth --------------------------------
    for j in range(1, depth + 1):
        for i in range(depth + 1 - j):
            skip_tensors = [nodes[f"{i}_{k}"] for k in range(j)]
            upsampled = layers.UpSampling2D(size=2, interpolation="bilinear",
                                             name=f"up_{i}_{j}")(nodes[f"{i+1}_{j-1}"])
            concat = layers.Concatenate(name=f"concat_{i}_{j}")(skip_tensors + [upsampled])
            nodes[f"{i}_{j}"] = conv_block(concat, filters[i], name_prefix=f"x{i}_{j}")

    # --- Output head(s) ----------------------------------------------------------
    if deep_supervision:
        outputs = [
            layers.Conv2D(out_channels, kernel_size=1, name=f"output_{j}")(nodes[f"0_{j}"])
            for j in range(1, depth + 1)
        ]
    else:
        outputs = layers.Conv2D(out_channels, kernel_size=1, name="output_final")(nodes[f"0_{depth}"])

    model = Model(inputs=inputs, outputs=outputs, name="LightUNetPlusPlus")
    return model


def count_parameters(model: Model) -> int:
    """Returns the total number of trainable parameters in the model."""
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


if __name__ == "__main__":
    # Quick self-test: verify forward pass shapes and print parameter count.
    model = build_unet_plus_plus(input_shape=(256, 256, 1), out_channels=1,
                                  base_filters=16, depth=4, deep_supervision=True)
    dummy = tf.random.normal((2, 256, 256, 1))
    outputs = model(dummy)
    for idx, o in enumerate(outputs):
        print(f"Deep-supervision output {idx}: shape={tuple(o.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
