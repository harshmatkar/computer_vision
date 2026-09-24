"""
models/attention_unet.py
--------------------------
Attention U-Net (Oktay et al., "Attention U-Net: Learning Where to Look for
the Pancreas," arXiv:1804.03999, 2018), implemented in TensorFlow/Keras
using the Functional API.

Core idea
---------
A standard U-Net skip connection copies the encoder feature map at a given
depth directly to the decoder. Attention U-Net inserts an "Attention Gate"
(AG) on every skip connection: the gate looks at both the incoming encoder
features AND the coarser, more semantically-refined decoder features from
one level deeper, and produces a per-pixel gating coefficient in [0, 1]
that suppresses activations in regions irrelevant to the segmentation
target (here: the cytosolic interior) while preserving activations near
the membrane/actin-edge region.

This is a genuinely different mechanism from the nested skip pathways used
in Member A's UNet++: attention gating reweights WHERE in the image to
trust encoder features, whereas UNet++ changes HOW MANY intermediate
processing steps a skip connection passes through.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters: int, name_prefix: str):
    """Two stacked standard 3x3 Conv -> BN -> ReLU layers."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    return x


def attention_gate(g, x, inter_channels: int, name_prefix: str):
    """
    Attention Gate (AG) as described in Oktay et al., 2018.

    Parameters
    ----------
    g : gating signal from the decoder (coarser resolution, already
        upsampled to match `x`'s spatial size before calling this function).
    x : encoder skip-connection features (finer resolution).
    inter_channels : number of channels in the shared intermediate space
        where `g` and `x` are compared.

    Returns
    -------
    x * alpha : the skip features re-weighted by the learned attention map,
                same shape as `x`.
    """
    theta_g = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name_prefix}_theta_g")(g)
    theta_g = layers.BatchNormalization(name=f"{name_prefix}_bn_g")(theta_g)

    phi_x = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name_prefix}_phi_x")(x)
    phi_x = layers.BatchNormalization(name=f"{name_prefix}_bn_x")(phi_x)

    # Additive attention: combine gate and skip signals, then squash to a
    # single-channel spatial attention map alpha in [0, 1].
    psi_in = layers.ReLU(name=f"{name_prefix}_relu")(layers.Add(name=f"{name_prefix}_add")([theta_g, phi_x]))
    psi = layers.Conv2D(1, 1, padding="same", name=f"{name_prefix}_psi")(psi_in)
    psi = layers.BatchNormalization(name=f"{name_prefix}_bn_psi")(psi)
    alpha = layers.Activation("sigmoid", name=f"{name_prefix}_sigmoid")(psi)

    return layers.Multiply(name=f"{name_prefix}_gated")([x, alpha])


def build_attention_unet(input_shape=(256, 256, 1), out_channels=1, base_filters=16,
                          depth=4) -> Model:
    """
    Builds the full Attention U-Net Keras model: standard convolutional
    encoder/decoder with an attention_gate() applied to every skip
    connection before concatenation.

    Parameters mirror models/unet_plus_plus.py for a fair parameter-count
    comparison across all three team-member models.
    """
    filters = [base_filters * (2 ** i) for i in range(depth + 1)]

    inputs = layers.Input(shape=input_shape, name="input_image")

    # --- Encoder, caching skip features at every depth -------------------------
    skips = []
    cur = inputs
    for i in range(depth + 1):
        cur = conv_block(cur, filters[i], name_prefix=f"enc_{i}")
        skips.append(cur)
        if i < depth:
            cur = layers.MaxPooling2D(pool_size=2, name=f"pool_{i}")(cur)

    # --- Decoder with attention-gated skip fusion -----------------------------
    d = skips[-1]  # bottleneck features
    for depth_i in reversed(range(depth)):
        g = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"up_{depth_i}")(d)
        skip = skips[depth_i]
        gated_skip = attention_gate(g, skip, inter_channels=filters[depth_i] // 2,
                                     name_prefix=f"ag_{depth_i}")
        d = layers.Concatenate(name=f"concat_{depth_i}")([gated_skip, g])
        d = conv_block(d, filters[depth_i], name_prefix=f"dec_{depth_i}")

    outputs = layers.Conv2D(out_channels, kernel_size=1, name="output_final")(d)

    model = Model(inputs=inputs, outputs=outputs, name="AttentionUNet")
    return model


def count_parameters(model: Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


if __name__ == "__main__":
    model = build_attention_unet(input_shape=(256, 256, 1), out_channels=1,
                                  base_filters=16, depth=4)
    dummy = tf.random.normal((2, 256, 256, 1))
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
