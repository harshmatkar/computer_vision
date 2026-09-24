"""
models/resunet_pp.py
----------------------
ResUNet++ (Jha, D. et al., "ResUNet++: An Advanced Architecture for Medical
Image Segmentation," IEEE International Symposium on Multimedia (ISM), 2019),
implemented in TensorFlow/Keras using the Functional API.

Combines four separate ideas from four separate papers into one network:

  1. Residual blocks        (He et al., ResNet lineage / Zhang et al. 2018
                              "Road Extraction by Deep Residual U-Net")
  2. Squeeze-and-Excitation  (Hu, Shen & Sun, CVPR 2018)
  3. Atrous Spatial Pyramid Pooling / ASPP (Chen et al., DeepLab lineage)
  4. Attention-gated decoder skip connections (same idea family as
     Attention U-Net, applied at the decoder stage only)

This is architecturally distinct from both Member A (nested dense skips,
depthwise-separable convs) and Member B (plain encoder/decoder + attention
gates only): ResUNet++ additionally reweights features along the CHANNEL
axis (via Squeeze-and-Excitation) and captures multi-scale context at the
bridge via parallel dilated convolutions (ASPP), neither of which the other
two models do.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def squeeze_excitation(x, reduction: int = 8, name_prefix: str = "se"):
    """
    Squeeze-and-Excitation block (Hu, Shen & Sun, CVPR 2018).

    "Squeeze": global average pooling collapses each channel's spatial
    extent (H x W) down to a single scalar, producing a channel descriptor
    that summarizes the global response of that channel across the image.

    "Excitation": a small two-layer bottleneck (1x1 convs here) maps that
    channel descriptor through a reduced dimension and back out to a
    per-channel scaling factor in [0, 1] via a sigmoid, broadcast-multiplied
    back onto the original feature map. Channels the network finds more
    informative for the current input are scaled up; less relevant channels
    are scaled down.
    """
    channels = x.shape[-1]
    reduced = max(1, channels // reduction)

    s = layers.GlobalAveragePooling2D(name=f"{name_prefix}_pool")(x)
    s = layers.Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(s)
    s = layers.Conv2D(reduced, 1, activation="relu", name=f"{name_prefix}_fc1")(s)
    s = layers.Conv2D(channels, 1, activation="sigmoid", name=f"{name_prefix}_fc2")(s)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, s])


def residual_se_block(x, out_channels: int, stride: int = 1, name_prefix: str = "res"):
    """
    Residual block with an SE block applied to the residual branch before
    the skip-addition, matching the "residual + squeeze-excitation" encoder
    unit used throughout ResUNet++.
    """
    in_channels = x.shape[-1]

    residual = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    residual = layers.ReLU(name=f"{name_prefix}_relu1")(residual)
    residual = layers.Conv2D(out_channels, 3, strides=stride, padding="same", use_bias=False,
                              name=f"{name_prefix}_conv1")(residual)

    residual = layers.BatchNormalization(name=f"{name_prefix}_bn2")(residual)
    residual = layers.ReLU(name=f"{name_prefix}_relu2")(residual)
    residual = layers.Conv2D(out_channels, 3, padding="same", use_bias=False,
                              name=f"{name_prefix}_conv2")(residual)

    residual = squeeze_excitation(residual, name_prefix=f"{name_prefix}_se")

    # Projection shortcut when input/output channel counts or spatial
    # resolution differ (standard ResNet-style identity/projection choice).
    if stride != 1 or in_channels != out_channels:
        shortcut = layers.Conv2D(out_channels, 1, strides=stride, use_bias=False,
                                  name=f"{name_prefix}_shortcut")(x)
    else:
        shortcut = x

    return layers.Add(name=f"{name_prefix}_add")([residual, shortcut])


def aspp_block(x, out_channels: int, dilations=(1, 6, 12, 18), name_prefix: str = "aspp"):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., DeepLab lineage).

    Runs several atrous (dilated) 3x3 convolutions in parallel at different
    dilation rates -- each rate expands the receptive field without adding
    parameters or losing spatial resolution the way pooling would -- plus a
    global-average-pooling branch for whole-image context, then fuses all
    branches with a final 1x1 convolution.
    """
    h, w = x.shape[1], x.shape[2]
    branches = []
    for idx, d in enumerate(dilations):
        if d == 1:
            b = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name_prefix}_b{idx}_conv")(x)
        else:
            b = layers.Conv2D(out_channels, 3, padding="same", dilation_rate=d, use_bias=False,
                               name=f"{name_prefix}_b{idx}_conv")(x)
        b = layers.BatchNormalization(name=f"{name_prefix}_b{idx}_bn")(b)
        b = layers.ReLU(name=f"{name_prefix}_b{idx}_relu")(b)
        branches.append(b)

    g = layers.GlobalAveragePooling2D(name=f"{name_prefix}_global_pool")(x)
    g = layers.Reshape((1, 1, x.shape[-1]), name=f"{name_prefix}_global_reshape")(g)
    g = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name_prefix}_global_conv")(g)
    g = layers.ReLU(name=f"{name_prefix}_global_relu")(g)
    g = layers.Resizing(h, w, interpolation="bilinear", name=f"{name_prefix}_global_resize")(g)
    branches.append(g)

    concat = layers.Concatenate(name=f"{name_prefix}_concat")(branches)
    out = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name_prefix}_project_conv")(concat)
    out = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(out)
    out = layers.ReLU(name=f"{name_prefix}_project_relu")(out)
    return out


def decoder_attention_block(skip, gate, name_prefix: str = "dec_attn"):
    """
    Lightweight attention block applied to the encoder skip connection
    before it is concatenated into the decoder, in the same spirit as
    Attention U-Net's gate but implemented via the simpler channel+spatial
    squeeze used in the original ResUNet++ decoder.
    """
    skip_channels = skip.shape[-1]
    h, w = skip.shape[1], skip.shape[2]

    g = layers.Resizing(h, w, interpolation="bilinear", name=f"{name_prefix}_resize")(gate)
    g = layers.BatchNormalization(name=f"{name_prefix}_bn_g")(g)
    g = layers.ReLU(name=f"{name_prefix}_relu_g")(g)
    g = layers.Conv2D(skip_channels, 3, padding="same", use_bias=False,
                       name=f"{name_prefix}_conv_g")(g)

    s = layers.BatchNormalization(name=f"{name_prefix}_bn_s")(skip)
    s = layers.ReLU(name=f"{name_prefix}_relu_s")(s)
    s = layers.Conv2D(skip_channels, 3, padding="same", use_bias=False,
                       name=f"{name_prefix}_conv_s")(s)

    combined = layers.Add(name=f"{name_prefix}_add")([g, s])
    combined = layers.BatchNormalization(name=f"{name_prefix}_bn_c")(combined)
    combined = layers.ReLU(name=f"{name_prefix}_relu_c")(combined)
    alpha = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name_prefix}_alpha")(combined)

    return layers.Multiply(name=f"{name_prefix}_gated")([skip, alpha])


def build_resunet_pp(input_shape=(256, 256, 1), out_channels=1, base_filters=16,
                      depth=3) -> Model:
    """
    Builds the full ResUNet++ Keras model: stem block -> `depth` residual-SE
    encoder stages -> ASPP bridge -> `depth` attention-gated decoder stages
    -> output ASPP -> 1x1 output head.

    Parameters mirror unet_plus_plus.py and attention_unet.py for a fair
    three-way parameter-count comparison in the final report.
    """
    filters = [base_filters * (2 ** i) for i in range(depth + 1)]  # e.g. [16, 32, 64, 128]

    inputs = layers.Input(shape=input_shape, name="input_image")

    # --- Stem block (initial feature extraction, not yet residual) -----------
    stem = layers.Conv2D(filters[0], 3, padding="same", use_bias=False, name="stem_conv1")(inputs)
    stem = layers.BatchNormalization(name="stem_bn")(stem)
    stem = layers.ReLU(name="stem_relu")(stem)
    stem = layers.Conv2D(filters[0], 3, padding="same", use_bias=False, name="stem_conv2")(stem)
    stem_shortcut = layers.Conv2D(filters[0], 1, use_bias=False, name="stem_shortcut")(inputs)
    stem_out = layers.Add(name="stem_add")([stem, stem_shortcut])

    # --- Encoder, caching skip features ---------------------------------------
    skips = [stem_out]
    cur = stem_out
    for i in range(1, depth + 1):
        cur = layers.MaxPooling2D(pool_size=2, name=f"pool_{i}")(cur)
        cur = residual_se_block(cur, filters[i], name_prefix=f"enc_{i}")
        skips.append(cur)

    # --- Bridge: ASPP -----------------------------------------------------------
    bridge = aspp_block(cur, filters[depth], name_prefix="aspp_bridge")

    # --- Decoder: attention-gated skip fusion + residual blocks ----------------
    d = bridge
    for depth_i in reversed(range(depth)):
        skip = skips[depth_i]
        gated_skip = decoder_attention_block(skip, d, name_prefix=f"dec_attn_{depth_i}")
        h, w = skip.shape[1], skip.shape[2]
        d_up = layers.Resizing(h, w, interpolation="bilinear", name=f"up_{depth_i}")(d)
        d = layers.Concatenate(name=f"concat_{depth_i}")([gated_skip, d_up])
        d = residual_se_block(d, filters[depth_i], name_prefix=f"dec_{depth_i}")

    # --- Output ASPP + head (as in the original paper's final stage) -----------
    out = aspp_block(d, filters[0], name_prefix="aspp_out")
    outputs = layers.Conv2D(out_channels, 1, name="output_final")(out)

    model = Model(inputs=inputs, outputs=outputs, name="ResUNetPlusPlus")
    return model


def count_parameters(model: Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


if __name__ == "__main__":
    model = build_resunet_pp(input_shape=(256, 256, 1), out_channels=1,
                              base_filters=16, depth=3)
    dummy = tf.random.normal((2, 256, 256, 1))
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
