"""
models/resunet_pp_light.py
--------------------------
Lightweight ResUNet++ (Jha et al., ISM 2019) in TensorFlow/Keras.

Same architecture as resunet_pp.py (stem -> residual-SE encoder -> ASPP bridge
-> attention-gated residual-SE decoder -> output ASPP), with three
parameter-saving changes, each controlled by a flag so every change can be
ablated against the original:

  1. separable=True        every 3x3 conv (incl. dilated ASPP branches and
                           the decoder attention block) becomes a
                           SeparableConv2D: 9*Cin + Cin*Cout weights instead
                           of 9*Cin*Cout.
  2. aspp_dilations        original (1, 6, 12, 18) was tuned for DeepLab's
                           large feature maps. At the bridge here the map is
                           only 32x32 (256 / 2^3), so rate 18 reaches 37 px
                           -- most taps land on zero padding. (1, 2, 4) covers
                           the same map with real content.
  3. aspp_branch_ratio     each ASPP branch outputs ratio * out_channels
                           instead of out_channels. The final 1x1 projection
                           then fuses fewer channels and shrinks too.

Setting separable=False, aspp_dilations=(1, 6, 12, 18), aspp_branch_ratio=1.0
reproduces the original ResUNet++ exactly (the "brute-force" version).

Not changed: SE blocks (already 1x1 and tiny), the 1x1 residual shortcuts,
and the 1x1 output head -- depthwise-separable gives nothing for 1x1 kernels.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv3x3(x, filters, separable, name, strides=1, dilation=1):
    """Single swap point between standard and depthwise-separable 3x3 conv."""
    if separable:
        return layers.SeparableConv2D(filters, 3, strides=strides, padding="same",
                                      dilation_rate=dilation, use_bias=False,
                                      name=name)(x)
    return layers.Conv2D(filters, 3, strides=strides, padding="same",
                         dilation_rate=dilation, use_bias=False, name=name)(x)


def squeeze_excitation(x, reduction=8, name_prefix="se"):
    channels = x.shape[-1]
    reduced = max(1, channels // reduction)
    s = layers.GlobalAveragePooling2D(name=f"{name_prefix}_pool")(x)
    s = layers.Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(s)
    s = layers.Conv2D(reduced, 1, activation="relu", name=f"{name_prefix}_fc1")(s)
    s = layers.Conv2D(channels, 1, activation="sigmoid", name=f"{name_prefix}_fc2")(s)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, s])


def residual_se_block(x, out_channels, separable, stride=1, name_prefix="res"):
    """Pre-activation residual block with SE on the residual branch."""
    in_channels = x.shape[-1]

    r = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    r = layers.ReLU(name=f"{name_prefix}_relu1")(r)
    r = _conv3x3(r, out_channels, separable, f"{name_prefix}_conv1", strides=stride)
    r = layers.BatchNormalization(name=f"{name_prefix}_bn2")(r)
    r = layers.ReLU(name=f"{name_prefix}_relu2")(r)
    r = _conv3x3(r, out_channels, separable, f"{name_prefix}_conv2")
    r = squeeze_excitation(r, name_prefix=f"{name_prefix}_se")

    if stride != 1 or in_channels != out_channels:
        shortcut = layers.Conv2D(out_channels, 1, strides=stride, use_bias=False,
                                 name=f"{name_prefix}_shortcut")(x)
    else:
        shortcut = x
    return layers.Add(name=f"{name_prefix}_add")([r, shortcut])


def aspp_block(x, out_channels, separable, dilations=(1, 2, 4),
               branch_ratio=0.5, name_prefix="aspp"):
    """
    ASPP: parallel dilated convs + global-pool branch, fused by a 1x1 conv.
    Each branch has `branch_ratio * out_channels` channels; the fused output
    has `out_channels`.
    """
    h, w = x.shape[1], x.shape[2]
    branch_ch = max(8, int(out_channels * branch_ratio))

    branches = []
    for idx, d in enumerate(dilations):
        if d == 1:
            b = layers.Conv2D(branch_ch, 1, use_bias=False,
                              name=f"{name_prefix}_b{idx}_conv")(x)
        else:
            b = _conv3x3(x, branch_ch, separable, f"{name_prefix}_b{idx}_conv", dilation=d)
        b = layers.BatchNormalization(name=f"{name_prefix}_b{idx}_bn")(b)
        b = layers.ReLU(name=f"{name_prefix}_b{idx}_relu")(b)
        branches.append(b)

    g = layers.GlobalAveragePooling2D(name=f"{name_prefix}_global_pool")(x)
    g = layers.Reshape((1, 1, x.shape[-1]), name=f"{name_prefix}_global_reshape")(g)
    g = layers.Conv2D(branch_ch, 1, use_bias=False, name=f"{name_prefix}_global_conv")(g)
    g = layers.ReLU(name=f"{name_prefix}_global_relu")(g)
    g = layers.Resizing(h, w, interpolation="bilinear", name=f"{name_prefix}_global_resize")(g)
    branches.append(g)

    out = layers.Concatenate(name=f"{name_prefix}_concat")(branches)
    out = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name_prefix}_project_conv")(out)
    out = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(out)
    return layers.ReLU(name=f"{name_prefix}_project_relu")(out)


def decoder_attention_block(skip, gate, separable, name_prefix="dec_attn"):
    """Gate the encoder skip with the coarser decoder feature (ResUNet++ style)."""
    c = skip.shape[-1]
    h, w = skip.shape[1], skip.shape[2]

    g = layers.Resizing(h, w, interpolation="bilinear", name=f"{name_prefix}_resize")(gate)
    g = layers.BatchNormalization(name=f"{name_prefix}_bn_g")(g)
    g = layers.ReLU(name=f"{name_prefix}_relu_g")(g)
    g = _conv3x3(g, c, separable, f"{name_prefix}_conv_g")

    s = layers.BatchNormalization(name=f"{name_prefix}_bn_s")(skip)
    s = layers.ReLU(name=f"{name_prefix}_relu_s")(s)
    s = _conv3x3(s, c, separable, f"{name_prefix}_conv_s")

    a = layers.Add(name=f"{name_prefix}_add")([g, s])
    a = layers.BatchNormalization(name=f"{name_prefix}_bn_c")(a)
    a = layers.ReLU(name=f"{name_prefix}_relu_c")(a)
    alpha = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name_prefix}_alpha")(a)
    return layers.Multiply(name=f"{name_prefix}_gated")([skip, alpha])


def build_resunet_pp_light(input_shape=(256, 256, 1), out_channels=1, base_filters=16,
                           depth=3, separable=True, aspp_dilations=(1, 2, 4),
                           aspp_branch_ratio=0.5) -> Model:
    filters = [base_filters * 2 ** i for i in range(depth + 1)]
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem
    stem = _conv3x3(inputs, filters[0], separable, "stem_conv1")
    stem = layers.BatchNormalization(name="stem_bn")(stem)
    stem = layers.ReLU(name="stem_relu")(stem)
    stem = _conv3x3(stem, filters[0], separable, "stem_conv2")
    stem_sc = layers.Conv2D(filters[0], 1, use_bias=False, name="stem_shortcut")(inputs)
    cur = layers.Add(name="stem_add")([stem, stem_sc])

    # Encoder
    skips = [cur]
    for i in range(1, depth + 1):
        cur = layers.MaxPooling2D(2, name=f"pool_{i}")(cur)
        cur = residual_se_block(cur, filters[i], separable, name_prefix=f"enc_{i}")
        skips.append(cur)

    # Bridge
    d = aspp_block(cur, filters[depth], separable, aspp_dilations,
                   aspp_branch_ratio, name_prefix="aspp_bridge")

    # Decoder
    for i in reversed(range(depth)):
        skip = skips[i]
        gated = decoder_attention_block(skip, d, separable, name_prefix=f"dec_attn_{i}")
        up = layers.Resizing(skip.shape[1], skip.shape[2], interpolation="bilinear",
                             name=f"up_{i}")(d)
        d = layers.Concatenate(name=f"concat_{i}")([gated, up])
        d = residual_se_block(d, filters[i], separable, name_prefix=f"dec_{i}")

    # Output ASPP + head (dilations capped to what a full-res 16-ch map needs)
    out = aspp_block(d, filters[0], separable, aspp_dilations,
                     aspp_branch_ratio, name_prefix="aspp_out")
    outputs = layers.Conv2D(out_channels, 1, name="output_final")(out)

    return Model(inputs, outputs,
                 name="LightResUNetPP" if separable else "ResUNetPP")


def count_parameters(model: Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


if __name__ == "__main__":
    dummy = tf.random.normal((2, 256, 256, 1))
    configs = {
        "original (brute force)":      dict(separable=False, aspp_dilations=(1, 6, 12, 18), aspp_branch_ratio=1.0),
        "+ separable convs only":      dict(separable=True,  aspp_dilations=(1, 6, 12, 18), aspp_branch_ratio=1.0),
        "+ small ASPP rates":          dict(separable=True,  aspp_dilations=(1, 2, 4),      aspp_branch_ratio=1.0),
        "+ half-width ASPP (final)":   dict(separable=True,  aspp_dilations=(1, 2, 4),      aspp_branch_ratio=0.5),
    }
    base = None
    for label, cfg in configs.items():
        m = build_resunet_pp_light(**cfg)
        assert m(dummy).shape == (2, 256, 256, 1)
        p = count_parameters(m)
        base = base or p
        print(f"{label:30s} {p:>10,}  ({base / p:.1f}x smaller)")