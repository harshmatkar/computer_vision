"""
models/attention_unet.py
--------------------------
Attention U-Net (Oktay et al., "Attention U-Net: Learning Where to Look for
the Pancreas," arXiv:1804.03999, 2018).

Core idea
---------
A standard U-Net skip connection copies the encoder feature map at a given
depth directly to the decoder. Attention U-Net inserts an "Attention Gate"
(AG) on every skip connection: the gate looks at both the incoming encoder
features AND the coarser, more semantically-refined decoder features from
one level deeper, and produces a per-pixel gating coefficient in [0, 1] that
suppresses activations in regions irrelevant to the segmentation target
(here: the cytosolic interior) while preserving activations near the
membrane/actin-edge region.

This is a genuinely different mechanism from the nested skip pathways used
in Member A's UNet++: attention gating reweights WHERE in the image to
trust encoder features, whereas UNet++ changes HOW MANY intermediate
processing steps a skip connection passes through.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two stacked standard 3x3 Conv -> BN -> ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (AG) as described in Oktay et al., 2018.

    Parameters
    ----------
    gate_channels : int
        Number of channels in the gating signal `g` (from the coarser,
        deeper decoder stage).
    skip_channels : int
        Number of channels in the skip-connection feature map `x` (from the
        matching encoder stage).
    inter_channels : int
        Number of channels in the shared intermediate space where `g` and
        `x` are compared.

    Forward
    -------
    g : gating signal from the decoder (coarser resolution, upsampled to
        match `x`'s spatial size before calling this module).
    x : encoder skip-connection features (finer resolution).

    Returns
    -------
    x * alpha : the skip features re-weighted by the learned attention map,
                same shape as `x`.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Additive attention: combine gate and skip signals, then squash to
        # a single-channel spatial attention map alpha in [0, 1].
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class AttentionUNet(nn.Module):
    """
    Full Attention U-Net: standard convolutional encoder/decoder with an
    AttentionGate applied to every skip connection before concatenation.

    Parameters
    ----------
    in_channels, out_channels, base_filters, depth : same meaning as in
        models/unet_plus_plus.py, kept consistent across all three
        team-member models for a fair parameter-count comparison.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=16, depth=4):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # --- Encoder -------------------------------------------------------
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth + 1):
            self.encoders.append(ConvBlock(in_ch, filters[i]))
            in_ch = filters[i]

        # --- Decoder + Attention Gates -------------------------------------
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            # Gate signal comes from filters[i+1] (one level deeper/coarser),
            # skip connection comes from filters[i] (encoder at this depth).
            self.attention_gates.append(
                AttentionGate(gate_channels=filters[i + 1], skip_channels=filters[i],
                              inter_channels=filters[i] // 2)
            )
            # After concatenating the gated skip (filters[i]) with the
            # upsampled decoder features (filters[i+1]), run a ConvBlock
            # back down to filters[i] channels.
            self.decoders.append(ConvBlock(filters[i] + filters[i + 1], filters[i]))

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder path, caching skip features at every depth -----------
        skips = []
        cur = x
        for i, enc in enumerate(self.encoders):
            cur = enc(cur)
            skips.append(cur)
            if i < self.depth:
                cur = self.pool(cur)

        # --- Decoder path with attention-gated skip fusion -----------------
        d = skips[-1]  # bottleneck features
        for idx, depth_i in enumerate(reversed(range(self.depth))):
            g = self.up(d)  # upsample the coarser decoder/bottleneck signal
            skip = skips[depth_i]
            gated_skip = self.attention_gates[idx](g=g, x=skip)
            d = torch.cat([gated_skip, g], dim=1)
            d = self.decoders[idx](d)

        return self.final_conv(d)  # single logits tensor, no deep supervision


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, out_channels=1, base_filters=16, depth=4)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
