"""
models/resunet_pp.py
----------------------
ResUNet++ (Jha, D. et al., "ResUNet++: An Advanced Architecture for Medical
Image Segmentation," IEEE International Symposium on Multimedia (ISM), 2019).

Combines four separate ideas from four separate papers into one network:

  1. Residual blocks       (He et al., ResNet lineage / Zhang et al. 2018
                             "Road Extraction by Deep Residual U-Net")
  2. Squeeze-and-Excitation (Hu, Shen & Sun, CVPR 2018)
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block (Hu, Shen & Sun, CVPR 2018).

    "Squeeze": global average pooling collapses each channel's spatial
    extent (H x W) down to a single scalar, producing a channel descriptor
    that summarizes the global response of that channel across the image.

    "Excitation": a small two-layer MLP (implemented as 1x1 convs here)
    maps that channel descriptor through a bottleneck and back out to a
    per-channel scaling factor in [0, 1] via a sigmoid, which is then
    broadcast-multiplied back onto the original feature map. Channels the
    network finds more informative for the current input are scaled up;
    less relevant channels are scaled down.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualSEBlock(nn.Module):
    """
    Residual block with an SE block applied to the residual branch before
    the skip-addition, matching the "residual + squeeze-excitation" encoder
    unit used throughout ResUNet++.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.se = SqueezeExcitation(out_channels)

        # Projection shortcut when input/output channel counts or spatial
        # resolution differ (standard ResNet-style identity/projection choice).
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.se(residual)
        return residual + self.shortcut(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., DeepLab lineage), placed at
    the network's bridge (bottleneck) exactly as in the original ResUNet++.

    Runs several atrous (dilated) 3x3 convolutions in parallel at different
    dilation rates -- each rate expands the receptive field without adding
    parameters or losing spatial resolution the way pooling would -- plus a
    global-average-pooling branch for whole-image context, then fuses all
    branches with a final 1x1 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                ))

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        branch_outs = [branch(x) for branch in self.branches]
        global_out = self.global_branch(x)
        global_out = F.interpolate(global_out, size=(h, w), mode="bilinear", align_corners=True)
        branch_outs.append(global_out)
        return self.project(torch.cat(branch_outs, dim=1))


class DecoderAttentionBlock(nn.Module):
    """
    Lightweight attention block applied to the encoder skip connection
    before it is concatenated into the decoder, in the same spirit as
    Attention U-Net's gate but implemented via the simpler channel+spatial
    squeeze used in the original ResUNet++ decoder.
    """

    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, skip_channels, kernel_size=3, padding=1, bias=False),
        )
        self.skip_conv = nn.Sequential(
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, bias=False),
        )
        self.attn = nn.Sequential(
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        g = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=True)
        g = self.gate_conv(g)
        s = self.skip_conv(skip)
        alpha = self.attn(g + s)
        return skip * alpha


class ResUNetPlusPlus(nn.Module):
    """
    Full ResUNet++: stem block -> 3 residual-SE encoder stages -> ASPP
    bridge -> 3 attention-gated decoder stages -> 1x1 output head.

    Parameters mirror models/unet_plus_plus.py and models/attention_unet.py
    for a fair three-way parameter-count comparison in the final report.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=16, depth=3):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]  # e.g. [16, 32, 64, 128]

        # --- Stem block (initial feature extraction, not yet residual) -----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
        )
        self.stem_shortcut = nn.Conv2d(in_channels, filters[0], kernel_size=1, bias=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # --- Encoder: residual + SE blocks, downsampling between stages ----
        self.encoders = nn.ModuleList()
        for i in range(1, depth + 1):
            self.encoders.append(ResidualSEBlock(filters[i - 1], filters[i]))

        # --- Bridge: ASPP -----------------------------------------------------
        self.aspp_bridge = ASPP(filters[depth], filters[depth])

        # --- Decoder: attention-gated skip fusion + residual blocks --------
        self.decoder_attn = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoder_attn.append(DecoderAttentionBlock(skip_channels=filters[i], gate_channels=filters[i + 1]))
            self.decoders.append(ResidualSEBlock(filters[i] + filters[i + 1], filters[i]))

        # --- Output ASPP + head (as in the original paper's final stage) ----
        self.aspp_out = ASPP(filters[0], filters[0])
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # --- Stem ------------------------------------------------------------
        stem_out = self.stem(x) + self.stem_shortcut(x)

        # --- Encoder, caching skip features ------------------------------
        skips = [stem_out]
        cur = stem_out
        for i, enc in enumerate(self.encoders):
            cur = self.pool(cur)
            cur = enc(cur)
            skips.append(cur)

        # --- Bridge -----------------------------------------------------------
        bridge = self.aspp_bridge(cur)

        # --- Decoder ----------------------------------------------------------
        d = bridge
        for idx, depth_i in enumerate(reversed(range(self.depth))):
            skip = skips[depth_i]
            gated_skip = self.decoder_attn[idx](skip=skip, gate=d)
            d_up = F.interpolate(d, size=skip.shape[2:], mode="bilinear", align_corners=True)
            d = torch.cat([gated_skip, d_up], dim=1)
            d = self.decoders[idx](d)

        out = self.aspp_out(d)
        return self.final_conv(out)  # single logits tensor, no deep supervision


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResUNetPlusPlus(in_channels=1, out_channels=1, base_filters=16, depth=3)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
