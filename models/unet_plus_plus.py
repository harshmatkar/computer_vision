"""
models/unet_plus_plus.py
-------------------------
A lightweight UNet++ (Zhou et al., 2018 "UNet++: A Nested U-Net Architecture
for Medical Image Segmentation") implemented with Depthwise Separable
Convolutions instead of standard convolutions in every conv block.

Why Depthwise Separable Convolutions?
--------------------------------------
A standard 3x3 conv with C_in -> C_out channels costs:
    3 * 3 * C_in * C_out            multiply-accumulate ops per pixel
A depthwise separable conv (depthwise 3x3 + pointwise 1x1) costs:
    3 * 3 * C_in  +  C_in * C_out   multiply-accumulate ops per pixel
For typical channel widths used here (16-256), this is roughly an 8-9x
reduction in FLOPs/parameters per conv layer, which is why MobileNet-style
architectures use it. This keeps the whole UNet++ "lightweight" enough to
train on a laptop GPU/CPU on microscopy image datasets that are often small.

Why UNet++ (nested skip connections) instead of vanilla UNet?
---------------------------------------------------------------
Vanilla UNet skips features directly from encoder stage i to decoder stage i,
forcing the decoder to fuse features of very different semantic depth in one
step. UNet++ inserts a grid of intermediate convolutional nodes (X^{i,j})
that progressively bridge the semantic gap between encoder and decoder
feature maps before the final fusion, which empirically improves boundary
delineation -- important here since we are segmenting thin actin edge bands.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    One depthwise separable convolution block:
        Depthwise 3x3 conv (groups=in_channels) -> BatchNorm -> ReLU
        -> Pointwise 1x1 conv (channel mixing)   -> BatchNorm -> ReLU

    This is the fundamental building block replacing the standard
    "Conv3x3 -> BN -> ReLU" used in the original UNet/UNet++ papers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """Two stacked DepthwiseSeparableConv layers -> the standard UNet++ node X^{i,j}."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class LightUNetPlusPlus(nn.Module):
    """
    Lightweight UNet++ with 'DEPTH' encoder stages and dense nested
    decoder skip pathways, using depthwise separable convolutions
    throughout.

    Notation follows the original paper: X^{i,j} is the node at encoder
    depth i and nested-skip position j (j=0 is the raw encoder output at
    depth i; j>0 are the nested decoder nodes).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for grayscale microscopy).
    out_channels : int
        Number of output segmentation channels (1 for binary edge mask).
    base_filters : int
        Number of filters in the first encoder stage; doubles each stage.
    depth : int
        Number of encoder down-sampling stages (network has depth+1 levels,
        indices 0..depth).
    deep_supervision : bool
        If True, forward() also returns intermediate decoder outputs
        (X^{0,1}, X^{0,2}, ..., X^{0,depth}) for UNet++ style deep
        supervision during training; if False, only the final X^{0,depth}
        output is returned.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=16, depth=4,
                 deep_supervision=True):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision

        # Number of channels at each encoder depth: e.g. [16, 32, 64, 128, 256]
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # nodes[i][j] holds the ConvBlock for X^{i,j}
        self.nodes = nn.ModuleDict()

        # --- Build the nested grid of conv blocks -------------------------------
        for i in range(depth + 1):
            # j = 0 column: plain encoder path X^{i,0}
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.nodes[f"x_{i}_0"] = ConvBlock(in_ch, filters[i])

        for j in range(1, depth + 1):
            for i in range(depth + 1 - j):
                # X^{i,j} takes: upsampled X^{i+1,j-1}  concatenated with
                # all previous nodes at the same depth i: X^{i,0..j-1}
                in_ch = filters[i] * j + filters[i + 1]
                self.nodes[f"x_{i}_{j}"] = ConvBlock(in_ch, filters[i])

        # --- Final 1x1 conv heads for (optionally deep-supervised) outputs -----
        if deep_supervision:
            self.final_heads = nn.ModuleList([
                nn.Conv2d(filters[0], out_channels, kernel_size=1)
                for _ in range(depth)
            ])
        else:
            self.final_heads = nn.ModuleList([
                nn.Conv2d(filters[0], out_channels, kernel_size=1)
            ])

    def forward(self, x):
        depth = self.depth
        # store all computed nodes: cache["i_j"] -> tensor
        cache = {}

        # --- Encoder column j = 0 ------------------------------------------
        cur = x
        for i in range(depth + 1):
            cur = self.nodes[f"x_{i}_0"](cur if i == 0 else self.pool(cache[f"{i-1}_0"]))
            cache[f"{i}_0"] = cur

        # --- Nested decoder columns j = 1 .. depth --------------------------
        for j in range(1, depth + 1):
            for i in range(depth + 1 - j):
                skip_tensors = [cache[f"{i}_{k}"] for k in range(j)]
                upsampled = self.up(cache[f"{i+1}_{j-1}"])
                concat = torch.cat(skip_tensors + [upsampled], dim=1)
                cache[f"{i}_{j}"] = self.nodes[f"x_{i}_{j}"](concat)

        # --- Output head(s) --------------------------------------------------
        if self.deep_supervision:
            outputs = [head(cache[f"0_{j}"]) for j, head in enumerate(self.final_heads, start=1)]
            return outputs  # list of logits tensors, one per supervised decoder depth
        else:
            return self.final_heads[0](cache[f"0_{depth}"])  # single logits tensor


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick self-test: verify forward pass shapes and print parameter count.
    model = LightUNetPlusPlus(in_channels=1, out_channels=1, base_filters=16, depth=4,
                               deep_supervision=True)
    dummy = torch.randn(2, 1, 256, 256)
    outputs = model(dummy)
    for idx, o in enumerate(outputs):
        print(f"Deep-supervision output {idx}: shape={tuple(o.shape)}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
