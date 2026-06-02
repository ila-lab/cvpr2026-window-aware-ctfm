"""
DenseNet121 3D encoder for CT representation learning.

This module defines a 3D DenseNet121-style encoder used in our
window-aware and multi-scale global-local self-supervised learning pipeline.
The encoder returns multi-level 3D feature maps and can also produce a
single concatenated embedding by applying global average pooling to each
feature level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer3D(nn.Module):
    """A single 3D DenseNet layer.

    Each layer applies BN-ReLU-1x1 Conv followed by BN-ReLU-3x3 Conv.
    The output is concatenated with the input along the channel dimension.
    """

    def __init__(self, in_channels, growth_rate=32, bn_size=4, dropout=0.2):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            inter_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm2 = nn.BatchNorm3d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            inter_channels,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        new_features = self.dropout(new_features)
        return torch.cat([x, new_features], dim=1)


class DenseBlock3D(nn.Module):
    """A 3D DenseNet block composed of multiple DenseLayer3D modules."""

    def __init__(self, num_layers, in_channels, growth_rate=32, bn_size=4, dropout=0.2):
        super().__init__()
        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layers.append(
                DenseLayer3D(
                    in_channels=channels,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    dropout=dropout,
                )
            )
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class Transition3D(nn.Module):
    """Transition layer used between DenseNet blocks.

    It reduces the number of channels with a 1x1 convolution and downsamples
    the spatial resolution using 3D average pooling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet1213DEncoder(nn.Module):
    """DenseNet121-style 3D encoder.

    Args:
        in_channels: Number of input image channels. For CT, this is usually 1.
        growth_rate: Channel growth rate of DenseNet layers.
        block_config: Number of layers in each dense block.
        num_init_features: Number of channels after the initial convolution.
        bn_size: Bottleneck multiplier used in DenseNet layers.
        dropout: Dropout probability used in dense layers.

    Returns:
        forward(x): a list of multi-level feature maps [f1, f2, f3, f4].
        extract_embedding(x): pooled and concatenated multi-level embedding.
    """

    def __init__(
        self,
        in_channels=1,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        dropout=0.2,
    ):
        super().__init__()

        # Initial 3D convolution and downsampling.
        self.conv0 = nn.Conv3d(
            in_channels,
            num_init_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm3d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        num_features = num_init_features

        self.block1 = DenseBlock3D(
            block_config[0],
            num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout,
        )
        num_features = self.block1.out_channels
        self.trans1 = Transition3D(num_features, num_features // 2)
        num_features = num_features // 2

        self.block2 = DenseBlock3D(
            block_config[1],
            num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout,
        )
        num_features = self.block2.out_channels
        self.trans2 = Transition3D(num_features, num_features // 2)
        num_features = num_features // 2

        self.block3 = DenseBlock3D(
            block_config[2],
            num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout,
        )
        num_features = self.block3.out_channels
        self.trans3 = Transition3D(num_features, num_features // 2)
        num_features = num_features // 2

        self.block4 = DenseBlock3D(
            block_config[3],
            num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            dropout=dropout,
        )
        num_features = self.block4.out_channels
        self.norm5 = nn.BatchNorm3d(num_features)

        # Channel dimensions of the four output feature levels.
        self.feature_dims = [
            self.block1.out_channels,
            self.block2.out_channels,
            self.block3.out_channels,
            num_features,
        ]

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights following common DenseNet practice."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
        """Compute multi-level feature maps from a 3D CT volume."""
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        f1 = self.block1(x)

        x = self.trans1(f1)
        f2 = self.block2(x)

        x = self.trans2(f2)
        f3 = self.block3(x)

        x = self.trans3(f3)
        f4 = self.block4(x)
        f4 = self.norm5(f4)
        f4 = F.relu(f4, inplace=True)

        return [f1, f2, f3, f4]

    def forward(self, x):
        """Return multi-level feature maps [f1, f2, f3, f4]."""
        return self.forward_features(x)

    def extract_embedding(self, x):
        """Return a pooled multi-level embedding.

        Each feature map is globally averaged and flattened. The pooled features
        from all levels are concatenated into a single vector.
        """
        feats = self.forward_features(x)
        pooled = [F.adaptive_avg_pool3d(f, 1).flatten(1) for f in feats]
        return torch.cat(pooled, dim=1)

    @property
    def out_dim(self):
        """Dimension of the concatenated multi-level embedding."""
        return sum(self.feature_dims)


def densenet121_3d_encoder(in_channels=1, dropout=0.2):
    """Factory function for the DenseNet121 3D encoder."""
    return DenseNet1213DEncoder(
        in_channels=in_channels,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        dropout=dropout,
    )
