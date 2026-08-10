"""Residual building blocks shared by the reconstruction models."""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block using reflection padding."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock2(nn.Module):
    """Residual block using zero padding."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
