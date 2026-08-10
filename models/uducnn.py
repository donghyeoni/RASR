"""UDUCNN: up-down-up 2x upscaler."""

import torch.nn as nn

from .blocks import ResidualBlock2


class UDUCNN(nn.Module):
    """Up-Down-Up CNN: upsample, refine, downsample, upsample again (2x)."""

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # Base structure: 128 -> 256 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)                     # 128x128
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)   # 256x256

        # Residual block 1: feature refinement at 256x256
        self.res1 = ResidualBlock2(64)

        self.down = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)           # 128x128
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 256x256

        # Residual block 2: just before the final output
        self.res2 = ResidualBlock2(32)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # 128x128
        x = self.relu(self.up1(x))       # 256x256
        x = self.res1(x)                 # residual block 1
        x = self.relu(self.down(x))      # 128x128
        x = self.relu(self.up2(x))       # 256x256
        x = self.res2(x)                 # residual block 2
        x = self.output_conv(x)          # 256x256
        return x.clamp(0, 1)
