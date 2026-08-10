"""MRIMCNN: multi-resolution importance-mask predictor.

Extends the IMCNN idea by predicting importance maps at three scales (full,
1/2, 1/4) and fusing them before selecting the top-K patches under the memory
budget ``total_memory = 128 * 128``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIMCNN(nn.Module):
    """Multi-Resolution Importance-Mask CNN: fuses importance maps predicted at
    three scales (full, 1/2, 1/4) before selecting the top-K patches."""

    def __init__(self, image_size, patch_size, temperature, use_hard_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.total_memory = 128 * 128
        self.K = self.total_memory // (patch_size ** 2)
        self.grid_size = image_size // patch_size
        self.temperature = temperature
        self.use_hard_mask = use_hard_mask

        def make_predictor():
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1)
            )

        self.imp256 = make_predictor()
        self.imp128 = make_predictor()
        self.imp64 = make_predictor()

    def forward(self, x):
        B, C, H, W = x.shape

        x128 = F.interpolate(x, scale_factor=0.5, mode='bicubic', align_corners=True, antialias=True)
        x64 = F.interpolate(x, scale_factor=0.25, mode='bicubic', align_corners=True, antialias=True)

        imp1 = self.imp256(x)
        imp2 = F.interpolate(self.imp128(x128), size=(H, W), mode='bicubic', align_corners=True, antialias=True)
        imp3 = F.interpolate(self.imp64(x64), size=(H, W), mode='bicubic', align_corners=True, antialias=True)
        importance_map = imp1 + imp2 + imp3

        pooled = F.avg_pool2d(importance_map, kernel_size=self.patch_size, stride=self.patch_size)
        flat = pooled.view(B, -1)

        soft_weights = F.softmax(flat / self.temperature, dim=1) * self.K
        mask_small = soft_weights.view(B, 1, self.grid_size, self.grid_size)

        if self.use_hard_mask:
            with torch.no_grad():
                topk_vals, topk_idx = torch.topk(flat, self.K, dim=1)
                hard_mask_flat = torch.zeros_like(flat)
                hard_mask_flat.scatter_(1, topk_idx, 1.0)
                mask_small = hard_mask_flat.view(B, 1, self.grid_size, self.grid_size)

        mask = F.interpolate(mask_small, scale_factor=self.patch_size, mode='nearest')
        return mask
