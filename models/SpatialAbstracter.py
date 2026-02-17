import torch
import torch.nn as nn
from skimage.transform import resize
import numpy as np

class SpatialAbstracter(nn.Module):
    def __init__(self, target_hw: tuple[int, int]):
        super().__init__()
        self.target_hw = target_hw  # (128, 128)
        self.register_buffer('fixed_target_hw', torch.tensor(target_hw, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> CPU numpy per skimage resize bilineare (order=1)
        B, C, D, H, W = x.shape
        x_np = x.detach().cpu().numpy()
        
        resized = np.zeros((B, C, D, *self.target_hw), dtype=x_np.dtype)
        for b in range(B):
            for c in range(C):
                resized[b, c] = resize(
                    x_np[b, c], 
                    (D, *self.target_hw), 
                    order=1,
                    preserve_range=True,
                    anti_aliasing=True
                )
        
        return torch.from_numpy(resized).to(x.device)