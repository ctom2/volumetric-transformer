# transformer blocks (encoder/decoder)

import torch.nn as nn
from patch_operations import PatchMerge, PatchExpand
from transformer import TransformerBlock

class BasicBlock_down(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads, depth, downsample):
        super().__init__()
        
        self.downsample = downsample

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads, window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])
        
        if self.downsample:
            self.down = PatchMerge(input_resolution=input_resolution, dim=dim)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        if self.downsample:
            x = self.down(x)

        return x

class BasicBlock_up(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads, depth, upsample):
        super().__init__()
        
        self.upsample = upsample

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads, window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)])
        
        if self.upsample:
            self.up = PatchExpand(input_resolution=input_resolution, dim=dim)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        if self.upsample:
            x = self.up(x)

        return x