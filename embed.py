# patch embedding

import torch.nn as nn

class Embedding(nn.Module):
    """ Volumetric data to patch embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Pd*Ph*Pw C

        if self.norm is not None:
            x = self.norm(x)
        
        return x    