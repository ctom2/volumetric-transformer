# patch transformations

import torch
import torch.nn as nn
from einops import rearrange

class PatchMerge(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # (2**3) because 3 dimensions for odd/even (#2) slices
        self.reduction = nn.Linear((2**3) * dim, (2**2) * dim, bias=False)
        self.norm = norm_layer((2**3) * dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*D/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """
    Expands each patch by a factor of 2 in each dimension and halves the number of channels
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False)
        self.norm = norm_layer(dim // 4)

    def forward(self, x):
        """
        x: B, L, C
        """
        D, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=2, p2=2, p3=2, c=C//8)
        x = x.view(B,-1,C//8)
        
        x = self.norm(x)

        return x