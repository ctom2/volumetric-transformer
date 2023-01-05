# transformer

import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module.
        It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, D, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        D (int): Depth of image
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D* H * W / window_size / window_size / window_size))
    x = windows.view(B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)

    x = x.permute(0, 1, 4, 2, 5, 3, 6, -1).contiguous().view(B, D, H, W, -1)

    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """ Transformer block
    Info:
        The input is an encoded set of patches.
    Args:
        dim (int): input dimension/channels
        input_resolution (int, int, int): H, W, D of the input
        window_size (int): window size for attention
        num_heads (int): number of attention heads
        shift_size (int): cyclic shift length
        norm_layer: normalisation layer
        mlp_ratio (int): ratio of hidden dim to embed dim in FFN
        act_layer: activation function in FFN
        drop (float): dropout rate in FFN
    """
    def __init__(self, dim, input_resolution, window_size, num_heads, shift_size=0,
                 norm_layer=nn.LayerNorm, mlp_ratio=4, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(self.dim, self.window_size, self.num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        """
        The input is an image encoded as a sequence (B,L,C).
         1. Layer normalisation on the channels
         2. Reshaping to a volumetric object to subsequently do a cyclic shift (B,H,W,D,C)
         3. Partitioning the volumetric object to 3D windows (B*nW, wH, wW, wD, C)
         4. Reshaping the windows into a sequence of voxels (B*nW, wH*wW*wD, C)
         5. Self-attention
         6. Reshaping the sequence of voxels back to windows (B*nW, wH, wW, wD, C)
         7. Reshaping to a volumetric object and reversing the cyclic shift (B,H,W,D,C)
         8. Reshaping into a sequence (B,L,C)
         9. Shortcut
        """
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # reshaping to windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)  # nW*B, window_size*window_size*window_size, C

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)  # B D H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x
        x = x.view(B, D * H * W, C) # B L C

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return shifted_x.shape