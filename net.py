# volumetric transformer network

import torch
import torch.nn as nn
from embed import Embedding
from blocks import BasicBlock_down, BasicBlock_up
from patch_operations import PatchExpand, FinalPatchExpand

class VolumetricTransformerNet(nn.Module):
    """ Volumetric Transformer Net
    Args:
        img_xy (int, int): height and width of an input
        img_z (int): depth of an input
        img_c (int): input image channels
        dim (int): initial patch embedding dimension
        patch_size (int): initial path size
        window_size (int): attention window size
        depths (list of ints): number of transformers per encoding/decoding block
        num_heads (list of ints): number of attention heads per encoding/decoding block
    """
    def __init__(self, img_xy, img_z, img_c, dim, patch_size=4, window_size=4, depths=[6,6], num_heads=[2,4,8]):
        super().__init__()

        self.dim = dim
        self.num_layers = len(depths)
        self.depths = depths
        # number of features in the bottleneck
        self.num_features = int(self.dim * 4 ** (self.num_layers - 1))

        patches_resolution = [img_z//patch_size, img_xy[0]//patch_size, img_xy[1]//patch_size]

        self.norm1 = nn.LayerNorm(self.num_features)
        self.norm2 = nn.LayerNorm(self.dim)

        self.embed = Embedding(patch_size=patch_size, in_chans=img_c, embed_dim=dim, norm_layer=nn.LayerNorm)

        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicBlock_down(dim=int(self.dim * 4 ** i_layer),
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                      patches_resolution[1] // (2 ** i_layer),
                                                      patches_resolution[2] // (2 ** i_layer)),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    downsample=True if (i_layer < self.num_layers - 1) else False)
            self.layers_down.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2*int(self.dim*4**(self.num_layers-1-i_layer)),
                int(self.dim*4**(self.num_layers-1-i_layer))
            ) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                layer = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers-1-i_layer)),
                        patches_resolution[2] // (2 ** (self.num_layers-1-i_layer)),
                    ), 
                    dim=int(self.dim * 4 ** (self.num_layers-1-i_layer)),  
                )
            else:
                layer = BasicBlock_up(dim=int(self.dim * 4 ** (self.num_layers-1-i_layer)),
                                      input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                        patches_resolution[1] // (2 ** (self.num_layers-1-i_layer)),
                                                        patches_resolution[2] // (2 ** (self.num_layers-1-i_layer))),
                                      depth=depths[i_layer],
                                      num_heads=num_heads[i_layer],
                                      window_size=window_size,
                                      upsample=True if (i_layer < self.num_layers - 1) else False)
            self.layers_up.append(layer)
            self.concat_back_dim.append(concat_linear)

        self.final_expand = FinalPatchExpand(
            input_resolution=(patches_resolution[0],patches_resolution[1],patches_resolution[2]),
            patch_size=patch_size, dim=self.dim
        )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=self.dim,out_channels=img_c,kernel_size=1,bias=False),
        )

    def forward(self, x):
        x = self.embed(x)

        x_downsample = []
        for layer_down in self.layers_down:
            x_downsample.append(x)
            x = layer_down(x)

        x = self.norm1(x)

        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[(self.num_layers-1)-i]],-1)
                x = self.concat_back_dim[i](x)
                x = layer_up(x)

        x = self.norm2(x)

        x = self.final_expand(x)
        x = self.output(x)

        return x