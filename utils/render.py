import numpy as np
import torch
import torch.nn as nn
import pytorch3d

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)

from pytorch3d.renderer.blending import hard_rgb_blend, softmax_rgb_blend, BlendParams



class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, hard_mode=False):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.hard_mode = hard_mode

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.hard_mode:
            images = hard_rgb_blend(texels, fragments, blend_params)
        else:
            images = softmax_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image