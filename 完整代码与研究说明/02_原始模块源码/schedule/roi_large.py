"""ROI sequence MAE, architecture-route comparison. No training outside worker."""
import torch
from torch import nn
from mamba_ssm import Mamba
from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D
class BiBlock(nn.Module):
 def __init__(self,d=288):
  super().__init__();self.norm1=nn.LayerNorm(d);self.mamba=Mamba(d_model=d,d_state=16,d_conv=4,expand=2);self.norm2=nn.LayerNorm(d);self.mlp=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
 def forward(self,x):
  z=self.norm1(x);f=self.mamba(z);r=self.mamba(z.flip(1)).flip(1);x=x+(f+r)*.5;return x+self.mlp(self.norm2(x))
class Encoder(nn.Module):
 def __init__(self,d=576):
  super().__init__();self.roi_identity=nn.Parameter(torch.empty(1,116,d));nn.init.trunc_normal_(self.roi_identity,std=.02);self.blocks=nn.ModuleList([BiBlock(d)for _ in range(4)]);self.norm=nn.LayerNorm(d)
 def forward(self,x):
  x=x+self.roi_identity
  for block in self.blocks:x=block(x)
  return self.norm(x)
class ROIMAE(nn.Module):
 def __init__(self):
  super().__init__();self.image_size=(116,116);self.mask_unit=4;self.mask_ratio=.5;self.mask_policy='paired_symmetric';self.value_projection=nn.Linear(116,576);self.mask_projection=nn.Linear(116,576,bias=False);self.encoder=Encoder();self.decoder=nn.Sequential(nn.Linear(576,288),BiBlock(288),BiBlock(288),nn.LayerNorm(288),nn.Linear(288,116));self.register_buffer('off_diagonal',~torch.eye(116,dtype=torch.bool)[None,None],persistent=False)
 sample_mask=OverlapMaskedAutoencoder2D.sample_mask
 def embedded_input(self,image,mask):
  assert image.shape[1:]==(1,116,116) and mask.shape==image.shape and mask.dtype==torch.bool
  assert torch.isfinite(image).all() and torch.equal(mask,mask.transpose(-1,-2))
  return self.value_projection(image.masked_fill(mask,0)[:,0])+self.mask_projection(mask[:,0].to(image.dtype))
 def encode_full(self,image,*,use_prompts=False):
  assert not use_prompts
  return self.encoder(self.embedded_input(image,torch.zeros_like(image,dtype=torch.bool)))[:,None]
 def forward(self,image,*,pixel_mask=None,generator=None):
  if pixel_mask is None:pixel_mask=self.sample_mask(len(image),device=image.device,generator=generator)
  z=self.encoder(self.embedded_input(image,pixel_mask));prediction=self.decoder(z)[:,None];score=pixel_mask & self.off_diagonal
  assert score.flatten(1).any(1).all()
  return {'prediction':prediction,'target':image.detach(),'pixel_mask':pixel_mask,'score_mask':score}
