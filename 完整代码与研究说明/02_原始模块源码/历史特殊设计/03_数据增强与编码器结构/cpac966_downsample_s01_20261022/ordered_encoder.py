import torch
from torch import nn
from torch.nn import functional as F

class SameWidthMerge(nn.Module):
 def __init__(self,geometry):
  super().__init__();self.geometry=geometry;self.input_shape=geometry.input_shape;self.output_shape=geometry.output_shape;self.norm=nn.LayerNorm(1152);self.reduction=nn.Linear(1152,288,bias=False)
 def forward(self,x):
  assert tuple(x.shape[1:])==(15,15,288)
  groups=self.geometry.gather(x).flatten(start_dim=2)
  return self.reduction(self.norm(groups)).reshape(x.shape[0],8,8,288)

class OrderedEncoder(nn.Module):
 """Shared fine-grid position before both routes; only block/merge order differs."""
 def __init__(self,base,late):
  super().__init__();self.late=late;self.stage_dims=base.stage_dims;self.depths=base.depths;self.stage_shapes=base.stage_shapes;self.stages=base.stages;self.pos_embeds=base.pos_embeds
  geometry=self.stages[2].downsample.geometry;self.stages[2].downsample=None
  self.bridge=nn.Linear(144,288,bias=False);self.merge=SameWidthMerge(geometry)
  coarse=self.pos_embeds[3].detach().permute(0,3,1,2)
  fine=F.interpolate(coarse,size=(15,15),mode='bilinear',align_corners=False).permute(0,2,3,1).contiguous()
  self.pos_embeds[3]=nn.Parameter(fine)
  # Same global-window rule on both grids: min(grid,15) => 8 or15; no shift/padding.
  for b in self.stages[3].blocks:b.window=(15,15);b.shift=(0,0)
 def iter_blocks(self):
  for s in self.stages:yield from s.blocks
 def forward(self,x,*,use_prompts=False):
  for i in range(3):x=self.stages[i](x+self.pos_embeds[i],use_prompts=use_prompts)
  x=self.bridge(x)+self.pos_embeds[3]
  if self.late:
   x=self.stages[3](x,use_prompts=use_prompts);x=self.merge(x)
  else:
   x=self.merge(x);x=self.stages[3](x,use_prompts=use_prompts)
  assert tuple(x.shape[1:])==(8,8,288)
  return x
