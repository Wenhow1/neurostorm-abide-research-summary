import torch,numpy as np
from torch import nn
class CoordinateEncoder(nn.Module):
 def __init__(self,base,arm,atlas):
  super().__init__();self.stages=base.stages;self.pos_embeds=base.pos_embeds;self.stage_dims=base.stage_dims;self.stage_shapes=base.stage_shapes;self.depths=base.depths
  xyz=torch.from_numpy(np.load(atlas/'coordinates_mm.npy')).float()/100.
  if arm=='P2':xyz=xyz[torch.from_numpy(np.load(atlas/'permutation.npy'))]
  delta=xyz[:,None,:]-xyz[None,:,:];pair=torch.cat([(xyz[:,None,:]+xyz[None,:,:])/2,delta.abs(),delta.norm(dim=-1,keepdim=True)],dim=-1)
  self.register_buffer('geo_pair',pair[None]);self.geo_mlp=nn.Sequential(nn.Linear(7,36),nn.GELU(),nn.Linear(36,36));self.geo_projections=nn.ModuleList([nn.Identity()]+[nn.Linear(36,d,bias=False)for d in self.stage_dims[1:]]);self.geo_gates=nn.Parameter(torch.zeros(4))
  from neurostorm_2d.models.overlap_geometry import GridGroups2D
  self.geo_patch=GridGroups2D((116,116),4,2,edge_align=False)
 def iter_blocks(self):
  for s in self.stages:yield from s.blocks
 def forward(self,x,*,use_prompts=False):
  g=self.geo_mlp(self.geo_pair);g=self.geo_patch.gather(g).mean(2).reshape(1,57,57,36)
  for i,(p,stage) in enumerate(zip(self.pos_embeds,self.stages)):
   x=stage(x+p+self.geo_gates[i]*self.geo_projections[i](g),use_prompts=use_prompts)
   if stage.downsample is not None:g=stage.downsample.geometry.gather(g).mean(2).reshape(1,*stage.downsample.output_shape,36)
  return x
