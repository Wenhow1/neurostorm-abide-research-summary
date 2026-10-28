import torch
from torch import nn
from common import seed
class Bi(nn.Module):
 def __init__(self,d):
  super().__init__();from mamba_ssm import Mamba
  import os
  if os.environ.get('DIRECTFC_SCAN_REF')=='1':
   import mamba_ssm.modules.mamba_simple as ms
   from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
   ms.selective_scan_fn=selective_scan_ref
  self.norm=nn.LayerNorm(d,eps=1e-5);self.mamba=Mamba(d_model=d,d_state=16,d_conv=4,expand=2,dt_rank='auto',dt_min=.001,dt_max=.1,dt_init='random',dt_scale=1.,dt_init_floor=.0001,bias=False,conv_bias=True)
 def forward(self,x):
  u=self.norm(x);return x+(self.mamba(u)+self.mamba(u.flip(1)).flip(1))*.5
class TB(nn.Module):
 def __init__(self,d):
  super().__init__();self.n1=nn.LayerNorm(d,eps=1e-5);self.att=nn.MultiheadAttention(d,4,dropout=0,bias=True,batch_first=True);self.n2=nn.LayerNorm(d,eps=1e-5);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(approximate='none'),nn.Linear(4*d,d))
 def forward(self,x):
  u=self.n1(x);x=x+self.att(u,u,u,need_weights=False)[0];return x+self.ff(self.n2(x))
class Net(nn.Module):
 def __init__(self,arm,d,condition="A"):
  super().__init__();self.arm=arm;self.condition=condition;self.dropgen=None
  if arm=='GCN':self.graph1=nn.Linear(116,d);self.graph2=nn.Linear(d,d)
  else:self.input=nn.Linear(116,d);self.blocks=nn.ModuleList([(Bi if arm=='Mamba' else TB)(d) for _ in range(2)])
  self.norm=nn.LayerNorm(d,eps=1e-5);self.head=nn.Linear(116*d,1)
 def forward(self,x,a=None,features=False):
  if self.arm=='GCN':
   # Bias must follow aggregation: AHW + b, not A(HW+b).
   h=torch.relu(torch.matmul(a,torch.nn.functional.linear(x,self.graph1.weight))+self.graph1.bias)
   h=torch.relu(torch.matmul(a,torch.nn.functional.linear(h,self.graph2.weight))+self.graph2.bias)
  else:
   h=self.input(x)
   for b in self.blocks:h=b(h)
  z=self.norm(h).flatten(1)
  if self.training and self.condition=='C':
   assert self.dropgen is not None
   z=z*(torch.rand(z.shape,device=z.device,generator=self.dropgen)>=.1)/.9
  return self.head(z).squeeze(-1)
def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
 for k in ['enable_flash_sdp','enable_mem_efficient_sdp','enable_cudnn_sdp']:getattr(torch.backends.cuda,k)(False)
 torch.backends.cuda.enable_math_sdp(True)
def build(j):torch.manual_seed(seed(j));return Net(j['arm'],j['width'],j.get('condition','A'))
def groups(m):
 normids={id(p) for mod in m.modules() if isinstance(mod,nn.LayerNorm) for p in mod.parameters(recurse=False)}
 yes=[];no=[];records=[]
 for n,p in m.named_parameters():
  if not p.requires_grad:continue
  excluded=getattr(p,'_no_weight_decay',False) or id(p) in normids or n.endswith('bias')
  (no if excluded else yes).append(p);records.append({'name':n,'shape':list(p.shape),'count':p.numel(),'weight_decay':0. if excluded else .01,'native_no_decay':bool(getattr(p,'_no_weight_decay',False))})
 assert len({id(p) for p in yes+no})==len(list(m.parameters()))
 if m.arm=='Mamba':assert all(v['weight_decay']==0 for v in records if v['name'].endswith(('A_log','.D')))
 return [{'params':yes,'weight_decay':.01},{'params':no,'weight_decay':0.}],records
