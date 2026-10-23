import torch
from torch import nn
# Use audited math attention on Blackwell; fused path failed repeatability pilot.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import roi_small,roi_large
class TransformerBlock(nn.Module):
 def __init__(self,d):
  super().__init__();self.norm1=nn.LayerNorm(d);self.attention=nn.MultiheadAttention(d,6,dropout=0.,batch_first=True);self.norm2=nn.LayerNorm(d);self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
 def forward(self,x):
  z=self.norm1(x);x=x+self.attention(z,z,z,need_weights=False)[0];return x+self.mlp(self.norm2(x))
def make(f,seed,arm):
 assert arm in ['MS','TS','ML','TL'];large=arm.endswith('L');d=576 if large else 288
 with torch.random.fork_rng():
  torch.manual_seed(2026102200+f+1000000*(seed-1));m=(roi_large if large else roi_small).ROIMAE()
  # Construct common modules first; preserve their exact paired initialization.
  if arm.startswith('T'):
   m.encoder.blocks=nn.ModuleList([TransformerBlock(d) for _ in range(5)])
   if large:
    m.decoder[1]=TransformerBlock(288);m.decoder[2]=TransformerBlock(288)
 return m
