import torch,types
from base_models import make as base_make
ORDER=torch.randperm(116,generator=torch.Generator().manual_seed(2026102301))
def ordered_forward(self,x):
 x=(x+self.roi_identity)[:,self.scan_order]
 for block in self.blocks:x=block(x)
 return self.norm(x[:,self.inverse_order])
def make(f,seed,arm):
 assert arm in ['MS','TS','ML','TL','MSO','MSZ','TSZ','MSF','TSF'];m=base_make(f,seed,arm[:2])
 if arm.endswith('O'):
  m.encoder.register_buffer('scan_order',ORDER.clone());m.encoder.register_buffer('inverse_order',torch.argsort(ORDER));m.encoder.forward=types.MethodType(ordered_forward,m.encoder)
 if arm.endswith('Z'):
  with torch.no_grad():m.encoder.roi_identity.zero_()
  m.encoder.roi_identity.requires_grad_(False)
 return m
