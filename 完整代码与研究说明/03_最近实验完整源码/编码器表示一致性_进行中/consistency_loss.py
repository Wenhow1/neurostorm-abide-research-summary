"""ROI-wise adaptation of variance/covariance regularization, not a VICReg reproduction.
Reference https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
No projector. Statistics across subjects at each ROI; no ROI average tokens.
Centered invariance excludes a fixed ROI identity offset. Target std=.1,
epsilon=1e-4, auxiliary=invariance+variance+.04*covariance, fixed a priori.
"""
import torch

def regularizer(h1,h2):
 assert h1.shape==h2.shape and h1.ndim==3 and h1.shape[1:]==(116,288)
 n,r,d=h1.shape;assert n>=2
 x=h1-h1.mean(0,keepdim=True);y=h2-h2.mean(0,keepdim=True)
 inv=(x-y).square().mean()
 def moments(z):
  v=z.square().sum(0)/(n-1);sd=torch.sqrt(v+1e-4)
  penalty=torch.relu(.1-sd).mean()
  # ||X'X||_F^2 == ||XX'||_F^2; avoid materializing 116 x288x288 matrices.
  t=z.transpose(0,1);gram=torch.bmm(t,t.transpose(1,2))/(n-1)
  off=(gram.square().sum((1,2))-v.square().sum(1)).clamp_min(0)
  cov=(off/d).mean()
  return penalty,cov,sd.mean(),(sd<.02).float().mean()
 vx,cx,sx,lx=moments(x);vy,cy,sy,ly=moments(y)
 var=(vx+vy)/2;cov=cx+cy
 return inv+var+.04*cov,{'invariance':float(inv.detach()),'variance_penalty':float(var.detach()),'covariance_penalty':float(cov.detach()),'subject_std_mean':float(((sx+sy)/2).detach()),'low_variance_fraction':float(((lx+ly)/2).detach())}

def mse(z):return torch.stack([(p-t).square().masked_select(s).mean()for p,t,s in zip(z['prediction'],z['target'].detach(),z['score_mask'])]).mean()
def total(z,weight):
 m=(mse(z)+mse(z['_view2']))/2;aux,diag=regularizer(z['_h1'],z['_h2'])
 return m+weight*aux,{'mse':float(m.detach()),'consistency_loss':float(aux.detach()),'weighted_consistency_loss':float((weight*aux).detach()),'lambda':weight,**diag}
