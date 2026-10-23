"""原逐人遮挡MSE + 遮挡上三角的个体残差Pearson损失。"""
import torch

def components(z,prior):
 pred=z['prediction'];target=z['target'].detach();mask=z['score_mask'].bool()
 mse=torch.stack([(p-t).square().masked_select(s).mean()for p,t,s in zip(pred,target,mask)]).mean()
 upper=torch.ones(pred.shape[-2:],dtype=torch.bool,device=pred.device).triu(1)
 losses=[];valid=0;degenerate=0
 for p,t,s in zip(pred,target,mask):
  q=s & upper
  center=(prior+prior.transpose(-1,-2))/2
  x=((p+p.transpose(-1,-2))/2-center).masked_select(q);y=((t+t.transpose(-1,-2))/2-center).masked_select(q)
  assert x.numel()>=2
  x=x-x.mean();y=y-y.mean();vy=y.square().mean()
  if float(vy)<=1e-12:
   losses.append(x.sum()*0);degenerate+=1;continue
  vx=x.square().mean();corr=(x*y).mean()/torch.sqrt(vx.clamp_min(1e-8)*vy.clamp_min(1e-8))
  losses.append(1-corr.clamp(-1,1));valid+=1
 corr_loss=torch.stack(losses).mean()
 return mse,corr_loss,{'valid_subjects':valid,'degenerate_targets':degenerate}

def total(z,prior,weight):
 mse,corr,diag=components(z,prior)
 return mse+weight*corr,{'mse':float(mse.detach()),'residual_correlation_loss':float(corr.detach()),'weighted_correlation_loss':float((weight*corr).detach()),'lambda':weight,**diag}
