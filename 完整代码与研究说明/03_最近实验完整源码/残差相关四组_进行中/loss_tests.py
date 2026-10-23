import torch
from residual_loss import components,total
torch.manual_seed(17)
t=torch.randn(2,1,6,6,dtype=torch.double);mu=torch.randn(1,6,6,dtype=torch.double);mask=torch.ones_like(t,dtype=torch.bool);mask[...,torch.arange(6),torch.arange(6)]=False
p=t.clone().requires_grad_();z=dict(prediction=p,target=t,score_mask=mask)
a,b,d=components(z,mu);assert a==0 and abs(float(b))<1e-10
p=(2*(t-mu)+mu).requires_grad_();a,b,d=components(dict(prediction=p,target=t,score_mask=mask),mu);assert abs(float(b))<1e-10 and a>0
p=(-t+2*mu).requires_grad_();a,b,d=components(dict(prediction=p,target=t,score_mask=mask),mu);assert abs(float(b)-2)<1e-10
p=torch.randn_like(t,requires_grad=True);z=dict(prediction=p,target=t,score_mask=mask);a,b,d=components(z,mu);loss,_=total(z,mu,0);assert torch.equal(a,loss);ga=torch.autograd.grad(a,p,retain_graph=True)[0];gb=torch.autograd.grad(loss,p)[0];assert torch.equal(ga,gb)
p=torch.zeros_like(t,requires_grad=True);loss,_=total(dict(prediction=p,target=t,score_mask=mask),torch.zeros_like(mu),.01);loss.backward();assert torch.isfinite(p.grad).all()
p=torch.randn_like(t,requires_grad=True);loss,d=total(dict(prediction=p,target=mu.expand_as(t),score_mask=mask),mu,.01);loss.backward();assert d['degenerate_targets']==2 and torch.isfinite(p.grad).all()
print('LOSS_TESTS_PASS')
