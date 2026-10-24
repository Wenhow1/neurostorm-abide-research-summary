import torch
from consistency_loss import regularizer,total,mse
torch.manual_seed(7);x=torch.randn(8,116,288);y=torch.randn_like(x);x.requires_grad_();y.requires_grad_()
a,d=regularizer(x,y);assert torch.isfinite(a);a.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
# Fixed ROI identity must not satisfy variance objective, nor change centered loss.
identity=torch.randn(1,116,288);zero=torch.zeros_like(x);v,diag=regularizer(zero+identity,zero+identity);assert v>.08 and diag['low_variance_fraction']==1
v1,_=regularizer(x,y);v2,_=regularizer(x+identity,y+identity);assert torch.allclose(v1,v2,atol=1e-5)
a,_=regularizer(x,x);b,_=regularizer(x,x.flip(0));assert b>a
# Gram implementation matches explicit ROI channel covariance.
z=x-x.mean(0);cov=torch.einsum('brd,bre->rde',z,z)/7;explicit=(cov.square().sum((1,2))-cov.diagonal(dim1=1,dim2=2).square().sum(1)).mean()/288
_,diag=regularizer(x,x);assert abs(diag['covariance_penalty']-float(2*explicit))<1e-4
p=torch.randn(2,1,116,116,requires_grad=True);t=torch.randn_like(p);mask=torch.ones_like(p,dtype=torch.bool);z={'prediction':p,'target':t,'score_mask':mask};zz=dict(z);z['_view2']=zz;z['_h1']=x;z['_h2']=y
l,_=total(z,0);m=mse(zz);assert torch.equal(l,m);assert torch.equal(torch.autograd.grad(l,p,retain_graph=True)[0],torch.autograd.grad(m,p)[0])
print('LOSS_TESTS_PASS')
