import torch,json
from neurostorm_2d.scripts.train_mae import auxiliary_loss,loss_rows
from pathlib import Path
R=Path(__file__).resolve().parent
torch.manual_seed(7);target=torch.randn(2,1,8,8);mask=(~torch.eye(8,dtype=torch.bool))[None,None].expand(2,1,8,8);prior=torch.zeros(1,8,8)
def out(p):return dict(prediction=p,target=target,score_mask=mask)
x=target.clone().requires_grad_();assert float(auxiliary_loss(out(x),prior,'B'))<1e-6;assert float(auxiliary_loss(out(x),prior,'C'))<1e-8
x=(target*.5).requires_grad_();b=auxiliary_loss(out(x),prior,'B');c=auxiliary_loss(out(x),prior,'C');assert float(b)<1e-6 and abs(float(c)-.25)<1e-6
x=torch.zeros_like(target,requires_grad=True);loss=loss_rows(out(x)).mean()+.05*auxiliary_loss(out(x),prior,'B');loss.backward();assert torch.isfinite(x.grad).all()
x=(target*.5).requires_grad_();auxiliary_loss(out(x),prior,'C').backward();assert float((x.grad*target).sum())<0
(R/'loss_unit_checks.json').write_text(json.dumps(dict(status='PASS',checks=['perfect_prediction','positive_scale_correlation_invariance','half_amplitude_penalty','finite_zero_prediction_gradient','amplitude_gradient_direction']),indent=2));print('PASS loss checks')
