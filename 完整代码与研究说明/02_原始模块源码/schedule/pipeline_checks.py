from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from supervisor import blocked
assert blocked(80,False,False,1,0,1)==(True,True)
assert blocked(66,True,False,1,0,1)==(True,True)
assert blocked(65,True,False,1,0,1)==(False,False)
assert blocked(30,False,True,0,0,1)[0]
assert not blocked(30,False,True,1,0,1)[0]
assert blocked(30,False,False,0,1,1)[0]
assert not blocked(30,False,False,1,1,1)[0]
from official_lr_scheduler import CosineAnnealingWarmUpRestarts as S
p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=.0001);s=S(opt,1830,1,.0001,1e-9,305,1);seq=[];saved=None
for step in range(4000):
 seq.append(opt.param_groups[0]['lr']);s.step()
 if step==2100:saved=(opt.state_dict(),s.state_dict())
assert seq[0]==1e-9 and abs(seq[305]-.0001)<1e-12 and seq[1830]==1e-9
opt2=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=.0001);s2=S(opt2,1830,1,.0001,1e-9,305,1);opt2.load_state_dict(saved[0]);s2.load_state_dict(saved[1]);seq2=[]
for step in range(2101,4000):seq2.append(opt2.param_groups[0]['lr']);s2.step()
assert np.array_equal(seq[2101:],seq2)
from classify import pipe,metrics
x=np.random.default_rng(1).normal(size=(60,288));y=np.arange(60)%2;m=pipe(.01,1);m.fit(x[:40],y[:40]);assert np.allclose(m[0].mean_,x[:40].mean(0));assert np.isfinite(list(metrics(y[40:],m.predict_proba(x[40:])[:,1]).values())).all()
(R/'pipeline_checks.json').write_text(json.dumps({'status':'PASS','schedule_4000steps_and_restart_resume':True,'thermal_80_65':True,'gpu0_cutoff_and_foreign_yield':True,'gpu1_no_foreign_yield':True,'classifier_synthetic_scaler_metrics':True,'actual_downstream_predictions_not_run_yet':True},indent=2))
print('PIPELINE_PASS')
