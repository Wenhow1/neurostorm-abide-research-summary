import os,json,hashlib
from pathlib import Path
import torch,numpy as np
import worker as w
from residual_loss import components
R=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-7ac39891-108b-f897-b92f-cd51e5392bab'
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
all_results={};weights={}
for fold in range(6):
 raw=np.load(R/'data/FC.npy',mmap_mode='r');ids=np.load(R/'data/IDs.npy');sp=w.read(R/'splits.json')[fold];tr=np.array(sp['inner_train']);assert not set(tr)&set(sp['inner_validation'])
 norm={'mean':float(raw[tr].mean()),'std':float(raw[tr].std())};prior=torch.from_numpy(((raw[tr].mean(0)-norm['mean'])/norm['std']).astype('float32'))[None]
 guard=w.ThermalGuard(R/'control/calibration_thermal.json',R/'control/calibration_events.jsonl',{'job':'calibration'},wait_limit=21600)
 records=[];ratios=[]
 for arm in ['MS','TS']:
  guard.check();m=w.make(fold,1,arm).cuda().eval();pars=list(m.encoder.parameters());values=[];before=w.old.state_hash(m)
  for b in range(4):
   guard.check();q=tr[b*8:(b+1)*8];x=torch.from_numpy(((raw[q]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda();mask=w.masks(m,ids[q],1,f'cpac966_fullcv_train_f{fold}_s1',.5).cuda();z=m(x,pixel_mask=mask);a,c,info=components(z,prior.cuda());assert torch.allclose(a,w.old.loss_mae(z),atol=0,rtol=0)
   ga=torch.autograd.grad(a,pars,retain_graph=True,allow_unused=True);gc=torch.autograd.grad(c,pars,allow_unused=True)
   na=float(torch.sqrt(sum(g.double().square().sum()for g in ga if g is not None)));nc=float(torch.sqrt(sum(g.double().square().sum()for g in gc if g is not None)));assert np.isfinite([na,nc]).all() and min(na,nc)>0
   values.append(na/nc);records.append(dict(arm=arm,batch=b,mse=float(a),correlation_loss=float(c),mse_encoder_grad=na,corr_encoder_grad=nc,ratio=na/nc,**info));del z,x,ga,gc,a,c
  assert before==w.old.state_hash(m);ratios.append(float(np.median(values)));del m;torch.cuda.empty_cache()
 weight=.25*min(ratios);assert np.isfinite(weight) and weight>0
 weights[str(fold)]=weight;all_results[str(fold)]={'shared_lambda':weight,'median_ratios':ratios,'batches':records,'training_only':True}
p=w.read(R/'protocol.json');p['residual_objective']['lambda_by_fold']=weights;p['residual_objective']['lambda']=None;p['residual_objective']['calibration']='Within each fold training only: 0.25*min(MS/TS median initial encoder gradient ratios); shared across architectures and seeds. Fold5 is full610.';w.put(R/'protocol.json',p);w.put(R/'control/calibration.json',{'status':'PASS','lambda_by_fold':weights,'folds':all_results,'no_optimizer_step':True});print(weights)
