from pathlib import Path
import sys,json,hashlib,subprocess,time
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
import numpy as np,torch
from common import read,put,sha
from rowmask import sample_rows
import supervisor as sup
A=Path(read(R/'protocol.json')['step1_reference']);assert sha(R/'roi_model.py')==sha(A/'roi_model.py');assert sha(R/'source/splits.json')==sha(A/'source/splits.json')
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
for f in [0,1,2]:
 pr=subprocess.run([sys.executable,'-m','neurostorm_2d.scripts.night_preflight','--data-dir',str(R/'source/data'),'--project-dir',f'/home/wenhao/neurostorm_2d/experiments/cpac966_r012_fullcv_20261021/preflight/f{f}'],cwd='/home/wenhao/neurostorm_2d',capture_output=True,text=True);j=json.loads(pr.stdout);put(R/f'preflight/f{f}.json',j);assert j['verdict']!='block'
initials=[]
for f in [0,1,2]:
 for s in [1,2]:
  m=w.make(f,s,'ROIR');h=w.old.state_hash(m);cfg=A/'runs'/f'ROI_f{f}_s{s}'/'config.json'
  if cfg.exists():assert h==read(cfg)['initial_parameters_sha256']
  initials.append({'fold':f,'seed':s,'initial_hash':h,'matched_step1_config':cfg.exists()})
m=w.make(0,1,'ROIR').cuda();ids=np.load(R/'source/data/IDs.npy');raw=np.load(R/'source/data/FC.npy',mmap_mode='r');sp=read(R/'source/splits.json')[0];tr=sp['inner_train'];q=sp['inner_validation'][:8];norm={'mean':float(raw[tr].mean()),'std':float(raw[tr].std())};x=torch.from_numpy(((raw[q]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
counts=[]
for seed in range(1000):
 mask,r=sample_rows(m,seed,.5);sel=torch.diagonal(mask[0,0]);assert torch.equal(mask[0,0],sel[:,None]|sel[None,:]);assert abs(r['expected_directed']-r['block_target_directed'])<1e-8;counts.append(r)
mask=w.masks(m,ids[q],1,'cpac966_fullcv_train_f0_s1',.5).cuda();assert torch.equal(mask,mask.transpose(-1,-2));m.eval()
with torch.no_grad():
 y=m(x,pixel_mask=mask);x2=x.clone();x2[mask]=100*torch.randn_like(x2[mask]);assert torch.equal(y['prediction'],m(x2,pixel_mask=mask)['prediction']);assert not torch.diagonal(y['score_mask'],dim1=-2,dim2=-1).any()
for ratio in [.5,.75]:
 a=w.masks(m,ids[q],0,'cpac966_fullcv_fixed_eval_v1',ratio)
 b=torch.cat([m.sample_mask(1,device=torch.device('cpu'),mask_ratio=ratio,generator=torch.Generator().manual_seed(int(hashlib.sha256(f'cpac966_fullcv_fixed_eval_v1|{sid}|0'.encode()).hexdigest()[:15],16)))for sid in ids[q]])
 assert torch.equal(a,b)
loss=w.old.loss_mae(m(x,pixel_mask=mask));loss.backward();assert float(m.encoder.roi_identity.grad.norm())>0;assert all(p.grad is None or torch.isfinite(p.grad).all()for p in m.parameters())
pol=read(R/'resource_policy.json');temp=lambda t:dict(temperature=t,memory_used=1,memory_total=100)
assert sup.resource_blocked(temp(80),False,[],0,pol)==(True,True) and sup.resource_blocked(temp(65),True,[],0,pol)==(False,False) and sup.resource_blocked(temp(40),False,[999],0,pol)==(True,False)
put(R/'engineering_static.json',{'status':'PASS','model_source_identical_to_step1':True,'initialization':initials,'mask_union_rows_columns':True,'hidden_values_no_effect':True,'original_block_eval_unchanged':True,'ratio_matching':'unbiased expected edge count, not exact per mask','sample1000_ROI_counts':dict(zip(*[list(map(int,x))for x in np.unique([r['ROI_count']for r in counts],return_counts=True)])),'sample1000_hidden_fraction':float(np.mean([r['hidden_edges_directed']/(116*115)for r in counts])),'sample1000_target_fraction':float(np.mean([r['block_target_directed']/(116*115)for r in counts])),'temperature80_pause65_resume':True,'GPU0_foreign_yield':True,'parameters':sum(p.numel()for p in m.parameters()),'outer_test_evaluated':False,'time':time.time()});print(json.dumps(read(R/'engineering_static.json'),indent=2),flush=True)
