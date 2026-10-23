from pathlib import Path
import sys,json,hashlib,subprocess,time
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
import numpy as np,torch
from common import read,put,sha
import supervisor as sup

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
for f in [0,1,2]:
 args=[sys.executable,'-m','neurostorm_2d.scripts.night_preflight','--data-dir',str(R/'source/data'),'--project-dir',f'/home/wenhao/neurostorm_2d/experiments/cpac966_r012_fullcv_20261021/preflight/f{f}']
 pr=subprocess.run(args,cwd='/home/wenhao/neurostorm_2d',capture_output=True,text=True);j=json.loads(pr.stdout);put(R/f'preflight/f{f}.json',j);assert j['verdict']!='block',j
raw=np.load(R/'source/data/FC.npy',mmap_mode='r');ids=np.load(R/'source/data/IDs.npy');assert raw.shape==(966,116,116) and len(set(ids.tolist()))==966
source_splits=read(R/'source/splits.json');coords=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_coordinate_pe_20261022')
assert sha(R/'source/splits.json')==sha(coords/'source/splits.json') and sha(R/'source/data/FC.npy')==sha(coords/'source/data/FC.npy')
records=[]
for f in [0,1,2]:
 sp=source_splits[f];assert not set(sp['inner_train'])&set(sp['inner_validation']) and not (set(sp['inner_train'])|set(sp['inner_validation']))&set(sp['outer_test'])
 records.append({k:len(v)for k,v in sp.items()if isinstance(v,list)})
m=w.make(0,1,'ROI').cuda();m2=w.make(0,1,'ROI').cuda();assert w.old.state_hash(m)==w.old.state_hash(m2);del m2
baseline=w.old.make('MAE',0).cuda();baseline.decoder.blocks[0]=torch.nn.ModuleList(list(baseline.decoder.blocks[0])[:2])
for r in [.5,.75]:
 for namespace in ['cpac966_fullcv_fixed_eval_v1','cpac966_fullcv_train_f0_s1']:
  a=w.masks(m,ids[:8],1,namespace,r);b=w.masks(baseline,ids[:8],1,namespace,r);assert torch.equal(a,b)
sp=source_splits[0];tr=sp['inner_train'];q=sp['inner_validation'][:8];mean=float(raw[tr].mean());std=float(raw[tr].std());x=torch.from_numpy(((raw[q]-mean)/std).astype('float32'))[:,None].cuda();mask=w.masks(m,ids[q],0,'cpac966_fullcv_fixed_eval_v1',.5).cuda()
m.eval()
with torch.no_grad():
 out=m(x,pixel_mask=mask);changed=x.clone();changed[mask]=torch.randn_like(changed[mask])*100;out2=m(changed,pixel_mask=mask);assert torch.equal(out['prediction'],out2['prediction']);assert out['prediction'].shape==(8,1,116,116);assert m.encode_full(x).shape==(8,1,116,288);assert torch.equal(out['score_mask'],out['score_mask'].transpose(-1,-2));assert not torch.diagonal(out['score_mask'],dim1=-2,dim2=-1).any()
 # Explicit one-row projection check and shared-weight bidirectional reference.
 emb=m.embedded_input(x,mask);reference=m.value_projection(x.masked_fill(mask,0)[:,0,13])+m.mask_projection(mask[:,0,13].float());assert torch.allclose(emb[:,13],reference,atol=2e-6)
 block=m.encoder.blocks[0];a=emb+m.encoder.roi_identity;z=block.norm1(a);v=a+(block.mamba(z)+block.mamba(z.flip(1)).flip(1))*.5;ref=v+block.mlp(block.norm2(v));assert torch.equal(block(a),ref)
opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.01);loss=w.old.loss_mae(m(x,pixel_mask=mask));loss.backward();assert float(m.encoder.roi_identity.grad.norm())>0
for block in m.encoder.blocks:assert any(p.grad is not None and float(p.grad.norm())>0 for p in block.mamba.parameters())
assert float(m.mask_projection.weight.grad.norm())>0;assert all(p.grad is None or torch.isfinite(p.grad).all()for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
pol=read(R/'resource_policy.json');temp=lambda t:dict(temperature=t,memory_used=1,memory_total=100)
assert sup.resource_blocked(temp(80),False,[],1,pol)==(True,True) and sup.resource_blocked(temp(79),True,[],1,pol)==(True,True) and sup.resource_blocked(temp(65),True,[],1,pol)==(False,False) and sup.resource_blocked(temp(40),False,[99],1,pol)==(False,False)
import mamba_ssm,inspect
put(R/'engineering_static.json',{'status':'PASS','data_subjects':966,'split_counts':records,'shape':[8,1,116,288],'identical_B_masks_50_75_train_eval':True,'hidden_value_perturbation_prediction_maxdiff':0,'same_seed_same_parameters':True,'ROI_row_mapping_pass':True,'bidirectional_restoration_pass':True,'real_Mamba_class':str(type(m.encoder.blocks[0].mamba)),'real_Mamba_file':inspect.getfile(type(m.encoder.blocks[0].mamba)),'real_Mamba_version':getattr(mamba_ssm,'__version__','unknown'),'parameters':sum(p.numel()for p in m.parameters()),'encoder_parameters':sum(p.numel()for p in m.encoder.parameters()),'decoder_parameters':sum(p.numel()for p in m.decoder.parameters()),'B_parameters':sum(p.numel()for p in baseline.parameters()),'B_encoder_parameters':sum(p.numel()for p in baseline.encoder.parameters()),'identity_grad_l2':float(m.encoder.roi_identity.grad.norm()),'initial_loss':float(loss),'temperature80_pause65_resume':True,'outer_test_evaluated':False,'time':time.time()})
print(json.dumps(read(R/'engineering_static.json'),indent=2),flush=True)
