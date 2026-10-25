from pathlib import Path
import json,hashlib,sys
import numpy as np,torch
R=Path(__file__).resolve().parent;SRC=R.parent/'cpac610_schedule_mechanism_20261023';OLD=R.parent/'cpac610_readout_heads_20261023'
sys.path.insert(0,str(R));from split_validation import validate_splits
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data={'fc':str(SRC/'downstream_data/FC.npy'),'labels':str(SRC/'downstream_data/Y.npy'),'ids':str(SRC/'downstream_data/IDs.npy'),'splits':str(SRC/'downstream_splits.json')}
orig=json.loads((OLD/'protocol.json').read_text());hashes={k:sha(Path(p))for k,p in data.items()}
for k in ['labels','ids','splits']:assert hashes[k]==orig['input_hashes'][k]
y=np.load(data['labels']);ids=np.load(data['ids']);splits=json.loads(Path(data['splits']).read_text());verified=validate_splits(splits,y,ids);x=np.load(data['fc'],mmap_mode='r');assert x.shape==(290,116,116)and np.isfinite(x).all()
checkpoints={};norm=None
for a in ['MS','TS']:
 for s in [1,2]:
  base=SRC/'refits'/f'{a}_full610_s{s}'
  for n in ['initial.pt','last.pt']:checkpoints[str(base/n)]=sha(base/n)
  c=torch.load(base/'last.pt',weights_only=False,map_location='cpu');v=c['norm'];assert len(v['fit_subject_ids'])==610 and not set(map(str,v['fit_subject_ids']))&set(map(str,ids))
  v={'mean':v['mean'],'std':v['std'],'source':'original upstream610 training only','fit_subject_count':610}
  if norm is None:norm=v
  else:assert norm==v
  assert json.loads((base/'completion.json').read_text())['last_sha256']==checkpoints[str(base/'last.pt')]
for d in ['control','logs','jobs','runs','pilot','benchmark','results']:(R/d).mkdir(exist_ok=True)
groups=['MS_random','MS_pretrained','TS_random','TS_pretrained','BNT_random'];jobs=[]
for fold in range(10):
 for seed in [1,2]:
  for lr in [1e-5,1e-4]:
   for inner in range(3):
    for group in groups:
     name=f'{group}_s{seed}_f{fold}_i{inner}_lr{lr:.0e}';job={'name':name,'group':group,'seed':seed,'fold':fold,'inner':inner,'lr':lr,'epochs':100};jobs.append(job);(R/'jobs'/f'{name}.json').write_text(json.dumps(job,indent=2))
p={'version':'cpac290-supervised-transfer-20261026-v1','groups':groups,'data':data,'data_hashes':hashes,'model_root':str(SRC),'checkpoint_hashes':checkpoints,'normalization':norm,'epochs_max':100,'inner_lr_grid':[1e-5,1e-4],'optimizer':{'name':'AdamW','weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'schedule':'fixed','batch':8,'clip':1.},'selection':'each group/seed/outer fold: select common LR and epoch maximizing mean AUROC across3inner folds; tie lower mean BCE then earlier epoch then smaller LR; refit all261 from original initialization to chosen epoch; outer29 only evaluated once after fit','seeds':[1,2],'planned_inner_fits':600,'planned_outer_fits':100,'label_use':'only respective inner/outer training partitions; validation for selection; heldout66 not accessed','limits':{'queue_hours':24,'job_hours':2,'thermal_wait_hours':6,'min_free_disk_GB':80,'gpu_index':1,'pause_C':80,'resume_C':65,'max_parallel':6},'BNT':{'official_commit':'8a588aadad0166209269fa114e5df4e42209e207','node_sz':116,'sizes':[116,100],'pooling':[False,True],'pos_encoding':'none','hidden_size':1024,'nhead':4,'dropout':.1,'freeze_center':True,'orthogonal':True,'project_assignment':True,'loss':'BCE on logit1-logit0, equivalent two-class CE, no auxiliary clustering KL','adaptation':'replace annotation-only OmegaConf import with Any; add optional is_causal=False compatibility argument; retain official modules and classifier; our split/normalization/optimizer differs, not paper-score reproduction'},'scope':'main random vs originalMSE initialization. BNT extra architecture reference, not isolated readout ablation. Frozen LR historical refs differ in optimizer, not matched freeze ablation.'}
if(R/'protocol.json').exists():assert json.loads((R/'protocol.json').read_text())==p
else:(R/'protocol.json').write_text(json.dumps(p,indent=2))
(R/'jobs.json').write_text(json.dumps(jobs,indent=2));(R/'control/data_preflight.json').write_text(json.dumps({'status':'PASS',**verified,'pretrain_downstream_disjoint':True,'pretrain_normalization_verified':True,'no_66_access':True,'data_hashes':hashes},indent=2));print('PREPARED',len(jobs))
