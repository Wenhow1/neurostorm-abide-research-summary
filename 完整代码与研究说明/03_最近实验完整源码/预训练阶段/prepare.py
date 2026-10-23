import json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'cpac610_readout_heads_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(v,indent=2))
p=json.loads((OLD/'protocol.json').read_text());p.update(version='2026-10-24-checkpoint-readout-v1',reference_root=str(OLD),epochs=[5,20,40],readouts=['flatten'],heads=['lr'],checkpoints={})
p['limits']['max_total_hours']=24;p['limits']['max_job_hours']=8
p['evaluation']['purpose']='Exploratory checkpoint trajectory; no outer-test checkpoint selection or claims of unbiased new generalization'
src=Path(p['data']['source_root']);p['fc_sha256']=sha(src/'downstream_data/FC.npy')
for d in ['control','features','feature_status','feature_events','receipts','pilot','jobs','logs','downstream']: (R/d).mkdir(exist_ok=True)
for arm in p['arms']:
 for seed in p['seeds']:
  ref=src/'refits'/f'{arm}_full610_s{seed}';entry={};cfg=json.loads((ref/'config.json').read_text())
  for epoch in [5,20,40,'last']:
   path=ref/('last.pt'if epoch=='last'else f'epoch_{epoch}.pt');assert path.is_file()
   entry[str(epoch)]={'path':str(path),'sha256':sha(path),'actual_epoch':cfg['epochs'] if epoch=='last'else epoch}
  p['checkpoints'][f'{arm}_s{seed}']=entry
jobs=[{'name':f'{a}_s{s}_e{e}_flatten_lr','feature_key':f'{a}_s{s}_e{e}_flatten','arm':a,'seed':s,'epoch':e,'head':'lr','readout':'flatten'}for a in p['arms']for s in p['seeds']for e in p['epochs']]
put(R/'protocol.json',p);put(R/'jobs.json',jobs)
for j in jobs:put(R/'jobs'/f"{j['name']}.json",j)
refs=[]
for j in json.loads((OLD/'jobs.json').read_text()):
 if j['head']=='lr'and(j['readout']=='flatten'or j['arm']=='R0'):
  cp=OLD/'downstream'/j['name']/'completion.json';rc=json.loads(cp.read_text());assert rc['status']=='COMPLETE'
  refs.append({'job':j,'completion_path':str(cp),'completion_sha256':sha(cp)})
assert len(refs)==17,len(refs)
put(R/'references.json',refs)
put(R/'control/frozen_files.json',{f:sha(R/f)for f in ['protocol.json','jobs.json','classifier.py','extract.py','frozen_helpers.py','supervisor.py']})
print(json.dumps({'new_features':24,'new_classifiers':24,'references':len(refs),'final_epochs':{k:v['last']['actual_epoch']for k,v in p['checkpoints'].items()}}))
