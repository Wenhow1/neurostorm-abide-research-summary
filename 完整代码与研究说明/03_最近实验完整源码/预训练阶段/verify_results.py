import json,csv,hashlib,time
from pathlib import Path
import numpy as np
import classifier as c
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=[];folds=[]
p=read(R/'protocol.json');refs=read(R/'references.json');jobs=read(R/'jobs.json');assert len(jobs)==24 and len(refs)==17
assert read(R/'control/status.json')['status']=='COMPLETE'
for fn,h in read(R/'control/frozen_files.json').items():assert c.sha(R/fn)==h
entries=[(R,j,False)for j in jobs]
for x in refs:
 assert c.sha(Path(x['completion_path']))==x['completion_sha256'];entries.append((Path(p['reference_root']),x['job'],True))
for root,j,reused in entries:
 ctx=c.build_context(root/'protocol.json',j);_,_,paths,x,y,ids,splits,name,head,verification,sigmat,signature=ctx
 out=root/'downstream'/name;done=read(out/'completion.json');assert done['status']=='COMPLETE' and done['signature']==signature
 if not reused:
  rc=read(R/'receipts'/f"{j['arm']}_s{j['seed']}_e{j['epoch']}.json")
  assert rc['frozen_verified'] and rc['no_backward'] and rc['no_decoder']
  assert rc['records'][0]['feature_sha256']==c.sha(paths['features'])
  cp=p['checkpoints'][f"{j['arm']}_s{j['seed']}"][str(j['epoch'])]
  assert rc['checkpoint_sha256']==cp['sha256']==c.sha(Path(cp['path']))
 stage='epoch' if not reused else ('raw' if j['arm']=='R0' else ('random' if j.get('kind',j.get('state',''))=='random' or '_random_'in name else 'final'))
 epoch=j.get('epoch',0 if stage in ['random','raw'] else p['checkpoints'].get(f"{j['arm']}_s{j.get('seed')}",{}).get('last',{}).get('actual_epoch'))
 results=[]
 for f,sp in enumerate(splits):
  z=c.fold_result_valid(out/f'fold_{f:02d}',signature,sp,ids,y);assert z is not None
  pred=z['predictions'];m=c.metrics(np.array([t['label']for t in pred]),np.array([t['score']for t in pred]),np.array([t['predicted_label']for t in pred]),np.array([t['probability']for t in pred]))
  for k in ['AUROC','ACC','BCE']:assert abs(m[k]-z['test_metrics'][k])<1e-10
  results.append(z);folds.append({'name':name,'arm':j['arm'],'seed':j.get('seed',0),'stage':stage,'epoch':epoch,'fold':f,'C':z['selected_value'],**{k:m[k]for k in ['AUROC','ACC','BCE']}})
 means={k:float(np.mean([z['test_metrics'][k]for z in results]))for k in ['AUROC','ACC','BCE']}
 for k,v in means.items():assert abs(v-done['fold_equal_mean'][k])<1e-10
 rows.append({'name':name,'arm':j['arm'],'seed':j.get('seed',0),'stage':stage,'epoch':epoch,'reused':reused,**means,'warnings':done['convergence_warning_fits']})
 del x
O=R/'verified_results';O.mkdir(exist_ok=True)
for name,data in [('trajectory.csv',rows),('folds.csv',folds)]:
 with (O/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(O/'verification.json').write_text(json.dumps({'status':'PASS','new_tasks':24,'references':17,'total_folds':len(folds),'checkpoint_and_feature_hashes':True,'fold_artifact_hashes':True,'prediction_metrics_recomputed':True,'frozen_code_hashes':True,'time':time.time()},indent=2))
print('STRICT_PASS',len(rows),len(folds))
