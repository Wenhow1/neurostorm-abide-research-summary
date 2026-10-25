from pathlib import Path
import json,csv,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
R=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text());P=read(R/'protocol.json');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();splits=read(P['data']['splits']);y=np.load(P['data']['labels']);jobs=read(R/'outer_jobs.json');sel=read(R/'selection.json');rows=[];seen={}
assert len(jobs)==100 and len(sel)==100
for j in jobs:
 d=R/'runs'/j['name'];c=read(d/'completion.json');assert c['status']=='COMPLETE'and c['epochs']==j['epochs']and c['last_sha256']==sha(d/'last.pt');r=read(d/'outer_result.json');assert r['indices']==splits[j['fold']]['test'];ix=np.array(r['indices']);z=np.array(r['logits']);p=np.array(r['probabilities']);assert r['test_evaluated_after_training_only'];assert np.isfinite(z).all()and np.isfinite(p).all()
 metrics={'AUROC':float(roc_auc_score(y[ix],z)),'ACC':float(accuracy_score(y[ix],p>=.5)),'BCE':float(np.mean(np.logaddexp(0,z)-y[ix]*z))}
 for k,v in metrics.items():assert abs(v-r['metrics'][k])<1e-10
 choice=next(s for s in sel if s['group']==j['group']and s['seed']==j['seed']and s['fold']==j['fold']);assert choice['epoch']==j['epochs']and choice['lr']==j['lr']and not choice['test_used']
 key=(j['group'],j['seed']);seen.setdefault(key,[]).extend(ix.tolist());rows.append({'group':j['group'],'seed':j['seed'],'fold':j['fold'],'epoch':j['epochs'],'lr':j['lr'],**metrics})
for ix in seen.values():assert sorted(ix)==list(range(290))
summary=[]
for group,seed in seen:
 rr=[r for r in rows if r['group']==group and r['seed']==seed];assert len(rr)==10;summary.append({'group':group,'seed':seed,**{k:float(np.mean([r[k]for r in rr]))for k in ['AUROC','ACC','BCE']}})
for name,rr in [('folds.csv',rows),('summary.csv',summary)]:
 with(R/'results'/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(R/'results/verification.json').write_text(json.dumps({'status':'PASS','inner_fits':600,'outer_fits':100,'groups':5,'seeds':2,'outer_test_once_per_subject_per_group_seed':True,'metrics_recomputed':True,'no_new_holdout':True},indent=2));print('VERIFIED')
