from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
P=read(R/'protocol.json')
for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h,f
for f,h in read(R/'control/upstream_source_hashes.json').items():assert sha(f)==h,f
for k,h in P['data_hashes'].items():assert sha(P['data'][k])==h
for f,h in P['checkpoint_hashes'].items():assert sha(f)==h
jobs=read(R/'jobs.json');assert len(jobs)==600;cache={};heads={};sources=[]
for j in jobs:
 d=R/'runs'/j['name'];c=read(d/'completion.json');assert c['status']=='COMPLETE'and c['epochs']==100 and c['last_sha256']==sha(d/'last.pt')
 h=[json.loads(l)for l in(d/'history.jsonl').read_text().splitlines()];assert [v['epoch']for v in h]==list(range(1,101));assert all('validation'in v for v in h)
 assert np.isfinite([[v['validation'][k]for k in ['BCE','AUROC','ACC']]for v in h]).all();cache[j['name']]=h
 cfg=read(d/'config.json');assert cfg['job']==j and cfg['protocol_sha256']==sha(R/'protocol.json') and cfg['test_evaluated_each_epoch']==False
 if j['group']!='BNT_random':
  key=(j['group'].split('_')[0],j['seed'],j['fold'],j['inner'],j['lr']);heads.setdefault(key,[]).append(cfg['initial_head_sha256'])
 sources.append({'name':j['name'],'history_sha256':sha(d/'history.jsonl'),'completion_sha256':sha(d/'completion.json'),'last_sha256':c['last_sha256']})
assert all(len(v)==2 and v[0]==v[1]for v in heads.values())
sel=read(R/'selection.json');assert len(sel)==100
for s in sel:
 choices=[]
 for lr in P['inner_lr_grid']:
  hs=[cache[j['name']]for j in jobs if j['group']==s['group']and j['seed']==s['seed']and j['fold']==s['fold']and j['lr']==lr];assert len(hs)==3
  for ep in range(100):choices.append({'lr':lr,'epoch':ep+1,'inner_AUROC':float(np.mean([h[ep]['validation']['AUROC']for h in hs])),'inner_BCE':float(np.mean([h[ep]['validation']['BCE']for h in hs]))})
 best=min(choices,key=lambda v:(-v['inner_AUROC'],v['inner_BCE'],v['epoch'],v['lr']))
 for k,v in best.items():assert s[k]==v
 assert not s['test_used']
for j in read(R/'outer_jobs.json'):
 d=R/'runs'/j['name'];c=read(d/'completion.json');h=[json.loads(l)for l in(d/'history.jsonl').read_text().splitlines()];assert [v['epoch']for v in h]==list(range(1,j['epochs']+1));assert all('validation'not in v and 'test'not in v for v in h);assert sha(d/'last.pt')==c['last_sha256']
import subprocess,sys
subprocess.run([sys.executable,'verify.py'],cwd=R,check=True)
(R/'results/full_audit.json').write_text(json.dumps({'status':'PASS','inner_fits':600,'outer_fits':100,'all_last_weight_hashes_checked':True,'selection_recomputed_from_inner_only':True,'paired_heads_checked':len(heads),'no_epoch_test_metrics':True,'source_data_and_code_hashes_unchanged':True,'time':time.time()},indent=2));(R/'results/source_receipts.json').write_text(json.dumps(sources,indent=2));print('FULL_AUDIT_PASS')
