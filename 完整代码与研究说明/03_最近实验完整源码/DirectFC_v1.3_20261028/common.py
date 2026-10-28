"""Production data/graph/metrics/selection functions for locked DirectFC v1.2."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,confusion_matrix
R=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):
 x=np.asarray(x,dtype=np.float64);assert x.shape[-2:]==(116,116) and np.isfinite(x).all()
 assert np.abs(x).max()<=1+1e-6 and np.max(np.abs(x-x.swapaxes(-1,-2)))<=1e-6
 x=(x+x.swapaxes(-1,-2))*.5;x=x.copy();x[...,np.arange(116),np.arange(116)]=0;return x

def graph(f):
 from scipy.sparse.csgraph import connected_components
 f=canonical(f);i,j=np.triu_indices(116,1);v=f[i,j];q=v>0;i,j,v=i[q],j[q],v[q];order=np.lexsort((j,i,-v))[:667]
 a=np.zeros((116,116));a[i[order],j[order]]=v[order];a+=a.T;degree=a.sum(1);b=a+np.eye(116);d=b.sum(1)
 diag={'edges':len(order),'positive_shortage':len(v)<667,'isolated_nodes':int((degree==0).sum()),'components':int(connected_components(a,directed=False,return_labels=False)),'degree_min':float(degree.min()),'degree_max':float(degree.max()),'negative_fraction':float((f[np.triu_indices(116,1)]<0).mean())}
 return (b/np.sqrt(d[:,None]*d[None,:])).astype('float32'),diag

def fit_scale(x):
 f=canonical(x);i,j=np.triu_indices(116,1);s=float(f[:,i,j].std(ddof=0));assert np.isfinite(s) and s>1e-8;return s

def metrics(y,z):
 y=np.asarray(y,dtype=int);z=np.asarray(z,dtype=np.float64);assert len(y)==len(z) and set(y)=={0,1} and np.isfinite(z).all()
 from scipy.special import expit
 p=expit(z);pred=z>=0;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'BCE':float(np.mean(np.logaddexp(0,z)-y*z)),'AUROC':float(roc_auc_score(y,z)),'ACC':float(accuracy_score(y,pred)),'BA':float(balanced_accuracy_score(y,pred)),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'Brier':float(np.mean((p-y)**2)),'class_BCE':{str(k):float(np.mean((np.logaddexp(0,z)-y*z)[y==k])) for k in [0,1]},'class_n':{str(k):int((y==k).sum()) for k in [0,1]},'confusion':[[int(tn),int(fp)],[int(fn),int(tp)]]}

def online_bce(records):
 assert records and sum(r['n'] for r in records)>0
 return sum(r['loss']*r['n'] for r in records)/sum(r['n'] for r in records)
def select(rows,lr_only=False):
 pool=list(rows);assert pool
 for key,maximize in ([('AUROC',True),('BCE',False),('C',False)] if lr_only else [('AUROC',True),('BCE',False),('epoch',False),('params',False),('strategy_priority',False)]):
  best=(max if maximize else min)(x.get(key,0) for x in pool);pool=[x for x in pool if abs(x.get(key,0)-best)<=1e-12]
 return pool[0]
def seed(job):return 2026102700+job['seed']*10000+(job['fold']+1)*100+job['width']
def statehash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def save(p,x):
 import torch
 p=Path(p);q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def restore_logs(out,history,steps):
 for name in ['history.jsonl','gradients.jsonl']:
  p=out/name
  if p.exists():
   lines=p.read_text().splitlines();kept=[l for l in lines if (json.loads(l)['step']<=steps if name=='gradients.jsonl' else json.loads(l)['epoch']<=len(history))]
   tail=lines[len(kept):]
   if tail:(out/(name+f'.incomplete_{time.time_ns()}')).write_text('\n'.join(tail)+'\n')
   p.write_text(('\n'.join(kept)+'\n') if kept else '')
 for p in out.glob('val_epoch_*.npz'):
  if int(p.stem.split('_')[-1])>len(history):p.rename(p.with_suffix(p.suffix+f'.incomplete_{time.time_ns()}'))
def gpu():
 import subprocess
 s=subprocess.check_output(['nvidia-smi','-i',os.environ.get('DIRECTFC_GPU','0'),'--query-gpu=temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=8);t,u,m=map(float,s.strip().split(','));return {'temperature':t,'used_MiB':u,'total_MiB':m}
class Limit(Exception):pass
class Guard:
 def __init__(self,out,formal=False):
  self.out=out;self.formal=formal;self.last=0;self.g={};self.start=read(out/'clock.json') if (out/'clock.json').exists() else {'first_start':time.time(),'thermal_seconds':0};put(out/'clock.json',self.start)
 def check(self):
  import shutil
  now=time.time()
  if now-self.last<5:return self.g
  self.last=now
  if now-self.start['first_start']>7200:raise Limit('single_fit_2h')
  if self.formal and now-read(R/'control/formal_clock.json')['start']>172800:raise Limit('queue_48h')
  if shutil.disk_usage(R).free<80*2**30:raise Limit('disk_below80GiB')
  self.g=gpu();wait_start=time.time();hot=self.g['temperature']>=80
  while hot or self.g['used_MiB']/self.g['total_MiB']>=.9:
   put(self.out/'status.json',{'status':'WAIT_RESOURCE','gpu':self.g,'time':time.time()});time.sleep(5);self.g=gpu()
   if hot and self.g['temperature']<=65:hot=False
   dt=time.time()-wait_start;self.start['thermal_seconds']+=dt;wait_start=time.time();put(self.out/'clock.json',self.start)
   if self.start['thermal_seconds']>=21600 or time.time()-self.start['first_start']>7200:raise Limit('resource_wait_limit')
  return self.g
