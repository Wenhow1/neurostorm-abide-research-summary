from pathlib import Path
import os,sys,json,time,subprocess,signal,hashlib,csv
S=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r012_fullcv_20261021');D=Path(__file__).resolve().parent
PY='/home/wenhao/neurostorm_2d/.venv/bin/python';GPUS={0:'GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7',1:'GPU-7ac39891-108b-f897-b92f-cd51e5392bab'}
def read(p):return json.loads(p.read_text())
def put(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows):
 with p.open('w') as h:w=csv.DictWriter(h,rows[0].keys());w.writeheader();w.writerows(rows)
def guard():
 g=int(os.environ['PHYSICAL_GPU']);raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 for line in raw.splitlines():
  u,pid=line.split(',')
  if u.strip()==GPUS[g] and int(pid)!=os.getpid():raise SystemExit(75)
 t=float(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))
 if t>=85:raise SystemExit(75)
def fold(f):
 import numpy as np,torch,joblib,warnings
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.exceptions import ConvergenceWarning
 sys.path.insert(0,str(S));from train_fold import make,metrics,state_hash
 warnings.filterwarnings('error',category=ConvergenceWarning)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=D/f'f{f}';out.mkdir(exist_ok=True);sp=read(S/'splits.json')[f];ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');raw=np.load(S/'data/FC.npy',mmap_mode='r');ep=read(S/f'runs/f{f}/mae_selection.json')['epoch'];proof=[]
 def extract(scope,tr,va):
  path=S/f'runs/f{f}/MAE_{scope}/epoch_{ep}.pt';idx=np.array(tr+va);cache=out/f'{scope}_features.npz';meta=out/f'{scope}_features.json';digest=sha(path)
  if cache.exists():
   assert read(meta)['checkpoint_sha256']==digest;a=np.load(cache);assert np.array_equal(a['ids'],ids[idx]);proof.append(read(meta));return a['features']
  guard();ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['norm']['fit_subject_ids']==ids[tr].tolist();assert not set(ids[tr])&set(ids[va]);m=make('MAE',f);m.load_state_dict(ck['model']);m.requires_grad_(False);m.cuda().eval();before=state_hash(m);features=[];calls=[0];hook=m.decoder.register_forward_hook(lambda *a:calls.__setitem__(0,calls[0]+1));norm=ck['norm']
  with torch.inference_mode():
   for k in range(0,len(idx),16):
    guard();x=torch.from_numpy(((raw[idx[k:k+16]]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda();z=m.encode_full(x,use_prompts=False);assert tuple(z.shape[1:])==(8,8,288);features.append(z.reshape(z.shape[0],2,4,2,4,288).mean((2,4)).reshape(z.shape[0],1152).cpu().numpy())
  x=np.concatenate(features);baseline=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r1_standardized_lr_20261021')/f'f{f}'/f'{scope}_features.npz';old=np.load(baseline);assert np.array_equal(old['ids'],ids[idx]);assert np.allclose(x.reshape(len(idx),4,288).mean(1),old['features'],rtol=1e-4,atol=1e-5);assert x.shape==(len(idx),1152) and np.isfinite(x).all();assert state_hash(m)==before and calls[0]==0
  rec=dict(checkpoint=str(path),checkpoint_sha256=digest,normalization_training_ids=ids[tr].tolist(),feature_ids=ids[idx].tolist(),unchanged_model=True,decoder_calls=0,shape=list(x.shape));np.savez_compressed(cache,features=x,ids=ids[idx]);put(meta,rec);proof.append(rec);hook.remove();del m,ck;torch.cuda.empty_cache();return x
 def pipeline(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=10000,tol=1e-6,random_state=2026102100+f))
 tr=sp['inner_train'];va=sp['inner_validation'];x=extract('inner',tr,va);scores=[]
 for c in [.01,.1,1.,10.]:
  m=pipeline(c);m.fit(x[:len(tr)],y[tr]);scores.append(dict(C=c,**metrics(y[va],m.predict_proba(x[len(tr):])[:,1])))
 best=min(scores,key=lambda z:(z['BCE'],z['C']));put(out/'selection.json',dict(selected=best,candidates=scores,outer_test_used=False));ot=sp['outer_train'];te=sp['outer_test'];x=extract('outer',ot,te);m=pipeline(best['C']);m.fit(x[:len(ot)],y[ot]);train=metrics(y[ot],m.predict_proba(x[:len(ot)])[:,1]);p=m.predict_proba(x[len(ot):])[:,1];joblib.dump(m,out/'classifier.joblib');np.savez_compressed(out/'outer_predictions.npz',ids=ids[te],labels=y[te],probability=p);put(out/'completion.json',dict(status='COMPLETE',fold=f,fits=5,backbone_training_steps=0,C=best['C'],validation=best,train=train,test=metrics(y[te],p),sources=proof,time_unix=time.time()))
def summarize():
 import numpy as np
 sys.path.insert(0,str(S));from train_fold import metrics
 ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');splits=read(S/'splits.json');p=np.full(len(y),np.nan);rows=[]
 for f,sp in enumerate(splits):
  c=read(D/f'f{f}/completion.json');a=np.load(D/f'f{f}/outer_predictions.npz');ix=sp['outer_test'];assert np.array_equal(a['ids'],ids[ix]) and np.isnan(p[ix]).all();p[ix]=a['probability'];rows.append(dict(fold=f,C=c['C'],**{'train_'+k:v for k,v in c['train'].items()},**{'test_'+k:v for k,v in c['test'].items()}))
 assert np.isfinite(p).all();table(D/'fold_results.csv',rows);table(D/'OOF.csv',[dict(subject_id=int(i),label=int(t),probability=float(v)) for i,t,v in zip(ids,y,p)])
 comparisons=[]
 for route in ['R0','R1']:
  with (S/f'{route}_OOF.csv').open() as h:a=list(csv.DictReader(h))
  q=np.array([float(x['probability']) for x in a]);comparisons.append(dict(route=route,**metrics(y,q)))
 prev=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r1_standardized_lr_20261021')
 with (prev/'OOF.csv').open() as h:a=list(csv.DictReader(h))
 assert [int(z['subject_id']) for z in a]==ids.tolist()
 q=np.array([float(z['probability']) for z in a]);comparisons.append(dict(route='R1_288_standardized_logistic',**metrics(y,q)))
 comparisons.append(dict(route='R1_1152_spatial2x2_logistic',**metrics(y,p)));table(D/'comparison.csv',comparisons);put(D/'results.json',dict(status='COMPLETE',summary=comparisons,new_classifier_fits=50,backbone_updates=0,limitations=['Compared to the prior 288-dimensional standardized-logistic arm only pooling layout changes; same C search rule, not necessarily same selected C.','Historically exposed development CV; single seed.'],time_unix=time.time()))
def worker():
 deadline=time.time()+3600;pending=list(range(10));active={};done=[];forced=set()
 try:
  while pending or active:
   if time.time()>deadline:raise TimeoutError('one hour bound')
   for g,(f,p,log) in list(active.items()):
    ret=p.poll()
    if ret is None:continue
    log.close();del active[g]
    if ret==0 and (D/f'f{f}/completion.json').exists():done.append(f)
    elif ret==75:pending.append(f);forced.add(f)
    else:raise RuntimeError(f'fold {f} exited {ret}')
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);busy={u.strip() for u,p in [z.split(',') for z in raw.splitlines()]}
   for g in [1,0]:
    if g in active or GPUS[g] in busy:continue
    options=[f for f in pending if g==1 or f not in forced]
    if not options:continue
    f=options[0];pending.remove(f)
    if (D/f'f{f}/completion.json').exists():done.append(f);continue
    env={**os.environ,'CUDA_VISIBLE_DEVICES':GPUS[g],'PHYSICAL_GPU':str(g),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};log=(D/f'f{f}.log').open('a');p=subprocess.Popen([PY,str(D/'run.py'),'fold',str(f)],env=env,cwd=D,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[g]=(f,p,log)
   put(D/'status.json',dict(state='RUNNING',done=done,active=[dict(gpu=g,fold=f,pid=p.pid) for g,(f,p,l) in active.items()],pending=pending,time_unix=time.time()));time.sleep(2)
  summarize();put(D/'status.json',dict(state='COMPLETE',folds=10,new_classifier_fits=50,backbone_updates=0,time_unix=time.time()))
 except BaseException as e:
  for f,p,l in active.values():
   if p.poll() is None:p.terminate()
  put(D/'status.json',dict(state='FAILED',error=repr(e)));raise
if __name__=='__main__':
 if sys.argv[1]=='fold':fold(int(sys.argv[2]))
 else:worker()
