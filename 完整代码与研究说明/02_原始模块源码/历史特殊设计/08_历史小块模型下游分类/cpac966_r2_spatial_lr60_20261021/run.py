from pathlib import Path
import os,sys,json,time,hashlib,csv,subprocess,signal,argparse
D=Path(__file__).resolve().parent;BASE=Path('/home/wenhao/neurostorm_2d/experiments');S=BASE/'cpac966_r012_fullcv_20261021';B=BASE/'cpac966_r1_spatial2x2_strongreg_20261021'
PY='/home/wenhao/neurostorm_2d/.venv/bin/python';GPUS={0:'GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7',1:'GPU-7ac39891-108b-f897-b92f-cd51e5392bab'}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def table(p,rows):
 with p.open('w') as h:w=csv.DictWriter(h,rows[0].keys());w.writeheader();w.writerows(rows)
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]!='Z'
 except FileNotFoundError:return False
def foreign(rows,active):
 own={a['pid'] for a in active if live(a['pid'])};busy={0:[],1:[]}
 for u,pid in rows:
  try:ours=os.getsid(pid) in own
  except ProcessLookupError:continue
  if not ours:
   for g,uid in GPUS.items():
    if u==uid:busy[g].append(pid)
 return busy
def apps():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10);return [(u.strip(),int(p)) for u,p in (l.split(',') for l in raw.splitlines())]
def validate():
 sys.path.insert(0,str(S));from common import verify
 verify();p=read(D/'plan.json');assert p['learning_rates']==[.0001,.0003,.001] and p['epochs']==60 and p['C']==[.001,.003,.01]
 assert p['splits_sha256']==sha(S/'splits.json') and p['script_sha256']==sha(D/'run.py');assert read(B/'status.json')['state']=='COMPLETE'
 return p
STOP=False
def stop(*args):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop)
def boundary():
 if STOP:raise SystemExit(75)
 if (D/'STOP').exists():raise SystemExit(76)
def training(f,smoke=False):
 import numpy as np,torch,joblib,warnings
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.exceptions import ConvergenceWarning
 sys.path.insert(0,str(S));from train_fold import make,metrics,state_hash
 warnings.filterwarnings('error',category=ConvergenceWarning);plan=validate();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=D/('smoke' if smoke else f'f{f}');root.mkdir(exist_ok=True);sp=read(S/'splits.json')[f];ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');raw=np.load(S/'data/FC.npy',mmap_mode='r');ep=read(S/f'runs/f{f}/mae_selection.json')['epoch'];epochs=2 if smoke else 60
 def save(path,x):
  q=path.with_suffix('.tmp');torch.save(x,q);q.replace(path)
 def image(ix,norm):return torch.from_numpy(((raw[ix]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
 def features(m,x):
  t=m.mae.encode_full(x,use_prompts=True);assert tuple(t.shape[1:])==(8,8,288);return t.reshape(t.shape[0],2,4,2,4,288).mean((2,4)).reshape(t.shape[0],1152)
 def evaluate(m,ix,norm):
  m.eval();p=[]
  with torch.inference_mode():
   for k in range(0,len(ix),16):p.append(torch.sigmoid(m.head(features(m,image(ix[k:k+16],norm))).squeeze(-1)).cpu().numpy())
  return metrics(y[ix],np.concatenate(p))
 def fit(scope,lr,tr,va):
  boundary();out=root/f'{scope}_lr{lr:g}';out.mkdir(exist_ok=True);source=S/f'runs/f{f}/MAE_{scope}/epoch_{ep}.pt';digest=sha(source);ck=torch.load(source,map_location='cpu',weights_only=False);norm=ck['norm'];assert set(norm['fit_subject_ids'])==set(ids[sp['inner_train'] if scope=='inner' else sp['outer_train']]);assert not set(ids[va])&set(norm['fit_subject_ids'])
  m=make('T1',f,ck['model']);torch.manual_seed(2026102200+f+900000);m.head=torch.nn.Linear(1152,1);m.cuda();params=[p for p in m.parameters() if p.requires_grad];names=[n for n,p in m.named_parameters() if p.requires_grad];assert all(n.startswith('head.') or n.endswith('.prompt') for n in names);assert sum(p.numel() for p in params)==83953
  frozen=state_hash(m,'frozen');initial=state_hash(m);cfg=dict(scope=scope,fold=f,lr=lr,epochs=epochs,train_ids=ids[tr].tolist(),val_ids=ids[va].tolist(),source_sha256=digest,source=str(source),norm=norm,initial_hash=initial,frozen_hash=frozen,plan_sha256=sha(D/'plan.json'),trainable=83953)
  if (out/'config.json').exists():assert read(out/'config.json')==cfg
  else:put(out/'config.json',cfg)
  opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01,betas=(.9,.999),eps=1e-8);start=1;history=[];steps=0
  if (out/'last.pt').exists():
   prev=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert prev['config']==cfg;m.load_state_dict(prev['model']);opt.load_state_dict(prev['optimizer']);start=prev['epoch']+1;steps=prev['steps'];history=prev['history'];del prev
  began=time.time();calls=[0];hook=m.mae.decoder.register_forward_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
  if not (out/'completion.json').exists():
   for epoch in range(start,epochs+1):
    boundary();m.train();order=np.random.default_rng(2026102100+f*10000+epoch).permutation(tr);total=0.;gn=[];pg=[];updates=[]
    for k in range(0,len(order),16):
     ix=order[k:k+16];opt.zero_grad(set_to_none=True);logits=m.head(features(m,image(ix,norm))).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(y[ix],device='cuda',dtype=torch.float32));assert torch.isfinite(loss);loss.backward();assert not any(p.grad is not None for p in m.parameters() if not p.requires_grad)
     prompt_norm=float(torch.sqrt(sum((p.grad.double().square().sum() for n,p in m.named_parameters() if n.endswith('.prompt') and p.grad is not None),torch.zeros((),device='cuda'))));g=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));assert prompt_norm>0 if steps==0 else True
     before=[p.detach().clone() for p in params] if steps%10==0 else None;opt.step();steps+=1
     if before is not None:updates.append(float(torch.sqrt(sum(((p.detach()-b).double().square().sum() for p,b in zip(params,before)),torch.zeros((),device='cuda')))))
     total+=float(loss.detach())*len(ix);gn.append(g);pg.append(prompt_norm)
    row=dict(epoch=epoch,online_BCE=total/len(tr),gradient_norm_mean=float(np.mean(gn)),prompt_gradient_norm_mean=float(np.mean(pg)),update_norm_sampled_mean=float(np.mean(updates)) if updates else None,elapsed_seconds=time.time()-began)
    if epoch%10==0 or epoch==epochs:row['train']=evaluate(m,tr,norm);row['validation']=evaluate(m,va,norm) if scope=='inner' and len(va) else None
    history.append(row);save(out/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,steps=steps,history=history,config=cfg));(out/'history.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history));put(out/'status.json',dict(epoch=epoch,target=epochs,steps=steps,time_unix=time.time(),elapsed_seconds=time.time()-began))
   assert state_hash(m,'frozen')==frozen and calls[0]==0;put(out/'completion.json',dict(status='COMPLETE',epochs=epochs,steps=steps,frozen_hash_unchanged=True,prompt_hash_changed=state_hash(m)!=initial,decoder_calls=0,checkpoint_sha256=sha(out/'last.pt'),time_unix=time.time()))
  else:assert read(out/'completion.json')['checkpoint_sha256']==sha(out/'last.pt')
  boundary();cache=out/'features.npz';ix=np.r_[tr,va]
  if not cache.exists():
   ff=[];m.eval()
   with torch.inference_mode():
    for k in range(0,len(ix),16):boundary();ff.append(features(m,image(ix[k:k+16],norm)).cpu().numpy())
   np.savez_compressed(cache,features=np.concatenate(ff),ids=ids[ix])
  a=np.load(cache);assert np.array_equal(a['ids'],ids[ix]);x=a['features'];hook.remove();del m,opt,ck;torch.cuda.empty_cache();return x
 def classifier(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=10000,tol=1e-6,random_state=2026102100+f))
 tr=np.array(sp['inner_train']);va=np.array(sp['inner_validation'])
 if smoke:
  tr=np.r_[tr[y[tr]==0][:16],tr[y[tr]==1][:16]];va=np.r_[va[y[va]==0][:8],va[y[va]==1][:8]]
  x=fit('inner',.0001,tr,va);assert x.shape==(48,1152);put(root/'completion.json',dict(status='PASS',scope='2 epochs engineering only; excluded from formal results',time_unix=time.time()));return
 choices=[]
 for lr in plan['learning_rates']:
  x=fit('inner',lr,tr,va)
  for c in plan['C']:
   boundary();m=classifier(c);m.fit(x[:len(tr)],y[tr]);z=metrics(y[va],m.predict_proba(x[len(tr):])[:,1]);choices.append(dict(lr=lr,C=c,**z))
 selected=min(choices,key=lambda a:(a['BCE'],a['lr'],a['C']));put(root/'selection.json',dict(selected=selected,candidates=choices,criterion='inner validation final-logistic BCE; ties lower lr then lower C',epochs_fixed=60,outer_test_used=False));ot=np.array(sp['outer_train']);te=np.array(sp['outer_test']);x=fit('outer',selected['lr'],ot,te);boundary();m=classifier(selected['C']);m.fit(x[:len(ot)],y[ot]);assert m[0].n_samples_seen_==len(ot);p=m.predict_proba(x[len(ot):])[:,1];joblib.dump(m,root/'classifier.joblib');np.savez_compressed(root/'outer_predictions.npz',ids=ids[te],labels=y[te],probability=p);put(root/'completion.json',dict(status='COMPLETE',selected=selected,train=metrics(y[ot],m.predict_proba(x[:len(ot)])[:,1]),test=metrics(y[te],p),prompt_fits=4,logistic_fits=10,backbone_updates=0,time_unix=time.time()))
def summarize():
 import numpy as np
 sys.path.insert(0,str(S));from train_fold import metrics
 ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');sp=read(S/'splits.json');p=np.full(len(ids),np.nan);rows=[]
 for f,s in enumerate(sp):
  c=read(D/f'f{f}/completion.json');a=np.load(D/f'f{f}/outer_predictions.npz');ix=s['outer_test'];assert np.array_equal(a['ids'],ids[ix]) and np.isnan(p[ix]).all();p[ix]=a['probability'];rows.append(dict(fold=f,lr=c['selected']['lr'],C=c['selected']['C'],**{'train_'+k:v for k,v in c['train'].items()},**{'test_'+k:v for k,v in c['test'].items()}))
 assert np.isfinite(p).all();table(D/'fold_results.csv',rows);table(D/'OOF.csv',[dict(subject_id=int(i),label=int(t),probability=float(v)) for i,t,v in zip(ids,y,p)]);comparison=[]
 for name,path in [('R0',S/'R0_OOF.csv'),('R1_1152_strongreg',B/'OOF.csv')]:
  with path.open() as h:a=list(csv.DictReader(h))
  assert [int(z['subject_id']) for z in a]==ids.tolist();comparison.append(dict(route=name,**metrics(y,np.array([float(z['probability']) for z in a]))))
 comparison.append(dict(route='R2_1152_LRselected_60epochs',**metrics(y,p)));table(D/'comparison.csv',comparison);put(D/'results.json',dict(status='COMPLETE',summary=comparison,prompt_fits=40,classifier_fits=100,backbone_updates=0,limitations=['one seed, historically exposed development CV','R2 searches 3 learning rates plus C; R1 only C; unequal tuning budget'],time_unix=time.time()))
def worker():
 import fcntl
 lock=(D/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=validate();assert read(D/'smoke/completion.json')['status']=='PASS';pending=list(range(10));active={};done=[];forced=set();deadline=time.time()+p['max_hours']*3600;cool={0:0,1:0}
 try:
  while pending or active:
   if time.time()>deadline or (D/'STOP').exists():raise RuntimeError('deadline or STOP')
   for f,a in list(active.items()):
    ret=a['proc'].poll()
    if ret is None:continue
    a['log'].close();del active[f]
    if ret==0 and (D/f'f{f}/completion.json').exists():done.append(f)
    elif ret==75 or (ret==-9 and a.get('stopping')):pending.append(f);forced.add(f)
    else:raise RuntimeError(f'fold {f} return {ret}')
   busy=foreign(apps(),list(active.values()));raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10);gpu={}
   for line in raw.splitlines():
    g,t,used,total,util=map(float,line.split(','));gpu[int(g)]=dict(temperature=t,memory=used,total=total,util=util)
    if t>=82 or used>total*.9:cool[int(g)]=time.time()+120
   for f,a in active.items():
    g=a['gpu'];yielding=(g==0 and bool(busy[0])) or cool[g]>time.time()
    if yielding and not a.get('stopping'):
     a['proc'].terminate();a['stopping']=time.time();put(D/f'yield_f{f}_{int(time.time())}.json',dict(gpu=g,foreign_pids=busy[g],telemetry=gpu[g],resume_gpu=1,time_unix=time.time()))
    if a.get('stopping') and time.time()-a['stopping']>60:a['proc'].kill()
   for g in [1,0]:
    while not busy[g] and cool[g]<time.time() and sum(a['gpu']==g for a in active.values())<p['parallel_per_gpu']:
     eligible=[f for f in pending if g==1 or f not in forced]
     if not eligible:break
     f=eligible[0];pending.remove(f)
     if (D/f'f{f}/completion.json').exists():done.append(f);continue
     env={**os.environ,'CUDA_VISIBLE_DEVICES':GPUS[g],'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};log=(D/f'f{f}.log').open('a');proc=subprocess.Popen([PY,str(D/'run.py'),'fold',str(f)],cwd=D,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[f]=dict(pid=proc.pid,gpu=g,proc=proc,log=log)
   put(D/'status.json',dict(state='RUNNING',done=done,pending=pending,active=[dict(fold=f,pid=a['pid'],gpu=a['gpu'],stopping=bool(a.get('stopping'))) for f,a in active.items()],gpu=gpu,foreign=busy,time_unix=time.time()));time.sleep(3)
  summarize();put(D/'status.json',dict(state='COMPLETE',folds=10,time_unix=time.time()))
 except BaseException as e:
  for a in active.values():
   if a['proc'].poll() is None:a['proc'].terminate()
  put(D/'status.json',dict(state='FAILED',error=repr(e),time_unix=time.time()));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['validate','smoke','fold','worker']);a.add_argument('fold',nargs='?',type=int,default=0);v=a.parse_args()
 if v.action=='validate':validate();print('PASS')
 elif v.action=='smoke':training(0,True)
 elif v.action=='fold':training(v.fold)
 else:worker()
