import os,json,subprocess,time,signal,fcntl,hashlib,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python';E=os.environ.copy();E['CUDA_VISIBLE_DEVICES']='GPU-7ac39891-108b-f897-b92f-cd51e5392bab'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:E[k]='1'
from thermal_helpers import query_gpu
from split_validation import put,sha
read=lambda p:json.loads(Path(p).read_text());CHILDREN=[];STOP=False

def stop(*a):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
def cmd(j,mode='runs',extra=[]):return [PY,'worker.py','--job',str(R/'jobs'/f"{j['name']}.json"),'--mode',mode]+extra

def stage(tasks,width,label,budget_start):
 active={};completed=[];pending=list(tasks);start=time.time();hot=False;paused_at=0;waited=0;peak=0
 while pending or active:
  if STOP or time.time()-budget_start>24*3600:raise RuntimeError('queue stopped or original24h budget')
  if shutil.disk_usage(R).free<80*1024**3:raise RuntimeError('free disk below80GiB')
  g=query_gpu();peak=max(peak,g['temperature_C'])
  if g['temperature_C']>=80 and not hot:
   hot=True;paused_at=time.time()
   for p,_,_,_ in active.values():
    if p.poll()is None:os.killpg(p.pid,signal.SIGSTOP)
  if hot and g['temperature_C']<=65:
   waited+=time.time()-paused_at;hot=False
   for p,_,_,_ in active.values():
    if p.poll()is None:os.killpg(p.pid,signal.SIGCONT)
  if waited+(time.time()-paused_at if hot else 0)>21600:raise RuntimeError('thermal wait6h budget')
  for name,(p,t,f,cp)in list(active.items()):
   if time.time()-t>7200:raise RuntimeError('task2h budget '+name)
   if p.poll()is not None:
    f.close();assert p.returncode==0 and cp.exists(),(name,p.returncode);completed.append(name);del active[name]
  while pending and len(active)<width and not hot and g['memory_fraction']<.9:
   name,args,cp=pending.pop(0)
   if cp.exists()and label in ['inner','outer']and read(cp).get('status')=='COMPLETE':completed.append(name);continue
   f=(R/'logs'/f'{name}.log').open('a');p=subprocess.Popen(args,cwd=R,env=E,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);active[name]=(p,time.time(),f,cp);CHILDREN.append(p)
  put(R/'control/status.json',{'status':'THERMAL_PAUSE'if hot else'RUNNING','phase':label,'completed':len(completed),'total':len(tasks),'active':list(active),'pids':{n:v[0].pid for n,v in active.items()},'pending':len(pending),'parallelism':width,'temperature_C':g['temperature_C'],'peak_C':peak,'thermal_wait_seconds':waited,'time':time.time()})
  time.sleep(3)
 return {'seconds':time.time()-start,'tasks':len(tasks),'peak_C':peak,'thermal_wait_seconds':waited}

def gate(start):
 jobs=read(R/'jobs.json');pilot=[]
 for group in read(R/'protocol.json')['groups']:
  j=next(x for x in jobs if x['group']==group and x['seed']==1 and x['fold']==0 and x['inner']==0 and x['lr']==1e-4)
  pilot.append((j['name'],cmd(j,'pilot',['--epochs','3']),R/'pilot'/j['name']/'completion.json'))
 stage(pilot,4,'pilot',start)
 for j,_,cp in pilot:
  h=[json.loads(l)for l in(cp.parent/'history.jsonl').read_text().splitlines()];assert len(h)==3
  for r in h:assert r['gradient']['encoder_gradient_l2']['mean']>0 and r['gradient']['head_gradient_l2']['mean']>0
 for a in ['MS','TS']:
  dirs=[R/'pilot'/next(j['name']for j in jobs if j['group']==a+'_'+o and j['seed']==1 and j['fold']==0 and j['inner']==0 and j['lr']==1e-4) for o in ['random','pretrained']]
  cs=[read(p/'config.json')for p in dirs];assert cs[0]['initial_head_sha256']==cs[1]['initial_head_sha256']and cs[0]['initial_model_sha256']!=cs[1]['initial_model_sha256']
 j=next(x for x in jobs if x['name']==pilot[0][0]);args=cmd(j,'pilot',['--epochs','3','--tag','resume_check'])
 for extra in [['--stop-after','1'],[]]:
  with(R/'logs/resume_check.log').open('a')as f:subprocess.run(args+extra,cwd=R,env=E,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
 import torch
 c1=torch.load(pilot[0][2].parent/'last.pt',map_location='cpu',weights_only=False);c2=torch.load(R/'pilot/resume_check/last.pt',map_location='cpu',weights_only=False);assert c1['steps']==c2['steps']
 assert all(torch.allclose(c1['model'][k],c2['model'][k],atol=2e-6,rtol=1e-5)for k in c1['model'])
 for k,st in c1['optimizer']['state'].items():
  for n,v in st.items():
   w=c2['optimizer']['state'][k][n];assert torch.allclose(v,w,atol=2e-6,rtol=1e-5)if torch.is_tensor(v)else v==w
 put(R/'control/gate.json',{'status':'PASS','pilot_groups':5,'pilot_epochs':3,'train_encoder_and_head_gradients_verified':True,'paired_head_initialization':True,'resume_checked':True,'resume_max_weight_diff':max(float((c1['model'][k]-c2['model'][k]).abs().max())for k in c1['model']),'time':time.time()})
 bench={}
 for width in [4,6]:
  tasks=[]
  for i in range(12):
   group=['MS_random','TS_random','BNT_random'][i%3];j=next(x for x in jobs if x['group']==group and x['seed']==1 and x['fold']==0 and x['inner']==0 and x['lr']==1e-4);tag=f'width{width}_{i}'
   tasks.append((tag,cmd(j,'benchmark',['--tag',tag,'--benchsteps','100']),R/'benchmark'/tag/'benchmark.json'))
  bench[str(width)]=stage(tasks,width,f'benchmark{width}',start)
 width=min([4,6],key=lambda w:bench[str(w)]['seconds']);put(R/'control/parallelism.json',{'chosen':width,'fixed_workload':'12 fits x100steps same MS/TS/BNT mixture','results':bench});return width

def run():
 lock=(R/'control/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h,f
 clockfile=R/'control/start.json'
 if not clockfile.exists():put(clockfile,{'time':time.time()})
 start=read(clockfile)['time'];width=read(R/'control/parallelism.json')['chosen']if(R/'control/parallelism.json').exists()else gate(start)
 jobs=read(R/'jobs.json');stage([(j['name'],cmd(j),R/'runs'/j['name']/'completion.json')for j in jobs],width,'inner',start)
 import numpy as np
 selected=[];outer=[]
 for group in read(R/'protocol.json')['groups']:
  for seed in [1,2]:
   for fold in range(10):
    choices=[]
    for lr in [1e-5,1e-4]:
     js=[j for j in jobs if j['group']==group and j['seed']==seed and j['fold']==fold and j['lr']==lr];assert len(js)==3
     hs=[]
     for j in js:
      d=R/'runs'/j['name'];c=read(d/'completion.json');assert c['last_sha256']==sha(d/'last.pt');h=[json.loads(l)for l in(d/'history.jsonl').read_text().splitlines()];assert [r['epoch']for r in h]==list(range(1,101));hs.append(h)
     for ep in range(100):choices.append({'lr':lr,'epoch':ep+1,'inner_AUROC':float(np.mean([h[ep]['validation']['AUROC']for h in hs])),'inner_BCE':float(np.mean([h[ep]['validation']['BCE']for h in hs]))})
    best=min(choices,key=lambda v:(-v['inner_AUROC'],v['inner_BCE'],v['epoch'],v['lr']));name=f'{group}_s{seed}_f{fold}_outer';j={'name':name,'group':group,'seed':seed,'fold':fold,'inner':-1,'lr':best['lr'],'epochs':best['epoch']};put(R/'jobs'/f'{name}.json',j);outer.append(j);selected.append({'group':group,'seed':seed,'fold':fold,**best,'test_used':False})
 put(R/'selection.json',selected);put(R/'outer_jobs.json',outer)
 stage([(j['name'],cmd(j),R/'runs'/j['name']/'completion.json')for j in outer],width,'outer',start)
 subprocess.run([PY,'verify.py'],cwd=R,env=E,check=True)
 put(R/'control/status.json',{'status':'COMPLETE','inner_fits':600,'outer_fits':100,'time':time.time(),'next':'report and archive; no extra experiments'})
if __name__=='__main__':
 try:run()
 except BaseException as e:
  for p in CHILDREN:
   if p.poll()is None:
    os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGTERM)
  put(R/'control/error.json',{'error':repr(e),'time':time.time()});put(R/'control/status.json',{'status':'NEEDS_ATTENTION','error':repr(e),'time':time.time()});raise
