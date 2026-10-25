"""Execution-only 4/6 throughput comparison; frozen worker/protocol unchanged."""
import supervisor as old
from supervisor import R,PY,env,read,put,sha,command
import subprocess,time,os,signal,fcntl,json
from thermal_helpers import query_gpu
LIMIT_START=1792883883.5037117
OWN_CHILDREN=[]

def manage(pending,width,label,formal=False):
 active={};completed=[];start=time.time();waiting=0.;hot=False;pause_start=None;peak=0
 while pending or active:
  if time.time()-LIMIT_START>48*3600:raise RuntimeError('original 48h budget')
  g=query_gpu();peak=max(peak,g['temperature_C'])
  if g['temperature_C']>=80 and not hot:
   hot=True;pause_start=time.time()
   for p,_,_,_ in active.values():os.killpg(p.pid,signal.SIGSTOP)
  if hot and g['temperature_C']<=65:
   waiting+=time.time()-pause_start;hot=False
   for p,_,_,_ in active.values():os.killpg(p.pid,signal.SIGCONT)
  if waiting+(time.time()-pause_start if hot else 0)>21600:raise RuntimeError('thermal 6h limit')
  for name,(p,t,log,cp) in list(active.items()):
   if time.time()-t>8*3600:raise RuntimeError('task 8h limit '+name)
   rc=p.poll()
   if rc is not None:
    log.close();assert rc==0 and cp.exists(),(name,rc);completed.append(name);del active[name]
  while pending and len(active)<width and not hot and g['memory_fraction']<.9:
   name,args,cp=pending.pop(0);log=(R/'logs'/f'{name}.log').open('a');p=subprocess.Popen(args,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[name]=(p,time.time(),log,cp);OWN_CHILDREN.append(p)
  put(R/'control'/f'{label}_status.json',{'status':'WAITING_TEMPERATURE' if hot else 'RUNNING','active':list(active),'pids':{n:v[0].pid for n,v in active.items()},'completed_in_stage':len(completed),'pending':len(pending),'parallelism':width,'temperature':g['temperature_C'],'peak_C':peak,'thermal_wait_seconds':waiting,'time':time.time()})
  if formal:
   done=sum(read(x).get('status')=='COMPLETE' for x in (R/'runs').glob('*/completion.json'))
   put(R/'control/status.json',{'status':'WAITING_TEMPERATURE' if hot else 'RUNNING','phase':'upstream_cv','completed':done,'total':40,'active':list(active),'parallelism':width,'time':time.time()})
  time.sleep(2)
 return {'seconds':time.time()-start,'peak_C':peak,'thermal_wait_seconds':waiting,'tasks':len(completed)}

def main():
 for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h,f
 migration=read(R/'control/parallel_migration.json');pid=migration['old_supervisor_pid'];child=migration['old_worker_pid']
 # Old supervisor is stopped; its child is allowed to finish without interrupting an epoch.
 while True:
  state=subprocess.run(['ps','-p',str(child),'-o','stat='],capture_output=True,text=True).stdout.strip()
  if not state or state.startswith('Z'):break
  time.sleep(2)
 out=R/'runs'/migration['active']
 assert (out/'last.pt').exists(),'checkpoint missing'
 import torch
 ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
 assert ck['epoch']>=1 and ck['config']==read(out/'config.json') and ck['steps']==ck['epoch']*61
 put(R/'control/parallel_checkpoint_verified.json',{'epoch':ck['epoch'],'steps':ck['steps'],'sha256':sha(out/'last.pt'),'reason':'SIGUSR1 may exit between epochs without paused.json; full checkpoint verified'});del ck
 try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
 time.sleep(2)
 lock=(R/'control/queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 results={}
 for width in [4,6]:
  jobs=[]
  for i in range(12):
   a=['MS','TS'][i%2];o=['base','consistency'][(i//2)%2];name=f'parallel_bench_{width}_{i}'
   args=command(a,o,0,1,'benchmark',100)+['--tag',name,'--steps','180']
   jobs.append((name,args,R/'benchmark'/name/'benchmark.json'))
  results[str(width)]=manage(jobs,width,f'benchmark{width}')
  results[str(width)]['training_samples']=12*180*8
  results[str(width)]['aggregate_samples_per_second']=12*180*8/results[str(width)]['seconds']
  put(R/'control/parallel_benchmark.json',results)
 width=min([4,6],key=lambda w:results[str(w)]['seconds'])
 put(R/'control/parallel_choice.json',{'chosen':width,'metric':'same 12 tasks x180 steps total wall seconds, includes startup','results':results,'worker_protocol_unchanged':True,'time':time.time()})
 jobs=[]
 for f in range(5):
  for s in [1,2]:
   for a in ['MS','TS']:
    for o in ['base','consistency']:
     name=f'{a}_{o}_f{f}_s{s}';cp=R/'runs'/name/'completion.json'
     if cp.exists() and read(cp)['status']=='COMPLETE':continue
     jobs.append((name,command(a,o,f,s,'runs',100),cp))
 manage(jobs,width,'parallel_formal',True)
 put(R/'control/status.json',{'status':'UPSTREAM_COMPLETE','completed':40,'total':40,'parallelism':width,'time':time.time(),'next':'Select current validation epoch/LR, 8 refits then frozen LR'})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  for child in OWN_CHILDREN:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGCONT);os.killpg(child.pid,signal.SIGTERM)
  put(R/'control/parallel_error.json',{'error':repr(e),'time':time.time()});raise
