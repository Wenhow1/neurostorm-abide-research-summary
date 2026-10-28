"""Validated v1.3 dual-GPU queue, max4 new jobs per GPU, explicit TEST_LOCKED end."""
import sys,subprocess,signal,shutil,traceback,fcntl
from common import *
STOP=False
LR_HANDLE=None

def stop(*a):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
def validate():
 jobs=read(R/'jobs.json');expected={(c,a,w,f,s) for c in ['A','B','C','D'] for a in (['GCN'] if c=='D' else ['Mamba','Transformer','GCN']) for w in [64,128] for f in range(5) for s in [1,2]}
 assert len(jobs)==200 and len({j['name'] for j in jobs})==200
 assert {(j['condition'],j['arm'],j['width'],j['fold'],j['seed']) for j in jobs}==expected
 assert all(j['epochs']==100 and j['lr']==.0001 and j['gpu']==j['seed']-1 for j in jobs)
 assert len(read(R/'lr_jobs.json'))==30
 assert read(R/'protocol.json')['budget']['formal_total']==237
 put(R/'control/manifest_validation.json',{'status':'PASS','neural_dev':200,'lr_dev':30,'final':7,'total':237,'gpus':[0,1],'new_parallel_per_gpu':4,'test_auto_score':False})
def gpus():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=8);return {int(v[0]):{'temperature':float(v[1]),'used_MiB':float(v[2]),'total_MiB':float(v[3])} for v in [line.split(',') for line in s.strip().splitlines()]}
def queue(jobs,phase,cpu=False):
 active=[];pending=list(jobs);clock=read(R/'control/formal_clock.json');thermal=clock.get('thermal_seconds',0);previous=time.time();hot={0:False,1:False}
 try:
  while pending or active:
   now=time.time();clock=read(R/'control/formal_clock.json');gg=gpus()
   assert not (R/'control/lr_error.json').exists(),'LR queue blocked; full development cannot continue silently'
   if any(hot.values()):thermal+=now-previous
   previous=now;clock['thermal_seconds']=thermal
   if not cpu:put(R/'control/formal_clock.json',clock)
   if STOP or now-clock['start']>172800 or thermal>21600 or shutil.disk_usage(R).free<80*2**30:raise Limit('queue_stop_or_resource_limit')
   for g in [0,1]:
    if gg[g]['temperature']>=80:hot[g]=True
    if hot[g] and gg[g]['temperature']<=65:hot[g]=False
   for item in active:
    proc,f,j,start,suspended,g=item
    if now-start>7200:raise Limit('fit_2h')
    if not cpu and hot[g] and not suspended:os.killpg(proc.pid,signal.SIGSTOP);item[4]=True
    if not cpu and not hot[g] and suspended:os.killpg(proc.pid,signal.SIGCONT);item[4]=False
   for g in ([0] if cpu else [0,1]):
    while sum(item[5]==g for item in active)<4 and not (not cpu and hot[g]) and (cpu or gg[g]['used_MiB']/gg[g]['total_MiB']<.9):
     candidates=[i for i,j in enumerate(pending) if cpu or j['gpu']==g]
     if not candidates:break
     j=pending.pop(candidates[0]);out=R/'runs'/j['name']
     if (out/'completion.json').exists():assert read(out/'completion.json')['status']=='COMPLETE';continue
     jp=R/'jobs'/f"{j['name']}.json";put(jp,j);out.mkdir(parents=True,exist_ok=True)
     args=[sys.executable,str(R/('lr_worker.py' if cpu else 'worker.py')),'--job',str(jp)]
     if not cpu:args+=['--out',str(out)]
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),DIRECTFC_GPU=str(g),DIRECTFC_SCAN_REF='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
     f=(R/'logs'/f"{j['name']}.log").open('a');proc=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);active.append([proc,f,j,time.time(),False,g])
   for item in list(active):
    proc,f,j,start,suspended,g=item
    if proc.poll() is not None:
     f.close();active.remove(item)
     if proc.returncode==75:raise Limit(j['name']+' resource_limit')
     assert proc.returncode==0,(j['name'],proc.returncode)
     assert read(R/'runs'/j['name']/'completion.json')['status']=='COMPLETE'
   done=sum((R/'runs'/j['name']/'completion.json').exists() for j in jobs)
   put(R/'control'/('lr_status.json' if cpu else 'status.json'),{'status':'THERMAL_PAUSED' if any(hot.values()) else 'RUNNING','phase':phase,'done':done,'total':len(jobs),'active':[{'name':v[2]['name'],'gpu':v[5],'pid':v[0].pid} for v in active],'gpu':gg,'time':time.time(),'thermal_seconds':thermal,'test_locked':True});time.sleep(3)
 finally:
  for proc,f,j,start,suspended,g in active:
   if proc.poll() is None:
    if suspended:os.killpg(proc.pid,signal.SIGCONT)
    os.killpg(proc.pid,signal.SIGTERM)
  for proc,f,j,start,suspended,g in active:
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   f.close()
def main():
 global LR_HANDLE
 assert not (R/'control/user_hold.json').exists(),'user hold'
 validate();gate=read(R/'control/gate_A.json');assert gate['status']=='PASS'
 for n,h in gate['sources'].items():assert sha(R/n)==h,n
 if not (R/'control/formal_clock.json').exists():put(R/'control/formal_clock.json',{'start':time.time(),'thermal_seconds':0})
 # A and CPU baseline overlap without changing GPU worker slots.
 lrproc=None
 if not (R/'control/lr_dev_done.json').exists():
  log=(R/'logs/lr_queue.log').open('a');lrproc=subprocess.Popen([sys.executable,str(R/'supervisor.py'),'lr-dev'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);LR_HANDLE=lrproc
 for cond in ['A','B','C','D']:
  queue([j for j in read(R/'jobs.json') if j['condition']==cond],'NEURAL_'+cond)
  put(R/'control'/f'stage_{cond}_done.json',{'status':'COMPLETE','time':time.time()})
 if lrproc is not None:assert lrproc.wait()==0,'LR phase';log.close()
 assert read(R/'control/lr_dev_done.json')['status']=='COMPLETE'
 subprocess.run([sys.executable,str(R/'finalize.py'),'select'],check=True)
 rr=read(R/'refits.json');queue([j for j in rr if not j['name'].startswith('LR_')],'NEURAL_REFIT');queue([j for j in rr if j['name'].startswith('LR_')],'LR_REFIT',True)
 subprocess.run([sys.executable,str(R/'finalize.py'),'freeze'],check=True)
 put(R/'control/status.json',{'status':'FROZEN_TEST_LOCKED','formal_fits':237,'test_scored':False,'time':time.time(),'next':'development report then explicit test release; no automatic next experiment'})
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='validate':validate();print('VALIDATE_PASS');sys.exit()
 if len(sys.argv)>1 and sys.argv[1]=='lr-dev':
  try:queue(read(R/'lr_jobs.json'),'LR_DEVELOPMENT',True);put(R/'control/lr_dev_done.json',{'status':'COMPLETE'})
  except Exception as e:put(R/'control/lr_error.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc()});raise
  sys.exit()
 lock=(R/'control/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 except Limit as e:put(R/'control/status.json',{'status':'PAUSED_LIMIT','complete':False,'reason':str(e),'time':time.time(),'test_locked':True})
 except Exception as e:put(R/'control/error.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

 finally:
  if LR_HANDLE is not None and LR_HANDLE.poll() is None:
   os.killpg(LR_HANDLE.pid,signal.SIGTERM)
   try:LR_HANDLE.wait(timeout=40)
   except subprocess.TimeoutExpired:pass
