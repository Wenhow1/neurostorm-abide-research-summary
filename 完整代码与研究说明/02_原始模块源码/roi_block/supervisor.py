import pathlib,json,subprocess,os,time,signal,hashlib,fcntl,argparse,shutil
R=pathlib.Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python'
def read(p):return json.loads(p.read_text())
def put(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
POL=read(R/'resource_policy.json');GPU=int(os.environ.get('COORD_GPU','1'));UUID=['GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7','GPU-7ac39891-108b-f897-b92f-cd51e5392bab'][GPU];SESSION=f'cpac966-roi-token-gpu{GPU}'
STOP=False
def stop(*a):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def event(kind,**kw):
 with (R/'control/events.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'time_unix':time.time(),**kw})+'\n')
def gpu():
 row=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',');assert row[0].strip()==UUID;return dict(zip(['temperature','utilization','memory_used','memory_total'],map(float,row[1:])))
def foreign(active):
 own={a['proc'].pid for a in active};rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10);busy=[]
 for l in rows.splitlines():
  u,p=l.split(',');pid=int(p)
  if u.strip()!=UUID:continue
  try:ours=os.getsid(pid) in own or pid in own
  except ProcessLookupError:continue
  if not ours:busy.append(pid)
 return busy

def validate():
 assert read(R/'engineering.json')['status']=='PASS'
 for rel,h in read(R/'frozen_files.json').items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel
 assert len(read(R/'jobs.json')['jobs'])==6
 for f in [0,1,2]:assert read(R/f'preflight/f{f}.json')['verdict']!='block'
 if 'G1' in read(R/'protocol.json')['arms']:assert read(R/'window_audit.json')['status']=='PASS'
 return True

def resource_blocked(temp,cool,busy,gpu,pol):
 if temp['temperature']>=pol['temperature_pause']:cool=True
 if cool and temp['temperature']<=pol['temperature_resume']:cool=False
 return (gpu in pol['foreign_task_yield_gpu_ids'] and bool(busy)) or cool or temp['memory_used']>=pol['memory_pause_fraction']*temp['memory_total'],cool

def worker():
 validate();lock=(R/f'control/queue_gpu{GPU}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pending=list(read(R/'jobs.json')['jobs']);assert GPU==1;active=[];done=[];start=read(R/'control/authorization.json')['time_unix'];cool=False
 try:
  while pending or active:
   if STOP or (R/'control/STOP').exists() or time.time()-start>12*3600:raise RuntimeError('stop or12h queue limit')
   temp=gpu();busy=foreign(active)
   blocked,cool=resource_blocked(temp,cool,busy,GPU,POL)
   for a in list(active):
    rc=a['proc'].poll()
    if rc is not None:
     a['log'].close();active.remove(a)
     if (R/'readout'/a['name']/'completion.json').exists():done.append(a['job'])
     elif a['pause']:pending.insert(0,a['job'])
     else:raise RuntimeError('unexpected '+a['name']+' exit '+str(rc))
     continue
    if time.time()-a['start']>3*3600:raise RuntimeError('job3h limit '+a['name'])
    if blocked and not a['pause']:os.killpg(a['proc'].pid,signal.SIGTERM);a['pause']=True;a['pause_at']=time.time();event('pause',job=a['job'],gpu=GPU,temperature=temp['temperature'],foreign_pids=busy)
    if a['pause'] and time.time()-a['pause_at']>45:os.killpg(a['proc'].pid,signal.SIGKILL)
   if shutil.disk_usage(R).free<20*1024**3:raise RuntimeError('disk below20GiB')
   while pending and len(active)<POL['parallel_limits'][str(GPU)] and not blocked:
    job=pending.pop(0);arm,f,seed=job;name=f'{arm}_f{f}_s{seed}'
    if (R/'readout'/name/'completion.json').exists():done.append(job);continue
    env={**os.environ,'CUDA_VISIBLE_DEVICES':UUID,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(R/'cuda_cache'),'TORCH_EXTENSIONS_DIR':str(R/'torch_extensions'),'TMPDIR':str(R/'tmp')};log=(R/'control'/f'{name}.log').open('a');p=subprocess.Popen([PY,str(R/'job.py'),arm,str(f),str(seed)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append({'job':job,'name':name,'proc':p,'log':log,'pause':False,'start':time.time()});event('dispatch',job=job,gpu=GPU,pid=p.pid)
   put(R/f'status_gpu{GPU}.json',{'state':'RUNNING','completed':done,'pending':pending,'active':[{'job':a['job'],'gpu':GPU,'pid':a['proc'].pid,'pausing':a['pause']} for a in active],'gpu':temp,'foreign_pids':busy,'foreign_task_yield_enabled':GPU in POL['foreign_task_yield_gpu_ids'],'temperature_pause':POL['temperature_pause'],'temperature_resume':POL['temperature_resume'],'time_unix':time.time()})
   with (R/'control/telemetry.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),'gpu':GPU,'resources':temp,'blocked':blocked})+'\n')
   time.sleep(3)
  env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''};
  if all((R/'readout'/f'{a}_f{f}_s{seed}'/'completion.json').exists() for a,f,seed in read(R/'jobs.json')['jobs']):
   lk=(R/'control/summary.lock').open('w');fcntl.flock(lk,fcntl.LOCK_EX);subprocess.run([PY,str(R/'readout.py'),'summary'],cwd=R,env=env,check=True)
  put(R/f'status_gpu{GPU}.json',{'state':'COMPLETE','completed_jobs':len(done),'new_MAE_fits':6,'time_unix':time.time()})
 except BaseException as e:
  for a in active:
   if a['proc'].poll() is None:os.killpg(a['proc'].pid,signal.SIGTERM)
  deadline=time.time()+45
  while any(a['proc'].poll() is None for a in active) and time.time()<deadline:time.sleep(1)
  for a in active:
   if a['proc'].poll() is None:os.killpg(a['proc'].pid,signal.SIGKILL)
  put(R/f'status_gpu{GPU}.json',{'state':'STOPPED_NEEDS_REVIEW','error':repr(e),'time_unix':time.time()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','launch','worker','status']);a=p.parse_args()
 if a.action=='validate':validate();print('PASS',SESSION,GPU)
 elif a.action=='worker':worker()
 elif a.action=='status':print(json.dumps(read(R/f'status_gpu{GPU}.json'),indent=2))
 else:
  validate();assert read(R/'control/authorization.json')['authorized'];subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(R),f'COORD_GPU={GPU} {PY} -u supervisor.py worker > control/supervisor_gpu{GPU}.log 2>&1'],check=True)
