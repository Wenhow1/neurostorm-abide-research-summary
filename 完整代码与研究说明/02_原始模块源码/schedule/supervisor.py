from pathlib import Path
import json,subprocess,os,time,signal,hashlib,fcntl,argparse,shutil,datetime
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python';SESSION='cpac610-schedule-mechanism-20261023';UUIDS=['GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7','GPU-7ac39891-108b-f897-b92f-cd51e5392bab'];STOP=False

def read(p):return json.loads(p.read_text())
def put(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def stop(*args):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def event(kind,**kw):
 with (R/'control/events.jsonl').open('a') as h:h.write(json.dumps({'event':kind,'time':time.time(),**kw})+'\n')
def validate():
 assert read(R/'engineering.json')['status']=='PASS'
 for rel,sha in read(R/'frozen_files.json').items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
 assert len(read(R/'jobs.json'))==230 and read(R/'data_binding.json')['status']=='PASS'
 for f in range(5):assert read(R/f'preflight/f{f}.json')['verdict']!='block'
 return True

def blocked(temp,cool,foreign,gpu,now,cutoff):
 if temp>=80:cool=True
 if cool and temp<=65:cool=False
 return cool or (gpu==0 and (foreign or now>=cutoff)),cool

def resources(active):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10);out={}
 for row in rows.splitlines():
  i,u,t,ut,mem,total=[s.strip() for s in row.split(',')];i=int(i);assert UUIDS[i]==u;out[i]={'temperature':float(t),'utilization':float(ut),'memory':float(mem),'memory_total':float(total),'foreign_pids':[]}
 own={a['proc'].pid for a in active};rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
 for row in rows.splitlines():
  u,pid=[s.strip() for s in row.split(',')];pid=int(pid)
  try:ours=pid in own or os.getsid(pid) in own
  except ProcessLookupError:continue
  if not ours and u in UUIDS:out[UUIDS.index(u)]['foreign_pids'].append(pid)
 return out

def commands(stage,j):
 if stage in ['pretrain','refit']:
  mode='runs' if stage=='pretrain' else 'refits';cmd=['worker.py','--arm',j['arm'],'--fold',str(j['fold']),'--seed',str(j['seed']),'--lr',str(j['lr']),'--schedule',j['schedule'],'--epochs',str(j['epochs']),'--mode',mode,'--tag',j['name']];done=R/mode/j['name']/'completion.json'
 elif stage=='features':cmd=['features.py',j['arm'],str(j['seed'])];done=R/'features'/f"{j['arm']}_s{j['seed']}.json"
 else:cmd=['classify.py',j['name']];done=R/'downstream'/j['name']/'completion.json'
 return cmd,done

def worker():
 validate();lk=(R/'control/queue.lock').open('w');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);auth=read(R/'control/authorization.json');start=auth['time'];cutoff=auth['gpu0_cutoff_unix'];active=[];cool={0:False,1:False};caps={0:4,1:4}
 try:
  for stage in ['pretrain','refit','features','classifier']:
   if stage=='pretrain':jobs=read(R/'jobs.json')
   elif stage=='refit':
    from select_refits import build
    jobs=build()
   elif stage=='features':jobs=[{'name':f'{a}_s{s}','arm':a,'seed':s} for a in ['MS','TS','ML','TL','MSO','MSZ','TSZ','MSF','TSF'] for s in [1,2]]
   else:jobs=[{'name':'R0'}]+[{'name':f'{a}_s{s}_{k}'} for a in ['MS','TS','ML','TL','MSO','MSZ','TSZ','MSF','TSF'] for s in [1,2] for k in ['pretrained','random']]
   pending=[j for j in jobs if not commands(stage,j)[1].exists()];event('stage_start',stage=stage,pending=len(pending))
   while pending or active:
    if STOP or (R/'control/STOP').exists() or time.time()-start>24*3600:raise RuntimeError('stop request or24h bound')
    if shutil.disk_usage(R).free<30*1024**3:raise RuntimeError('disk below30GiB')
    rs=resources(active);blocked_gpu={}
    for g in [0,1]:
     v=rs[g];
     if v['temperature']>=77 and caps[g]>2:caps[g]=2;event('thermal_parallel_downshift',gpu=g,cap=2,temperature=v['temperature'])
     blocked_gpu[g],cool[g]=blocked(v['temperature'],cool[g],bool(v['foreign_pids']),g,time.time(),cutoff);blocked_gpu[g]|=v['memory']>=.9*v['memory_total']
    for a in list(active):
     rc=a['proc'].poll()
     if rc is not None:
      a['log'].close();active.remove(a)
      if a['done'].exists():event('job_complete',stage=stage,job=a['job']['name'],gpu=a['gpu'])
      elif a['paused']:pending.insert(0,a['job'])
      else:raise RuntimeError(f"unexpected exit {a['job']['name']} rc{rc}")
      continue
     if time.time()-a['start']>4*3600:raise RuntimeError('4h task bound')
     if a['gpu'] is not None and blocked_gpu[a['gpu']] and not a['paused']:
      os.killpg(a['proc'].pid,signal.SIGTERM);a['paused']=True;a['pause_at']=time.time();event('pause',job=a['job']['name'],gpu=a['gpu'],resources=rs[a['gpu']])
     if a['paused'] and time.time()-a['pause_at']>45:os.killpg(a['proc'].pid,signal.SIGKILL)
    for g in ([None] if stage=='classifier' else [0,1]):
     while pending and sum(a['gpu']==g for a in active)<(2 if g is None else min(caps[g],read(R/'resource_policy.json')['parallel'][str(g)])) and (g is None or not blocked_gpu[g]):
      j=pending.pop(0);cmd,done=commands(stage,j);env={**os.environ,'CUDA_VISIBLE_DEVICES':UUIDS[g] if g is not None else '', 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(R/'cuda_cache'),'TMPDIR':str(R/'tmp')};log=(R/'control'/f"{stage}_{j['name']}.log").open('a');proc=subprocess.Popen([PY,'-u']+cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append({'job':j,'gpu':g,'proc':proc,'log':log,'done':done,'paused':False,'start':time.time()});event('dispatch',job=j['name'],stage=stage,gpu=g,pid=proc.pid)
    put(R/'status.json',{'state':'RUNNING','stage':stage,'completed_in_stage':sum(commands(stage,j)[1].exists() for j in jobs),'total_in_stage':len(jobs),'pending':len(pending),'active':[{'name':a['job']['name'],'pid':a['proc'].pid,'gpu':a['gpu'],'paused':a['paused']} for a in active],'resources':rs,'gpu0_cutoff_unix':cutoff,'time':time.time()})
    with (R/'control/telemetry.jsonl').open('a') as h:h.write(json.dumps({'time':time.time(),'stage':stage,'resources':rs,'blocked':blocked_gpu})+'\n')
    time.sleep(5)
  subprocess.run([PY,'summarize.py'],cwd=R,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},check=True);put(R/'status.json',{'state':'COMPLETE','pretraining_fits':230,'refits':18,'classifiers':37,'time':time.time()})
 except BaseException as e:
  for a in active:
   if a['proc'].poll() is None:os.killpg(a['proc'].pid,signal.SIGTERM)
  limit=time.time()+45
  while any(a['proc'].poll() is None for a in active) and time.time()<limit:time.sleep(1)
  for a in active:
   if a['proc'].poll() is None:os.killpg(a['proc'].pid,signal.SIGKILL)
  put(R/'status.json',{'state':'STOPPED_NEEDS_REVIEW','error':repr(e),'time':time.time()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','launch','worker','status']);a=p.parse_args()
 if a.action=='validate':print(validate())
 elif a.action=='status':print(json.dumps(read(R/'status.json'),indent=2))
 elif a.action=='worker':worker()
 else:
  validate();assert read(R/'control/authorization.json')['authorized'];subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(R),f'{PY} -u supervisor.py worker > control/supervisor.log 2>&1'],check=True);print(SESSION)
