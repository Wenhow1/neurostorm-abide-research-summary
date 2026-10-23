"""Bounded project-local queue for frozen inference and nested CPU classifiers."""
from pathlib import Path
import argparse,json,time,subprocess,os,signal,hashlib,fcntl,sys
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python';UUID='GPU-7ac39891-108b-f897-b92f-cd51e5392bab';STOP=False

def read(p):return json.loads(p.read_text())
def put(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def event(kind,**kw):
 with (R/'control/events.jsonl').open('a')as f:f.write(json.dumps({'event':kind,'time':time.time(),**kw})+'\n')
def stop(*x):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def validate():
 p=read(R/'protocol.json');assert len(read(R/'jobs.json'))==99
 assert read(R/'control/engineering.json')['status']=='PASS'
 for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h,f
 assert p['limits']['gpu0_used']is False and p['limits']['temperature_pause']==80
 return p

def launch():
 validate();session='cpac610-readout-heads-20261023'
 if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise RuntimeError('queue already running')
 subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(R),PY,str(R/'supervisor.py'),'worker'],check=True)
 print(json.dumps({'status':'LAUNCHED','session':session}))

def worker():
 p=validate();lk=(R/'control/queue.lock').open('w');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=read(R/'jobs.json');fjobs=[{'name':f'{a}_s{s}','arm':a,'seed':s}for a in p['arms']for s in p['seeds']];active=[];failed={};start=time.time();attempted=set();env=os.environ.copy()
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
 def done(stage,j):return R/('receipts'if stage=='features'else'downstream')/(j['name']+'.json'if stage=='features'else j['name']+'/completion.json')
 ready_cache={}
 def complete(stage,j):
  path=done(stage,j)
  return path.exists() and read(path).get('status')=='COMPLETE'
 def ready(j):
  feature=R/'features'/(j['feature_key']+'.npy')
  if not feature.exists():return False
  if j['readout']!='flatten':return True
  receipt=R/'receipts'/f"{j['arm']}_s{j['seed']}.json"
  if not receipt.exists():return False
  rc=read(receipt)
  if rc.get('status')!='COMPLETE':return False
  rec=next((v for v in rc['records'] if v['name']==j['feature_key']),None)
  if rec is None:return False
  stat=(feature.stat().st_size,feature.stat().st_mtime_ns,rec['feature_sha256'])
  if ready_cache.get(str(feature))!=stat:
   if sha(feature)!=rec['feature_sha256']:raise RuntimeError('Feature receipt SHA mismatch')
   ready_cache[str(feature)]=stat
  return True
 def terminate(a,reason):
  event('TERMINATE',name=a['job']['name'],reason=reason)
  try:os.killpg(a['proc'].pid,signal.SIGTERM);a['proc'].wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(a['proc'].pid,signal.SIGKILL);a['proc'].wait()
  except ProcessLookupError:pass
 try:
  while True:
   if STOP or (R/'control/STOP').exists()or time.time()-start>p['limits']['max_total_hours']*3600:
    for a in active:terminate(a,'stop or queue time budget')
    put(R/'control/status.json',{'status':'STOPPED','time':time.time(),'failed':failed});return
   for a in active[:]:
    if time.time()-a['start']>p['limits']['max_job_hours']*3600:terminate(a,'job time budget')
    code=a['proc'].poll()
    if code is not None:
     a['log'].close();active.remove(a);path=done(a['stage'],a['job']);ok=code==0 and path.exists()and read(path)['status']=='COMPLETE'
     if not ok:failed[a['stage']+':'+a['job']['name']]={'exit_code':code,'log':str(a['path']),'time':time.time()}
     event('FINISH',stage=a['stage'],name=a['job']['name'],ok=ok,seconds=time.time()-a['start'])
   completedf=[j['name']for j in fjobs if complete('features',j)];completedc=[j['name']for j in jobs if complete('classifier',j)]
   all_finished=len(set(completedf)|{k.split(':',1)[1] for k in failed if k.startswith('features:')})==8 and len(set(completedc)|{k.split(':',1)[1] for k in failed if k.startswith('classifier:')})==99
   pendingc=[j for j in jobs if j['name']not in completedc and 'classifier:'+j['name']not in attempted]
   pendingf=[j for j in fjobs if j['name']not in completedf and 'features:'+j['name']not in attempted]
   # Failed feature dependencies remain explicitly blocked, never silently dropped.
   if not active and not pendingf and pendingc:
    for j in pendingc:
     if not ready(j):failed['classifier:'+j['name']]={'reason':'feature dependency unavailable or incomplete'};attempted.add('classifier:'+j['name'])
    pendingc=[j for j in pendingc if 'classifier:'+j['name']not in attempted]
    all_finished=not pendingc
   if all_finished and not active:
    put(R/'control/status.json',{'status':'COMPLETE'if not failed else'NEEDS_ATTENTION','features_done':len(completedf),'classifiers_done':len(completedc),'failed':failed,'time':time.time(),'elapsed_seconds':time.time()-start});event('QUEUE_END',failed=len(failed));return
   mem={x.split(':')[0]:int(x.split()[1])for x in Path('/proc/meminfo').read_text().splitlines()};allowcpu=mem['MemAvailable']>8*1024*1024
   for stage,waiting,cap in [('features',pendingf,1),('classifier',pendingc,p['limits']['cpu_parallel'])]:
    slots=cap-sum(a['stage']==stage for a in active)
    if stage=='classifier'and not allowcpu:slots=0
    for j in waiting:
     if slots<=0:break
     if stage=='classifier':
      if not ready(j):continue
      jp=R/'jobs'/(j['name']+'.json');args=['classifier.py','--protocol','protocol.json','--job',str(jp)];e=env.copy();e['CUDA_VISIBLE_DEVICES']=''
     else:args=['feature_extract.py','--protocol','protocol.json','--arm',j['arm'],'--seed',str(j['seed'])];e=env.copy();e['CUDA_VISIBLE_DEVICES']=UUID
     path=R/'logs'/(stage+'_'+j['name']+'.log');log=path.open('a');proc=subprocess.Popen([PY]+args,cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     active.append({'job':j,'stage':stage,'proc':proc,'start':time.time(),'log':log,'path':path});attempted.add(stage+':'+j['name']);slots-=1;event('START',stage=stage,name=j['name'],pid=proc.pid)
   put(R/'control/status.json',{'status':'RUNNING','time':time.time(),'elapsed_seconds':time.time()-start,'features_done':len(completedf),'classifiers_done':len(completedc),'features_total':8,'classifiers_total':99,'cpu_parallel':p['limits']['cpu_parallel'],'active':[{'stage':a['stage'],'name':a['job']['name'],'pid':a['proc'].pid,'seconds':time.time()-a['start']}for a in active],'failed':failed,'memory_available_GiB':mem['MemAvailable']/1024**2})
   time.sleep(5)
 except BaseException as error:
  put(R/'control/status.json',{'status':'NEEDS_ATTENTION','error':repr(error),'failed':failed,'time':time.time()});event('QUEUE_ERROR',error=repr(error));raise
 finally:
  for a in active:
   if a['proc'].poll()is None:terminate(a,'supervisor exit')
  fcntl.flock(lk,fcntl.LOCK_UN)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='validate':validate();print('VALIDATION_PASS')
 elif action=='launch':launch()
 elif action=='worker':worker()
 elif action=='status':print((R/'control/status.json').read_text())
 else:raise ValueError(action)
