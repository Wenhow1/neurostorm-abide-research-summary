"""四组固定队列，仅本进程创建的子进程可被停止。"""
import os,json,time,subprocess,sys,hashlib,signal,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='GPU-7ac39891-108b-f897-b92f-cd51e5392bab'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
def read(p):return json.loads(p.read_text())
def put(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(a,o,f,s,mode,ep):return [PY,'worker.py','--arm',a,'--objective',o,'--fold',str(f),'--seed',str(s),'--mode',mode,'--epochs',str(ep),'--schedule','plateau']
def call(args,path):
 with path.open('a')as out:subprocess.run(args,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
def gate():
 call([PY,'loss_tests.py'],R/'control/loss_tests.log')
 for a in ['MS','TS']:
  for o in ['base','corr']:call(command(a,o,0,1,'pilot',2),R/'logs'/f'pilot_{a}_{o}.log')
  x=read(R/'pilot'/f'{a}_base_f0_s1/config.json');y=read(R/'pilot'/f'{a}_corr_f0_s1/config.json')
  for k in ['initial_parameters_sha256','encoder_initial_sha256','norm','validation_ids','prior_sha256','parameters','decoder_parameters']:assert x[k]==y[k],k
  for o in ['base','corr']:
   hist=[json.loads(t)for t in (R/'pilot'/f'{a}_{o}_f0_s1/history.jsonl').read_text().splitlines()];assert len(hist)==2
   assert all(h['validation_mask50']['MSE']>0 for h in hist)
  assert read(R/'pilot'/f'{a}_base_f0_s1/train_mask50_epoch_1.json')['mask_sha256']==read(R/'pilot'/f'{a}_corr_f0_s1/train_mask50_epoch_1.json')['mask_sha256']
 files=['worker.py','residual_loss.py','models.py','base_models.py','roi_small.py','roi_large.py','protocol.json','splits.json','supervisor.py','thermal_helpers.py']
 put(R/'control/frozen_files.json',{f:sha(R/f)for f in files});put(R/'control/engineering.json',{'status':'PASS','paired_initialization':True,'paired_masks':True,'loss_tests':True,'pilot_epochs':2,'pilot_models':4,'lambda':read(R/'control/calibration.json')['lambda_by_fold']})
 subprocess.run(['tmux','new-session','-d','-s','cpac610-residual-objective-20261024','-c',str(R),PY,str(R/'supervisor.py'),'run'],check=True)
def run():
 assert read(R/'control/engineering.json')['status']=='PASS'
 for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h
 lock=(R/'control/queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=[(a,o,f,s)for f in range(5)for s in [1,2]for a in ['MS','TS']for o in ['base','corr']];done=[];start=time.time();child=None
 try:
  for a,o,f,s in jobs:
   name=f'{a}_{o}_f{f}_s{s}';cp=R/'runs'/name/'completion.json'
   if cp.exists() and read(cp)['status']=='COMPLETE':done.append(name);continue
   put(R/'control/status.json',{'status':'RUNNING','phase':'upstream_cv','completed':len(done),'total':40,'active':name,'time':time.time()})
   with (R/'logs'/f'{name}.log').open('a')as log:
    child=subprocess.Popen(command(a,o,f,s,'runs',100),cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll()is None:
     if (R/'control/STOP').exists()or time.time()-start>48*3600:
      os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=60);raise RuntimeError('Queue stop or48h budget')
     time.sleep(5)
    assert child.returncode==0 and cp.exists(),name
   assert read(cp)['status']=='COMPLETE';done.append(name)
  put(R/'control/status.json',{'status':'UPSTREAM_COMPLETE','completed':40,'total':40,'next':'Select epoch/LR only from upstream validation; paired full610 refits then fixed flatten LR on290','time':time.time()})
 except BaseException as e:
  put(R/'control/status.json',{'status':'NEEDS_ATTENTION','error':repr(e),'completed':len(done),'time':time.time()});raise
if __name__=='__main__':
 try:gate()if sys.argv[1]=='gate'else run()
 except BaseException as e:put(R/'control/error.json',{'error':repr(e),'time':time.time()});raise
