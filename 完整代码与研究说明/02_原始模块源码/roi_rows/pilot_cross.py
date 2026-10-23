from pathlib import Path
import subprocess,sys,time,json,os,signal
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
import torch
from common import read,put
# Every short process is monitored; pause by terminating our own group and resume from its saved epoch.
def run(args):
 while True:
  logfile=(R/'control/pilot_subprocess.log').open('a');p=subprocess.Popen([sys.executable]+args,cwd=R,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True);hot=False
  while p.poll()is None:
   t=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True));
   with (R/'control/pilot_temperature.jsonl').open('a')as f:f.write(json.dumps({'time':time.time(),'temperature':t})+'\n')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);busy=[]
   for line in apps.splitlines():
    u,pid=line.split(',');pid=int(pid)
    if u.strip()!='GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7':continue
    try:
     if os.getsid(pid)!=p.pid:busy.append(pid)
    except ProcessLookupError:pass
   if t>=80 or busy:os.killpg(p.pid,signal.SIGTERM);hot=True;break
   time.sleep(2)
  if hot:
   try:p.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   while float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True))>65 or 'GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7' in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
   logfile.close();continue
  logfile.close();assert p.returncode==0,(args,p.returncode);return
run(['cross_eval.py','--fold','0','--seed','1','--pilot'])
print('CROSS_EVAL_PILOT_PASS',flush=True)
