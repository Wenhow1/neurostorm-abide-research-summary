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
common=['worker.py','--arm','ROIR','--fold','0','--seed','1','--mode','pilot','--epochs','2']
run(common+['--tag','ROIR_full']);run(common+['--tag','ROIR_resume','--stop-after','1']);run(common+['--tag','ROIR_resume'])
a=torch.load(R/'pilot/ROIR_full/last.pt',map_location='cpu',weights_only=False);b=torch.load(R/'pilot/ROIR_resume/last.pt',map_location='cpu',weights_only=False);delta=max(float((a['model'][k]-b['model'][k]).abs().max())for k in a['model']);err=abs(a['history'][-1]['validation_mask50']['MSE']-b['history'][-1]['validation_mask50']['MSE']);assert delta<1e-5 and err<1e-7
run(['readout.py','run','--arm','ROIR','--fold','0','--seed','1','--pilot'])
r=read(R/'pilot_readout/ROIR_f0_s1/completion.json');assert r['feature_shape']==[869,288] and r['backbone_unchanged'] and r['decoder_calls']==0 and not r['outer_test_accessed']
put(R/'pilot_checks.json',{'status':'PASS','epochs':2,'resume_parameter_maxdiff':delta,'resume_validation_MSE_diff':err,'history':[{'epoch':x['epoch'],'loss':x['online_training_loss'],'train_epoch_seconds':x['train_epoch_seconds']}for x in a['history']],'readout_shape':r['feature_shape'],'decoder_calls':r['decoder_calls'],'frozen_backbone_unchanged':True,'no_outer_test':True,'time':time.time()});print(json.dumps(read(R/'pilot_checks.json'),indent=2),flush=True)
