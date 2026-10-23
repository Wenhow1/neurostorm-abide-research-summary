from pathlib import Path
import subprocess,os,json,time,sys
R=Path(__file__).resolve().parent
# Four jobs, exactly two independent workers on each card. All are engineering-only.
procs=[]
for arm,gpu in [('MS',0),('TS',0),('ML',1),('TL',1),('MSO',0),('MSZ',1),('TSZ',1)]:
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
 log=(R/'control'/f'pilot_{arm}.log').open('w')
 p=subprocess.Popen([sys.executable,'pilot_one.py',arm],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append((p,log,arm))
try:
 while any(p.poll() is None for p,l,a in procs):
  temps=[float(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
  with (R/'control/pilot_thermal.jsonl').open('a') as h:h.write(json.dumps({'time':time.time(),'temperatures':temps})+'\n')
  if max(temps)>=80:raise RuntimeError('Pilot temperature80: stop and review')
  if any(p.poll() not in [None,0] for p,l,a in procs):raise RuntimeError('pilot failure')
  time.sleep(3)
 assert all(p.returncode==0 for p,l,a in procs)
 (R/'pilot_checks.json').write_text(json.dumps({'status':'PASS','arms':[json.loads((R/f'pilot/{a}_receipt.json').read_text()) for p,l,a in procs]},indent=2))
finally:
 import signal
 for p,l,a in procs:
  if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  l.close()
