import sys,subprocess,signal,os,json
from pathlib import Path
R=Path(__file__).resolve().parent;PY=sys.executable;STOP=False;child=None

def sig(*a):
 global STOP;STOP=True
 if child and child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,sig)
def run(cmd):
 global child
 if STOP:sys.exit(75)
 child=subprocess.Popen([PY]+cmd,cwd=R);rc=child.wait()
 if STOP:sys.exit(75)
 if rc:sys.exit(rc)
a,f,s=sys.argv[1:];name=f'{a}_f{f}_s{s}';mode='reuse' if a=='G0' else 'runs'
if not (R/mode/name/'completion.json').exists():run(['worker.py','--arm',a,'--fold',f,'--seed',s,'--mode',mode])
run(['readout.py','run','--arm',a,'--fold',f,'--seed',s])
