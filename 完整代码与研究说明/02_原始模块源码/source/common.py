from pathlib import Path
import json,hashlib,csv,os,time
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));q.replace(p)
def csvout(p,rows):
 with Path(p).open('w',newline='') as h:w=csv.DictWriter(h,rows[0].keys());w.writeheader();w.writerows(rows)
def append(p,x):
 with Path(p).open('a') as h:h.write(json.dumps(x,allow_nan=False)+'\n')
def verify():
 g=read(R/'release_gate.json');assert g['status']=='PASS'
 for k,v in g['files'].items():assert sha(R/k)==v,k
 return read(R/'protocol.json')
