from pathlib import Path
import json,hashlib,shutil,numpy as np
R=Path(__file__).resolve().parent;E=R.parent;U=E/'cpac610_upstream_capacity_20261022';D=E/'cpac966_downstream_linear_20261021'
def put(p,x):p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in ['control','preflight','tmp','cuda_cache','torch_extensions','runs','refits','features','downstream','pilot']: (R/p).mkdir(exist_ok=True)
for src,dst in [(U/'source',R/'source'),(U/'data',R/'data'),(D/'data',R/'downstream_data')]:
 if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=True)
splits=json.loads((U/'splits.json').read_text());splits.append({'inner_train':list(range(610)),'inner_validation':[],'outer_test':[]});put(R/'splits.json',splits)
shutil.copy2(D/'splits.json',R/'downstream_splits.json');shutil.copy2(U/'separation_audit.json',R/'separation_audit.json')
u=np.load(R/'data/IDs.npy');v=np.load(R/'downstream_data/IDs.npy');assert len(u)==610 and len(v)==290 and not set(u)&set(v)
sp=json.loads((R/'downstream_splits.json').read_text());tests=[]
for f in sp:
 tr=set(f['train']);te=set(f['test']);assert len(tr)==261 and len(te)==29 and not tr&te;tests+=list(te)
 for i in f['inner']:assert not set(i['train'])&set(i['validation']) and set(i['train'])|set(i['validation'])==tr
assert sorted(tests)==list(range(290))
a=np.load(R/'data/FC.npy');assert a.shape==(610,116,116) and np.isfinite(a).all() and np.allclose(a,a.transpose(0,2,1))
y=np.load(R/'downstream_data/Y.npy');assert set(y)=={0,1}
import csv
rows=list(csv.DictReader((R/'downstream_data/manifest.csv').open()))
# Label contract verified from stored diagnostic metadata when available.
if 'DX_GROUP' in rows[0]:assert all(int(y[i])==int(float(r['DX_GROUP'])==1) for i,r in enumerate(rows))
put(R/'data_binding.json',{'status':'PASS','development':610,'downstream':290,'holdout_loaded':False,'downstream_label_positive':'ASD (DX_GROUP=1)','files':{str(p.relative_to(R)):sha(p) for sub in ['data','downstream_data'] for p in (R/sub).glob('*') if p.is_file()},'split_sha256':sha(R/'splits.json'),'downstream_split_sha256':sha(R/'downstream_splits.json')})
print('DATA_PASS')
