from pathlib import Path
import sys,time,importlib.util,shutil
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w;import supervisor as sup
from common import read,put,sha
A=Path(read(R/'protocol.json')['step1_reference'])
assert read(R/'engineering_static.json')['status']=='PASS' and read(R/'pilot_checks.json')['status']=='PASS'
assert read(R/'pilot_cross_eval/ROIR_f0_s1/completion.json')['status']=='COMPLETE'
assert sha(R/'roi_model.py')==sha(A/'roi_model.py')
spec=importlib.util.spec_from_file_location('step1_frozen_worker',A/'worker.py');first=importlib.util.module_from_spec(spec);spec.loader.exec_module(first)
initials=[]
for f in [0,1,2]:
 for s in [1,2]:
  m=w.make(f,s,'ROIR');a=first.make(f,s,'ROI');h=w.old.state_hash(m);assert h==w.old.state_hash(a);initials.append({'fold':f,'seed':s,'sha256':h});del m,a
put(R/'paired_initialization.json',{'status':'PASS','all_six_exactly_identical_to_step1_factory':True,'rows':initials})
for f in [0,1,2]:assert read(R/f'preflight/f{f}.json')['verdict']!='block'
assert read(R/'jobs.json')['jobs']==[['ROIR',f,s]for f in [0,1,2]for s in [1,2]]
shutil.copy2(A/'baseline_reference.json',R/'baseline_reference.json');shutil.copy2(A/'ROI身份与固定顺序.csv',R/'ROI身份与固定顺序.csv')
put(R/'engineering.json',{'status':'PASS','checks':sha(R/'engineering_static.json'),'pilot':sha(R/'pilot_checks.json'),'paired_initialization':sha(R/'paired_initialization.json'),'cross_eval_pilot':sha(R/'pilot_cross_eval/ROIR_f0_s1/completion.json'),'outer_test_evaluated':False,'time':time.time()})
files=['protocol.json','jobs.json','resource_policy.json','engineering.json','engineering_static.json','pilot_checks.json','paired_initialization.json','baseline_reference.json','ROI身份与固定顺序.csv']+[p.name for p in R.glob('*.py')]
files+=['source/data/FC.npy','source/data/IDs.npy','source/data/Y.npy','source/data/manifest.csv','source/splits.json','source/protocol.json']+[str(p.relative_to(R))for p in (R/'source').rglob('*.py')]
files+=[f'preflight/f{f}.json'for f in [0,1,2]]
put(R/'frozen_files.json',{f:sha(R/f)for f in sorted(set(files))});put(R/'control/authorization.json',{'authorized':True,'scope':'Step2 only ROIR;6fits folds0-2 seeds1-2;fresh paired initialization;100epochs;GPU0 2parallel;80C pause65C resume;foreign GPU0 task yield;paired block+row eval;no outer test','time_unix':time.time()});sup.validate();print('SEALED_PASS',flush=True)
