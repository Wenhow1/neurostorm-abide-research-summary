from pathlib import Path
import json,sys,time,hashlib,shutil,inspect
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w;import supervisor as sup
from common import read,put,sha
assert read(R/'engineering_static.json')['status']=='PASS' and read(R/'pilot_checks.json')['status']=='PASS'
assert len(read(R/'jobs.json')['jobs'])==6 and all(j[0]=='ROI' for j in read(R/'jobs.json')['jobs'])
for f in [0,1,2]:assert read(R/f'preflight/f{f}.json')['verdict']!='block'
C=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_coordinate_pe_20261022');shutil.copy2(C/'results.json',R/'baseline_reference.json');shutil.copy2(C/'atlas/AAL116_列标签与图谱质心.csv',R/'ROI身份与固定顺序.csv')
put(R/'engineering.json',{'status':'PASS','static':sha(R/'engineering_static.json'),'pilot':sha(R/'pilot_checks.json'),'new_training_budget':6,'folds':[0,1,2],'seeds':[1,2],'outer_test_evaluated':False,'time':time.time()})
files=['protocol.json','resource_policy.json','jobs.json','worker.py','roi_model.py','readout.py','job.py','augment.py','supervisor.py','engineering.json','engineering_static.json','pilot_checks.json','checks.py','pilot.py','seal.py','baseline_reference.json','ROI身份与固定顺序.csv']
files+=['source/data/FC.npy','source/data/IDs.npy','source/data/Y.npy','source/data/manifest.csv','source/splits.json','source/protocol.json']
files+=[str(p.relative_to(R))for p in (R/'source').rglob('*.py')]
files+=['preflight/f0.json','preflight/f1.json','preflight/f2.json']
put(R/'frozen_files.json',{f:sha(R/f)for f in sorted(set(files))})
from mamba_ssm import Mamba
put(R/'runtime.json',{'Mamba_class':str(Mamba),'Mamba_source':inspect.getfile(Mamba),'Mamba_source_sha256':sha(Path(inspect.getfile(Mamba))),'python':sys.version,'source_resolved':str((R/'source').resolve())})
put(R/'control/authorization.json',{'authorized':True,'scope':'First step ROI-token Mamba only. Six 100epoch fits, fold0-2 seed1-2, old block mask/global288/frozen classifier;GPU1 two parallel,80C pause65C resume; no ROI-row-mask training, no outer test','time_unix':time.time()})
sup.validate();print('SEALED PASS',flush=True)
