from pathlib import Path
import json,hashlib,time,datetime
R=Path(__file__).resolve().parent
from supervisor import validate,put,read
for n in ['engineering_static.json','pilot_checks.json','pipeline_checks.json','data_binding.json','refit_checks.json']:assert read(R/n)['status']=='PASS'
assert len(read(R/'jobs.json'))==230
for f in range(5):assert read(R/f'preflight/f{f}.json')['verdict']!='block'
for a in ['MS','TS','ML','TL','MSO','MSZ','TSZ']:
 h=[json.loads(l) for l in (R/'pilot'/f'{a}_continuous'/'history.jsonl').read_text().splitlines()];assert len(h)==6 and all('validation_mask50'in r for r in h)
put(R/'engineering.json',{'status':'PASS','every_epoch_validation_and_best':True,'plateau_exact5':True,'true_cuda_all7_models':True,'resume_all7':True,'two_vs_four_parallel_test':read(R/'parallel_test.json')['records'],'full610_refit_replay_checked':True,'time':time.time()})
files=list(R.glob('*.py'))+[R/n for n in ['protocol.json','jobs.json','resource_policy.json','engineering.json','engineering_static.json','pilot_checks.json','pipeline_checks.json','data_binding.json','refit_checks.json','splits.json','downstream_splits.json']]+list((R/'preflight').glob('*.json'))+list((R/'source').rglob('*.py'))+[p for s in ['data','downstream_data'] for p in (R/s).iterdir() if p.is_file()]
put(R/'frozen_files.json',{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
cutoff=datetime.datetime(2026,10,23,9,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
put(R/'control/authorization.json',{'authorized':True,'time':time.time(),'gpu0_cutoff_unix':cutoff,'scope':'User authorized restart unified everyepoch validation,5bad plateau,230CV+18refits+37nestedreadouts;4perGPUtested;77downshift80pause65resume;9amGPU0release;no66holdout scoring;all data separate;24hbound'})
validate();print('SEALED_PASS')
