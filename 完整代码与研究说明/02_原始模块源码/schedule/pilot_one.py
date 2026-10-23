import sys,subprocess,time,json,torch
from pathlib import Path
R=Path(__file__).resolve().parent;arm=sys.argv[1];start=time.time()
def run(tag,stop=0):
 cmd=[sys.executable,'worker.py','--arm',arm,'--epochs','6','--schedule','plateau','--mode','pilot','--tag',tag]
 if stop:cmd+=['--stop-after',str(stop)]
 subprocess.run(cmd,cwd=R,check=True)
run(arm+'_continuous');run(arm+'_resume',3);run(arm+'_resume')
a=torch.load(R/f'pilot/{arm}_continuous/last.pt',map_location='cpu',weights_only=False);b=torch.load(R/f'pilot/{arm}_resume/last.pt',map_location='cpu',weights_only=False)
assert a['steps']==b['steps']==366
err=max(float((v-b['model'][k]).abs().max()) for k,v in a['model'].items());assert err<1e-5,err
for tag in [arm+'_continuous',arm+'_resume']:
 p=R/'pilot'/tag;assert (p/'epoch_5.pt').exists() and (p/'epoch_6.pt').exists() and (p/'best.pt').exists()
 bs=json.loads((p/'best.json').read_text());assert bs['epoch']==min([r for r in a['history'] if 'validation_mask50'in r],key=lambda r:(r['validation_mask50']['MSE'],r['epoch']))['epoch']
(R/f'pilot/{arm}_receipt.json').write_text(json.dumps({'arm':arm,'resume_max_parameter_difference':err,'steps':366,'every5_and_best':'PASS','seconds_for_two_6epoch_runs':time.time()-start,'mean_train_epoch_seconds':sum(r['train_epoch_seconds'] for r in a['history'])/6,'status':'PASS'},indent=2))
