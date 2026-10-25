"""Server-side continuation: validation-only selection, refit, frozen LR, strict verification."""
from pathlib import Path
import json,time,os,subprocess,fcntl,signal,hashlib,shutil
import numpy as np
import parallel_supervisor as ps
R=ps.R;PY=ps.PY;read=ps.read;put=ps.put;sha=ps.sha
D=R.parent/'cpac610_consistency_downstream_20261025'

def status(phase,**kw):put(R/'control/pipeline_status.json',dict(status='RUNNING',phase=phase,time=time.time(),**kw))
def main():
 lock=(R/'control/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status('waiting_upstream')
 while read(R/'control/status.json')['status']!='UPSTREAM_COMPLETE':
  if (R/'control/parallel_error.json').exists():raise RuntimeError('upstream parallel error')
  if time.time()-ps.LIMIT_START>48*3600:raise TimeoutError('original48h')
  time.sleep(5)
 queue_lock=(R/'control/queue.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for f,h in read(R/'control/frozen_files.json').items():assert sha(R/f)==h,f
 status('verify_and_select')
 epochs={};lrs={};summary={};per_run=[]
 for arm in ['MS','TS']:
  for obj in ['base','consistency']:
   key=arm+'_'+obj;histories=[]
   for fold in range(5):
    for seed in [1,2]:
     d=R/'runs'/f'{key}_f{fold}_s{seed}';c=read(d/'completion.json');assert c['status']=='COMPLETE' and c['epochs']==100 and sha(d/'last.pt')==c['last_sha256']
     h=[json.loads(l)for l in (d/'history.jsonl').read_text().splitlines()];assert [v['epoch']for v in h]==list(range(1,101));histories.append(h)
     assert np.isfinite([[v['validation_mask50'][m]for m in ['MSE','residual_correlation','residual_RMS_ratio']]for v in h]).all()
   curve=np.mean([[v['validation_mask50']['MSE']for v in h]for h in histories],axis=0);n=int(np.argmin(curve))+1;epochs[key]=n
   lrs[key]=np.median([[v['lr_last_update']for v in h[:n]]for h in histories],axis=0).tolist()
   summary[key]={'selected_epoch':n,'validation_curve':curve.tolist(),**{m:float(np.mean([h[n-1]['validation_mask50'][m]for h in histories]))for m in ['MSE','residual_correlation','residual_RMS_ratio']}}
   for i,h in enumerate(histories):per_run.append({'group':key,'fold':i//2,'seed':i%2+1,'epoch':n,**h[n-1]['validation_mask50']})
 sel={'epochs':epochs,'replay_lrs':lrs,'criterion':'mean raw validation MSE over 5 folds x2 seeds; ties earliest; LR median of actual update LR','downstream_accessed':False}
 if (R/'selection.json').exists():assert read(R/'selection.json')==sel
 else:put(R/'selection.json',sel)
 put(R/'upstream_summary.json',{'groups':summary,'per_run':per_run,'verified_runs':40})
 src=(R/'worker.py').read_text();needle="read(R/'selection.json')['replay_lrs'][a.arm]";assert src.count(needle)==1
 src=src.replace(needle,"read(R/'selection.json')['replay_lrs'][a.arm+'_'+a.objective]");compile(src,'refit_worker.py','exec');(R/'refit_worker.py').write_text(src)
 width=read(R/'control/parallel_choice.json')['chosen'];jobs=[]
 for a in ['MS','TS']:
  for o in ['base','consistency']:
   name=f'refit_gate_{a}_{o}';n=epochs[a+'_'+o]
   args=[PY,'refit_worker.py','--arm',a,'--objective',o,'--fold','5','--seed','1','--mode','pilot','--tag',name,'--schedule','replay','--epochs',str(n),'--stop-after','1']
   jobs.append((name,args,R/'pilot'/name/'last.pt'))
 status('refit_short_check',epochs=epochs);ps.manage(jobs,4,'refit_gate')
 for a in ['MS','TS']:
  for o in ['base','consistency']:
   d=R/'pilot'/f'refit_gate_{a}_{o}';h=json.loads((d/'history.jsonl').read_text().splitlines()[0]);c=read(d/'config.json')
   assert h['epoch']==1 and h['lr_last_update']==lrs[a+'_'+o][0] and c['fold']==5 and len(c['norm']['fit_subject_ids'])==610 and c['validation_ids']==[]
   assert c['consistency_lambda']==(read(R/'protocol.json')['consistency_objective']['lambda_by_fold']['5'] if o=='consistency'else 0.)
 put(R/'control/refit_gate_verified.json',{'status':'PASS','groups':4,'epochs':1,'replay_lr_checked':True,'train_subjects':610,'source_sha256':sha(R/'worker.py'),'refit_source_sha256':sha(R/'refit_worker.py'),'time':time.time()})
 jobs=[]
 for a in ['MS','TS']:
  for o in ['base','consistency']:
   for s in [1,2]:
    name=f'{a}_{o}_f5_s{s}';cp=R/'refits'/name/'completion.json'
    if cp.exists() and read(cp)['status']=='COMPLETE':continue
    jobs.append((name,[PY,'refit_worker.py','--arm',a,'--objective',o,'--fold','5','--seed',str(s),'--mode','refits','--schedule','replay','--epochs',str(epochs[a+'_'+o])],cp))
 status('refits',parallelism=width,epochs=epochs);ps.manage(jobs,width,'refits')
 for a in ['MS','TS']:
  for o in ['base','consistency']:
   for s in [1,2]:
    d=R/'refits'/f'{a}_{o}_f5_s{s}';c=read(d/'completion.json');assert c['status']=='COMPLETE'and c['epochs']==epochs[a+'_'+o]and c['last_sha256']==sha(d/'last.pt')
 put(R/'control/refit_status.json',{'status':'COMPLETE','completed':8,'total':8,'time':time.time()})
 status('downstream_prepare')
 subprocess.run([PY,'prepare.py'],cwd=D,env=ps.env,check=True)
 # Classifier implementation is byte-identical to previous approved LR protocol.
 assert sha(D/'classifier.py')==sha(R.parent/'cpac610_residual_downstream_20261025/classifier.py')
 status('features_and_classification',downstream_root=str(D))
 subprocess.run([PY,'supervisor.py'],cwd=D,env=ps.env,check=True,timeout=max(1,48*3600-(time.time()-ps.LIMIT_START)))
 status('strict_verification');subprocess.run([PY,'verify.py'],cwd=D,env=ps.env,check=True)
 assert read(D/'results/verification.json')['status']=='PASS'
 put(R/'control/pipeline_status.json',{'status':'COMPUTATION_COMPLETE','upstream':40,'refits':8,'features':8,'classification':8,'folds':80,'time':time.time(),'next':'Analyze and produce Word/HTML and sync public/private; no new experiments'})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  for p in ps.OWN_CHILDREN:
   if p.poll()is None:os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGTERM)
  put(R/'control/pipeline_status.json',{'status':'NEEDS_ATTENTION','error':repr(e),'time':time.time()});raise
