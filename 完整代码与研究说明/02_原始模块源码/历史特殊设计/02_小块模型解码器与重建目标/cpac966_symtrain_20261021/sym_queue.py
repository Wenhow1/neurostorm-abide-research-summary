from pathlib import Path
import argparse,json,subprocess,time,csv,copy,sys,os
import numpy as np
from neurostorm_2d.overnight import parallel as m
from neurostorm_2d.overnight.supervisor import validate_manifest
R=m.ROOT;C=m.CONTROL;normal_command=m.command

def validate():
 p=m.read(R/'protocol.json');assert p['allowed_gpu_ids']==[1] and p['max_scientific_fits']==5 and p['primary_epoch']==100
 assert m.allowed_slots(time.time(),m.cutoff())=={0:0,1:5}
 a=validate_manifest(m.read(R/'formal_manifest.json'),R);assert len(a['jobs'])==5
 for f,j in enumerate(a['jobs']):assert j['epochs']==100 and j['parameters']['lr']==.0001 and j['parameters']['seed']==1 and f'/f{f}/' in j['parameters']['split_dir']
 g=m.read(R/'release_gate.json');assert g['status']=='PASS' and g['protocol_sha256']==m.sha(R/'protocol.json')
 for k,v in g['frozen_files'].items():assert m.sha(R/k)==v,k
 return a

def eval_jobs(kind):
 jobs=[]
 for f in range(5):
  if kind=='baseline':path=Path(m.read(R/'baseline_paths.json')[str(f)])/'epoch_100.pt'
  else:path=R/f'runs/sym_f{f}_s1/epoch_100.pt'
  out=R/f'runs/eval_{kind}_f{f}';j={'id':f'eval_{kind}_f{f}','kind':kind,'fold':f,'checkpoint':str(path),'output_dir':str(out)}
  m.put(R/f'{j["id"]}.json',j);jobs.append(j)
 return jobs

def run_eval(kind,deadline):
 m.command=lambda j,resume:[str(m.PY),str(R/'evaluate_sym.py'),str(R/f'{j["id"]}.json')]
 try:m.run_jobs(eval_jobs(kind),'evaluate_'+kind,deadline)
 finally:m.command=normal_command

def load_people(kind,arm):
 people=[]
 for f in range(5):
  with (R/f'runs/eval_{kind}_f{f}/subject_view_metrics.csv').open() as h:rows=[v for v in csv.DictReader(h) if v['arm']==arm]
  for sid in sorted({v['subject_id'] for v in rows}):
   v=[z for z in rows if z['subject_id']==sid];assert len(v)==3
   people.append({'subject_id':int(sid),'fold':f,**{k:float(np.mean([float(z[k]) for z in v])) for k in ['MSE','residual_correlation','amplitude_ratio','abs_log_amplitude_ratio','antisymmetric_MSE']}})
 return people

def summarize(final=False):
 raw=load_people('baseline','raw');base=load_people('baseline','sym');assert len(base)==610
 reduction=1-np.mean([x['MSE'] for x in base])/np.mean([x['MSE'] for x in raw])
 result={'baseline_raw_MSE':float(np.mean([x['MSE'] for x in raw])),'baseline_sym_MSE':float(np.mean([x['MSE'] for x in base])),'baseline_relative_MSE_reduction':float(reduction),'subjects':610,'views':3,'heldout_evaluated':False,'downstream_evaluated':False}
 m.csvout(R/'baseline_sym_subject_metrics.csv',base)
 if final:
  candidate=load_people('candidate','sym');m.csvout(R/'candidate_sym_subject_metrics.csv',candidate)
  cmean=float(np.mean([x['MSE'] for x in candidate]));gain=1-cmean/result['baseline_sym_MSE'];folds=[]
  for f in range(5):
   b=float(np.mean([x['MSE'] for x in base if x['fold']==f]));v=float(np.mean([x['MSE'] for x in candidate if x['fold']==f]));folds.append({'fold':f,'baseline_sym_MSE':b,'candidate_sym_MSE':v,'relative_improvement':1-v/b})
  corr_drop=float(np.median([x['residual_correlation'] for x in base])-np.median([x['residual_correlation'] for x in candidate]));amp_worse=float(np.median([x['abs_log_amplitude_ratio'] for x in candidate])-np.median([x['abs_log_amplitude_ratio'] for x in base]))
  passed=bool(gain>=.01 and sum(x['relative_improvement']>0 for x in folds)>=4 and corr_drop<=.01 and amp_worse<=.05)
  result.update(candidate_MSE=cmean,relative_improvement_vs_sym_baseline=gain,folds=folds,residual_correlation_median_drop=corr_drop,amplitude_log_error_median_increase=amp_worse,screen_pass=passed,additional_seeds_started=False,interpretation='development screen; site and subject-tail review required before further training')
  m.csvout(R/'paired_fivefold_results.csv',folds)
 m.put(R/('final_comparison.json' if final else 'baseline_diagnostics.json'),result)
 return result

def worker():
 a=validate();deadline=time.time()+240*60
 subprocess.run(['tmux','new-session','-d','-s','cpac966-sym-resources','-c',str(R),f'{m.PY} resource_monitor.py > control/resources.log 2>&1'],check=True)
 try:
  run_eval('baseline',deadline);b=summarize()
  if b['baseline_relative_MSE_reduction']<.01:
   m.put(R/'night_status.json',{'state':'STOPPED_DIAGNOSTIC_SMALL_EFFECT','baseline':b,'new_formal_fits':0});return
  e=validate_manifest(m.read(R/'engineering_manifest.json'),R)
  m.run_jobs(e['jobs'],'engineering_five_parallel_two_epochs',deadline)
  for f,j in enumerate(e['jobs']):
   c=m.read(Path(j['output_dir'])/'config.json');o=m.read(Path(m.read(R/'baseline_paths.json')[str(f)])/'config.json');assert c['initial_parameters_sha256']==o['initial_parameters_sha256']
  # Resource test passed before launching scientific fits. Engineering outputs are never reused as training starts.
  m.put(R/'engineering_pass.json',{'status':'PASS','fits':5,'epochs':2,'same_initialization_as_original':True,'formal_starts_from_scratch':True,'GPU1_only':True,'time_unix':time.time()})
  m.run_jobs(a['jobs'],'formal_symmetry_fivefold',deadline)
  run_eval('candidate',deadline);result=summarize(True)
  m.put(R/'night_status.json',{'state':'COMPLETED','formal_fits':5,'primary_epoch':100,'screen_pass':result['screen_pass'],'time_unix':time.time()})
 except BaseException as exc:
  m.put(R/'night_status.json',{'state':'STOPPED_NEEDS_REVIEW','error':repr(exc),'time_unix':time.time()});raise
 finally:(C/'RESOURCE_MONITOR_DONE').touch();(C/'FINISHED').touch()

def launch():
 validate();C.mkdir(exist_ok=True)
 with (C/'launch.lock').open('x') as f:f.write(str(time.time()))
 subprocess.run(['tmux','new-session','-d','-s','cpac966-symtrain5','-c',str(R),f'{m.PY} -u sym_queue.py worker > control/worker.log 2>&1'],check=True)
 m.put(C/'launched.json',{'session':'cpac966-symtrain5','allowed_gpu_ids':[1],'formal_fits':5,'time_unix':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','launch','worker','status']);a=p.parse_args()
 if a.action=='validate':validate();print('PASS GPU1-only fivefold symmetric-output plan')
 elif a.action=='launch':launch()
 elif a.action=='worker':worker()
 else:print(json.dumps(m.read(R/'night_status.json'),indent=2))
