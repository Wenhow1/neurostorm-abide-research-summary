from pathlib import Path
import argparse,json,subprocess,time,sys
from neurostorm_2d.overnight import parallel as m
from neurostorm_2d.overnight.supervisor import validate_manifest
from summarize_abc import summarize
R=m.ROOT;C=m.CONTROL;normal=m.command

def validate():
 p=m.read(R/'protocol.json');assert p['max_scientific_fits']==10 and p['allowed_gpu_ids']==[1] and p['primary_epoch']==100 and p['aux_weight']==.05
 assert m.allowed_slots(time.time(),m.cutoff())=={0:0,1:5}
 a=validate_manifest(m.read(R/'formal_manifest.json'),R);assert len(a['jobs'])==10
 for arm in ['B','C']:
  for f in range(5):
   j=next(x for x in a['jobs'] if x['id']==f'{arm}_f{f}_s1');assert j['epochs']==100 and j['parameters']['lr']==.0001 and j['parameters']['seed']==1 and f'/f{f}/' in j['parameters']['split_dir']
 g=m.read(R/'release_gate.json');assert g['status']=='PASS' and g['protocol_sha256']==m.sha(R/'protocol.json')
 for k,v in g['frozen_files'].items():assert m.sha(R/k)==v,k
 return a

def run_eval(kind,deadline):
 jobs=[]
 for f in range(5):
  cp=Path(m.read(R/'baseline_paths.json')[str(f)])/'epoch_100.pt' if kind=='A' else R/f'runs/{kind}_f{f}_s1/epoch_100.pt'
  j=dict(id=f'eval_{kind}_f{f}',kind=kind,fold=f,checkpoint=str(cp),output_dir=str(R/f'runs/eval_{kind}_f{f}'));m.put(R/f'{j["id"]}.json',j);jobs.append(j)
 m.command=lambda j,resume:[str(m.PY),str(R/'evaluate_abc.py'),str(R/f'{j["id"]}.json')]
 try:m.run_jobs(jobs,'evaluation_'+kind,deadline)
 finally:m.command=normal

def worker():
 a=validate();deadline=time.time()+240*60
 subprocess.run(['tmux','new-session','-d','-s','cpac966-abc-resources','-c',str(R),f'{m.PY} resource_monitor.py > control/resources.log 2>&1'],check=True)
 try:
  run_eval('A',deadline);summarize()
  e=validate_manifest(m.read(R/'engineering_manifest.json'),R);m.run_jobs(e['jobs'],'engineering_BC_two_epochs',deadline)
  for j in e['jobs']:
   c=m.read(Path(j['output_dir'])/'config.json');base=m.read(Path(m.read(R/'baseline_paths.json')['0'])/'config.json');assert c['initial_parameters_sha256']==base['initial_parameters_sha256'];g=m.read(Path(j['output_dir'])/'auxiliary_gradient_check.json');assert g['status']=='PASS'
   bpath=Path(m.read(R/'baseline_paths.json')['0']);first=json.loads((bpath/'gradients.jsonl').open().readline());assert abs(g['main_gradient_norm']-first['preclip_gradient_l2'])<max(1e-5,abs(first['preclip_gradient_l2'])*1e-4), 'A first-step main gradient mismatch'
  m.put(R/'engineering_pass.json',dict(status='PASS',engineering_fits=2,epochs=2,training_only_checks=True,formal_starts_from_scratch=True,GPU1_only=True))
  m.run_jobs(a['jobs'],'formal_BC_ten_fits',deadline)
  for j in a['jobs']:
   c=m.read(Path(j['output_dir'])/'config.json');base=m.read(Path(m.read(R/'baseline_paths.json')[str(c['fold'])])/'config.json');assert c['initial_parameters_sha256']==base['initial_parameters_sha256'];assert m.read(Path(j['output_dir'])/'completion.json')['optimizer_steps']==6100
  run_eval('B',deadline);run_eval('C',deadline);s=summarize(True)
  m.put(R/'night_status.json',dict(state='COMPLETED',scientific_fits=10,primary_epoch=100,B_screen_pass=s['B']['full_screen_pass'],C_screen_pass=s['C']['full_screen_pass'],additional_seeds_started=False,time_unix=time.time()))
 except BaseException as exc:
  m.put(R/'night_status.json',dict(state='STOPPED_NEEDS_REVIEW',error=repr(exc),time_unix=time.time()));raise
 finally:(C/'RESOURCE_MONITOR_DONE').touch();(C/'FINISHED').touch()
def launch():
 validate();C.mkdir(exist_ok=True)
 if 1 in m.external_busy(m.gpu_info(),{}):raise RuntimeError('GPU1 busy before launch')
 with (C/'launch.lock').open('x') as h:h.write(str(time.time()))
 subprocess.run(['tmux','new-session','-d','-s','cpac966-abc10','-c',str(R),f'{m.PY} -u abc_queue.py worker > control/worker.log 2>&1'],check=True)
 m.put(C/'launched.json',dict(session='cpac966-abc10',allowed_gpu_ids=[1],scientific_fits=10,time_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','launch','worker','status']);a=p.parse_args()
 if a.action=='validate':validate();print('PASS GPU1-only BC10')
 elif a.action=='launch':launch()
 elif a.action=='worker':worker()
 else:print(json.dumps(m.read(R/'night_status.json'),indent=2))
