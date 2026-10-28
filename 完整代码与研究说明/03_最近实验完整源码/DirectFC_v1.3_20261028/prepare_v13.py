"""Seal v1.3 exact manifest, reuse only audited data split, no training reuse."""
from common import *
import subprocess

def main():
 old=R.parent/'directfc_v12_20261028';pre=read(R/'control/prepared.json');assert sha(R/'data/development.npz')==pre['development_hash'] and sha(R/'data/cv.json')==pre['cv_hash'];assert sha(R/'data/development.npz')==sha(old/'data/development.npz')
 raw=read(R/'planning_manifest.json')['jobs'];jobs=[];lr=[]
 for r in raw:
  if r['phase']!='development':continue
  if r['stage']=='LR':j={'name':r['id'],'C':r['C'],'fold':r['fold'],'stage':'development'};lr.append(j)
  else:j={'name':r['id'],'condition':r['stage'],'arm':r['arm'],'width':r['width'],'fold':r['fold'],'seed':r['seed'],'epochs':100,'lr':.0001,'stage':'development','gpu':r['seed']-1};jobs.append(j)
  put(R/'jobs'/f"{j['name']}.json",j)
 assert len(jobs)==200 and len(lr)==30;put(R/'jobs.json',jobs);put(R/'lr_jobs.json',lr)
 put(R/'control/manifest_validation.json',{'status':'PASS','neural_development':200,'LR_development':30,'refits':7,'total':237,'stages':{s:sum(j['condition']==s for j in jobs) for s in ['A','B','C','D']},'holdout_not_copied_or_loaded':True})
 r=subprocess.run(['/home/wenhao/neurostorm_2d/.venv/bin/python','-m','neurostorm_2d.scripts.night_preflight','--project-dir','/home/wenhao/neurostorm_2d'],cwd='/home/wenhao/neurostorm_2d',capture_output=True,text=True,check=True);p=json.loads(r.stdout);assert p['verdict']!='block';put(R/'control/current_preflight.json',p)
 print('PREPARED237')
if __name__=='__main__':main()
