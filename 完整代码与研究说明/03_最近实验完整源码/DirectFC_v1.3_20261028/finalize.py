"""Gate B selection and Gate C freezing; test scoring is a separate command."""
import argparse,pickle
from common import *

def select_all():
 jobs=read(R/'jobs.json');by={};data=np.load(R/'data/development.npz');cv=read(R/'data/cv.json')
 gate=read(R/'control/gate_A.json')
 for name,h in gate['sources'].items():assert sha(R/name)==h,name
 for j in jobs:
  out=R/'runs'/j['name'];c=read(out/'completion.json');assert c['status']=='COMPLETE' and c['epochs']==100 and sha(out/'last.pt')==c['last_sha256'];cfg=read(out/'config.json');assert cfg['job']==j
  hist=[json.loads(l) for l in (out/'history.jsonl').read_text().splitlines()];assert [r['epoch'] for r in hist]==list(range(1,101));values=[]
  for ep,row in enumerate(hist,1):
   z=np.load(out/f'val_epoch_{ep:03d}.npz');ix=np.array(cv[j['fold']]['validation']);assert np.array_equal(z['ids'],data['ids'][ix]) and np.array_equal(z['labels'],data['y'][ix]);m=metrics(z['labels'],z['logits']);assert all(abs(m[k]-row['validation'][k])<1e-12 for k in ['AUROC','BCE','ACC']);values.append(m)
  key=(j['arm'],j['width'],j['condition']);by.setdefault(key,[]).append((j,values,cfg['parameters']))
 choices={};refits=[]
 for arm in ['Mamba','Transformer','GCN']:
  rows=[]
  for (ar,width,condition),runs in by.items():
   if ar!=arm or condition=='D':continue
   assert {(j['fold'],j['seed']) for j,v,p in runs}=={(f,s) for f in range(5) for s in [1,2]}
   for ep in range(1,101):rows.append({'arm':arm,'width':width,'lr':.0001,'condition':condition,'strategy_priority':{'A':0,'B':1,'C':2}[condition],'epoch':ep,'params':runs[0][2],'AUROC':float(np.mean([v[ep-1]['AUROC'] for j,v,p in runs])),'BCE':float(np.mean([v[ep-1]['BCE'] for j,v,p in runs]))})
  choices[arm]=select(rows)
  for sd in [1,2]:
   v=choices[arm];j={'arm':arm,'width':v['width'],'lr':v['lr'],'fold':-1,'seed':sd,'epochs':v['epoch'],'stage':'refit','condition':v['condition'],'gpu':sd-1,'name':f'{arm}_refit_s{sd}'};
   if v['condition']=='B':
    paths=[R/'runs'/r['name']/'history.jsonl' for r in jobs if r['arm']==arm and r['width']==v['width'] and r['condition']=='B'];traces=[[json.loads(t)['lr'] for t in path.read_text().splitlines()] for path in paths];assert len(traces)==10;j['lr_replay']=np.sort(np.array(traces),axis=0)[4,:v['epoch']].tolist()
   refits.append(j);put(R/'jobs'/f"{j['name']}.json",j)
 vals=[]
 for C in [.001,.01,.1,1.,10.,100.]:
  vv=[]
  for f in range(5):
   out=R/'runs'/f'LR_C{C:g}_f{f}';c=read(out/'completion.json');assert c['status']=='COMPLETE' and sha(out/'model.pkl')==c['model_sha256'];assert not read(out/'warnings.json');z=np.load(out/'validation.npz');vv.append(metrics(z['labels'],z['logits']))
  vals.append({'C':C,'AUROC':float(np.mean([v['AUROC'] for v in vv])),'BCE':float(np.mean([v['BCE'] for v in vv]))})
 choices['LR']=select(vals,True);j={'C':choices['LR']['C'],'fold':-1,'stage':'refit','name':'LR_refit'};refits.append(j);put(R/'jobs/LR_refit.json',j);put(R/'refits.json',refits);put(R/'results/selection.json',choices);put(R/'control/gate_B.json',{'status':'PASS','neural_development':200,'lr_development':30,'all_epoch_metrics_recomputed':True,'selection_sha256':sha(R/'results/selection.json'),'test_accessed':False})

def freeze():
 assert read(R/'control/gate_B.json')['status']=='PASS';frozen={}
 for j in read(R/'refits.json'):
  out=R/'runs'/j['name'];c=read(out/'completion.json');assert c['status']=='COMPLETE';p=out/('model.pkl' if j['name']=='LR_refit' else 'final.pt');assert p.exists();frozen[str(p.relative_to(R))]=sha(p)
 assert len(frozen)==7
 put(R/'control/gate_C.json',{'status':'PASS','models':frozen,'selection_sha256':sha(R/'results/selection.json'),'cv_sha256':sha(R/'data/cv.json'),'development_sha256':sha(R/'data/development.npz'),'source_hashes':{f.name:sha(f) for f in R.glob('*.py')},'threshold':.5,'time':time.time(),'test_scores_read':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['select','freeze']);a=p.parse_args();select_all() if a.action=='select' else freeze()
