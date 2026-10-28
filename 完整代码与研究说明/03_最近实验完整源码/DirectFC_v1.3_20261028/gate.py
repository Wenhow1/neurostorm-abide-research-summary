"""Real production-function and CUDA gate. No test-set reads."""
import subprocess,sys,platform,inspect,importlib.metadata,traceback
from common import *
from models import *

def call(j,tag,stop=None,bench=0):
 jpath=R/'pilot_jobs'/f'{tag}.json';put(jpath,j);out=R/'pilot'/tag;out.mkdir(parents=True,exist_ok=True)
 args=[sys.executable,str(R/'worker.py'),'--job',str(jpath),'--out',str(out)]
 if stop:args+=['--stop-after',str(stop)]
 if bench:args+=['--benchsteps',str(bench)]
 with (R/'logs'/f'{tag}.log').open('a') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(tag,r.returncode)
 return out

def test_production():
 checks={};f=np.full((116,116),.5);np.fill_diagonal(f,0);a,d=graph(f);checks['graph_667']=d['edges']==667;checks['graph_symmetry']=np.allclose(a,a.T);checks['negative_identity']=np.array_equal(graph(-f)[0],np.eye(116))
 one=np.zeros_like(f);one[0,1]=one[1,0]=.2;n,dd=graph(one);checks['single_edge_analytic']=np.allclose(n[:2,:2],np.array([[1,.2],[.2,1]])/1.2) and dd['edges']==1
 # Lexicographic ties and no repeated normalization are evaluated against actual graph implementation.
 ii,jj=np.triu_indices(116,1);aa=np.zeros_like(f);aa[ii[:667],jj[:667]]=.5;aa+=aa.T;bb=aa+np.eye(116);deg=bb.sum(1);checks['tie_order_exact']=np.allclose(a,bb/np.sqrt(deg[:,None]*deg[None,:]))
 checks['weighted_online_tail_batch']=online_bce([{'loss':1.,'n':8},{'loss':3.,'n':2}])==1.4
 checks['whole_collection_AUC']=metrics([0,1,0,1],[-2,1,-1,2])['AUROC']==1.
 try:fit_scale([np.zeros_like(f)]);checks['zero_std_block']=False
 except AssertionError:checks['zero_std_block']=True
 rows=[{'AUROC':.7,'BCE':.6,'epoch':4,'params':100,'lr':.001,'strategy_priority':2},{'AUROC':.7,'BCE':.6,'epoch':4,'params':100,'lr':.0001,'strategy_priority':0}];checks['production_selector_tie']=select(rows)['lr']==.0001
 sandbox=R/'pilot/log_recovery_test';sandbox.mkdir(parents=True,exist_ok=True);(sandbox/'history.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n');(sandbox/'gradients.jsonl').write_text('{"step":1}\n{"step":2}\n');restore_logs(sandbox,[{'epoch':1}],1);checks['production_log_tail_isolated']=len((sandbox/'history.jsonl').read_text().splitlines())==1 and len(list(sandbox.glob('*.incomplete_*')))>=2
 assert all(checks.values()),checks;put(R/'control/production_rules.json',{'status':'PASS','checks':checks})

def compare(a,b):
 if isinstance(a,torch.Tensor):return a.shape==b.shape and torch.allclose(a.cpu(),b.cpu(),atol=1e-5,rtol=1e-4)
 if isinstance(a,dict):return a.keys()==b.keys() and all(compare(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
 return a==b

def main():
 setup();test_production();import mamba_ssm
 versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scikit-learn','mamba-ssm']};versions['optional_causal_conv1d']='absent: native torch Conv1d plus selective_scan CUDA path';versions['mamba_file']=inspect.getfile(mamba_ssm.Mamba);versions['mamba_source_sha256']=sha(versions['mamba_file']);put(R/'control/versions.json',versions)
 structure={}
 for arm in ['Mamba','Transformer','GCN']:
  for width in [64,128]:
   j={'arm':arm,'width':width,'seed':1,'fold':0,'lr':.0001,'epochs':3,'stage':'pilot','name':'pilot'};m=build(j);g,records=groups(m);structure[f'{arm}_{width}']={'parameters':sum(p.numel() for p in m.parameters()),'groups':records,'initial_hash':statehash(m.state_dict())};del m
 put(R/'control/structures.json',structure)
 results=[]
 for arm in ['Mamba','Transformer','GCN']:
  for width in [64,128]:
   outs=[]
   for lr in [.0001,.001]:
    j={'arm':arm,'width':width,'seed':1,'fold':0,'lr':lr,'epochs':3,'stage':'pilot','name':f'{arm}_{width}_{lr}'};tag=j['name'];out=call(j,tag);assert read(out/'completion.json')['epochs']==3;outs.append(out);results.append({'condition':tag,'status':'PASS','last_hash':sha(out/'last.pt')})
   assert read(outs[0]/'config.json')['initial_hash']==read(outs[1]/'config.json')['initial_hash']
   if width==128:
    j={'arm':arm,'width':width,'seed':1,'fold':0,'lr':.001,'epochs':3,'stage':'pilot','name':f'{arm}_{width}_{.001}'};out=call(j,f'{arm}_restore',2);call(j,f'{arm}_restore');a=torch.load(outs[1]/'last.pt',weights_only=False,map_location='cpu');b=torch.load(out/'last.pt',weights_only=False,map_location='cpu')
    assert compare(a['model'],b['model']) and compare(a['optimizer'],b['optimizer']) and a['steps']==b['steps'] and a['best']['epoch']==b['best']['epoch']
    za=np.load(outs[1]/'val_epoch_003.npz')['logits'];zb=np.load(out/'val_epoch_003.npz')['logits'];assert np.allclose(za,zb,atol=1e-5,rtol=1e-4)
    put(R/'control'/f'resume_{arm}.json',{'status':'PASS','logit_maxdiff':float(abs(za-zb).max()),'model_optimizer_allclose':True,'steps':a['steps'],'duplicate_epochs':False})
 put(R/'control/short_runs.json',{'status':'PASS','conditions':results,'used_for_recipe_selection':False})
 # Same12 tasks,40 updates each; max parallel1/2/4/6 only resource tuning.
 benchmarks=[]
 for par in [1,2,4,6]:
  jobs=[]
  for arm in ['Mamba','Transformer','GCN']:
   for width in [64,128]:
    for sd in [1,2]:jobs.append({'arm':arm,'width':width,'seed':sd,'fold':0,'lr':.001,'epochs':3,'stage':'benchmark','name':f'{arm}_{width}_{sd}'})
  active=[];pending=list(jobs);beg=time.time();peak=0
  while pending or active:
   gg=gpu();peak=max(peak,gg['temperature'])
   while pending and len(active)<par and gg['temperature']<80 and gg['used_MiB']/gg['total_MiB']<.9:
    j=pending.pop(0);tag=f'bench{par}_{j["name"]}';jp=R/'pilot_jobs'/f'{tag}.json';put(jp,j);o=R/'pilot'/tag;o.mkdir(parents=True,exist_ok=True);f=(R/'logs'/f'{tag}.log').open('w');p=subprocess.Popen([sys.executable,str(R/'worker.py'),'--job',str(jp),'--out',str(o),'--benchsteps','40'],stdout=f,stderr=subprocess.STDOUT);active.append((p,f,tag))
   for item in list(active):
    p,f,tag=item
    if p.poll() is not None:assert p.returncode==0,tag;f.close();active.remove(item)
   time.sleep(1)
  benchmarks.append({'parallelism':par,'wall_seconds':time.time()-beg,'peak_C':peak,'fixed_steps':480})
 selected=min([b for b in benchmarks if b['peak_C']<80],key=lambda x:x['wall_seconds']);put(R/'control/benchmark.json',{'status':'PASS','measurements':benchmarks,'selected_parallelism':selected['parallelism']})
 hashes={f.name:sha(f) for f in R.glob('*.py')};hashes['protocol.json']=sha(R/'protocol.json');put(R/'control/gate_A.json',{'status':'PASS','short_conditions':12,'resume_architectures':3,'production_rules':True,'data_status':read(R/'control/data_audit.json')['status'],'sources':hashes,'parallelism':selected['parallelism'],'time':time.time()})
 print('GATE_A_PASS')
if __name__=='__main__':
 try:main()
 except Exception as e:put(R/'control/gate_error.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise
