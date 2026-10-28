import sys,subprocess,shutil,traceback,importlib.metadata,inspect
from common import *
from models import *
from scheduler import Plateau
from gate import compare,test_production

def call(j,tag,gpu_id=0,capture=False):
 out=R/'acceptance'/tag;out.mkdir(parents=True,exist_ok=True);jp=R/'pilot_jobs'/f'{tag}.json';put(jp,j)
 args=[sys.executable,str(R/'worker.py'),'--job',str(jp),'--out',str(out)]
 if capture:args+=['--capture-after','2']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu_id),DIRECTFC_GPU=str(gpu_id),DIRECTFC_SCAN_REF='1')
 with (R/'logs'/f'{tag}.log').open('a') as f:r=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(tag,r.returncode)
 return out

def main():
 setup();test_production();p=Plateau();used=[]
 for i in range(50):used.append(p.lr);p.step(1.)
 assert used[6]==5e-5 and used[12]==2.5e-5 and min(used)==1e-6
 p=Plateau()
 for i in range(6):p.step(1.)
 q=Plateau();q.load_state_dict(p.state_dict())
 for v in [1,.8,.8,.8,.8,.8,.8]:assert p.step(v)==q.step(v)
 put(R/'control/scheduler_production.json',{'status':'PASS','constant_sequence_first_lr_epochs':[7,13],'minimum':1e-6,'restore_exact':True})
 # Analytical scan backend comparison on synthetic tensors: no clinical metric used.
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref,selective_scan_fn
 torch.manual_seed(42);shapes=[(2,128,116),(2,256,116)];scanchecks=[]
 for b,d,l in shapes:
  vals=[torch.randn(b,d,l,device='cuda')*.1,torch.rand(b,d,l,device='cuda')*.05,-torch.rand(d,16,device='cuda'),torch.randn(b,16,l,device='cuda')*.1,torch.randn(b,16,l,device='cuda')*.1,torch.randn(d,device='cuda')*.1]
  aa=[x.detach().clone().requires_grad_() for x in vals];bb=[x.detach().clone().requires_grad_() for x in vals];x=selective_scan_fn(*aa,delta_softplus=True);y=selective_scan_ref(*bb,delta_softplus=True);assert torch.allclose(x,y,atol=1e-5,rtol=1e-4)
  x.square().mean().backward();y.square().mean().backward();assert all(torch.allclose(a.grad,b.grad,atol=1e-5,rtol=1e-4) for a,b in zip(aa,bb));scanchecks.append({'shape':[b,d,l],'max_forward_diff':float((x-y).abs().max()),'gradient_allclose':True})
 import mamba_ssm
 put(R/'control/backend_equivalence.json',{'status':'PASS','checks':scanchecks,'backend':'official Mamba1 selective_scan_ref running FP32 on CUDA; unchanged architecture','source':inspect.getfile(selective_scan_ref),'source_sha256':sha(inspect.getfile(selective_scan_ref)),'old_blocker':'native selective_scan_cuda output resume failed; preserved in v1.2','strict_same_prefix_reference_resume':read(R/'control/identical_prefix_resume.json')})
 structures={}
 for arm in ['Mamba','Transformer','GCN']:
  for width in [64,128]:
   j={'arm':arm,'width':width,'seed':1,'fold':0,'condition':'A'};m=build(j);gg,rr=groups(m);structures[f'{arm}_{width}']={'parameters':sum(p.numel() for p in m.parameters()),'groups':rr};del m
 put(R/'control/structures_v13.json',structures)
 done=[];rest=[]
 for cond in ['A','B','C','D']:
  for arm in (['GCN'] if cond=='D' else ['Mamba','Transformer','GCN']):
   for width in [64,128]:
    j={'arm':arm,'width':width,'fold':0,'seed':1,'condition':cond,'lr':.0001,'epochs':3,'stage':'pilot','name':f'{cond}_{arm}_{width}'};tag=j['name'];out=call(j,tag,capture=width==128);cfg=read(out/'config.json');assert read(out/'completion.json')['epochs']==3
    if cond!='A':
     base=R/'acceptance'/f'A_{arm}_{width}';assert cfg['initial_hash']==read(base/'config.json')['initial_hash'];h=[json.loads(v) for v in (out/'history.jsonl').read_text().splitlines()];hb=[json.loads(v) for v in (base/'history.jsonl').read_text().splitlines()];assert [v['sample_order_sha256'] for v in h]==[v['sample_order_sha256'] for v in hb]
    done.append({'condition':tag,'status':'PASS','parameters':cfg['parameters'],'last_hash':sha(out/'last.pt')});put(R/'control/acceptance_status.json',{'status':'RUNNING','short_done':len(done),'short_total':20,'resume_done':len(rest),'time':time.time()})
    if width==128:
     target=R/'acceptance'/(tag+'_resume');target.mkdir(exist_ok=True)
     for n in ['last.pt','config.json']:shutil.copy2(out/'fork_checkpoint'/n,target/n)
     call(j,tag+'_resume');a=torch.load(out/'last.pt',map_location='cpu',weights_only=False);b=torch.load(target/'last.pt',map_location='cpu',weights_only=False);za=np.load(out/'val_epoch_003.npz')['logits'];zb=np.load(target/'val_epoch_003.npz')['logits'];assert compare(a['model'],b['model']) and compare(a['optimizer'],b['optimizer']) and compare(a['scheduler'],b['scheduler']) and torch.equal(a['dropout_rng'],b['dropout_rng']);assert np.allclose(za,zb,atol=1e-5,rtol=1e-4);assert a['steps']==b['steps'] and a['best']['epoch']==b['best']['epoch'];rest.append({'condition':tag,'status':'PASS','logit_maxdiff':float(abs(za-zb).max())})
 put(R/'control/short_runs_v13.json',{'status':'PASS','conditions':done,'count':20,'used_to_select_recipe':False});put(R/'control/resume_v13.json',{'status':'PASS','cases':rest,'count':10})
 # Verify unchanged full model output/gradient on GPU1 too, with same production evaluator.
 for arm in ['Mamba','Transformer','GCN']:
  j={'arm':arm,'width':128,'fold':0,'seed':1,'condition':'C','lr':.0001,'epochs':3,'stage':'pilot','name':f'C_{arm}_128'};out=call(j,'GPU1_'+arm,1);a=np.load(R/'acceptance'/f'C_{arm}_128/val_epoch_003.npz')['logits'];b=np.load(out/'val_epoch_003.npz')['logits'];assert np.allclose(a,b,atol=1e-5,rtol=1e-4),('cross_GPU',arm)
 put(R/'control/cross_gpu.json',{'status':'PASS','models':3,'gpu0_gpu1_outputs_allclose':True})
 # Fixed workload eight jobs at each of2 and4 simultaneous per card. Retain existing GPU1 task.
 measurements=[]
 for parallel in [2,4]:
  specs=[]
  for gid in [0,1]:
   for i in range(4):
    arm=['Mamba','Transformer','GCN','Mamba'][i];j={'arm':arm,'width':128,'fold':0,'seed':1,'condition':'C','lr':.0001,'epochs':3,'stage':'benchmark','name':f'bench_p{parallel}_g{gid}_{i}'};specs.append((gid,j))
  active=[];beg=time.time()
  while specs or active:
   for gid in [0,1]:
    while sum(v[2]==gid for v in active)<parallel and any(v[0]==gid for v in specs):
     idx=next(i for i,v in enumerate(specs) if v[0]==gid);_,j=specs.pop(idx);o=R/'acceptance'/j['name'];o.mkdir(exist_ok=True);jp=R/'pilot_jobs'/f"{j['name']}.json";put(jp,j);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gid),DIRECTFC_GPU=str(gid),DIRECTFC_SCAN_REF='1');f=(R/'logs'/f"{j['name']}.log").open('w');p=subprocess.Popen([sys.executable,str(R/'worker.py'),'--job',str(jp),'--out',str(o),'--benchsteps','40'],env=env,stdout=f,stderr=subprocess.STDOUT);active.append((p,f,gid,j))
   for item in list(active):
    p,f,gid,j=item
    if p.poll() is not None:assert p.returncode==0,j['name'];f.close();active.remove(item)
   time.sleep(1)
  measurements.append({'per_card_parallelism':parallel,'wall_seconds':time.time()-beg,'total_steps':320,'gpus':[0,1]})
 put(R/'control/dual_gpu_benchmark.json',{'status':'PASS','measurements':measurements,'planned_max_parallel_per_card':4})
 # Count every production source and config only after all uploads complete.
 assert (R/'control/source_ready.json').exists();expected=read(R/'control/source_ready.json')
 for n,h in expected.items():assert sha(R/n)==h,n
 put(R/'control/gate_A.json',{'status':'PASS','short_conditions':20,'resume_cases':10,'cross_gpu_models':3,'sources':expected,'parallel_per_gpu':4,'data_audit':'PASS_WITH_DISCLOSED_LIMITATIONS','time':time.time(),'test_released':False})
 put(R/'control/acceptance_status.json',{'status':'PASS','short_done':20,'resume_done':10,'time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:put(R/'control/gate_v13_error.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise
