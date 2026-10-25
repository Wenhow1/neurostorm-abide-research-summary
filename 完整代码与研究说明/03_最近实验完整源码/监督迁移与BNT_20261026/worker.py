import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,argparse,signal,math
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss
from thermal_helpers import ThermalGuard
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]));P=json.loads((R/'protocol.json').read_text());STOP=False

def sig(*x):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGUSR1,sig)
def read(p):return json.loads(Path(p).read_text())
def put(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def append(p,x):
 with Path(p).open('a')as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def statehash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def save(p,x):q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def seed_for(seed,fold,inner):return 2026102600+seed*10000+fold*100+(inner+1)*3
class Transfer(nn.Module):
 def __init__(self,group,seed,fold,inner):
  super().__init__();arm=group.split('_')[0];src=Path(P['model_root']);sys.path[:0]=[str(src),str(src/'source'),str(src/'source/adapters')];import base_models
  m=base_models.make(5,seed,arm);cp=src/'refits'/f'{arm}_full610_s{seed}'/('last.pt'if group.endswith('pretrained')else'initial.pt');assert sha(cp)==P['checkpoint_hashes'][str(cp)];ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck
  self.backbone=m;self.backbone.decoder=nn.Identity();self.backbone.mask_projection.requires_grad_(False)
  torch.manual_seed(seed_for(seed,fold,inner)+71);self.head=nn.Linear(116*288,1)
 def forward(self,x):return self.head(self.backbone.encode_full(x).flatten(1)).squeeze(-1)
class BNT(nn.Module):
 def __init__(self,seed,fold,inner):
  super().__init__();from types import SimpleNamespace as N
  from vendor_bnt.BNT.bnt import BrainNetworkTransformer
  torch.manual_seed(seed_for(seed,fold,inner));self.net=BrainNetworkTransformer(N(dataset=N(node_sz=116),model=N(pos_encoding='none',sizes=[116,100],pooling=[False,True],orthogonal=True,freeze_center=True,project_assignment=True)))
 def forward(self,x):z=self.net(None,x[:,0]);return z[:,1]-z[:,0]
def build(group,seed,fold,inner):
 torch.manual_seed(seed_for(seed,fold,inner));m=BNT(seed,fold,inner)if group=='BNT_random'else Transfer(group,seed,fold,inner)
 torch.manual_seed(seed_for(seed,fold,inner)+99);return m

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
 for k in ['enable_flash_sdp','enable_mem_efficient_sdp','enable_cudnn_sdp']:getattr(torch.backends.cuda,k)(False)
 torch.backends.cuda.enable_math_sdp(True)
 job=read(a.job);group=job['group'];sd=job['seed'];fold=job['fold'];inner=job['inner'];lr=job['lr'];n=a.epochs or job['epochs'];out=R/a.mode/(a.tag or job['name']);out.mkdir(parents=True,exist_ok=True)
 paths=P['data'];splits=read(paths['splits']);sp=splits[fold];tr=np.array(sp['train']if inner<0 else sp['inner'][inner]['train']);va=np.array([]if inner<0 else sp['inner'][inner]['validation'],dtype=int);te=np.array(sp['test']);assert not set(tr)&set(te)and not set(va)&set(te)and not set(tr)&set(va)
 for k,path in paths.items():assert sha(path)==P['data_hashes'][k],k
 raw=np.load(paths['fc'],mmap_mode='r');y=np.load(paths['labels']).astype(int);norm=P['normalization'];x=np.asarray((raw-norm['mean'])/norm['std'],dtype=np.float32);del raw
 guard=ThermalGuard(out/'thermal.json',out/'thermal_events.jsonl',{'job':job['name']},wait_limit=21600);guard.check(phase='before_cuda')
 m=build(group,sd,fold,inner);init=statehash(m.state_dict());headhash=statehash(m.head.state_dict())if hasattr(m,'head')else None;m=m.cuda();opt=torch.optim.AdamW([p for p in m.parameters()if p.requires_grad],lr=lr,weight_decay=.01,betas=(.9,.999),eps=1e-8);lossfn=nn.BCEWithLogitsLoss()
 cfg=dict(job=job,epochs=n,normalization=norm,train_count=len(tr),validation_count=len(va),outer_test_count=len(te),initial_model_sha256=init,initial_head_sha256=headhash,parameters=sum(p.numel()for p in m.parameters()),trainable_parameters=sum(p.numel()for p in m.parameters()if p.requires_grad),protocol_sha256=sha(R/'protocol.json'),training_input='complete FC no masking',test_evaluated_each_epoch=False)
 if(out/'config.json').exists():assert read(out/'config.json')==cfg
 else:put(out/'config.json',cfg)
 history=[];steps=0;start=1;best=-math.inf;elapsed=0
 if(out/'last.pt').exists()and not a.benchsteps:
  ck=torch.load(out/'last.pt',weights_only=False,map_location='cpu');assert ck['config']==cfg;m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);history=ck['history'];steps=ck['steps'];start=ck['epoch']+1;best=ck['best'];elapsed=ck['elapsed'];del ck
  (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in history))
  if(out/'gradients.jsonl').exists():(out/'gradients.jsonl').write_text(''.join(l+'\n'for l in(out/'gradients.jsonl').read_text().splitlines()if json.loads(l)['step']<=steps))
 def tensor(ix):return torch.from_numpy(np.ascontiguousarray(x[ix]))[:,None].cuda()
 @torch.no_grad()
 def evaluate(ix,return_predictions=False):
  m.eval();zs=[]
  for off in range(0,len(ix),8):guard.check(phase='eval');z=m(tensor(ix[off:off+8]));zs.extend(z.cpu().tolist())
  logits=np.asarray(zs);prob=1/(1+np.exp(-np.clip(logits,-700,700)));yy=y[ix];bce=float(np.mean(np.logaddexp(0,logits)-yy*logits))
  metrics={'BCE':bce,'AUROC':float(roc_auc_score(yy,logits)),'ACC':float(accuracy_score(yy,prob>=.5))}
  return (metrics,prob.tolist(),logits.tolist())if return_predictions else metrics
 def update(q,epoch):
  nonlocal steps
  guard.check(phase='train',epoch=epoch);m.train();opt.zero_grad(set_to_none=True);z=m(tensor(q));loss=lossfn(z,torch.from_numpy(y[q].astype('float32')).cuda());assert torch.isfinite(loss);loss.backward()
  gs={};pre=0.
  for name,p in m.named_parameters():
   if p.grad is not None:
    v=float(p.grad.detach().double().square().sum());pre+=v;k='head'if name.startswith('head.')or name.startswith('net.fc.')else'encoder';gs[k]=gs.get(k,0)+v
  gn=math.sqrt(pre);assert math.isfinite(gn);torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
  before=[(p,p.detach().clone())for p in m.parameters()if p.requires_grad]if steps%10==0 else[];opt.step();steps+=1
  record={'step':steps,'epoch':epoch,'loss':float(loss),'lr':lr,'all_preclip_l2':gn,'postclip_l2':min(gn,1.),'clipped':gn>1,**{k+'_gradient_l2':math.sqrt(v)for k,v in gs.items()}}
  if before:record['actual_parameter_update_l2']=math.sqrt(sum(float((p.detach()-b).double().square().sum())for p,b in before))
  return record
 beg=time.monotonic()
 if a.benchsteps:
  for step in range(a.benchsteps):update(tr[(np.arange(8)+step*8)%len(tr)],0)
  torch.cuda.synchronize();put(out/'benchmark.json',{'steps':a.benchsteps,'seconds':time.monotonic()-beg,'samples':a.benchsteps*8,'group':group});return
 for ep in range(start,n+1):
  if STOP:return
  order=np.random.default_rng(seed_for(sd,fold,inner)+ep*17).permutation(tr);rr=[];total=0.;seen=0;eb=time.monotonic()
  for off in range(0,len(order),8):
   if STOP:put(out/'paused.json',{'last_checkpoint':max([0]+[r['epoch']for r in history if r['epoch']%5==0]),'interrupted_epoch':ep});return
   q=order[off:off+8];r=update(q,ep);append(out/'gradients.jsonl',r);rr.append(r);total+=r['loss']*len(q);seen+=len(q)
  row={'epoch':ep,'steps':steps,'online_train_BCE':total/seen,'train_eval':evaluate(tr),'lr':lr,'gradient':{k:{'mean':float(np.mean([r[k]for r in rr])),'median':float(np.median([r[k]for r in rr])),'max':float(max(r[k]for r in rr)),'p05':float(np.quantile([r[k]for r in rr],.05)),'p95':float(np.quantile([r[k]for r in rr],.95))}for k in ['all_preclip_l2','encoder_gradient_l2','head_gradient_l2']},'clipping_fraction':float(np.mean([r['clipped']for r in rr])),'sampled_update_l2_mean':float(np.mean([r['actual_parameter_update_l2']for r in rr if 'actual_parameter_update_l2'in r])),'epoch_seconds':time.monotonic()-eb,'elapsed_seconds':elapsed+time.monotonic()-beg}
  if len(va):row['validation']=evaluate(va)
  history.append(row);append(out/'history.jsonl',row)
  if len(va)and row['validation']['AUROC']>best:best=row['validation']['AUROC'];save(out/'best.pt',{'model':m.state_dict(),'epoch':ep,'config':cfg})
  if ep%5==0 or ep==n or a.stop_after==ep:
   save(out/'last.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'steps':steps,'config':cfg,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best':best,'elapsed':row['elapsed_seconds']})
  put(out/'status.json',{'status':'RUNNING','epoch':ep,'epochs':n,'time':time.time()})
  if a.stop_after==ep:return
 if inner<0 and a.mode=='runs':
  metrics,prob,logits=evaluate(te,True);put(out/'outer_result.json',{'metrics':metrics,'indices':te.tolist(),'probabilities':prob,'logits':logits,'selected_epoch':n,'selected_lr':lr,'test_evaluated_after_training_only':True})
 put(out/'completion.json',{'status':'COMPLETE','epochs':n,'steps':steps,'last_sha256':sha(out/'last.pt'),'elapsed_seconds':elapsed+time.monotonic()-beg,'time':time.time()});put(out/'status.json',{'status':'COMPLETE','epoch':n,'time':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--mode',default='runs');p.add_argument('--tag');p.add_argument('--epochs',type=int);p.add_argument('--benchsteps',type=int,default=0);p.add_argument('--stop-after',type=int);run(p.parse_args())
