import argparse,signal,sys,traceback
from common import *
from models import *
from scheduler import Plateau
STOP=False
def stop(*a):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)

def run(j,out,stop_after=None,benchsteps=0,capture_after=None):
 setup();out.mkdir(exist_ok=True,parents=True);formal=j['stage'] in ['development','refit'];guard=Guard(out,formal)
 if formal:
  gate=read(R/'control/gate_A.json');assert gate['status']=='PASS'
  for name,h in gate['sources'].items():assert sha(R/name)==h,name
 prep=read(R/'control/prepared.json');assert sha(R/'data/development.npz')==prep['development_hash'] and sha(R/'data/cv.json')==prep['cv_hash']
 data=np.load(R/'data/development.npz');raw=data['fc'];y=data['y'];ids=data['ids'];cv=read(R/'data/cv.json')
 tr=np.array(cv[j['fold']]['train'] if j['fold']>=0 else np.arange(len(y)));va=np.array(cv[j['fold']]['validation'] if j['fold']>=0 else [],dtype=int)
 s=fit_scale(raw[tr]);x=(raw/s).astype('float32');a=np.stack([graph(f)[0] for f in raw]) if j['arm']=='GCN' else None
 if j.get('condition')=='D':a=np.repeat(np.eye(116,dtype='float32')[None],len(raw),axis=0)
 guard.check();m=build(j);optgroups,gr=groups(m);initial=statehash(m.state_dict());cfg={'job':j,'initial_hash':initial,'scale':s,'scale_fit_ids_hash':hashlib.sha256(ids[tr].tobytes()).hexdigest(),'parameters':sum(p.numel() for p in m.parameters()),'optimizer_groups':gr,'training_n':len(tr),'validation_n':len(va),'data_sha256':prep['development_hash'],'cv_sha256':prep['cv_hash']}
 if (out/'config.json').exists():assert read(out/'config.json')==cfg
 else:put(out/'config.json',cfg);save(out/'initial.pt',{'model':m.state_dict()})
 m=m.cuda();m.dropgen=torch.Generator(device='cuda').manual_seed(seed(j)+900000);sched=Plateau(j['lr']);opt=torch.optim.AdamW(optgroups,lr=j['lr'],betas=(.9,.999),eps=1e-8);history=[];steps=0;start=1;best=None
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg;m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);history=ck['history'];steps=ck['steps'];start=ck['epoch']+1;best=ck['best'];sched.load_state_dict(ck['scheduler']);m.dropgen.set_state(ck['dropout_rng']);restore_logs(out,history,steps)
 def tensors(ix):return torch.from_numpy(np.ascontiguousarray(x[ix])).cuda(),None if a is None else torch.from_numpy(np.ascontiguousarray(a[ix])).cuda()
 @torch.no_grad()
 def evaluate(ix):
  m.eval();zs=[]
  for off in range(0,len(ix),8):guard.check();xx,aa=tensors(ix[off:off+8]);zs.extend(m(xx,aa).cpu().double().tolist())
  z=np.array(zs,dtype=np.float64);return metrics(y[ix],z),z
 def norm(ps,grad=False):return math.sqrt(sum(float((p.grad if grad else p).detach().double().square().sum()) for p in ps if not grad or p.grad is not None))
 def update(q,ep):
  nonlocal steps
  g=guard.check();m.train();opt.zero_grad(set_to_none=True);xx,aa=tensors(q);z=m(xx,aa);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.from_numpy(y[q].astype('float32')).cuda());assert torch.isfinite(loss);loss.backward()
  names={k:[] for k in ['input','block0','block1','norm','head']}
  for name,p in m.named_parameters():key='head' if name.startswith('head.') else 'input' if name.startswith('input.') else 'norm' if name.startswith('norm.') else 'block0' if name.startswith(('blocks.0.','graph1.')) else 'block1';names[key].append(p)
  pre=norm(m.parameters(),True);assert math.isfinite(pre);gn={k:norm(v,True) for k,v in names.items()};torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);post=norm(m.parameters(),True)
  before=[(p,p.detach().clone()) for p in m.parameters()] if steps%10==0 else [];pn=norm(m.parameters());opt.step();steps+=1
  rec={'step':steps,'epoch':ep,'n':len(q),'loss':float(loss),'lr':opt.param_groups[0]['lr'],'total_pre':pre,'total_post':post,'clipped':pre>1,'parameter_l2':pn,**{k+'_gradient':v for k,v in gn.items()}}
  if before:
   delta=math.sqrt(sum(float((p.detach()-v).double().square().sum()) for p,v in before));rec.update(update_l2=delta,update_parameter_ratio=delta/(pn+1e-12))
  return rec
 if benchsteps:
  beg=time.time()
  for k in range(benchsteps):update(tr[(np.arange(8)+8*k)%len(tr)],0)
  torch.cuda.synchronize();put(out/'benchmark.json',{'seconds':time.time()-beg,'steps':benchsteps,'job':j});return
 for ep in range(start,j['epochs']+1):
  if STOP:raise Limit('external_stop')
  if j.get('lr_replay'):
   for pg in opt.param_groups:pg['lr']=j['lr_replay'][ep-1]
  used_lr=opt.param_groups[0]['lr'];beg=time.time();order=np.random.default_rng(seed(j)+ep*17).permutation(tr);records=[]
  for off in range(0,len(order),8):
   if STOP:raise Limit('external_stop')
   rec=update(order[off:off+8],ep);append(out/'gradients.jsonl',rec);records.append(rec)
  train,ztr=evaluate(tr);row={'epoch':ep,'steps':steps,'online_BCE':online_bce(records),'eval_train':train,'lr':used_lr,'gradient':{},'seconds':time.time()-beg,'gpu':guard.g,'parameter_l2':norm(m.parameters()),'clip_fraction':float(np.mean([v['clipped'] for v in records]))}
  for key in ['total_pre','total_post','input_gradient','block0_gradient','block1_gradient','norm_gradient','head_gradient']:
   vv=[r[key] for r in records];row['gradient'][key]={k:float(v) for k,v in zip(['mean','median','p05','p95','max'],[np.mean(vv),np.median(vv),np.quantile(vv,.05),np.quantile(vv,.95),np.max(vv)])}
  row['sample_update_l2_mean']=float(np.mean([r['update_l2'] for r in records if 'update_l2' in r]))
  if len(va):
   val,z=evaluate(va);row['validation']=val;np.savez(out/f'val_epoch_{ep:03d}.npz',logits=z,labels=y[va],ids=ids[va]);candidate={'AUROC':val['AUROC'],'BCE':val['BCE'],'epoch':ep,'params':cfg['parameters'],'lr':j['lr']}
   if best is None or select([candidate,best]) is candidate:best=candidate;save(out/'best.pt',{'model':m.state_dict(),'epoch':ep,'config':cfg})
  row['sample_order_sha256']=hashlib.sha256(ids[order].tobytes()).hexdigest();row['dropout_rng_sha256']=hashlib.sha256(m.dropgen.get_state().cpu().numpy().tobytes()).hexdigest()
  if j.get('condition')=='B' and len(va):
   row['scheduler']=sched.step(val['BCE'])
   for pg in opt.param_groups:pg['lr']=sched.lr
  row['next_lr']=opt.param_groups[0]['lr'];history.append(row)
  # checkpoint is the commit point; restore will reconstruct committed history if logging was interrupted.
  save(out/'last.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'steps':steps,'history':history,'best':best,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'scheduler':sched.state_dict(),'dropout_rng':m.dropgen.get_state(),'config':cfg})
  (out/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history));put(out/'status.json',{'status':'RUNNING','epoch':ep,'target':j['epochs'],'time':time.time()})
  if capture_after==ep:
   import shutil
   fork=out/'fork_checkpoint';fork.mkdir(exist_ok=True);shutil.copy2(out/'last.pt',fork/'last.pt');shutil.copy2(out/'config.json',fork/'config.json')
  if stop_after==ep:put(out/'status.json',{'status':'ENGINEERING_STOP','epoch':ep});return
 if j['stage']=='refit':save(out/'final.pt',{'model':m.state_dict(),'config':cfg,'epoch':j['epochs']})
 put(out/'completion.json',{'status':'COMPLETE','epochs':j['epochs'],'last_sha256':sha(out/'last.pt'),'steps':steps,'test_accessed':False});put(out/'status.json',{'status':'COMPLETE','epoch':j['epochs']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);p.add_argument('--stop-after',type=int);p.add_argument('--benchsteps',type=int,default=0);p.add_argument('--capture-after',type=int);args=p.parse_args();out=Path(args.out)
 try:run(read(args.job),out,args.stop_after,args.benchsteps,args.capture_after)
 except Limit as e:put(out/'status.json',{'status':'PAUSED_LIMIT','complete':False,'reason':str(e),'time':time.time()});sys.exit(75)
 except Exception as e:put(out/'error.json',{'status':'BLOCKED','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise
