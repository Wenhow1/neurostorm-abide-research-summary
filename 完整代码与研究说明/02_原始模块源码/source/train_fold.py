from common import *
import sys,math,signal
import numpy as np,torch
from sklearn.metrics import roc_auc_score,log_loss,accuracy_score,balanced_accuracy_score,confusion_matrix
sys.path[:0]=[str(R),str(R/'adapters')]
from model_adapter import make_model
from neurostorm_2d.scripts.train_mae import masks
STOP=False

def stop(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,stop)
def check_stop():
 if STOP or (R/'control/STOP').exists():raise RuntimeError('STOP at checkpoint-safe boundary')
def save(p,x):
 q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def state_hash(model,which='all'):
 h=hashlib.sha256()
 for n,p in sorted(model.named_parameters()):
  if which=='frozen' and p.requires_grad:continue
  if which=='head' and not n.startswith('head.'):continue
  h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def metrics(y,pr):
 pred=pr>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return dict(AUROC=float(roc_auc_score(y,pr)),BCE=float(log_loss(y,pr,labels=[0,1])),ACC=float(accuracy_score(y,pred)),BA=float(balanced_accuracy_score(y,pred)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))
def loss_mae(result):return torch.stack([(p-t).square().masked_select(m).mean() for p,t,m in zip(result['prediction'],result['target'],result['score_mask'])]).mean()
def make(mode,fold,pretrained=None):
 return make_model(mode,'A2',1,dict(initialization_seed=2026102200+fold,classification_seed=2026102200+fold+800000,mechanisms_path=str(R/'model_source/code_v1/mechanisms.py'),mechanisms_sha256=read(R/'protocol.json')['mechanisms_sha256'],base_code_path=str(R)),pretrained_state=pretrained)[0]
def gradstats(model):
 out={}
 for group,predicate in [('all',lambda n:True),('encoder',lambda n:n.startswith('encoder.')),('decoder',lambda n:n.startswith('decoder.')),('head',lambda n:n.startswith('head.')),('prompt',lambda n:n.endswith('.prompt'))]:
  g=[p.grad.detach() for n,p in model.named_parameters() if p.grad is not None and predicate(n)];out[group+'_preclip_l2']=float(torch.sqrt(sum((x.double().square().sum() for x in g),torch.zeros((),device='cuda')))) if g else 0.
 assert all(np.isfinite(list(out.values())));assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad);return out

class Fold:
 def __init__(self,f,engineering=False):
  self.p=verify();assert os.environ['CUDA_VISIBLE_DEVICES'] in read(R/'resource_policy.json')['gpu_uuids'].values();self.f=f;self.eng=engineering;self.raw=np.load(R/'data/FC.npy',mmap_mode='r');self.ids=np.load(R/'data/IDs.npy');self.y=np.load(R/'data/Y.npy');self.sp=read(R/'splits.json')[f];self.root=R/('engineering' if engineering else 'runs')/f'f{f}';self.root.mkdir(parents=True,exist_ok=True)
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def image(self,ix,norm):return torch.from_numpy(((self.raw[ix]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
 def normalization(self,ix):
  a=np.asarray(self.raw[ix]);return dict(mean=float(a.mean()),std=float(a.std()),fit_subject_ids=self.ids[ix].tolist())
 @torch.no_grad()
 def recon(self,m,ix,norm,prior):
  m.eval();vals=[];refs=[];symvals=[]
  for view in range(3):
   for start in range(0,len(ix),8):
    q=ix[start:start+8];mask=masks(m,self.ids[q],view,'cpac966_fullcv_fixed_eval_v1').cuda();z=m(self.image(q,norm),pixel_mask=mask);pr=z['prediction']*norm['std']+norm['mean'];target=z['target']*norm['std']+norm['mean'];sym=(pr+pr.transpose(-1,-2))/2;score=z['score_mask'];assert torch.equal(score,score.transpose(-1,-2)) and not torch.diagonal(score[:,0],dim1=-2,dim2=-1).any()
    for k,i in enumerate(q):
     sel=score[k];vals.append(float((pr[k]-target[k]).square().masked_select(sel).mean()));symvals.append(float((sym[k]-target[k]).square().masked_select(sel).mean()));s=sel.cpu().numpy()[0];refs.append(float(np.mean((prior[s]-self.raw[i][s])**2)))
  return dict(MSE=float(np.mean(vals)),sym_MSE=float(np.mean(symvals)),mean_FC_MSE=float(np.mean(refs)),subjects=len(ix),views=3)
 @torch.no_grad()
 def features(self,m,ix,norm):
  m.eval();return torch.cat([m.mae.encode_full(self.image(ix[s:s+16],norm),use_prompts=False).mean((1,2)).detach().cpu() for s in range(0,len(ix),16)])
 @torch.no_grad()
 def predict(self,m,ix,norm):
  m.eval();return np.concatenate([torch.sigmoid(m(self.image(ix[s:s+16],norm))).cpu().numpy() for s in range(0,len(ix),16)])
 def fit(self,name,kind,ix,val,epochs,lr,mae_source=None):
  out=self.root/name;out.mkdir(exist_ok=True)
  if (out/'completion.json').exists():
   c=read(out/'completion.json');assert c['status']=='COMPLETE' and c['protocol_sha256']==sha(R/'protocol.json') and c['last_sha256']==sha(out/'last.pt');return out
  check_stop();norm=self.normalization(ix);prior=np.asarray(self.raw[ix]).mean(0)
  ismae=kind=='MAE';source=None
  if not ismae:
   source=torch.load(mae_source,map_location='cpu',weights_only=False);assert set(source['norm']['fit_subject_ids'])==set(norm['fit_subject_ids']);assert not set(self.ids[val])&set(norm['fit_subject_ids']);m=make('T0' if kind=='R1' else 'T1',self.f,source['model']);del source
  else:m=make('MAE',self.f)
  m=m.cuda();params=[p for p in m.parameters() if p.requires_grad];initial=state_hash(m);frozen=state_hash(m,'frozen');head=state_hash(m,'head');trainable=[n for n,p in m.named_parameters() if p.requires_grad]
  if not ismae:assert all(n.startswith('head.') or (kind=='R2' and n.endswith('.prompt')) for n in trainable)
  opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01,betas=(.9,.999),eps=1e-8);batch=8 if ismae else 16
  candidates=[epochs] if self.eng else (self.p['mae']['selection_epochs'] if ismae else self.p['classification']['selection_epochs']);candidates=[e for e in candidates if e<=epochs]
  cfg=dict(name=name,kind=kind,fold=self.f,engineering=self.eng,train_ids=self.ids[ix].tolist(),validation_ids=self.ids[val].tolist(),norm=norm,epochs=epochs,lr=lr,batch=batch,source=str(mae_source) if mae_source else None,source_sha256=sha(mae_source) if mae_source else None,initial_parameters_sha256=initial,initial_head_sha256=head,frozen_sha256=frozen,trainable_parameters=sum(p.numel() for p in params),trainable_names=trainable,protocol_sha256=sha(R/'protocol.json'))
  put(out/'config.json',cfg);start=1;steps=0;history=[]
  if (out/'last.pt').exists():
   old=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert old['config']==cfg;m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);start=old['epoch']+1;steps=old['steps'];history=old['history'];del old
   (out/'history.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history))
   if (out/'gradients.jsonl').exists():
    keep=[x for x in (out/'gradients.jsonl').read_text().splitlines() if json.loads(x)['step']<=steps];(out/'gradients.jsonl').write_text('\n'.join(keep)+ ('\n' if keep else ''))
  cache=None;cacheval=None
  if kind=='R1':cache=self.features(m,ix,norm).cuda();cacheval=self.features(m,val,norm).cuda() if len(val) else None;mapping={int(v):i for i,v in enumerate(ix)}
  probe=np.array(sorted(ix)[:min(64,len(ix))]);beg=time.time()
  if ismae and start==1 and len(val):put(out/'initial_evaluation.json',dict(train=self.recon(m,probe,norm,prior),validation=self.recon(m,val,norm,prior)))
  for epoch in range(start,epochs+1):
   check_stop();m.train();order=np.random.default_rng(2026102100+self.f*10000+epoch).permutation(ix);loss_sum=0.;seen=0
   for offset in range(0,len(ix),batch):
    q=order[offset:offset+batch];opt.zero_grad(set_to_none=True)
    if ismae:
     mask=masks(m,self.ids[q],epoch,f'cpac966_fullcv_train_f{self.f}_s1').cuda();z=m(self.image(q,norm),pixel_mask=mask);loss=loss_mae(z)
    else:
     logits=m.head(cache[[mapping[int(i)] for i in q]]).squeeze(-1) if kind=='R1' else m(self.image(q,norm));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(self.y[q],device='cuda',dtype=torch.float32))
    assert torch.isfinite(loss);loss.backward();gs=gradstats(m)
    if steps==0:
     assert gs['all_preclip_l2']>0
     if kind=='R2':assert gs['prompt_preclip_l2']>0 and gs['head_preclip_l2']>0
     if ismae:assert gs['encoder_preclip_l2']>0 and gs['decoder_preclip_l2']>0
    normgrad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));detailed=steps%10==0;before=[p.detach().clone() for p in params] if detailed else None
    opt.step();steps+=1;rec=dict(epoch=epoch,step=steps,lr=lr,clipped=normgrad>1,**gs)
    if detailed:rec['parameter_update_l2']=float(torch.sqrt(sum(((p.detach()-b).double().square().sum() for p,b in zip(params,before)),torch.zeros((),device='cuda'))));del before
    append(out/'gradients.jsonl',rec);loss_sum+=float(loss.detach())*len(q);seen+=len(q)
   assert seen==len(ix);row=dict(epoch=epoch,online_training_loss=loss_sum/seen,optimizer_steps=steps,elapsed_seconds=time.time()-beg)
   if ismae:row['online_train_raw_MSE']=row['online_training_loss']*norm['std']**2
   if epoch%10==0 or epoch==epochs or self.eng:
    if ismae:
     row['train_fixed_mask']=self.recon(m,probe,norm,prior)
     if len(val):row['validation_fixed_mask']=self.recon(m,val,norm,prior)
    else:
     if kind=='R1':
      with torch.no_grad():pr=torch.sigmoid(m.head(cache).squeeze(-1)).cpu().numpy();pv=torch.sigmoid(m.head(cacheval).squeeze(-1)).cpu().numpy() if cacheval is not None else None
     else:pr=self.predict(m,ix,norm);pv=self.predict(m,val,norm) if len(val) else None
     row['train']=metrics(self.y[ix],pr)
     if len(val):row['validation']=metrics(self.y[val],pv)
    # Fixed training probe gradient: same subjects/masks and eval mode at each observation.
    m.eval();opt.zero_grad(set_to_none=True);q=probe[:8]
    if ismae:z=m(self.image(q,norm),pixel_mask=masks(m,self.ids[q],0,'cpac966_fixed_gradient_probe_v1').cuda());probe_loss=loss_mae(z)
    else:probe_loss=torch.nn.functional.binary_cross_entropy_with_logits(m(self.image(q,norm)),torch.as_tensor(self.y[q],device='cuda',dtype=torch.float32))
    probe_loss.backward();row['fixed_probe_gradient']=gradstats(m);opt.zero_grad(set_to_none=True)
   history.append(row);append(out/'history.jsonl',row)
   checkpoint=dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,steps=steps,history=history,config=cfg,norm=norm)
   save(out/'last.pt',checkpoint)
   if epoch in candidates:save(out/f'epoch_{epoch}.pt',dict(model=m.state_dict(),epoch=epoch,norm=norm,config=cfg))
   put(out/'status.json',dict(state='RUNNING',kind=kind,epoch=epoch,target=epochs,steps=steps,elapsed_seconds=time.time()-beg,time_unix=time.time()))
  assert ismae or state_hash(m,'frozen')==frozen
  put(out/'completion.json',dict(status='COMPLETE',kind=kind,epochs=epochs,steps=steps,train_count=len(ix),validation_count=len(val),protocol_sha256=sha(R/'protocol.json'),last_sha256=sha(out/'last.pt'),frozen_unchanged=not ismae,trainable_parameters=cfg['trainable_parameters'],time_unix=time.time()))
  del m,opt,cache,cacheval;torch.cuda.empty_cache();return out
 def selected_mae(self,out):
  h=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()];epochs=[2] if self.eng else self.p['mae']['selection_epochs'];best=min([x for x in h if x['epoch'] in epochs],key=lambda x:(x['validation_fixed_mask']['MSE'],x['epoch']));put(self.root/'mae_selection.json',dict(epoch=best['epoch'],criterion='inner_validation_MSE',score=best['validation_fixed_mask']['MSE'],outer_test_read=False));return best['epoch']
 def R0(self):
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.exceptions import ConvergenceWarning
  import warnings,joblib
  warnings.filterwarnings('error',category=ConvergenceWarning);tr=self.sp['inner_train'];v=self.sp['inner_validation'];ot=self.sp['outer_train'];te=self.sp['outer_test'];u=np.triu_indices(116,1);x=self.raw[:,u[0],u[1]];out=self.root/'R0';out.mkdir(exist_ok=True)
  if (out/'completion.json').exists():return
  def pipeline(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=10000,tol=1e-6,random_state=2026102100+self.f))
  scores=[]
  for c in self.p['R0']['C']:
   m=pipeline(c);m.fit(x[tr],self.y[tr]);scores.append(dict(C=c,**metrics(self.y[v],m.predict_proba(x[v])[:,1])))
  best=min(scores,key=lambda z:(z['BCE'],z['C']));put(out/'selection.json',dict(selected=best,candidates=scores,outer_test_used=False));m=pipeline(best['C']);m.fit(x[ot],self.y[ot]);pr=m.predict_proba(x[te])[:,1];joblib.dump(m,out/'model.joblib');np.savez_compressed(out/'outer_predictions.npz',ids=self.ids[te],labels=self.y[te],probability=pr);put(out/'completion.json',dict(status='COMPLETE',fits=5,test_n=len(te),metrics=metrics(self.y[te],pr),time_unix=time.time()))
 def run(self):
  tr=np.array(self.sp['inner_train']);v=np.array(self.sp['inner_validation']);ot=np.array(self.sp['outer_train']);te=np.array(self.sp['outer_test'])
  if self.eng:tr=np.r_[tr[self.y[tr]==0][:16],tr[self.y[tr]==1][:16]];v=np.r_[v[self.y[v]==0][:8],v[self.y[v]==1][:8]];ot=np.array([],int)
  maxmae=2 if self.eng else 100;inner=self.fit('MAE_inner','MAE',tr,v,maxmae,.0001);ep=self.selected_mae(inner);source=inner/f'epoch_{ep}.pt';selections={};firstheads=[]
  for route in ['R1','R2']:
   options=[]
   for lr in ([.0003] if self.eng else self.p['classification']['lrs']):
    out=self.fit(f'{route}_inner_lr{lr:g}',route,tr,v,2 if self.eng else 80,lr,source);firstheads.append(read(out/'config.json')['initial_head_sha256']);h=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    for z in h:
     if z['epoch'] in ([2] if self.eng else self.p['classification']['selection_epochs']):options.append(dict(lr=lr,epoch=z['epoch'],BCE=z['validation']['BCE'],AUROC=z['validation']['AUROC']))
   minimum=min(x['BCE'] for x in options);best=min([x for x in options if x['BCE']-minimum<=1e-8],key=lambda x:(x['lr'],x['epoch']));selections[route]=best;put(self.root/f'{route}_selection.json',dict(selected=best,candidates=options,criterion='inner BCE',outer_test_read=False))
  assert len(set(firstheads))==1
  if self.eng:
   put(self.root/'completion.json',dict(status='PASS_ENGINEERING',head_initialization_matched=True,frozen_backbone_preserved=True,R2_prompt_gradients_nonzero=True,scientific_results=False));return
  self.R0();outer=self.fit('MAE_outer','MAE',ot,np.array([],int),ep,.0001);source=outer/f'epoch_{ep}.pt'
  for route in ['R1','R2']:
   z=selections[route];out=self.fit(f'{route}_outer',route,ot,np.array([],int),z['epoch'],z['lr'],source);ck=torch.load(out/f'epoch_{z["epoch"]}.pt',map_location='cpu',weights_only=False)
   m=make('T0' if route=='R1' else 'T1',self.f,torch.load(source,map_location='cpu',weights_only=False)['model']);m.load_state_dict(ck['model']);m=m.cuda().eval();pr=self.predict(m,te,ck['norm']);np.savez_compressed(out/'outer_predictions.npz',ids=self.ids[te],labels=self.y[te],probability=pr);put(out/'outer_evaluation.json',dict(metrics=metrics(self.y[te],pr),test_ids=self.ids[te].tolist(),checkpoint_sha256=sha(out/f'epoch_{z["epoch"]}.pt'),training_ids=self.ids[ot].tolist(),time_unix=time.time()));del m,ck;torch.cuda.empty_cache()
  put(self.root/'completion.json',dict(status='COMPLETE_R012_FOLD',fold=self.f,neural_fits=8,logistic_fits=5,mae_epoch=ep,classifier_selections=selections,test_count=len(te),time_unix=time.time()))
if __name__=='__main__':Fold(int(sys.argv[1]),'--engineering' in sys.argv).run()
