import pathlib,sys,os,json,hashlib,time,signal,argparse
R=pathlib.Path(__file__).resolve().parent;S=R/'source';sys.path[:0]=[str(S),str(S/'adapters')]
import numpy as np,torch
import train_fold as old
from common import put,append,sha,read
P=read(R/'protocol.json');STOP=False

def sig(*a):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGUSR1,sig)
def hash_encoder(m):
 h=hashlib.sha256()
 for n,p in sorted(m.encoder.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make(f,seed,arm):
 if seed==1:m=old.make('MAE',f)
 else:m=old.make_model('MAE','A2',1,dict(initialization_seed=2026102200+f+1000000*(seed-1),classification_seed=2026102200+f+800000,mechanisms_path=str(S/'model_source/code_v1/mechanisms.py'),mechanisms_sha256=read(S/'protocol.json')['mechanisms_sha256'],base_code_path=str(S)))[0]
 if arm in ['B','D']:m.decoder.blocks[0]=torch.nn.ModuleList(list(m.decoder.blocks[0])[:2])
 return m

def masks(m,ids,epoch,namespace,ratio):
 return torch.cat([m.sample_mask(1,device=torch.device('cpu'),mask_ratio=ratio,generator=torch.Generator().manual_seed(int(hashlib.sha256(f'{namespace}|{sid}|{epoch}'.encode()).hexdigest()[:15],16))) for sid in ids])

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 raw=np.load(S/'data/FC.npy',mmap_mode='r');ids=np.load(S/'data/IDs.npy');split=read(S/'splits.json')[a.fold];tr=np.array(split['inner_train']);va=np.array(split['inner_validation']);te=np.array(split['outer_test']);assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
 job=f'{a.arm}_f{a.fold}_s{a.seed}';out=R/a.mode/(a.tag or job);out.mkdir(parents=True,exist_ok=True)
 norm={'mean':float(raw[tr].mean()),'std':float(raw[tr].std()),'fit_subject_ids':ids[tr].tolist()};prior=np.asarray(raw[tr]).mean(0)
 def image(q):return torch.from_numpy(((raw[q]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
 m=make(a.fold,a.seed,a.arm);enc=hash_encoder(m);init=old.state_hash(m);params=sum(p.numel() for p in m.parameters());dec=sum(p.numel() for p in m.decoder.parameters());m=m.cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 cfg=dict(job=job,arm=a.arm,fold=a.fold,seed=a.seed,mode=a.mode,epochs=a.epochs,mask_ratio=P['arms'][a.arm]['mask_ratio'],lr=.0001,batch=8,weight_decay=.01,gradient_clip=1,encoder_initial_sha256=enc,initial_parameters_sha256=init,parameters=params,decoder_parameters=dec,norm=norm,validation_ids=ids[va].tolist(),outer_test_accessed=False,protocol_sha256=sha(R/'protocol.json'))
 if (out/'config.json').exists():assert read(out/'config.json')==cfg
 else:put(out/'config.json',cfg)
 @torch.no_grad()
 def evaluate(ix,ratio,views=3):
  m.eval();rows=[];mh=hashlib.sha256()
  for view in range(views):
   for start in range(0,len(ix),8):
    q=ix[start:start+8];mask=masks(m,ids[q],view,'cpac966_fullcv_fixed_eval_v1',ratio);mh.update(mask.numpy().tobytes());z=m(image(q),pixel_mask=mask.cuda());score=z['score_mask'][:,0].cpu().numpy();pr=z['prediction'][:,0].cpu().numpy()*norm['std']+norm['mean']
    assert np.array_equal(score,score.transpose(0,2,1)) and not np.diagonal(score,axis1=1,axis2=2).any()
    for k,i in enumerate(q):
     s=score[k];t=raw[i][s];p=pr[k][s];res=t-prior[s];predres=p-prior[s];den=np.sqrt(np.mean(res**2));corr=np.corrcoef(res,predres)[0,1] if np.std(predres)>1e-12 and np.std(res)>1e-12 else 0.
     rows.append(dict(subject_id=str(ids[i]),view=view,MSE=float(np.mean((p-t)**2)),meanFC_MSE=float(np.mean(res**2)),residual_correlation=float(corr),residual_RMS_ratio=float(np.sqrt(np.mean(predres**2))/max(den,1e-12)),score_fraction=float(s.sum()/(116*115))))
  return dict(MSE=float(np.mean([r['MSE'] for r in rows])),meanFC_MSE=float(np.mean([r['meanFC_MSE'] for r in rows])),residual_correlation=float(np.mean([r['residual_correlation'] for r in rows])),residual_RMS_ratio=float(np.mean([r['residual_RMS_ratio'] for r in rows])),score_fraction=float(np.mean([r['score_fraction'] for r in rows])),mask_sha256=mh.hexdigest(),subjects=len(ix),views=views,rows=rows)
 if a.mode=='reuse':
  path=pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r012_fullcv_20261021/runs')/f'f{a.fold}/MAE_inner/last.pt';ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==100 and ck['norm']==norm and ck['config']['initial_parameters_sha256']==init;m.load_state_dict(ck['model']);r=evaluate(va,.5);oldscore=ck['history'][-1]['validation_fixed_mask']['MSE'];assert abs(r['MSE']-oldscore)<1e-7,(r['MSE'],oldscore);r75=evaluate(va,.75);put(out/'evaluation.json',{'mask50':r,'mask75':r75});put(out/'completion.json',dict(status='COMPLETE_REUSED_BASELINE',source=str(path),sha256=sha(path),epochs=100,baseline_MSE_matches=True));return
 history=[];start=1;steps=0;elapsed_before=0
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg;m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];start=ck['epoch']+1;steps=ck['steps'];elapsed_before=ck['elapsed_seconds'];del ck
  (out/'history.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history))
  if (out/'gradients.jsonl').exists():
   lines=[l for l in (out/'gradients.jsonl').read_text().splitlines() if json.loads(l)['step']<=steps];(out/'gradients.jsonl').write_text('\n'.join(lines)+'\n')
 beg=time.monotonic();ratio=P['arms'][a.arm]['mask_ratio'];namespace=f'cpac966_fullcv_train_f{a.fold}_s{a.seed}'
 if a.mode=='benchmark':
  m.train();begin=time.monotonic();times=[]
  for step in range(a.steps):
   if STOP:break
   q=tr[(np.arange(8)+step*8)%len(tr)];t=time.monotonic();opt.zero_grad(set_to_none=True);mask=masks(m,ids[q],step+1,namespace,ratio).cuda();z=m(image(q),pixel_mask=mask);loss=old.loss_mae(z);assert torch.isfinite(loss);loss.backward();old.gradstats(m);torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
  put(out/'benchmark.json',dict(completed_steps=len(times),warmup_steps=10,seconds=time.monotonic()-begin,steady_seconds=sum(times[10:]),steady_samples=max(0,len(times)-10)*8,initial_encoder_sha256=enc,decoder_parameters=dec));return
 for epoch in range(start,a.epochs+1):
  if STOP:return
  m.train();order=np.random.default_rng(2026102100+a.fold*10000+epoch+(a.seed-1)*1000000).permutation(tr);total=0.;seen=0;ebeg=time.monotonic()
  for off in range(0,len(order),8):
   if STOP:
    put(out/'paused.json',dict(epoch_rolled_back=epoch,last_complete_epoch=epoch-1,reason='signal',time_unix=time.time()));return
   q=order[off:off+8];opt.zero_grad(set_to_none=True);mask=masks(m,ids[q],epoch,namespace,ratio).cuda();z=m(image(q),pixel_mask=mask);loss=old.loss_mae(z);assert torch.isfinite(loss);loss.backward();gs=old.gradstats(m);assert gs['encoder_preclip_l2']>0 and gs['decoder_preclip_l2']>0
   gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True));before=[p.detach().clone() for p in m.parameters()] if steps%10==0 else None;opt.step();steps+=1;rec=dict(epoch=epoch,step=steps,loss=float(loss),clipped=gn>1,**gs)
   if before is not None:rec['parameter_update_l2']=float(torch.sqrt(sum(((p.detach()-b).double().square().sum() for p,b in zip(m.parameters(),before)),torch.zeros((),device='cuda'))));del before
   append(out/'gradients.jsonl',rec);total+=float(loss)*len(q);seen+=len(q)
  row=dict(epoch=epoch,online_training_loss=total/seen,online_train_raw_MSE=total/seen*norm['std']**2,steps=steps,train_epoch_seconds=time.monotonic()-ebeg,elapsed_seconds=elapsed_before+time.monotonic()-beg)
  if epoch%10==0 or epoch==a.epochs:
   for label,ix in [('train',np.array(sorted(tr)[:64])),('validation',va)]:
    ev=evaluate(ix,.5);row[label+'_mask50']={k:v for k,v in ev.items() if k!='rows'}
    if epoch==a.epochs:put(out/f'{label}_mask50_subjects.json',ev)
   q=np.array(sorted(tr)[:8]);m.eval();opt.zero_grad(set_to_none=True);z=m(image(q),pixel_mask=masks(m,ids[q],0,'cpac966_fixed_gradient_probe_v1',.5).cuda());old.loss_mae(z).backward();row['fixed_probe_gradient']=old.gradstats(m);opt.zero_grad(set_to_none=True)
  if epoch%20==0 or epoch==a.epochs:
   ev=evaluate(va,.75);row['validation_mask75']={k:v for k,v in ev.items() if k!='rows'}
   if epoch==a.epochs:put(out/'validation_mask75_subjects.json',ev)
  history.append(row);append(out/'history.jsonl',row)
  old.save(out/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,steps=steps,config=cfg,norm=norm,history=history,elapsed_seconds=elapsed_before+time.monotonic()-beg))
  if epoch in [20,40,60,80,100]:old.save(out/f'epoch_{epoch}.pt',dict(model=m.state_dict(),epoch=epoch,norm=norm,config=cfg))
  put(out/'status.json',dict(state='RUNNING',epoch=epoch,target=a.epochs,steps=steps,time_unix=time.time()))
  if a.stop_after and epoch==a.stop_after:return
 put(out/'completion.json',dict(status='COMPLETE',epochs=a.epochs,steps=steps,last_sha256=sha(out/'last.pt'),time_unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list('ABCD'),required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=1);p.add_argument('--mode',choices=['pilot','benchmark','runs','reuse'],default='runs');p.add_argument('--tag');p.add_argument('--epochs',type=int,default=100);p.add_argument('--steps',type=int,default=100);p.add_argument('--stop-after',type=int,default=0);run(p.parse_args())
