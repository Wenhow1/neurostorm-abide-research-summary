from pathlib import Path
import sys,json,time,subprocess,hashlib,os
import numpy as np,torch,joblib
BASE=Path('/home/wenhao/neurostorm_2d/experiments/cpac966_coordinate_pe_20261022');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE));import worker as w
from common import read,put,sha

torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
raw=np.load(BASE/'source/data/FC.npy',mmap_mode='r');ids=np.load(BASE/'source/data/IDs.npy');labels=np.load(BASE/'source/data/Y.npy');splits=read(BASE/'source/splits.json')
last_check=0; paused=False
def guard():
 global last_check,paused
 if time.time()-last_check<5:return
 while True:
  temp=float(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip());last_check=time.time()
  with (OUT/'temperature.jsonl').open('a')as f:f.write(json.dumps({'time':last_check,'GPU':1,'temperature':temp})+'\n')
  if temp>=80:paused=True
  if paused and temp<=65:paused=False
  if not paused:return
  put(OUT/'status.json',{'state':'PAUSED_TEMPERATURE','temperature':temp,'time':time.time()});time.sleep(10)

def pair(shuffle):
 xyz=torch.from_numpy(np.load(BASE/'atlas/coordinates_mm.npy')).float()/100
 if shuffle:xyz=xyz[torch.from_numpy(np.load(BASE/'atlas/permutation.npy'))]
 delta=xyz[:,None]-xyz[None,:]
 return torch.cat([(xyz[:,None]+xyz[None,:])/2,delta.abs(),delta.norm(dim=-1,keepdim=True)],-1)[None].cuda()
rows=[]
for arm in ['P1','P2']:
 for seed in [1,2]:
  for fold in [0,1,2]:
   name=f'{arm}_f{fold}_s{seed}';dest=OUT/name;dest.mkdir(exist_ok=True)
   if (dest/'completion.json').exists():rows.extend(read(dest/'completion.json')['rows']);continue
   guard();path=BASE/'runs'/name/'epoch_100.pt';diskhash=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);norm=ck['norm'];tr=splits[fold]['inner_train'];va=splits[fold]['inner_validation'];assert norm['fit_subject_ids']==ids[tr].tolist();assert not set(va)&set(splits[fold]['outer_test'])
   m=w.make(fold,seed,arm).cuda().eval();m.load_state_dict(ck['model']);m.requires_grad_(False);before=w.old.state_hash(m);gates=m.encoder.geo_gates.detach().clone();geometry=m.encoder.geo_pair.detach().clone();prior=np.asarray(raw[tr]).mean(0);clf=joblib.load(BASE/'readout'/name/'global_classifier.joblib');clfhash=sha(BASE/'readout'/name/'global_classifier.joblib');baseline=None;unit=[]
   def image(q):return torch.from_numpy(((raw[q]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
   with torch.inference_mode():
    for mode in ['original','off','swap']:
     guard();put(OUT/'status.json',{'state':'RUNNING','model':name,'mode':mode,'completed_rows':len(rows),'time':time.time()})
     m.encoder.geo_gates.copy_(gates if mode!='off' else torch.zeros_like(gates));m.encoder.geo_pair.copy_(geometry if mode!='swap' else pair(arm=='P1'))
     features=[];injection=[[]for _ in range(4)];hooks=[];projected={}
     for j,proj in enumerate(m.encoder.geo_projections):
      hooks.append(proj.register_forward_hook(lambda mod,args,out,j=j:projected.__setitem__(j,out)))
     for j,stage in enumerate(m.encoder.stages):
      def hook(mod,args,j=j):
       inc=m.encoder.geo_gates[j]*projected[j];z=args[0];den=(z-inc).double().square().sum();num=inc.double().square().sum()*z.shape[0];injection[j].append([float(num),float(den)])
      hooks.append(stage.register_forward_pre_hook(hook))
     for start in range(0,len(va),8):
      guard();z=m.encode_full(image(va[start:start+8]),use_prompts=False);features.append(z.mean((1,2)).cpu().numpy())
     for h in hooks:h.remove()
     feat=np.concatenate(features);prob=clf.predict_proba(feat)[:,1];rec={};preds={};targets={};refs={}
     for ratio in [.5,.75]:
      pp=[];tt=[];rr=[];errors=[];cors=[];amps=[];mh=hashlib.sha256()
      for view in range(3):
       for start in range(0,len(va),8):
        guard();q=va[start:start+8];mask=w.masks(m,ids[q],view,'cpac966_fullcv_fixed_eval_v1',ratio);mh.update(mask.numpy().tobytes());z=m(image(q),pixel_mask=mask.cuda());score=z['score_mask'][:,0].cpu().numpy();p=z['prediction'][:,0].cpu().numpy()*norm['std']+norm['mean']
        assert not np.diagonal(score,axis1=1,axis2=2).any() and np.array_equal(score,score.transpose(0,2,1))
        for k,i in enumerate(q):
         sel=score[k];a=p[k][sel];b=raw[i][sel];r=prior[sel];pp.append(a);tt.append(b);rr.append(r);errors.append(float(np.mean((a-b)**2)));cors.append(float(np.corrcoef(a-r,b-r)[0,1]));amps.append(float(np.sqrt(np.mean((a-r)**2)/np.mean((b-r)**2))))
      key=str(ratio);preds[key]=np.concatenate(pp);targets[key]=np.concatenate(tt);refs[key]=np.concatenate(rr);rec[key]={'MSE':float(np.mean(errors)),'residual_correlation':float(np.mean(cors)),'amplitude_ratio':float(np.mean(amps)),'mask_sha256':mh.hexdigest()}
      saved=read(BASE/'runs'/name/f'validation_mask{int(ratio*100)}_subjects.json');assert rec[key]['mask_sha256']==saved['mask_sha256']
      if mode=='original':assert abs(rec[key]['MSE']-saved['MSE'])<1e-7
     if mode=='original':
      baseline={'features':feat,'prob':prob,'preds':preds};saved=np.load(BASE/'readout'/name/'global_validation.npz');assert np.array_equal(saved['ids'],ids[va]);assert np.max(abs(prob-saved['probability']))<1e-5
     result={'arm':arm,'fold':fold,'seed':seed,'mode':mode,'subjects':len(va),'reconstruction':rec,'classification':w.old.metrics(labels[va],prob),'feature_relative_L2':float(np.linalg.norm(feat-baseline['features'])/np.linalg.norm(baseline['features'])),'probability_MAE_change':float(np.mean(abs(prob-baseline['prob']))),'probability_max_change':float(np.max(abs(prob-baseline['prob']))),'prediction_flips':int(np.sum((prob>=.5)!=(baseline['prob']>=.5))),'injection_relative_L2_by_stage':[float(np.sqrt(np.sum(v,axis=0)[0]/np.sum(v,axis=0)[1]))for v in injection],'reconstruction_change':{k:{'RMSE_change':float(np.sqrt(np.mean((preds[k]-baseline['preds'][k])**2))),'relative_to_residual_L2':float(np.linalg.norm(preds[k]-baseline['preds'][k])/np.linalg.norm(targets[k]-refs[k]))}for k in preds}}
     np.savez_compressed(dest/f'{mode}_predictions.npz',ids=ids[va],labels=labels[va],probability=prob,features=feat);unit.append(result);put(dest/f'{mode}.json',result)
    m.encoder.geo_gates.copy_(gates);m.encoder.geo_pair.copy_(geometry)
   assert w.old.state_hash(m)==before and sha(path)==diskhash and sha(BASE/'readout'/name/'global_classifier.joblib')==clfhash
   put(dest/'completion.json',{'status':'COMPLETE','rows':unit,'checkpoint_sha256':diskhash,'classifier_sha256':clfhash,'weights_restored_unchanged':True,'outer_test_accessed':False,'no_optimizer':True,'time':time.time()});rows.extend(unit);put(OUT/'results.json',{'status':'RUNNING','rows':rows});del m,ck;torch.cuda.empty_cache()
put(OUT/'results.json',{'status':'COMPLETE','rows':rows,'time':time.time(),'limitations':['fixed weights inference sensitivity only','internal validation only','same fixed shuffle as original experiment','no classifier refitting or new training']});put(OUT/'status.json',{'state':'COMPLETE','models':12,'conditions':36,'time':time.time()})
