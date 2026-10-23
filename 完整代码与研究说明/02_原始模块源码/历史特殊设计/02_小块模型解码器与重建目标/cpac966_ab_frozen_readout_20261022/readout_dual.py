import pathlib,sys,os,json,hashlib,csv,time,warnings,argparse
R=pathlib.Path(__file__).resolve().parent;S=R/'source';P=pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_decoder_mask_20261022');OLD=pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r012_fullcv_20261021');CACHED=pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r1_spatial2x2_lr_20261021');STRONG=pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_r1_spatial2x2_strongreg_20261021')
sys.path[:0]=[str(R),str(S),str(S/'adapters')]
import numpy as np,torch,joblib
import outer_worker_dual as worker
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from common import put,read,sha
from train_fold import metrics,state_hash

def checkpoint(arm,fold,seed,scope):
 if arm=='A' and seed==1:return OLD/f'runs/f{fold}/MAE_{scope}/epoch_100.pt'
 if scope=='inner':return P/f'runs/{arm}_f{fold}_s{seed}/epoch_100.pt'
 return R/f'runs/{arm}_f{fold}_s{seed}/epoch_100.pt'

def fold(arm,f,seed,engineering=False):
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ['GPU-8524fa8f-e912-9e1b-1e81-65be799c90f7','GPU-7ac39891-108b-f897-b92f-cd51e5392bab']
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;warnings.filterwarnings('error',category=ConvergenceWarning)
 ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');raw=np.load(S/'data/FC.npy',mmap_mode='r');sp=read(S/'splits.json')[f];out=R/('pilot_readout' if engineering else 'readout')/f'{arm}_f{f}_s{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 def extract(scope,tr,va):
  path=checkpoint(arm,f,seed,scope);digest=sha(path);idx=np.array(tr+va);cache=out/f'{scope}_features.npz';meta=out/f'{scope}_features.json'
  if cache.exists() and meta.exists():
   rec=read(meta);assert rec['checkpoint_sha256']==digest and rec['feature_sha256']==sha(cache);a=np.load(cache);assert np.array_equal(a['ids'],ids[idx]);return a['features']
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==100 and ck['norm']['fit_subject_ids']==ids[tr].tolist();assert not set(tr)&set(va);m=worker.make(f,seed,arm);m.load_state_dict(ck['model']);m.requires_grad_(False);m=m.cuda().eval();before=state_hash(m);called=[0];hook=m.decoder.register_forward_hook(lambda *a:called.__setitem__(0,called[0]+1));norm=ck['norm'];x=[]
  with torch.inference_mode():
   for k in range(0,len(idx),16):
    image=torch.from_numpy(((raw[idx[k:k+16]]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda();z=m.encode_full(image,use_prompts=False);assert tuple(z.shape[1:])==(8,8,288);x.append(z.reshape(len(z),2,4,2,4,288).mean((2,4)).reshape(len(z),1152).cpu().numpy())
  x=np.concatenate(x);assert np.isfinite(x).all() and state_hash(m)==before and called[0]==0 and not any(p.requires_grad for p in m.parameters())
  # A1 is the old matched baseline. Preserve its cached feature bytes for exact classifier reproduction,
  # while checking that a fresh extraction agrees within the measured float32 kernel tolerance.
  baseline_proof=None
  if arm=='A' and seed==1:
   oldmeta=read(CACHED/f'f{f}/{scope}_features.json');old=np.load(CACHED/f'f{f}/{scope}_features.npz');assert oldmeta['checkpoint_sha256']==digest and np.array_equal(old['ids'],ids[idx]);delta=float(np.max(np.abs(x-old['features'])));assert np.allclose(x,old['features'],rtol=1e-4,atol=1e-5),delta;baseline_proof={'cache':str(CACHED/f'f{f}/{scope}_features.npz'),'fresh_max_abs_difference':delta};x=old['features']
  temp=out/f'{scope}_features.tmp.npz';np.savez_compressed(temp,features=x,ids=ids[idx]);temp.replace(cache);put(meta,{'checkpoint':str(path),'checkpoint_sha256':digest,'feature_sha256':sha(cache),'normalization_training_ids':ids[tr].tolist(),'feature_ids':ids[idx].tolist(),'shape':list(x.shape),'decoder_calls':called[0],'frozen_parameters_unchanged':True,'A1_cached_feature_check':baseline_proof});hook.remove();del m,ck;torch.cuda.empty_cache();return x
 def pipe(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=10000,tol=1e-6,random_state=2026102100+f))
 tr=sp['inner_train'];va=sp['inner_validation'];x=extract('inner',tr,va);scores=[]
 for c in [.001,.003,.01]:
  m=pipe(c);m.fit(x[:len(tr)],y[tr]);assert m[0].n_samples_seen_==len(tr);scores.append({'C':c,'train':metrics(y[tr],m.predict_proba(x[:len(tr)])[:,1]),'validation':metrics(y[va],m.predict_proba(x[len(tr):])[:,1])})
 best=min(scores,key=lambda a:(a['validation']['BCE'],a['C']));put(out/'selection.json',{'selected_C':best['C'],'candidates':scores,'selection_criterion':'inner validation BCE','outer_test_used':False})
 if engineering:
  put(out/'completion.json',{'status':'ENGINEERING_INNER_ONLY_PASS','classifier_fits':3,'outer_test_evaluated':False,'time_unix':time.time()});return
 ot=sp['outer_train'];te=sp['outer_test'];x=extract('outer',ot,te);m=pipe(best['C']);m.fit(x[:len(ot)],y[ot]);assert m[0].n_samples_seen_==len(ot);assert np.allclose(m[0].mean_,x[:len(ot)].astype(float).mean(0),rtol=1e-10,atol=1e-10);pred=m.predict_proba(x[len(ot):])[:,1];train=metrics(y[ot],m.predict_proba(x[:len(ot)])[:,1]);test=metrics(y[te],pred);reproduced=None
 if arm=='A' and seed==1:
  previous=np.load(STRONG/f'f{f}/outer_predictions.npz');assert np.array_equal(previous['ids'],ids[te]);assert np.max(np.abs(previous['probability']-pred))<1e-10;reproduced=True
 joblib.dump(m,out/'classifier.joblib');np.savez_compressed(out/'outer_predictions.npz',ids=ids[te],labels=y[te],probability=pred);put(out/'completion.json',{'status':'COMPLETE','arm':arm,'fold':f,'seed':seed,'selected_C':best['C'],'classifier_fits':4,'backbone_updates_during_readout':0,'train':train,'test':test,'A1_historical_predictions_reproduced':reproduced,'time_unix':time.time()})

def summarize():
 rows=[];summ=[];predictions={};ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');sp=read(S/'splits.json')
 for seed in [1,2]:
  for arm in 'AB':
   pred=np.full(len(y),np.nan)
   for f in range(10):
    out=R/'readout'/f'{arm}_f{f}_s{seed}';c=read(out/'completion.json');a=np.load(out/'outer_predictions.npz');te=sp[f]['outer_test'];assert np.array_equal(a['ids'],ids[te]) and np.isnan(pred[te]).all();pred[te]=a['probability'];rows.append({'arm':arm,'seed':seed,'fold':f,'C':c['selected_C'],**{'train_'+k:v for k,v in c['train'].items()},**{'test_'+k:v for k,v in c['test'].items()}})
   assert np.isfinite(pred).all();predictions[(arm,seed)]=pred;summ.append({'arm':arm,'seed':seed,**metrics(y,pred)})
   with (R/f'OOF_{arm}_s{seed}.csv').open('w') as fh:w=csv.writer(fh);w.writerow(['subject_id','label','probability']);w.writerows(zip(ids,y,pred))
 with (R/'fold_results.csv').open('w') as fh:w=csv.DictWriter(fh,rows[0].keys());w.writeheader();w.writerows(rows)
 paired=[]
 for seed in [1,2]:
  a=metrics(y,predictions[('A',seed)]);b=metrics(y,predictions[('B',seed)]);paired.append({'seed':seed,'B_minus_A':{k:b[k]-a[k] for k in a},'fold_AUROC_wins_B':sum(next(x for x in rows if x['arm']=='B' and x['seed']==seed and x['fold']==f)['test_AUROC']>next(x for x in rows if x['arm']=='A' and x['seed']==seed and x['fold']==f)['test_AUROC'] for f in range(10))})
 put(R/'results.json',{'status':'COMPLETE','summary':summ,'paired_comparisons':paired,'new_outer_MAE_fits':30,'readout_classifier_fits':160,'pretrained_inner_subjects':[695,696],'pretrained_outer_subjects':[869,870],'limitations':['historically exposed development CV; not new independent confirmation','do not pool two seeds as1932 independent subjects','no automatic tuning or new training based on outer results'],'time_unix':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fold','summarize']);p.add_argument('--arm');p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);p.add_argument('--engineering',action='store_true');a=p.parse_args();summarize() if a.action=='summarize' else fold(a.arm,a.fold,a.seed,a.engineering)
