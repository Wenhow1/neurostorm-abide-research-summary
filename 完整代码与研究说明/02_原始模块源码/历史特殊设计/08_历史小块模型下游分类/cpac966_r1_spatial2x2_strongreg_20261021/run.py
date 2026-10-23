from pathlib import Path
import os,json,csv,hashlib,time,warnings
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,log_loss,confusion_matrix
D=Path(__file__).resolve().parent;BASE=Path('/home/wenhao/neurostorm_2d/experiments');S=BASE/'cpac966_r012_fullcv_20261021';P=BASE/'cpac966_r1_spatial2x2_lr_20261021';Q=BASE/'cpac966_r1_standardized_lr_20261021'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def table(p,rows):
 with p.open('w') as h:w=csv.DictWriter(h,rows[0].keys());w.writeheader();w.writerows(rows)
def metrics(y,p):
 tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel();return dict(AUROC=float(roc_auc_score(y,p)),BCE=float(log_loss(y,p,labels=[0,1])),ACC=float(accuracy_score(y,p>=.5)),BA=float(balanced_accuracy_score(y,p>=.5)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)))
def run():
 warnings.filterwarnings('error',category=ConvergenceWarning);assert read(P/'status.json')['state']=='COMPLETE';assert sha(S/'splits.json')==read(P/'plan.json')['splits_sha256'];sp=read(S/'splits.json');ids=np.load(S/'data/IDs.npy');y=np.load(S/'data/Y.npy');pred=np.full(len(y),np.nan);foldrows=[];scoresall=[];proof=[]
 plan=dict(source_features=str(P),splits_sha256=sha(S/'splits.json'),candidate_C=[.001,.003,.01],selection='minimum inner validation BCE; tie smaller C',classifier=dict(solver='liblinear',max_iter=10000,tol=1e-6,penalty='l2'),max_new_classifier_fits=40,backbone_updates=0,feature_extraction_runs=0,scope='fixed development follow-up; no automatic further expansion')
 put(D/'plan.json',plan)
 for f,s in enumerate(sp):
  out=D/f'f{f}';out.mkdir(exist_ok=True);src=P/f'f{f}'
  for scope,tr,va in [('inner',s['inner_train'],s['inner_validation']),('outer',s['outer_train'],s['outer_test'])]:
   meta=read(src/f'{scope}_features.json');a=np.load(src/f'{scope}_features.npz');assert a['features'].shape==(len(tr)+len(va),1152);assert np.array_equal(a['ids'],ids[tr+va]);assert meta['normalization_training_ids']==ids[tr].tolist();assert meta['checkpoint_sha256']==sha(Path(meta['checkpoint']));assert not set(tr)&set(va)
   proof.append(dict(fold=f,scope=scope,feature_sha256=sha(src/f'{scope}_features.npz'),checkpoint_sha256=meta['checkpoint_sha256']))
  def pipeline(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,solver='liblinear',max_iter=10000,tol=1e-6,random_state=2026102100+f))
  x=np.load(src/'inner_features.npz')['features'];tr=s['inner_train'];va=s['inner_validation'];scores=[]
  for c in plan['candidate_C']:
   m=pipeline(c);m.fit(x[:len(tr)],y[tr]);v=metrics(y[va],m.predict_proba(x[len(tr):])[:,1]);t=metrics(y[tr],m.predict_proba(x[:len(tr)])[:,1]);r=dict(C=c,**v);scores.append(r);scoresall.append(dict(fold=f,C=c,**{'train_'+k:z for k,z in t.items()},**{'val_'+k:z for k,z in v.items()}));assert m[0].n_samples_seen_==len(tr)
   if c==.01:
    old=next(z for z in read(src/'selection.json')['candidates'] if z['C']==.01);assert abs(old['BCE']-v['BCE'])<1e-10 and abs(old['AUROC']-v['AUROC'])<1e-10
  best=min(scores,key=lambda r:(r['BCE'],r['C']));put(out/'selection.json',dict(selected=best,candidates=scores,outer_test_used=False));ot=s['outer_train'];te=s['outer_test'];x=np.load(src/'outer_features.npz')['features'];m=pipeline(best['C']);m.fit(x[:len(ot)],y[ot]);assert m[0].n_samples_seen_==len(ot);assert np.allclose(m[0].mean_,x[:len(ot)].astype(float).mean(0),rtol=1e-10,atol=1e-10);p=m.predict_proba(x[len(ot):])[:,1];assert np.isnan(pred[te]).all();pred[te]=p;train=metrics(y[ot],m.predict_proba(x[:len(ot)])[:,1]);test=metrics(y[te],p)
  joblib.dump(m,out/'classifier.joblib');np.savez_compressed(out/'outer_predictions.npz',ids=ids[te],labels=y[te],probability=p);foldrows.append(dict(fold=f,C=best['C'],**{'train_'+k:v for k,v in train.items()},**{'test_'+k:v for k,v in test.items()}));put(out/'completion.json',dict(status='COMPLETE',classifier_fits=4,selected=best,train=train,test=test,time_unix=time.time()));put(D/'status.json',dict(state='RUNNING',completed_folds=f+1));print('COMPLETE_FOLD',f,flush=True)
 assert np.isfinite(pred).all();table(D/'OOF.csv',[dict(subject_id=int(i),label=int(t),probability=float(p)) for i,t,p in zip(ids,y,pred)]);table(D/'fold_results.csv',foldrows);table(D/'inner_candidates.csv',scoresall)
 summary=[];previous={}
 for name,path in [('R0',S/'R0_OOF.csv'),('R1_288',Q/'OOF.csv'),('R1_1152_original',P/'OOF.csv')]:
  with path.open() as h:a=list(csv.DictReader(h))
  assert [int(z['subject_id']) for z in a]==ids.tolist() and [int(z['label']) for z in a]==y.tolist();p=np.array([float(z['probability']) for z in a]);previous[name]=p;summary.append(dict(route=name,**metrics(y,p)))
 summary.append(dict(route='R1_1152_stronger_regularization',**metrics(y,pred)));table(D/'comparison.csv',summary)
 rng=np.random.default_rng(20261021);g=[np.where(y==v)[0] for v in [0,1]];intervals=[]
 for name in ['R1_1152_original','R0']:
  a=[]
  for _ in range(2000):
   ix=np.concatenate([rng.choice(z,len(z),replace=True) for z in g]);a.append(roc_auc_score(y[ix],pred[ix])-roc_auc_score(y[ix],previous[name][ix]))
  intervals.append(dict(comparison='stronger_reg minus '+name,delta=float(roc_auc_score(y,pred)-roc_auc_score(y,previous[name])),conditional95CI=np.quantile(a,[.025,.975]).tolist()))
 put(D/'results.json',dict(status='COMPLETE',summary=summary,comparisons=intervals,limitation='post-hoc developmental follow-up, one seed; intervals condition on fixed predictions, not training/split uncertainty',time_unix=time.time()));put(D/'audit_receipt.json',dict(status='PASS',OOF_subjects=966,baseline_C001_inner_scores_reproduced=True,train_only_scalers=True,new_classifier_fits=40,backbone_updates=0,features_reused=proof));put(D/'status.json',dict(state='COMPLETE',completed_folds=10,time_unix=time.time()))
if __name__=='__main__':
 try:run()
 except BaseException as e:put(D/'status.json',dict(state='FAILED',error=repr(e)));raise
