import argparse,warnings,pickle
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();j=read(a.job);out=R/'runs'/j['name'];out.mkdir(exist_ok=True,parents=True)
 assert read(R/'control/gate_A.json')['status']=='PASS';d=np.load(R/'data/development.npz');assert sha(R/'data/development.npz')==read(R/'control/prepared.json')['development_hash'];fc=d['fc'];y=d['y'];ii,jj=np.triu_indices(116,1);x=fc[:,ii,jj];sp=read(R/'data/cv.json');tr=np.array(sp[j['fold']]['train']) if j['fold']>=0 else np.arange(len(y));va=np.array(sp[j['fold']]['validation']) if j['fold']>=0 else np.array([],dtype=int)
 scaler=StandardScaler().fit(x[tr]);m=LogisticRegression(C=j['C'],solver='lbfgs',penalty='l2',max_iter=5000,tol=1e-6,random_state=20261027)
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');m.fit(scaler.transform(x[tr]),y[tr])
 warn=[str(w.message) for w in ww if issubclass(w.category,ConvergenceWarning)];put(out/'warnings.json',warn)
 if warn:put(out/'status.json',{'status':'BLOCKED','reason':'LR_convergence_warning'});raise RuntimeError('LR convergence gate')
 q=out/'model.pkl.tmp';q.write_bytes(pickle.dumps({'model':m,'scaler':scaler,'job':j},protocol=5));q.replace(out/'model.pkl')
 if len(va):
  z=m.decision_function(scaler.transform(x[va]));np.savez(out/'validation.npz',logits=z,labels=y[va],ids=d['ids'][va]);put(out/'metrics.json',metrics(y[va],z))
 put(out/'completion.json',{'status':'COMPLETE','model_sha256':sha(out/'model.pkl'),'test_accessed':False});put(out/'status.json',{'status':'COMPLETE'})
