from pathlib import Path
import numpy as np,hashlib,json
R=Path(__file__).resolve().parent

def rng_for(kind,fold,seed,epoch,sid):
 return np.random.default_rng(int(hashlib.sha256(f'augv1|{kind}|{fold}|{seed}|{epoch}|{sid}'.encode()).hexdigest()[:16],16))
def selected(f,s,e,sid):return rng_for('switch',f,s,e,sid).random()<.5
class Augment:
 def __init__(self,arm,fold,seed,ids,raw,norm):self.arm,self.fold,self.seed,self.ids,self.raw,self.norm=arm,fold,seed,ids,raw,norm;self.cache={}
 def arrays(self,q,epoch,mask):
  target=np.array(self.raw[q],copy=True);inputs=target.copy();changed=0;noise_sq=0.;noise_n=0;clipped=0;window_starts=[]
  for k,i in enumerate(q):
   sid=str(self.ids[i])
   if self.arm not in ['G1','G2'] or not selected(self.fold,self.seed,epoch,sid):continue
   changed+=1
   if self.arm=='G1':
    u,v=np.triu_indices(116,1);r=rng_for('noise',self.fold,self.seed,epoch,sid);eps=r.normal(0,.02,len(u));visible=~mask[k,0,u,v];u,v,eps=u[visible],v[visible],eps[visible];a=inputs[k,u,v]+eps;clipped+=int(np.sum(np.abs(a)>1));a=np.clip(a,-1,1);actual=a-inputs[k,u,v];inputs[k,u,v]=a;inputs[k,v,u]=a;noise_sq+=float(np.sum(actual**2));noise_n+=len(a)
   else:
    if sid not in self.cache:self.cache[sid]=np.load(R/'windows'/f'{sid}.npy',mmap_mode='r')
    w=self.cache[sid];start=int(rng_for('window',self.fold,self.seed,epoch,sid).integers(len(w)));target[k]=w[start];inputs[k]=target[k];window_starts.append([sid,start])
  return inputs,target,{'augmented':changed,'subjects':len(q),'noise_squared_sum':noise_sq,'noise_edges':noise_n,'noise_clipped_edges':clipped,'windows':window_starts}
