from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
from models import make
from common import put,sha
arm,seed=sys.argv[1],int(sys.argv[2]);root=R/'refits'/f'{arm}_full610_s{seed}';ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False);assert (root/'completion.json').exists();norm=ck['norm'];assert len(norm['fit_subject_ids'])==610
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
raw=np.load(R/'downstream_data/FC.npy');ids=np.load(R/'downstream_data/IDs.npy');assert not set(ids)&set(norm['fit_subject_ids']);m=make(5,seed,arm).cuda();out=R/'features';records=[]
for kind,path in [('pretrained',root/'last.pt'),('random',root/'initial.pt')]:
 c=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(c['model']);m.eval();m.requires_grad_(False);before=w.old.state_hash(m);xs=[]
 with torch.no_grad():
  for st in range(0,len(raw),8):
   x=torch.from_numpy(((raw[st:st+8]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda();xs.append(m.encode_full(x).mean((1,2)).cpu().numpy())
 assert before==w.old.state_hash(m);x=np.concatenate(xs);assert x.shape==(290,576 if arm.endswith('L') else 288) and np.isfinite(x).all();name=f'{arm}_s{seed}_{kind}';np.save(out/f'{name}.npy',x);records.append({'name':name,'shape':list(x.shape),'feature_sha256':sha(out/f'{name}.npy'),'checkpoint':str(path),'checkpoint_sha256':sha(path),'frozen_verified':True})
put(out/f'{arm}_s{seed}.json',{'status':'COMPLETE','records':records,'ids_sha256':sha(R/'downstream_data/IDs.npy')})
