import sys,time,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
from common import read,put,sha
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();name=f'ROIR_f{a.fold}_s{a.seed}';out=R/('pilot_cross_eval' if a.pilot else 'cross_eval')/name;out.mkdir(exist_ok=True,parents=True)
if (out/'completion.json').exists():sys.exit(0)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
A=Path(read(R/'protocol.json')['step1_reference']);paths={'step1_block':A/'runs'/f'ROI_f{a.fold}_s{a.seed}'/'epoch_100.pt','step2_rows':R/'runs'/name/'epoch_100.pt'}
if a.pilot:paths={'step1_block':A/'pilot/ROI_full/last.pt','step2_rows':R/'pilot/ROIR_full/last.pt'}
# First-step work is independent; never write into or restart its directory.
needed=list(paths.values())
if not a.pilot:needed.append(A/'readout'/f'ROI_f{a.fold}_s{a.seed}'/'completion.json')
while not all(q.is_file()for q in needed):time.sleep(3)
raw=np.load(R/'source/data/FC.npy',mmap_mode='r');ids=np.load(R/'source/data/IDs.npy');sp=read(R/'source/splits.json')[a.fold];tr=sp['inner_train'];va=sp['inner_validation'];prior=np.asarray(raw[tr]).mean(0);summ=[]
for label,path in paths.items():
 ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==(2 if a.pilot else 100) and ck['norm']['fit_subject_ids']==ids[tr].tolist();norm=ck['norm'];m=w.make(a.fold,a.seed,'ROIR').cuda().eval();m.load_state_dict(ck['model']);m.requires_grad_(False);before=w.old.state_hash(m)
 with torch.inference_mode():
  for ratio in [.5,.75]:
   rows=[];mh=hashlib.sha256()
   for view in range(3):
    for start in range(0,len(va),8):
     q=va[start:start+8];mask=w.row_masks(m,ids[q],view,'cpac966_roi_fixed_eval_v1',ratio);mh.update(mask.numpy().tobytes());x=torch.from_numpy(((raw[q]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda();z=m(x,pixel_mask=mask.cuda());score=z['score_mask'][:,0].cpu().numpy();pr=z['prediction'][:,0].cpu().numpy()*norm['std']+norm['mean']
     for k,i in enumerate(q):
      s=score[k];t=raw[i][s];p=pr[k][s];r=t-prior[s];v=p-prior[s];rows.append({'subject_id':str(ids[i]),'view':view,'MSE':float(np.mean((p-t)**2)),'sym_MSE':float(np.mean((((pr[k]+pr[k].T)*.5)[s]-t)**2)),'meanFC_MSE':float(np.mean(r**2)),'residual_correlation':float(np.corrcoef(r,v)[0,1]),'residual_RMS_ratio':float(np.sqrt(np.mean(v**2)/np.mean(r**2))),'score_fraction':float(s.sum()/(116*115))})
   rec={'model':label,'mask':'row_column','ratio':ratio,'mask_sha256':mh.hexdigest(),'subjects':len(va),'views':3,**{k:float(np.mean([r[k]for r in rows]))for k in ['MSE','sym_MSE','meanFC_MSE','residual_correlation','residual_RMS_ratio','score_fraction']}}
   put(out/f'{label}_rows{int(ratio*100)}.json',{**rec,'rows':rows});summ.append(rec)
 assert w.old.state_hash(m)==before
 del m,ck;torch.cuda.empty_cache()
for ratio in [.5,.75]:assert len({x['mask_sha256']for x in summ if x['ratio']==ratio})==1
put(out/'completion.json',{'status':'COMPLETE','rows':summ,'weights_unchanged':True,'outer_test_accessed':False,'no_training':True,'time':time.time()})
