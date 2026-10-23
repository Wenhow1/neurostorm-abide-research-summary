"""冻结既有中间检查点；仅完整FC编码，不调用解码器、不反向传播。"""
import os,sys,json,argparse,time,signal
from pathlib import Path
import numpy as np
from frozen_helpers import sha,put,atomic_npy,ThermalGuard,validate_source,flatten_tokens,GPU_UUID
R=Path(__file__).resolve().parent

def run(a):
 p=json.loads((R/'protocol.json').read_text());src=Path(p['data']['source_root']); old=Path(p['reference_root'])
 name=f'{a.arm}_s{a.seed}_e{a.epoch}'; ref=src/'refits'/f'{a.arm}_full610_s{a.seed}'
 cp=ref/(f'epoch_{a.epoch}.pt' if not a.pilot else 'last.pt'); ch=sha(cp)
 assert ch==p['checkpoints'][f'{a.arm}_s{a.seed}'][str(a.epoch) if not a.pilot else 'last']['sha256']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
 guard=ThermalGuard(R/'feature_status'/f'{name}.json',R/'feature_events'/f'{name}.jsonl',{'name':name},wait_limit=21600)
 guard.check(phase='load')
 sys.path.insert(0,str(src));import worker,models,torch
 signal.signal(signal.SIGUSR1,signal.SIG_DFL);signal.signal(signal.SIGTERM,signal.SIG_DFL)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ck=torch.load(cp,map_location='cpu',weights_only=False); norm=ck['norm']
 raw=np.load(src/'downstream_data/FC.npy');ids=np.load(src/'downstream_data/IDs.npy');validate_source(raw,ids,norm,np)
 assert sha(src/'downstream_data/IDs.npy')==p['input_hashes']['ids']
 assert sha(src/'downstream_data/FC.npy')==p['fc_sha256']
 assert norm==json.loads((ref/'config.json').read_text())['norm']
 if not a.pilot: assert int(ck['epoch'])==a.epoch
 model=models.make(5,a.seed,a.arm).cuda().eval();model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False);del ck
 before=worker.old.state_hash(model)
 def forbidden(*x):raise AssertionError('Decoder invoked')
 hook=model.decoder.register_forward_pre_hook(forbidden)
 n=8 if a.pilot else 290;d=576 if a.arm.endswith('L')else 288;parts=[]
 for off in range(0,n,2):
  guard.check(phase='encode',completed_subjects=off,total_subjects=n)
  with torch.inference_mode():
   x=torch.from_numpy(((raw[off:min(off+2,n)]-norm['mean'])/norm['std']).astype('float32'))[:,None].cuda()
   z=model.encode_full(x).cpu().numpy();parts.append(flatten_tokens(z,d,np));del x,z
 flat=np.concatenate(parts)
 assert np.isfinite(flat).all() and before==worker.old.state_hash(model) and ch==sha(cp)
 assert all(not x.requires_grad and x.grad is None for x in model.parameters())
 if a.pilot:
  expected=np.load(old/'features'/f'{a.arm}_s{a.seed}_pretrained_flatten.npy',mmap_mode='r')[:n]
  assert np.allclose(flat,expected,atol=2e-6,rtol=1e-5)
  put(R/'pilot'/f'{a.arm}_s{a.seed}.json',{'status':'PASS','max_error':float(np.abs(flat-expected).max()),'checkpoint_sha256':ch})
 else:
  path=R/'features'/f'{name}_flatten.npy';atomic_npy(path,flat,np)
  put(R/'receipts'/f'{name}.json',{'status':'COMPLETE','name':name,'checkpoint_sha256':ch,'epoch':a.epoch,'ids_sha256':sha(src/'downstream_data/IDs.npy'),'frozen_verified':True,'no_decoder':True,'no_backward':True,'max_temperature_C':guard.max_temperature,'records':[{'name':f'{name}_flatten','feature_sha256':sha(path),'shape':list(flat.shape)}]})
 hook.remove()
 print(name,'PASS',flush=True)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--arm',required=True);q.add_argument('--seed',type=int,required=True);q.add_argument('--epoch',type=int,required=True);q.add_argument('--pilot',action='store_true');run(q.parse_args())
