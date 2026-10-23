from pathlib import Path
import sys,json,torch,numpy as np,time,types
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import worker as w
from models import make,ORDER,ordered_forward
from plateau import Plateau
from common import put
p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=.0001);s=Plateau(opt)
for i in range(5):s.step(1.);assert opt.param_groups[0]['lr']==.0001
s.step(1.);assert opt.param_groups[0]['lr']==.00005 and s.events==1
s.step(1.);assert s.bad==0 and s.cooldown==0
for i in range(4):s.step(1.);assert s.events==1
saved=s.state_dict();s.step(1.);assert s.events==2
s2=Plateau(torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=.00005));s2.load_state_dict(saved);s2.step(1.);assert s2.state_dict()==s.state_dict()
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
raw=np.load(R/'data/FC.npy');x=torch.tensor(raw[:2],dtype=torch.float32)[:,None].cuda();records={}
for arm in ['MS','TS','ML','TL','MSO','MSZ','TSZ']:
 m=make(0,1,arm).cuda();mask=w.masks(m,['a','b'],0,'check',.5).cuda();alt=x.clone();alt[mask]=99.;assert torch.equal(m.embedded_input(x,mask),m.embedded_input(alt,mask))
 z=m(x,pixel_mask=mask);w.old.loss_mae(z).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());records[arm]={'parameters':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
 if arm.endswith('Z'):assert m.encoder.roi_identity.grad is None and m.encoder.roi_identity.abs().max()==0
 del m,z;torch.cuda.empty_cache()
# Mamba scan changes order only; shared tensors match baseline exactly.
a=make(0,1,'MS');b=make(0,1,'MSO');assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items());assert torch.equal(ORDER[torch.argsort(ORDER)],torch.arange(116))
# A full noncausal Transformer should be equivariant to sequence permutation with identity attached.
m=make(0,1,'TS').cuda().eval()
with torch.no_grad():
 y=m.encode_full(x);m.encoder.register_buffer('scan_order',ORDER.cuda());m.encoder.register_buffer('inverse_order',torch.argsort(ORDER).cuda());m.encoder.forward=types.MethodType(ordered_forward,m.encoder);yp=m.encode_full(x);delta=float((y-yp).abs().max());assert delta<1e-5
put(R/'engineering_static.json',{'status':'PASS','models':records,'plateau_exact5_bad_epochs_and_resume':True,'hidden_value_invariance':True,'scan_order_inverse':True,'transformer_permutation_max_difference':delta,'time':time.time()});print('ENGINEERING_PASS',flush=True)
