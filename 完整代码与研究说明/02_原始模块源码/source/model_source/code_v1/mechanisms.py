"""Engineering candidate, NOT a frozen formal experiment.

2D reductions of NeuroSTORM commit 4971d5d. Retains raw symmetric masking
and edge-aligned overlap geometry. No image/time/anatomical equivalence claimed.
"""
import sys, math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'base_code'))
from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D, overlap_masked_mse
from neurostorm_2d.models.overlap_backbone import OverlapPatchMerging2D, OverlapPatchExpanding2D
from neurostorm_2d.models.mamba_block import _real_mamba_factory


class OfficialWindowBlock2D(nn.Module):
    """LN/pad/roll/window-Mamba/reverse/crop/residual/LN-MLP-residual.

    Official SwinTransformerBlock4D.forward_part1 does not use attn_mask
    in the Mamba branch. That behavior is preserved, including cyclic wrap.
    No STRD: selected official configuration defaults use_strd=False.
    """
    def __init__(self, dim, window=4, shift=0):
        super().__init__()
        self.dim = dim
        self.window = (window, window) if isinstance(window, int) else tuple(window)
        self.shift = (shift, shift) if isinstance(shift, int) else tuple(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = _real_mamba_factory()(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
        self.register_parameter('prompt', None)

    def add_task_prompt(self, length):
        assert self.prompt is None
        self.prompt = nn.Parameter(self.norm1.weight.new_empty(1, length, self.dim))
        nn.init.trunc_normal_(self.prompt, std=.02)

    def forward(self, x, *, use_prompts=False):
        b,h,w,c = x.shape
        wh,ww = min(h,self.window[0]), min(w,self.window[1])
        sh = self.shift[0] if h > self.window[0] else 0
        sw = self.shift[1] if w > self.window[1] else 0
        z = self.norm1(x)
        ph,pw = (-h)%wh, (-w)%ww
        z = F.pad(z, (0,0,0,pw,0,ph))
        hp,wp = h+ph,w+pw
        if sh or sw: z = torch.roll(z, (-sh,-sw), (1,2))
        z = z.reshape(b,hp//wh,wh,wp//ww,ww,c).permute(0,1,3,2,4,5).reshape(-1,wh*ww,c)
        if use_prompts:
            assert self.prompt is not None
            n = self.prompt.shape[1]
            z = torch.cat([self.prompt.expand(z.shape[0],-1,-1),z],dim=1)
            z = self.mamba(z)[:,n:]
        else: z = self.mamba(z)
        z = z.reshape(b,hp//wh,wp//ww,wh,ww,c).permute(0,1,3,2,4,5).reshape(b,hp,wp,c)
        if sh or sw: z = torch.roll(z, (sh,sw), (1,2))
        z = x + z[:,:h,:w]
        return z + self.mlp(self.norm2(z))


class OfficialStage2D(nn.Module):
    def __init__(self,dim,shape,depth,window,downsample,full=False):
        super().__init__()
        self.blocks = nn.ModuleList([OfficialWindowBlock2D(dim,shape if full else window,0 if full or i%2==0 else window//2) for i in range(depth)])
        self.downsample = OverlapPatchMerging2D(dim,shape) if downsample else None
    def forward(self,x,*,use_prompts=False):
        for block in self.blocks: x=block(x,use_prompts=use_prompts)
        return self.downsample(x) if self.downsample is not None else x


class OfficialEncoder2D(nn.Module):
    """E1 selected YAML: width36, depths2/2/6/2, window4, last full.

    Retains edge-aligned 2D merging because 57/115 grids are odd. Temporal
    positions have no counterpart in FC; only spatial positions are adapted.
    """
    def __init__(self,shape):
        super().__init__()
        self.stage_dims=(36,72,144,288)
        self.depths=(2,2,6,2)
        self.stages=nn.ModuleList(); self.pos_embeds=nn.ParameterList(); shapes=[]
        for i,(dim,dep) in enumerate(zip(self.stage_dims,self.depths)):
            shapes.append(shape)
            pos=nn.Parameter(torch.empty(1,*shape,dim)); nn.init.trunc_normal_(pos,std=.02)
            self.pos_embeds.append(pos)
            stage=OfficialStage2D(dim,shape,dep,4,i<3,full=i==3)
            self.stages.append(stage)
            if stage.downsample is not None: shape=stage.downsample.output_shape
        self.stage_shapes=tuple(shapes)
    def iter_blocks(self):
        for s in self.stages: yield from s.blocks
    def add_task_prompts(self,n):
        for b in self.iter_blocks(): b.add_task_prompt(n)
    def forward(self,x,*,use_prompts=False):
        for pos,stage in zip(self.pos_embeds,self.stages): x=stage(x+pos,use_prompts=use_prompts)
        return x


class OfficialExpand2D(OverlapPatchExpanding2D):
    def __init__(self,inc,outc,shape):
        super().__init__(inc,outc,shape)
        # Official expansion has no bias. In 2D, 4*(C/2)=2C outputs.
        self.proj=nn.Linear(inc,4*outc,bias=False)


class OfficialDecoder2D(nn.Module):
    """first expand -> block(s)/expand -> block(s)/expand -> block(s) -> LN/head.

    Official BasicLayerUp depths run encoder stages 2,1,0 in that order.
    Only real native grid cells are averaged when odd merge groups overlap.
    """
    def __init__(self,encoder,depths,area):
        super().__init__(); ds=encoder.stage_dims; ss=encoder.stage_shapes
        self.first_patch_expanding=OfficialExpand2D(ds[3],ds[2],ss[2])
        self.blocks=nn.ModuleList(); self.expands=nn.ModuleList()
        for i in (2,1,0):
            self.blocks.append(nn.ModuleList([OfficialWindowBlock2D(ds[i],4,0 if j%2==0 else 2) for j in range(depths[i])]))
            if i>0:self.expands.append(OfficialExpand2D(ds[i],ds[i-1],ss[i-1]))
        self.norm_up=nn.LayerNorm(ds[0]); self.decoder_pred=nn.Linear(ds[0],area)
    def forward(self,x):
        x=self.first_patch_expanding(x)
        for i,blocks in enumerate(self.blocks):
            for b in blocks:x=b(x)
            if i<len(self.expands):x=self.expands[i](x)
        return self.decoder_pred(self.norm_up(x)).flatten(1,2)


class MechanismMAE(OverlapMaskedAutoencoder2D):
    def __init__(self,encoder='E0',decoder='D0',mask='M0',patch=4,stride=2):
        self.encoder_kind,self.decoder_kind,self.mask_kind=encoder,decoder,mask
        dim=16 if encoder=='E0' else 36
        depths=(2,2,1,1) if encoder=='E0' else (2,2,6,2)
        # The full constructor is deterministically seeded by fold and seed.
        # E1 replaces the initial legacy-shaped encoder; D1 replaces D0.
        # Pairwise A/B jobs load the exact same resulting MAE checkpoint.
        super().__init__(patch_size=patch,stride=stride,embed_dim=dim,depths=depths,
                         window_sizes=(8,8,8,8),mask_unit=4,mask_policy='paired_symmetric')
        if encoder=='E1': self.encoder=OfficialEncoder2D(self.geometry.output_shape)
        if decoder=='D1': self.decoder=OfficialDecoder2D(self.encoder,depths,patch*patch)
        if mask=='M1': self.mask_token=nn.Parameter(torch.zeros(1,1,dim))
        self.depths=depths
    def embedded_input(self,image,pixel_mask):
        tokens=super().embedded_input(image,pixel_mask)
        if self.mask_kind=='M1':
            full=self.geometry.gather(pixel_mask.permute(0,2,3,1)).all(dim=2)
            flat=tokens.flatten(1,2)
            flat=torch.where(full,self.mask_token.to(flat.dtype),flat)
            tokens=flat.reshape_as(tokens)
        return tokens


class Classifier(nn.Module):
    def __init__(self,mae,mode):
        super().__init__(); assert mode in ('T0','T1','T2')
        self.mae=mae; self.mode=mode; mae.requires_grad_(False)
        if mode!='T0':
            mae.encoder.add_task_prompts(50)
            for b in mae.encoder.iter_blocks(): b.prompt.requires_grad_(True)
        if mode=='T2':mae.encoder.stages[-1].requires_grad_(True)
        self.head=nn.Linear(mae.encoder.stage_dims[-1],1)
    def train(self,mode=True):
        super().train(mode); self.mae.eval(); return self
    def forward(self,image):
        return self.head(self.mae.encode_full(image,use_prompts=self.mode!='T0').mean((1,2))).squeeze(-1)
