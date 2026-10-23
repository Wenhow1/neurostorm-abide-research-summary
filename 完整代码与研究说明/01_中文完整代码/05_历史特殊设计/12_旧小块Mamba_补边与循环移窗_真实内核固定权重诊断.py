# 12_旧小块Mamba_补边与循环移窗_真实内核固定权重诊断
# 归档主题：主题04
# 保留真实Mamba内核的历史边界诊断：baseline、去补边token、拆分循环跨界片段、两者同时。
# 输入/输出：旧B第0折两个种子，32个预定内部验证受试者，完整FC与固定50%方块两个条件。
# 本实验改动：只临时替换编码器forward，不更新参数；boundary只作用15×15网格的shift2块；decoder保持原实现。
# 参数在哪里：plan()、forward()、evaluate()；历史导入依赖主题03的S01 worker.py，其make(...,G0)构造B。
# 证据范围：此文件来自本地此前已保存且SHA一致的历史诊断；不是本轮重新运行。存在跨主题依赖和顶层执行。
# 源文件：04_位置编码与补边移窗/mamba_boundary_diagnostic_20261022/audit.py
# 原始字节 SHA256：6c234fd6d73a4c0b716010de021228fe99f24dc85c9b4a069328e026a2440027
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

from pathlib import Path
import sys, json, time, types, hashlib, subprocess, os
import numpy as np, torch
from torch.nn import functional as F
from collections import defaultdict
ROOT = Path('/home/wenhao/neurostorm_2d/experiments/cpac966_downsample_s01_20261022')
sys.path.insert(0, str(ROOT))
import worker as w
OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cache = {}
rows = []
local = []
checks = []
temps = []

def put(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def thermal():
    for attempt in range(60):
        t = int(subprocess.check_output(['nvidia-smi',
             '-i',
             '1',
             '--query-gpu=temperature.gpu',
             '--format=csv,noheader,nounits'],
             text=True).strip())
        temps.append(t)
        if t < 80:
            return
        put('thermal_pause.json', {'temperature': t, 'time': time.time()})
        time.sleep(5)
    raise RuntimeError('GPU1 temperature remained >=80C; stopped without training')

def plan(h, ww, sh, mode):
    key = (h, ww, sh, mode)
    if key in cache:
        return cache[key]
    hp = h + -h % ww
    valid = np.zeros((hp, hp), bool)
    valid[:h, :h] = True
    rr, cc = np.indices((hp, hp))
    region = (rr < sh).astype(int) * 2 + (cc < sh).astype(int) if sh else np.zeros((hp, hp), int)

    def pack(a):
        a = np.roll(a, (-sh, -sh), (0, 1))
        return a.reshape(hp // ww, ww, hp // ww, ww).transpose(0, 2, 1, 3).reshape(-1, ww * ww)
    vs, rs = (pack(valid), pack(region))
    group = defaultdict(list)
    for i in range(len(vs)):
        for regionid in np.unique(rs[i]) if mode in ['boundary', 'both'] else [None]:
            take = np.ones(ww * ww, bool) if regionid is None else rs[i] == regionid
            if mode in ['pad', 'both']:
                take &= vs[i]
            idx = tuple(np.flatnonzero(take).tolist())
            if idx:
                group[idx].append(i)
    result = (len(vs),
         [(torch.tensor(ix,
         device='cuda'),
         torch.tensor(wi,
         device='cuda')) for ix,
         wi in group.items()])
    cache[key] = result
    return result

@torch.no_grad()
def forward(self, x, *, use_prompts=False):
    assert not use_prompts
    b, h, wd, c = x.shape
    assert h == wd
    ww = min(h, self.window[0])
    sh = self.shift[0] if h > self.window[0] else 0
    z = self.norm1(x)
    ph = -h % ww
    z = F.pad(z, (0, 0, 0, ph, 0, ph))
    hp = h + ph
    if sh:
        z = torch.roll(z, (-sh, -sh), (1, 2))
    z = z.reshape(b, hp // ww, ww, hp // ww, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ww * ww, c)
    mode = self._audit_mode
    if mode == 'boundary' and (not (h == 15 and sh == 2)):
        mode = 'baseline'
    if mode == 'both' and (not (h == 15 and sh == 2)):
        mode = 'pad'
    if mode == 'baseline':
        z = self.mamba(z)
    else:
        nw, groups = plan(h, ww, sh, mode)
        out = torch.zeros_like(z)
        for idx, win in groups:
            wi = (torch.arange(b, device=x.device)[:, None] * nw + win[None, :]).flatten()
            y = self.mamba(z[wi[:, None], idx[None, :]])
            out[wi[:, None], idx[None, :]] = y
        z = out
    z = z.reshape(b, hp // ww, hp // ww, ww, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, hp, c)
    if sh:
        z = torch.roll(z, (sh, sh), (1, 2))
    z = x + z[:, :h, :h]
    return z + self.mlp(self.norm2(z))

def evaluate(m, x, mask):
    if bool(mask.any()):
        return m(x, pixel_mask=mask)
    feat = m.encode_full(x, use_prompts=False)
    patch = m.decoder(feat)
    pred = m.geometry.average(patch.unsqueeze(-1)).permute(0, 3, 1, 2)
    return {'prediction': pred, 'score_mask': m.off_diagonal.expand_as(x)}

def metrics(a, b):
    a = a.reshape(a.shape[0], -1).double()
    b = b.reshape(b.shape[0], -1).double()
    return {'relative_L2': float(((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).mean()),
         'cosine': float(F.cosine_similarity(a,
         b,
         dim=1).mean()),
         'max_abs': float((b - a).abs().max())}
raw = np.load(ROOT / 'source/data/FC.npy', mmap_mode='r')
ids = np.load(ROOT / 'source/data/IDs.npy')
sp = json.loads((ROOT / 'source/splits.json').read_text())[0]
q = sorted(sp['inner_validation'], key=lambda i: int(ids[i]))[:32]
assert not set(q) & set(sp['inner_train']) and (not set(q) & set(sp['outer_test']))
put('protocol.json',
     {'models': 'B fold0 seeds1,2 epoch100',
     'subjects': 32,
     'selection': 'first32 internal validation by subject ID, fixed before outcome',
     'input_modes': ['fullFC',
     'fixed mask50 view0'],
     'variants': {'baseline': 'original encoder',
     'pad': 'delete padding tokens within each window; keep true token order',
     'boundary': 'only 15x15 shifted blocks: split wrapped components into separate Mamba sequences; keep padding slots within components; 57/29 grids unchanged',
     'both': 'both operations'},
     'scope': 'encoder only; decoder and parameters unchanged; inference only; no classifier fitted',
     'GPU': 1,
     'pause_temperature': 80,
     'ids': ids[q].tolist()})
with torch.inference_mode():
    for seed in [1, 2]:
        thermal()
        path = Path('/home/wenhao/neurostorm_2d/experiments/cpac966_decoder_mask_20261022/runs') / f'B_f0_s{seed}/epoch_100.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = w.make(0, seed, 'G0')
        m.load_state_dict(ck['model'], strict=True)
        norm = ck['norm']
        assert norm['fit_subject_ids'] == ids[sp['inner_train']].tolist()
        m = m.cuda().eval().requires_grad_(False)
        before = w.old.state_hash(m)
        blocks = list(m.encoder.iter_blocks())
        original = [b.forward for b in blocks]
        for start in range(0, len(q), 4):
            thermal()
            qq = q[start:start + 4]
            x = torch.from_numpy(((raw[qq] - norm['mean']) / norm['std']).astype('float32'))[:, None].cuda()
            for inputmode in ['fullFC', 'masked50']:
                mask = torch.zeros_like(x,
                     dtype=torch.bool) if inputmode == 'fullFC' else w.masks(m,
                     ids[qq],
                     0,
                     'cpac966_fullcv_fixed_eval_v1',
                     0.5).cuda()
                for b, fn in zip(blocks, original):
                    b.forward = fn
                feats = []
                hk = m.encoder.register_forward_hook(lambda mod, inp, out: feats.append(out.detach().clone()))
                z = evaluate(m, x, mask)
                hk.remove()
                bp = z['prediction'].detach().clone()
                bf = feats[0]
                score = z['score_mask']
                if inputmode == 'fullFC':
                    score = (~torch.eye(116, dtype=torch.bool, device='cuda'))[None, None].expand_as(x)
                for b in blocks:
                    b._audit_mode = 'baseline'
                    b.forward = types.MethodType(forward, b)
                ctl = evaluate(m, x, mask)['prediction']
                err = float((ctl - bp).abs().max())
                assert err < 1e-05, err
                checks.append(err)
                if start == 0 and inputmode == 'masked50':
                    for b, fn in zip(blocks, original):
                        b.forward = fn
                    captures = {}
                    handles = []
                    for ix in [1, 3, 5]:
                        handles.append(blocks[ix].register_forward_pre_hook(lambda mod,
                             inp,
                             ix=ix: captures.__setitem__(ix,
                             inp[0].detach().clone())))
                    m(x, pixel_mask=mask)
                    for h in handles:
                        h.remove()
                    for ix, xx in captures.items():
                        base = original[ix](xx)
                        for mode in ['pad', 'boundary', 'both']:
                            blocks[ix]._audit_mode = mode
                            alt = forward(blocks[ix], xx)
                            local.append({'seed': seed,
                                 'block_index': ix,
                                 'grid': xx.shape[1],
                                 'variant': mode,
                                 **metrics(base,
                                 alt)})
                for mode in ['baseline', 'pad', 'boundary', 'both']:
                    if mode == 'baseline':
                        pred, feat = (bp, bf)
                    else:
                        for b in blocks:
                            b._audit_mode = mode
                            b.forward = types.MethodType(forward, b)
                        fs = []
                        hk = m.encoder.register_forward_hook(lambda mod, inp, out: fs.append(out.detach().clone()))
                        a = evaluate(m, x, mask)
                        hk.remove()
                        pred, feat = (a['prediction'], fs[0])
                    for i, sid in enumerate(ids[qq]):
                        sc = score[i]
                        orig = bp[i][sc] * norm['std'] + norm['mean']
                        v = pred[i][sc] * norm['std'] + norm['mean']
                        target = torch.as_tensor(np.array(raw[qq[i]]), device='cuda')[None][sc]
                        p0 = bf[i:i + 1].reshape(1, 2, 4, 2, 4, 288).mean((2, 4))
                        p1 = feat[i:i + 1].reshape(1, 2, 4, 2, 4, 288).mean((2, 4))
                        rows.append({'seed': seed,
                             'subject_id': int(sid),
                             'input': inputmode,
                             'variant': mode,
                             'MSE': float((v - target).square().mean()),
                             'prediction_change_RMS_FC': float((v - orig).square().mean().sqrt()),
                             'prediction_relative_L2': metrics(orig[None],
                             v[None])['relative_L2'],
                             'feature_relative_L2': metrics(bf[i:i + 1],
                             feat[i:i + 1])['relative_L2'],
                             'readout_relative_L2': metrics(p0,
                             p1)['relative_L2'],
                             'readout_cosine': metrics(p0,
                             p1)['cosine']})
            put('partial_results.json', {'rows': rows, 'local': local})
            print(json.dumps({'seed': seed, 'subjects_done': start + len(qq), 'temp': temps[-1]}), flush=True)
        assert before == w.old.state_hash(m)
        put(f'seed{seed}_receipt.json',
             {'weights_unchanged': True,
             'state_sha256': before,
             'checkpoint_sha256': w.sha(path),
             'production_control_maxabs': max(checks)})
        del m, ck
        torch.cuda.empty_cache()
put('results.json',
     {'status': 'COMPLETE',
     'rows': rows,
     'local': local,
     'max_control_difference': max(checks),
     'max_GPU1_temperature': max(temps),
     'training_performed': False})
print('COMPLETE', flush=True)
