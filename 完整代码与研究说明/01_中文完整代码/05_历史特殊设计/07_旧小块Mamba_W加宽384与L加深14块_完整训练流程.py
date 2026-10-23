# 07_旧小块Mamba_W加宽384与L加深14块_完整训练流程
# 归档主题：主题03
# W/L实际结构变体的完整装配和训练，避免用当前脑区模型替代历史容量实验。
# 输入/输出：共同116×116 FC、4×4重叠patch；W末端8×8×384，L仍8×8×288。
# 本实验改动：W通道48/96/192/384，首解码扩展384→144，后续解码宽度保留；L通道不变，深度2/2/8/2；均先将旧解码器首组6块减至2块。
# 参数在哪里：make()的W/L分支与同目录protocol.json；不能把W说成全编码器解码器同步加宽。
# 证据范围：旧CPAC966内部开发的预定3折×双种子；与B基线比较，不是完整外层十折确认。
# 源文件：03_数据增强与编码器结构/cpac966_encoder_wl_20261022/worker.py
# 原始字节 SHA256：700425f4e85b9853477357f5df9f49e835e7507e2901da229986227152936e24
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

import pathlib, sys, os, json, hashlib, time, signal, argparse
R = pathlib.Path(__file__).resolve().parent
S = R / 'source'
sys.path[:0] = [str(S), str(S / 'adapters')]
import numpy as np, torch
import train_fold as old
from common import put, append, sha, read
from augment import Augment
P = read(R / 'protocol.json')
STOP = False

def sig(*a):
    global STOP
    STOP = True
signal.signal(signal.SIGTERM, sig)
signal.signal(signal.SIGUSR1, sig)

def hash_encoder(m):
    h = hashlib.sha256()
    for n, p in sorted(m.encoder.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def make(f, seed, arm):
    if seed == 1:
        m = old.make('MAE', f)
    else:
        m = old.make_model('MAE',
             'A2',
             1,
             dict(initialization_seed=2026102200 + f + 1000000 * (seed - 1),
             classification_seed=2026102200 + f + 800000,
             mechanisms_path=str(S / 'model_source/code_v1/mechanisms.py'),
             mechanisms_sha256=read(S / 'protocol.json')['mechanisms_sha256'],
             base_code_path=str(S)))[0]
    m.decoder.blocks[0] = torch.nn.ModuleList(list(m.decoder.blocks[0])[:2])
    if arm in ['W', 'L']:
        mod = sys.modules[type(m.encoder).__module__]
        shared = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.random.fork_rng():
            torch.manual_seed(2026102200 + f + 1000000 * (seed - 1) + 7000000)
            if arm == 'L':
                for shift in [0, 2]:
                    m.encoder.stages[2].blocks.append(mod.OfficialWindowBlock2D(144, 4, shift))
                m.encoder.depths = (2, 2, 8, 2)
                m.depths = (2, 2, 8, 2)
            else:
                m.embed_dim = 48
                m.value_projection = torch.nn.Linear(16, 48)
                m.mask_projection = torch.nn.Linear(16, 48, bias=False)
                m.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 48))
                enc = m.encoder
                enc.stage_dims = (48, 96, 192, 384)
                enc.stages = torch.nn.ModuleList()
                enc.pos_embeds = torch.nn.ParameterList()
                for i, (dim, dep, shape) in enumerate(zip(enc.stage_dims, enc.depths, enc.stage_shapes)):
                    pos = torch.nn.Parameter(torch.empty(1, *shape, dim))
                    torch.nn.init.trunc_normal_(pos, std=0.02)
                    enc.pos_embeds.append(pos)
                    enc.stages.append(mod.OfficialStage2D(dim, shape, dep, 4, i < 3, full=i == 3))
                m.decoder.first_patch_expanding = mod.OfficialExpand2D(384, 144, m.encoder.stage_shapes[2])
        new = m.state_dict()
        for k, v in shared.items():
            if k in new and new[k].shape == v.shape:
                new[k] = v
        m.load_state_dict(new)
    return m

def masks(m, ids, epoch, namespace, ratio):
    return torch.cat([m.sample_mask(1,
         device=torch.device('cpu'),
         mask_ratio=ratio,
         generator=torch.Generator().manual_seed(int(hashlib.sha256(f'{namespace}|{sid}|{epoch}'.encode()).hexdigest()[:15],
         16))) for sid in ids])

def run(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = np.load(S / 'data/FC.npy', mmap_mode='r')
    ids = np.load(S / 'data/IDs.npy')
    split = read(S / 'splits.json')[a.fold]
    tr = np.array(split['inner_train'])
    va = np.array(split['inner_validation'])
    te = np.array(split['outer_test'])
    assert not set(tr) & set(va) and (not set(tr) & set(te)) and (not set(va) & set(te))
    job = f'{a.arm}_f{a.fold}_s{a.seed}'
    out = R / a.mode / (a.tag or job)
    out.mkdir(parents=True, exist_ok=True)
    norm = {'mean': float(raw[tr].mean()), 'std': float(raw[tr].std()), 'fit_subject_ids': ids[tr].tolist()}
    prior = np.asarray(raw[tr]).mean(0)

    def image(q):
        return torch.from_numpy(((raw[q] - norm['mean']) / norm['std']).astype('float32'))[:, None].cuda()
    m = make(a.fold, a.seed, a.arm)
    enc = hash_encoder(m)
    init = old.state_hash(m)
    params = sum((p.numel() for p in m.parameters()))
    dec = sum((p.numel() for p in m.decoder.parameters()))
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-08)
    cfg = dict(job=job,
         arm=a.arm,
         fold=a.fold,
         seed=a.seed,
         mode=a.mode,
         epochs=a.epochs,
         mask_ratio=P['arms'][a.arm]['mask_ratio'],
         lr=0.0001,
         batch=8,
         weight_decay=0.01,
         gradient_clip=1,
         encoder_initial_sha256=enc,
         initial_parameters_sha256=init,
         parameters=params,
         decoder_parameters=dec,
         norm=norm,
         validation_ids=ids[va].tolist(),
         outer_test_accessed=False,
         protocol_sha256=sha(R / 'protocol.json'))
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == cfg
    else:
        put(out / 'config.json', cfg)

    @torch.no_grad()
    def evaluate(ix, ratio, views=3):
        m.eval()
        rows = []
        mh = hashlib.sha256()
        for view in range(views):
            for start in range(0, len(ix), 8):
                q = ix[start:start + 8]
                mask = masks(m, ids[q], view, 'cpac966_fullcv_fixed_eval_v1', ratio)
                mh.update(mask.numpy().tobytes())
                z = m(image(q), pixel_mask=mask.cuda())
                score = z['score_mask'][:, 0].cpu().numpy()
                pr = z['prediction'][:, 0].cpu().numpy() * norm['std'] + norm['mean']
                assert np.array_equal(score,
                     score.transpose(0,
                     2,
                     1)) and (not np.diagonal(score,
                     axis1=1,
                     axis2=2).any())
                for k, i in enumerate(q):
                    s = score[k]
                    t = raw[i][s]
                    p = pr[k][s]
                    res = t - prior[s]
                    predres = p - prior[s]
                    den = np.sqrt(np.mean(res ** 2))
                    corr = np.corrcoef(res,
                         predres)[0,
                         1] if np.std(predres) > 1e-12 and np.std(res) > 1e-12 else 0.0
                    rows.append(dict(subject_id=str(ids[i]),
                         view=view,
                         MSE=float(np.mean((p - t) ** 2)),
                         sym_MSE=float(np.mean((((pr[k] + pr[k].T) / 2)[s] - t) ** 2)),
                         meanFC_MSE=float(np.mean(res ** 2)),
                         residual_correlation=float(corr),
                         residual_RMS_ratio=float(np.sqrt(np.mean(predres ** 2)) / max(den,
                         1e-12)),
                         score_fraction=float(s.sum() / (116 * 115))))
        return dict(sym_MSE=float(np.mean([r['sym_MSE'] for r in rows])),
             MSE=float(np.mean([r['MSE'] for r in rows])),
             meanFC_MSE=float(np.mean([r['meanFC_MSE'] for r in rows])),
             residual_correlation=float(np.mean([r['residual_correlation'] for r in rows])),
             residual_RMS_ratio=float(np.mean([r['residual_RMS_ratio'] for r in rows])),
             score_fraction=float(np.mean([r['score_fraction'] for r in rows])),
             mask_sha256=mh.hexdigest(),
             subjects=len(ix),
             views=views,
             rows=rows)
    if a.mode == 'reuse':
        path = pathlib.Path('/home/wenhao/neurostorm_2d/experiments/cpac966_decoder_mask_20261022/runs') / f'B_f{a.fold}_s{a.seed}/epoch_100.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['epoch'] == 100 and ck['norm'] == norm and (ck['config']['initial_parameters_sha256'] == init)
        assert ck['config']['validation_ids'] == ids[va].tolist()
        m.load_state_dict(ck['model'])
        r = evaluate(va, 0.5)
        r75 = evaluate(va, 0.75)
        prev = read(path.parent / 'validation_mask50_subjects.json')
        assert abs(r['MSE'] - prev['MSE']) < 1e-07
        put(out / 'validation_mask50_subjects.json', r)
        put(out / 'validation_mask75_subjects.json', r75)
        put(out / 'completion.json',
             dict(status='COMPLETE_REUSED_BASELINE',
             source=str(path),
             sha256=sha(path),
             epochs=100,
             baseline_MSE_matches=True,
             inner_train_count=len(tr)))
        return
    history = []
    start = 1
    steps = 0
    elapsed_before = 0
    if (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == cfg
        m.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        history = ck['history']
        start = ck['epoch'] + 1
        steps = ck['steps']
        elapsed_before = ck['elapsed_seconds']
        del ck
        (out / 'history.jsonl').write_text(''.join((json.dumps(x) + '\n' for x in history)))
        if (out / 'gradients.jsonl').exists():
            lines = [l for l in (out / 'gradients.jsonl').read_text().splitlines() if json.loads(l)['step'] <= steps]
            (out / 'gradients.jsonl').write_text('\n'.join(lines) + '\n')
    aug = Augment(a.arm, a.fold, a.seed, ids, raw, norm)

    def training_forward(q, epoch, mask):
        x, t, ar = aug.arrays(q, epoch, mask.cpu().numpy())
        x = torch.from_numpy(((x - norm['mean']) / norm['std']).astype('float32'))[:, None].cuda()
        t = torch.from_numpy(((t - norm['mean']) / norm['std']).astype('float32'))[:, None].cuda()
        z = m(x, pixel_mask=mask)
        z['target'] = t.detach()
        return (z, ar)
    beg = time.monotonic()
    ratio = P['arms'][a.arm]['mask_ratio']
    namespace = f'cpac966_fullcv_train_f{a.fold}_s{a.seed}'
    if a.mode == 'benchmark':
        m.train()
        begin = time.monotonic()
        times = []
        for step in range(a.steps):
            if STOP:
                break
            q = tr[(np.arange(8) + step * 8) % len(tr)]
            t = time.monotonic()
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], step + 1, namespace, ratio).cuda()
            z, ar = training_forward(q, step + 1, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            old.gradstats(m)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - t)
        put(out / 'benchmark.json',
             dict(completed_steps=len(times),
             warmup_steps=10,
             seconds=time.monotonic() - begin,
             steady_seconds=sum(times[10:]),
             steady_samples=max(0,
             len(times) - 10) * 8,
             initial_encoder_sha256=enc,
             decoder_parameters=dec))
        return
    for epoch in range(start, a.epochs + 1):
        if STOP:
            return
        m.train()
        order = np.random.default_rng(2026102100 + a.fold * 10000 + epoch + (a.seed - 1) * 1000000).permutation(tr)
        total = 0.0
        seen = 0
        ebeg = time.monotonic()
        for off in range(0, len(order), 8):
            if STOP:
                put(out / 'paused.json',
                     dict(epoch_rolled_back=epoch,
                     last_complete_epoch=epoch - 1,
                     reason='signal',
                     time_unix=time.time()))
                return
            q = order[off:off + 8]
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], epoch, namespace, ratio).cuda()
            z, ar = training_forward(q, epoch, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            gs = old.gradstats(m)
            assert gs['encoder_preclip_l2'] > 0 and gs['decoder_preclip_l2'] > 0
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True))
            before = [p.detach().clone() for p in m.parameters()] if steps % 10 == 0 else None
            opt.step()
            steps += 1
            rec = dict(epoch=epoch, step=steps, loss=float(loss), clipped=gn > 1, **gs)
            if before is not None:
                rec['parameter_update_l2'] = float(torch.sqrt(sum(((p.detach() - b).double().square().sum() for p,
                     b in zip(m.parameters(),
                     before)),
                     torch.zeros((),
                     device='cuda'))))
                del before
            rec['augmentation'] = ar
            append(out / 'gradients.jsonl', rec)
            total += float(loss) * len(q)
            seen += len(q)
        row = dict(epoch=epoch,
             online_training_loss=total / seen,
             online_train_raw_MSE=total / seen * norm['std'] ** 2,
             steps=steps,
             train_epoch_seconds=time.monotonic() - ebeg,
             elapsed_seconds=elapsed_before + time.monotonic() - beg)
        if epoch % 10 == 0 or epoch == a.epochs:
            for label, ix in [('train', np.array(sorted(tr)[:64])), ('validation', va)]:
                ev = evaluate(ix, 0.5)
                row[label + '_mask50'] = {k: v for k, v in ev.items() if k != 'rows'}
                if epoch == a.epochs:
                    put(out / f'{label}_mask50_subjects.json', ev)
            q = np.array(sorted(tr)[:8])
            m.eval()
            opt.zero_grad(set_to_none=True)
            z = m(image(q), pixel_mask=masks(m, ids[q], 0, 'cpac966_fixed_gradient_probe_v1', 0.5).cuda())
            old.loss_mae(z).backward()
            row['fixed_probe_gradient'] = old.gradstats(m)
            opt.zero_grad(set_to_none=True)
        if epoch % 20 == 0 or epoch == a.epochs:
            ev = evaluate(va, 0.75)
            row['validation_mask75'] = {k: v for k, v in ev.items() if k != 'rows'}
            if epoch == a.epochs:
                put(out / 'validation_mask75_subjects.json', ev)
        history.append(row)
        append(out / 'history.jsonl', row)
        old.save(out / 'last.pt',
             dict(model=m.state_dict(),
             optimizer=opt.state_dict(),
             epoch=epoch,
             steps=steps,
             config=cfg,
             norm=norm,
             history=history,
             elapsed_seconds=elapsed_before + time.monotonic() - beg))
        if epoch in [20, 40, 60, 80, 100]:
            old.save(out / f'epoch_{epoch}.pt', dict(model=m.state_dict(), epoch=epoch, norm=norm, config=cfg))
        put(out / 'status.json',
             dict(state='RUNNING',
             epoch=epoch,
             target=a.epochs,
             steps=steps,
             time_unix=time.time()))
        if a.stop_after and epoch == a.stop_after:
            return
    put(out / 'completion.json',
         dict(status='COMPLETE',
         epochs=a.epochs,
         steps=steps,
         last_sha256=sha(out / 'last.pt'),
         time_unix=time.time()))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['G0', 'G1', 'G2', 'W', 'L'], required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mode', choices=['pilot', 'benchmark', 'runs', 'reuse'], default='runs')
    p.add_argument('--tag')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--stop-after', type=int, default=0)
    run(p.parse_args())
